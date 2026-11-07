"""Offline regression tests for orchestration. Accelerator jobs are replaced at the process boundary."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import one_click
from test_numerics_log import diagnostic_log


class OneClickTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name) / "disk with spaces"
        (self.work / "logs").mkdir(parents=True)
        self.calls = []
        self.fatal = False
        self.failure_message = None
        self.diagnostic_message = "Diagnostic replay completed successfully"
        self.diagnostic_rc = 0
        self.interrupt = False
        self.missing_acceptance = False
        self.bad_acceptance = False
        self.env = patch.dict(
            os.environ, CUDA_VISIBLE_DEVICES="a,b,c,d", VSI_BENCH_DIR=str(self.work / "vsi"), VEGA_NO_OVERLAP="0"
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("VEGA_MIN_FREE_GIB", None)
        os.environ.pop("VEGA_REUSE_SWEEP", None)
        proc = patch.object(one_click.subprocess, "run", side_effect=self.run_stage)
        proc.start()
        self.addCleanup(proc.stop)

    def run_stage(self, cmd, **opts):
        self.calls.append(cmd)

        def arg(name):
            return cmd[cmd.index(name) + 1]

        if "--benchmark-steps" in cmd:
            width, stride = int(arg("--micro-batch")), int(arg("--checkpoint-every"))
            if "--diagnose-numerics" in cmd:
                print(self.diagnostic_message, file=opts["stdout"])
                if self.diagnostic_rc:
                    return subprocess.CompletedProcess(cmd, self.diagnostic_rc)
            elif self.failure_message is not None:
                opts["stdout"].write(self.failure_message)
                return subprocess.CompletedProcess(cmd, 1)
            if self.fatal or (width, stride) == (2, 0):
                print(
                    "bad data" if self.fatal else "torch.OutOfMemoryError: CUDA out of memory",
                    file=opts["stdout"],
                )
                return subprocess.CompletedProcess(cmd, 1)
            out = Path(arg("--output"))
            out.mkdir()
            speed = 100 if (width, stride) == (2, 2) else 10 * width + stride
            ranks = [
                {"rank": i, "min_device_free_gib": 3 if speed == 100 and i == 3 else 12,
                 "training_acceptance": {"accepted": True, "expected_steps": 5, "final_engine_global_steps": 5}}
                for i in range(4)
            ]
            if self.missing_acceptance:
                ranks[3].pop("training_acceptance")
            if self.bad_acceptance:
                ranks[2]["training_acceptance"]["accepted"] = False
            one_click.write_json(
                out / "throughput.json", dict(qa_per_s=speed, qa=256, measured_steps=4, ranks=ranks,
                                              protocol={"overlap_comm": "--no-overlap" not in cmd})
            )
        elif "--micro-batch" in cmd or cmd[1].endswith("pipeline.py"):
            out = Path(arg("--output"))
            out.mkdir(exist_ok=True)
            if self.interrupt:
                one_click.write_json(out / "protocol.json", {"overlap_comm": "--no-overlap" not in cmd})
                one_click.write_json(out / "step-001165.json", dict(step=1165))
                raise subprocess.CalledProcessError(1, cmd)
            (out / "sp16k_vsi").mkdir()
            one_click.write_json(
                out / "sp16k_vsi" / "results.json",
                dict(questions=5130, overall={"baseline": 1, "sp": 2}),
            )
        return subprocess.CompletedProcess(cmd, 0)

    def completed_sweep(self, name="sweep-reuse"):
        source = self.work / "logs" / name
        source.mkdir()
        data = self.work / "data"
        data.mkdir(exist_ok=True)
        hashes = {}
        for part in ("train", "tune"):
            raw = json.dumps([{"id": part}]).encode()
            (data / (part + ".json")).write_bytes(raw)
            hashes[part + "_sha256"] = hashlib.sha256(raw).hexdigest()
        measurements = [
            (1, 1, .851647304, 6.78967285),
            (1, 2, .876562407, 1.025390625),
            (1, 0, .900551552, 2.3984375),
            (2, 1, .838145732, 6.787719726),
            (2, 2, .88, 1.2),
            (2, 0, None, None),
        ]
        trials = []
        for width, stride, speed, free in measurements:
            out = source / f"m{width}-c{stride}"
            trial = dict(micro_batch=width, checkpoint_every=stride,
                         log=str(out.with_suffix(".log")), status="cuda_oom")
            if speed is not None:
                out.mkdir()
                protocol = dict(hashes, world=4, micro_batch=width, effective_batch=64,
                                smoke=False, cpu_adam=False, benchmark_steps=5, overlap_comm=False)
                if stride != 1:
                    protocol["checkpoint_every"] = stride
                report = dict(qa_per_s=speed, qa=256, measured_steps=4, warmup_steps=1,
                              protocol=protocol, ranks=[dict(
                                  rank=i, min_device_free_gib=free + .1*i,
                                  training_acceptance=dict(accepted=True, expected_steps=5,
                                                           final_engine_global_steps=5))
                                  for i in range(4)])
                one_click.write_json(out / "throughput.json", report)
                trial.update(qa_per_s=speed, min_free_gib=free, status="low_headroom",
                             report=str(out / "throughput.json"))
            trials.append(trial)
        one_click.write_json(source / "trials.json", trials)
        return source

    def test_reuse_sweep_at_four_gib_avoids_benchmark_and_preserves_source(self):
        source = self.completed_sweep()
        original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*.json")}
        os.environ.update(VEGA_NO_OVERLAP="1", VEGA_MIN_FREE_GIB="4", VEGA_REUSE_SWEEP=str(source))
        one_click.launch(self.work)
        choice = json.loads((self.work / "speed_selection.json").read_text())
        self.assertEqual((choice["winner"]["micro_batch"], choice["winner"]["checkpoint_every"]), (1, 1))
        self.assertAlmostEqual(choice["winner"]["qa_per_s"], .851647304)
        self.assertEqual(choice["min_free_gib"], 4.)
        self.assertEqual(choice["reused_sweep"], str(source.resolve()))
        self.assertEqual([t["status"] for t in choice["trials"]],
                         ["eligible", "low_headroom", "low_headroom", "eligible", "low_headroom", "cuda_oom"])
        self.assertFalse(any("--benchmark-steps" in cmd for cmd in self.calls))
        self.assertIn("--no-overlap", self.calls[-1])
        for path, before in original.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_reuse_rejects_incomplete_or_incompatible_measurements(self):
        for defect in ("incomplete", "hash", "overlap", "acceptance", "nan_headroom", "diagnostic", "batch"):
            with self.subTest(defect=defect):
                source = self.completed_sweep("sweep-" + defect)
                path = source / "m1-c1" / "throughput.json"
                report = json.loads(path.read_text())
                if defect == "incomplete":
                    trials = json.loads((source / "trials.json").read_text())
                    one_click.write_json(source / "trials.json", trials[:-1])
                elif defect == "hash":
                    report["protocol"]["train_sha256"] = "different-dataset"
                elif defect == "overlap":
                    report["protocol"]["overlap_comm"] = True
                elif defect == "acceptance":
                    report["ranks"][2]["training_acceptance"]["accepted"] = False
                elif defect == "nan_headroom":
                    report["ranks"][3]["min_device_free_gib"] = float("nan")
                elif defect == "diagnostic":
                    report["protocol"]["diagnose_numerics"] = True
                else:
                    report["protocol"]["effective_batch"] = 32
                path.write_text(json.dumps(report))
                os.environ.update(VEGA_NO_OVERLAP="1", VEGA_MIN_FREE_GIB="4", VEGA_REUSE_SWEEP=str(source))
                self.calls.clear()
                with self.assertRaises(ValueError):
                    one_click.launch(self.work)
                self.assertFalse(any("--micro-batch" in cmd for cmd in self.calls))
                self.assertFalse((self.work / "speed_selection.json").exists())
                self.assertFalse(list((self.work / "runs").glob("full-*")))

    def test_new_headroom_and_reuse_do_not_change_existing_checkpoint_config(self):
        os.environ["VEGA_NO_OVERLAP"] = "1"
        self.interrupt = True
        with self.assertRaises(subprocess.CalledProcessError):
            one_click.launch(self.work)
        selection = self.work / "speed_selection.json"
        original = selection.read_bytes()
        source = self.completed_sweep()
        os.environ.update(VEGA_MIN_FREE_GIB="4", VEGA_REUSE_SWEEP=str(source))
        self.interrupt = False
        self.calls.clear()
        one_click.launch(self.work)
        command = self.calls[-1]
        self.assertEqual(command[command.index("--micro-batch") + 1], "2")
        self.assertEqual(command[command.index("--checkpoint-every") + 1], "1")
        self.assertEqual(command[-2:], ["--resume", "step-001165"])
        self.assertFalse(any("--benchmark-steps" in cmd for cmd in self.calls))
        self.assertEqual(selection.read_bytes(), original)

    def test_invalid_headroom_stops_before_preparation(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value):
                os.environ["VEGA_MIN_FREE_GIB"] = value
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    one_click.launch(self.work)
                self.assertEqual(self.calls, [])

    def test_select_and_resume(self):
        self.interrupt = True
        with self.assertRaises(subprocess.CalledProcessError):
            one_click.launch(self.work)
        choice = json.loads((self.work / "speed_selection.json").read_text())
        self.assertEqual(
            (choice["winner"]["micro_batch"], choice["winner"]["checkpoint_every"]), (2, 1)
        )
        self.assertEqual([t["status"] for t in choice["trials"]][-2:], ["low_headroom", "cuda_oom"])
        self.assertFalse(any("--diagnose-numerics" in cmd for cmd in self.calls))
        self.assertTrue(any("--sp-text" in cmd for cmd in self.calls))
        self.interrupt = False
        self.calls.clear()
        one_click.launch(self.work)
        self.assertEqual(self.calls[-1][-2:], ["--resume", "step-001165"])
        self.assertFalse(any("--benchmark-steps" in cmd for cmd in self.calls))
        self.calls.clear()
        one_click.launch(self.work)
        self.assertEqual(self.calls, [])

    def test_finished_base_only_runs_post_training(self):
        out = self.work / "runs" / "full-0001"
        out.mkdir(parents=True)
        one_click.write_json(out / "step-004657.json", {"best": {"step": 2329, "tune_ce": 1.2}})
        one_click.write_json(
            self.work / "speed_selection.json", {"winner": {"micro_batch": 1, "checkpoint_every": 1}}
        )
        one_click.launch(self.work)
        self.assertTrue(self.calls[-1][1].endswith("pipeline.py"))
        self.assertEqual(json.loads((out / "selected.json").read_text())["tag"], "step-002329")

    def test_no_overlap_reaches_every_trial_training_and_resume(self):
        os.environ["VEGA_NO_OVERLAP"] = "1"
        self.interrupt = True
        with self.assertRaises(subprocess.CalledProcessError):
            one_click.launch(self.work)
        training = [cmd for cmd in self.calls if "--micro-batch" in cmd]
        self.assertEqual(len(training), 7)
        self.assertTrue(all("--no-overlap" in cmd for cmd in training))
        choice = json.loads((self.work / "speed_selection.json").read_text())
        self.assertIs(choice["overlap_comm"], False)
        self.assertEqual(choice["acceptance_version"], 1)
        del os.environ["VEGA_NO_OVERLAP"]
        self.interrupt = False
        self.calls.clear()
        one_click.launch(self.work)
        self.assertIn("--no-overlap", self.calls[-1])
        self.assertEqual(self.calls[-1][-2:], ["--resume", "step-001165"])
        self.assertFalse(any("--benchmark-steps" in cmd for cmd in self.calls))

    def test_resume_rejects_changing_checkpoint_overlap_before_preparation(self):
        self.interrupt = True
        with self.assertRaises(subprocess.CalledProcessError):
            one_click.launch(self.work)
        self.calls.clear()
        os.environ["VEGA_NO_OVERLAP"] = "1"
        with self.assertRaisesRegex(ValueError, "differs from the existing checkpoint"):
            one_click.launch(self.work)
        self.assertEqual(self.calls, [])

    def test_old_selection_is_remeasured_and_preserved(self):
        old = dict(gpus="a,b,c,d", winner={"micro_batch": 1, "checkpoint_every": 1})
        one_click.write_json(self.work / "speed_selection.json", old)
        original = (self.work / "speed_selection.json").read_bytes()
        os.environ["VEGA_NO_OVERLAP"] = "1"
        one_click.launch(self.work)
        self.assertEqual(sum("--benchmark-steps" in cmd for cmd in self.calls), 6)
        backups = list((self.work / "logs").glob("speed-selection-prior-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertIn("--no-overlap", self.calls[-1])

    def test_incomplete_or_failed_update_acceptance_never_starts_training(self):
        for flag in ("missing_acceptance", "bad_acceptance"):
            with self.subTest(flag=flag):
                setattr(self, flag, True)
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, "Five real optimizer updates"):
                    one_click.launch(self.work)
                self.assertFalse((self.work / "speed_selection.json").exists())
                self.assertFalse(list((self.work / "runs").glob("full-*")))
                self.assertEqual(sum("--benchmark-steps" in cmd for cmd in self.calls), 1)
                setattr(self, flag, False)

    def test_overlap_failure_replay_keeps_flag_and_stays_stopped(self):
        os.environ["VEGA_NO_OVERLAP"] = "1"
        self.failure_message = "[rank2]: FloatingPointError: Nonfinite loss at step 2\n"
        with self.assertRaisesRegex(RuntimeError, "training remains stopped"):
            one_click.launch(self.work)
        commands = [cmd for cmd in self.calls if "--benchmark-steps" in cmd]
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--no-overlap" in cmd for cmd in commands))
        self.assertFalse(list((self.work / "runs").glob("full-*")))

    def test_non_oom_failure_stops(self):
        self.fatal = True
        with self.assertRaisesRegex(RuntimeError, "bad data"):
            one_click.launch(self.work)
        self.assertFalse((self.work / "speed_selection.json").exists())
        self.assertEqual(sum("--benchmark-steps" in cmd for cmd in self.calls), 1)
        self.assertFalse(any("--diagnose-numerics" in cmd for cmd in self.calls))

    def test_nonfinite_gets_one_replay_even_when_replay_succeeds(self):
        for suffix, expected_steps in [("at step 2", 2), ("at step 1", 2), ("before optimizer update", 5)]:
            with self.subTest(suffix=suffix):
                self.calls.clear()
                self.failure_message = f"[rank2]: FloatingPointError: Nonfinite loss {suffix}\n"
                with self.assertRaisesRegex(RuntimeError, "training remains stopped") as caught:
                    one_click.launch(self.work)
                commands = [cmd for cmd in self.calls if "--benchmark-steps" in cmd]
                self.assertEqual(len(commands), 2)
                original, diagnostic = commands
                expected = original.copy()
                expected[expected.index("--output") + 1] += "-numerics"
                expected[expected.index("--benchmark-steps") + 1] = str(expected_steps)
                expected.append("--diagnose-numerics")
                self.assertEqual(diagnostic, expected)
                original_out = Path(original[original.index("--output") + 1])
                self.assertIn(str(original_out.with_suffix(".log")), str(caught.exception))
                self.assertIn(str(original_out.with_name(original_out.name + "-numerics.log")), str(caught.exception))
                self.assertIn(self.failure_message.strip(), str(caught.exception))
                self.assertIn(self.diagnostic_message, str(caught.exception))
                self.assertFalse((self.work / "speed_selection.json").exists())
                self.assertEqual(list((self.work / "runs").glob("full-*")), [])

    def test_nonfinite_replay_error_is_preserved_without_repeating(self):
        self.failure_message = "[rank0]: FloatingPointError: Nonfinite gradient at step 2\n"
        self.diagnostic_message = "\n".join([
            "[rank0]: Traceback (most recent call last):",
            "[rank0]: backward source context",
            "[rank0]: RuntimeError: Function 'AttentionBackward' returned nan values in its 0th output.",
            *("NCCL shutdown" for _ in range(300)),
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
        ])
        self.diagnostic_rc = 1
        with self.assertRaises(RuntimeError) as caught:
            one_click.launch(self.work)
        self.assertIn(self.failure_message.strip(), str(caught.exception))
        self.assertIn("backward source context", str(caught.exception))
        self.assertIn("Function 'AttentionBackward' returned nan", str(caught.exception))
        self.assertNotIn("ChildFailedError", str(caught.exception))
        commands = [cmd for cmd in self.calls if "--benchmark-steps" in cmd]
        self.assertEqual(len(commands), 2)
        logs = list((self.work / "logs").glob("sweep-*/m1-c1-numerics.log"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].read_text(), self.diagnostic_message + "\n")
        self.assertFalse((self.work / "speed_selection.json").exists())

    def test_nonfinite_diagnostic_still_stops_after_one_replay(self):
        self.failure_message = "[rank0]: FloatingPointError: Nonfinite loss at step 2\n"
        self.diagnostic_message = "[rank1]: FloatingPointError: Nonfinite gradient at step 1, group=0"
        self.diagnostic_rc = 1
        with self.assertRaises(RuntimeError) as caught:
            one_click.launch(self.work)
        self.assertIn(self.diagnostic_message, str(caught.exception))
        self.assertEqual(sum("--benchmark-steps" in cmd for cmd in self.calls), 2)
        self.assertEqual(sum("--diagnose-numerics" in cmd for cmd in self.calls), 1)
        self.assertFalse((self.work / "speed_selection.json").exists())

    def test_anomaly_forward_traceback_and_sample_context_reach_launcher_error(self):
        self.failure_message = "[rank2]: FloatingPointError: Nonfinite optimizer_gradient at step 1\n"
        self.diagnostic_message = diagnostic_log()
        self.diagnostic_rc = 1
        with self.assertRaises(RuntimeError) as caught:
            one_click.launch(self.work)
        summary = str(caught.exception)
        self.assertIn("FORWARD TRACEBACK (rank 2)", summary)
        self.assertIn("forward_frame_0.py", summary)
        self.assertIn("train_model.py", summary)
        self.assertIn("failing-qa", summary)
        self.assertIn("MmBackward0' returned nan", summary)
        self.assertNotIn("other-qa", summary)
        self.assertEqual(sum("--diagnose-numerics" in cmd for cmd in self.calls), 1)

    def test_uuid_error_survives_nccl_cleanup_and_torchrun_wrapper(self):
        error = "[rank3]: ValueError: invalid literal for int() with base 10: 'GPU-uuid-3'"
        self.failure_message = "\n".join([
            "[rank3]: Traceback (most recent call last):",
            *[f"[rank3]: original context {i:02d}" for i in range(15)],
            error,
            *[f"[rank3]: following context {i}" for i in range(5)],
            *("NCCL shutdown: cleaning communicator and waiting for peers" for _ in range(300)),
            "Traceback (most recent call last):",
            '  File "/env/bin/torchrun", line 8, in <module>',
            "    raise ChildFailedError(name=entrypoint, failures=result.failures)",
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
            "train.py FAILED; rank: 3; exitcode: 1",
        ]) + "\n"
        with self.assertRaises(RuntimeError) as caught:
            one_click.launch(self.work)
        summary = str(caught.exception)
        self.assertIn(error, summary)
        self.assertIn("original context 01", summary)
        self.assertIn("following context 3", summary)
        self.assertNotIn("original context 00", summary)
        self.assertNotIn("following context 4", summary)
        self.assertNotIn("NCCL shutdown", summary)
        self.assertNotIn("ChildFailedError", summary)
        logs = list((self.work / "logs").glob("sweep-*/m1-c1.log"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].read_text(), self.failure_message)
        self.assertFalse((self.work / "speed_selection.json").exists())


if __name__ == "__main__":
    unittest.main()
