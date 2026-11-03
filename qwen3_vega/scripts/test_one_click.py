"""Offline regression tests for orchestration. Accelerator jobs are replaced at the process boundary."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import one_click


class OneClickTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name) / "disk with spaces"
        (self.work / "logs").mkdir(parents=True)
        self.calls = []
        self.fatal = False
        self.interrupt = False
        self.env = patch.dict(
            os.environ, CUDA_VISIBLE_DEVICES="a,b,c,d", VSI_BENCH_DIR=str(self.work / "vsi")
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        proc = patch.object(one_click.subprocess, "run", side_effect=self.run_stage)
        proc.start()
        self.addCleanup(proc.stop)

    def run_stage(self, cmd, **opts):
        self.calls.append(cmd)

        def arg(name):
            return cmd[cmd.index(name) + 1]

        if "--benchmark-steps" in cmd:
            width, stride = int(arg("--micro-batch")), int(arg("--checkpoint-every"))
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
                {"rank": i, "min_device_free_gib": 3 if speed == 100 and i == 3 else 12}
                for i in range(4)
            ]
            one_click.write_json(
                out / "throughput.json", dict(qa_per_s=speed, qa=256, measured_steps=4, ranks=ranks)
            )
        elif "--micro-batch" in cmd or cmd[1].endswith("pipeline.py"):
            out = Path(arg("--output"))
            out.mkdir(exist_ok=True)
            if self.interrupt:
                one_click.write_json(out / "step-001165.json", dict(step=1165))
                raise subprocess.CalledProcessError(1, cmd)
            (out / "sp16k_vsi").mkdir()
            one_click.write_json(
                out / "sp16k_vsi" / "results.json",
                dict(questions=5130, overall={"baseline": 1, "sp": 2}),
            )
        return subprocess.CompletedProcess(cmd, 0)

    def test_select_and_resume(self):
        self.interrupt = True
        with self.assertRaises(subprocess.CalledProcessError):
            one_click.launch(self.work)
        choice = json.loads((self.work / "speed_selection.json").read_text())
        self.assertEqual(
            (choice["winner"]["micro_batch"], choice["winner"]["checkpoint_every"]), (2, 1)
        )
        self.assertEqual([t["status"] for t in choice["trials"]][-2:], ["low_headroom", "cuda_oom"])
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

    def test_non_oom_failure_stops(self):
        self.fatal = True
        with self.assertRaisesRegex(RuntimeError, "bad data"):
            one_click.launch(self.work)
        self.assertFalse((self.work / "speed_selection.json").exists())


if __name__ == "__main__":
    unittest.main()
