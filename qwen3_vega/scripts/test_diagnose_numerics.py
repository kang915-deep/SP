import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnose_numerics as diagnostic


class DiagnosticCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.disk = Path(self.temp.name)
        self.work = self.disk / "qwen3-vega"
        for name in ("env/bin", "upstream/VEGA-3D/src", "assets/qwen", "assets/wan", "data"):
            (self.work / name).mkdir(parents=True)
        (self.work / "env/bin/python").touch()
        (self.work / "data/media_ready.json").write_text("{}")
        self.original = self.work / "assets/qwen/retain.bin"
        self.original.write_bytes(b"keep existing model")
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()

    def tearDown(self):
        self.environ.stop()
        self.temp.cleanup()

    def invoke(self, rc=0, extra=()):
        seen = {}

        def run(command, **kwargs):
            seen.update(command=command, env=kwargs["env"])
            self.assertEqual(command[command.index("--diagnostic-micros") + 1], "2")
            self.assertIn("--diagnose-numerics", command)
            self.assertIn("--max-restarts=0", command)
            self.assertTrue(any(x.endswith("/train.py") for x in command))
            self.assertFalse(any("start.sh" in x or "pipeline.py" in x for x in command))
            if rc:
                out = Path(command[command.index("--output") + 1])
                report = out / "linear-rank2/first_nonfinite.json"
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({"name": "layers.7.mlp.up_proj", "phase": "G_incoming", "context": {"micro": 2}}))
            return type("Result", (), {"returncode": rc})()

        output = io.StringIO()
        with patch("sys.argv", ["diagnose", str(self.disk), "0,1,2,3", *extra]), \
                patch.object(diagnostic.subprocess, "check_output", side_effect=[f"GPU-test-{i}\n" for i in range(4)]), \
                patch.object(diagnostic.subprocess, "run", side_effect=run), contextlib.redirect_stdout(output):
            result = diagnostic.main()
        return result, output.getvalue(), seen

    def test_existing_install_bounded_and_immutable(self):
        result, output, seen = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(seen["env"]["CUDA_VISIBLE_DEVICES"], "GPU-test-0,GPU-test-1,GPU-test-2,GPU-test-3")
        self.assertEqual(self.original.read_bytes(), b"keep existing model")
        self.assertIn("Formal training remains stopped", output)
        self.assertEqual(len(list((self.work / "logs").glob("linear-diagnostic-*/command.json"))), 1)
        self.assertFalse(list(self.work.rglob("throughput.json")))

    def test_specific_failure_report_printed(self):
        result, output, _ = self.invoke(rc=1)
        self.assertEqual(result, 1)
        self.assertIn("FIRST_NONFINITE", output)
        self.assertIn("layers.7.mlp.up_proj", output)
        self.assertIn("G_incoming", output)
        self.assertNotIn("Prefix completed", output)

    def test_only_overlap_flag_changes_replay(self):
        _, _, original = self.invoke()
        _, output, changed = self.invoke(extra=("--no-overlap",))
        before = original["command"].copy()
        after = changed["command"].copy()
        before[before.index("--output") + 1] = "NEW_OUTPUT"
        after[after.index("--output") + 1] = "NEW_OUTPUT"
        self.assertEqual(after, before + ["--no-overlap"])
        self.assertEqual(original["env"], changed["env"])
        self.assertIn("overlap_comm=False", output)

    def test_reads_existing_incoming_gradient_without_mutating_logs(self):
        rank = self.work / "logs/linear-diagnostic-existing/result/linear-rank2"
        rank.mkdir(parents=True)
        x = dict(shape=[1, 2125, 4096], min=-286., max=43.25, finite=True)
        failure = dict(name="layers.35.mlp.up_proj", phase="dWeightTranspose_mm", forward_call=323,
                       context=dict(rank=2, micro=2), original_forward={"x": x})
        path = rank / "first_nonfinite.json"
        path.write_text(json.dumps(failure))
        records = [dict(name=failure["name"], forward_call=320, phase="G_incoming", stats=dict(min=-100., max=100., finite=True)),
                   dict(name=failure["name"], forward_call=323, phase="G_incoming", stats=dict(min=-.125, max=.25, finite=True))]
        events = rank / "events.jsonl"
        events.write_text("\n".join(json.dumps(event) for event in records) + "\n")
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, events)}
        _, output, _ = self.invoke(extra=("--no-overlap",))
        summary = json.loads(next(line.split(" ", 1)[1] for line in output.splitlines() if line.startswith("PREVIOUS_FAILURE ")))
        self.assertEqual(summary["incoming_gradient"]["max"], .25)
        self.assertEqual(summary["observed_dot_product_abs_bound"], 2125 * 286 * .25)
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, events)}, before)

    def test_active_workspace_not_restarted(self):
        with (self.work / "launch.lock").open("a") as lock:
            diagnostic.fcntl.flock(lock, diagnostic.fcntl.LOCK_EX | diagnostic.fcntl.LOCK_NB)
            with patch("sys.argv", ["diagnose", str(self.disk), "0,1,2,3"]), \
                    patch.object(diagnostic.subprocess, "run") as run:
                with self.assertRaises(BlockingIOError):
                    diagnostic.main()
                run.assert_not_called()

    def test_duplicate_devices_rejected_before_launch(self):
        with patch("sys.argv", ["diagnose", str(self.disk), "0,1,2,02"]), \
                patch.object(diagnostic.subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                diagnostic.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
