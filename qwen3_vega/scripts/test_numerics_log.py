"""Existing-log extraction only; no torch/model imports or training processes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from numerics_log import failure_summary


def diagnostic_log():
    return "\n".join([
        'NUMERICS_CONTEXT {"rank": 2, "step": 1, "micro": 1, "qids": ["old-qa"]}',
        'NUMERICS_CONTEXT {"rank": 2, "step": 1, "micro": 2, "qids": ["failing-qa"]}',
        '[rank1]: UserWarning: Error detected in OtherBackward. Traceback of forward call that caused the error:',
        '[rank1]:   File "other_rank.py", line 99, in forward',
        '[rank1]:  (Triggered internally at anomaly.cpp:122.)',
        '[rank2]: UserWarning: Error detected in MmBackward0. Traceback of forward call that caused the error:',
        *[f'[rank2]:   File "forward_frame_{i}.py", line {i}, in forward' for i in range(70)],
        'NUMERICS_CONTEXT {"rank": 3, "step": 1, "micro": 3, "qids": ["other-qa"]}',
        '[rank3]: Traceback (most recent call last):',
        '[rank2]:   File "/workspace/SP/qwen3_vega/train_model.py", line 39, in forward',
        '[rank2]:     logits = self.qwen.lm_head(hidden)',
        '[rank2]:  (Triggered internally at anomaly.cpp:122.)',
        '[rank2]: Traceback (most recent call last):',
        *[f'[rank2]:   File "backward_frame_{i}.py", line {i}, in backward' for i in range(30)],
        "[rank2]: RuntimeError: Function 'MmBackward0' returned nan values in its 0th output.",
        *["NCCL shutdown" for _ in range(300)],
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
    ]) + "\n"


class NumericsLogTest(unittest.TestCase):
    def test_full_forward_and_context_survive_long_backward_stack(self):
        summary = failure_summary(diagnostic_log())
        self.assertIn('"qids": ["failing-qa"]', summary)
        self.assertNotIn("old-qa", summary)
        self.assertNotIn("other-qa", summary)
        self.assertNotIn("other_rank.py", summary)
        self.assertIn("FORWARD TRACEBACK (rank 2)", summary)
        self.assertIn("forward_frame_0.py", summary)
        self.assertIn("forward_frame_69.py", summary)
        self.assertIn("train_model.py", summary)
        self.assertIn("logits = self.qwen.lm_head(hidden)", summary)
        self.assertIn("backward_frame_0.py", summary)
        self.assertIn("backward_frame_29.py", summary)
        self.assertIn("MmBackward0' returned nan", summary)
        self.assertNotIn("NCCL", summary)
        self.assertNotIn("ChildFailedError", summary)

    def test_unprefixed_forward_stack_is_retained_without_claiming_rank(self):
        content = "\n".join([
            'NUMERICS_CONTEXT {"rank": 1, "step": 1, "micro": 2, "qids": ["qa-1"]}',
            'UserWarning: Error detected in MmBackward0. Traceback of forward call that caused the error:',
            '  File "unprefixed_forward.py", line 41, in forward',
            '    return module(x)',
            ' (Triggered internally at anomaly.cpp:122.)',
            '[rank1]: Traceback (most recent call last):',
            "[rank1]: RuntimeError: Function 'MmBackward0' returned nan values in its 0th output.",
        ])
        summary = failure_summary(content)
        self.assertIn("FORWARD TRACEBACK (unattributed; shared log)", summary)
        self.assertIn("unprefixed_forward.py", summary)
        self.assertIn('"qids": ["qa-1"]', summary)

    def test_cli_selects_latest_failed_log_and_only_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "qwen3-vega" / "logs" / "sweep-existing"
            logs.mkdir(parents=True)
            old = logs / "m1-c1-numerics.log"
            latest = logs / "m1-c2-numerics.log"
            successful = logs / "m2-c1-numerics.log"
            old.write_text("[rank0]: FloatingPointError: Nonfinite old failure at step 1\n")
            latest.write_text(diagnostic_log())
            successful.write_text("BENCHMARK completed without errors\n")
            for stamp, path in enumerate((old, latest, successful), 10):
                os.utime(path, (stamp, stamp))
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in logs.iterdir()}
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("numerics_log.py")), str(root)],
                                    capture_output=True, text=True, check=True)
            self.assertIn(str(latest.resolve()), result.stdout)
            self.assertIn("train_model.py", result.stdout)
            self.assertIn("failing-qa", result.stdout)
            self.assertNotIn("Nonfinite old failure", result.stdout)
            after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in logs.iterdir()}
            self.assertEqual(before, after)
            self.assertEqual(sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file()),
                             sorted(p.relative_to(root) for p in before))

    def test_no_failed_log_returns_without_starting_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("numerics_log.py")), str(root)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("No jobs were started", result.stderr)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
