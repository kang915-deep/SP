"""Real four-process CPU/Gloo checks for the validation collective contract.

No model, GPU, dataset download, or remote connection is involved. Each group
uses spawn and a 20-second collective timeout; only its own child processes
are terminated if the 60-second process deadline is exceeded.
"""
import contextlib
from datetime import timedelta
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest
import uuid

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import validate


WORLD = 4


def validation_worker(rank, rendezvous, output, modes):
    """Global spawn target: run identical mode/collective order on every rank."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
        timeout=timedelta(seconds=20),
    )
    reports = {}
    try:
        for mode in modes:
            size = 2036 if mode == "complete" else 11
            rows = [{"id": f"qid-{i}", "index": i, "value": (i + 1) / 16}
                    for i in range(size)]
            prepared = []
            seen = []
            logs = io.StringIO()

            def prepare(row, processor):
                assert processor is None
                prepared.append(row["id"])
                if rank == 3 and row["index"] == 3:
                    if mode == "delay":
                        time.sleep(.05)
                    elif mode == "prepare_error":
                        raise RuntimeError("intentional prepare failure")
                return row

            def forward(batch, phase, progress):
                assert phase == "validation" and len(batch) == 1
                qid, valid, row = batch[0]
                seen.append({"qid": qid, "valid": valid,
                             "grad_enabled": torch.is_grad_enabled()})
                progress("encode")
                progress("forward")
                if rank == 3 and row["index"] == 3:
                    if mode == "forward_error":
                        raise RuntimeError("intentional forward failure")
                    if mode == "nan":
                        return torch.tensor([float("nan")])
                return torch.tensor([row["value"]], dtype=torch.float32)

            with contextlib.redirect_stdout(logs):
                try:
                    mean = validate(rows, None, prepare, forward, rank=rank,
                                    world=WORLD, device=torch.device("cpu"), step=1165)
                    outcome = {"mean": mean}
                except Exception as error:
                    outcome = {"error_type": type(error).__name__, "error": str(error)}
            events = [json.loads(line.removeprefix("VALIDATION_PROGRESS "))
                      for line in logs.getvalue().splitlines()
                      if line.startswith("VALIDATION_PROGRESS ")]
            reports[mode] = dict(outcome, events=events, prepared=prepared, seen=seen)
    finally:
        # A direct prepare/forward exception is deliberately not represented as
        # a collective success: the peers receive Gloo's disconnect/timeout.
        (Path(output) / f"rank-{rank}.json").write_text(json.dumps(reports))
        dist.destroy_process_group()


def run_group(modes):
    context = multiprocessing.get_context("spawn")
    # TemporaryDirectory's restrictive mkdir mode is inaccessible to spawned
    # children in the observed Windows sandbox. Ordinary mkdir works, and the
    # new test receipts are retained rather than deleting any existing files.
    root = Path(tempfile.gettempdir()) / f"vega-validation-rounds-{uuid.uuid4().hex}"
    root.mkdir()
    print(f"Validation test receipts: {root}", flush=True)
    rendezvous = (root / "gloo-store").as_uri()
    processes = [context.Process(target=validation_worker,
                                 args=(rank, rendezvous, str(root), modes))
                 for rank in range(WORLD)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 60
    for process in processes:
        process.join(max(0., deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    if alive:
        for process in alive:
            process.terminate()
        for process in alive:
            process.join(5)
        raise AssertionError("Validation child processes exceeded the 60-second deadline")
    codes = [process.exitcode for process in processes]
    if codes != [0] * WORLD:
        raise AssertionError(f"Validation children failed: exit codes {codes}")
    return [json.loads((root / f"rank-{rank}.json").read_text())
            for rank in range(WORLD)]


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is required")
class ValidationRoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regular = run_group(["complete", "padding", "delay", "nan"])

    def assert_complete(self, mode, size):
        expected = (size + 1) / 32
        seen_valid = []
        for rank, report in enumerate(self.regular):
            result = report[mode]
            self.assertNotIn("error", result)
            self.assertEqual(result["mean"], expected)
            self.assertEqual(len(result["prepared"]), (size + WORLD - 1) // WORLD)
            self.assertTrue(all(not item["grad_enabled"] for item in result["seen"]))
            seen_valid.extend(item["qid"] for item in result["seen"] if item["valid"])
            for slot in range((size + WORLD - 1) // WORLD):
                events = [event for event in result["events"] if event["slot"] == slot + 1]
                self.assertEqual([event["phase"] for event in events],
                                 ["read", "encode", "forward", "reduce", "done"])
                self.assertTrue(all(event["rank"] == rank and event["step"] == 1165
                                    and event["qid"] == f"qid-{min(slot * WORLD + rank, size - 1)}"
                                    for event in events))
        self.assertCountEqual(seen_valid, [f"qid-{i}" for i in range(size)])

    def test_complete_2036_questions_are_counted_once_on_all_ranks(self):
        self.assert_complete("complete", 2036)

    def test_11_questions_padding_does_not_change_mean(self):
        self.assert_complete("padding", 11)
        self.assertEqual(self.regular[3]["padding"]["seen"][-1],
                         {"qid": "qid-10", "valid": False, "grad_enabled": False})

    def test_slow_rank_still_returns_the_same_complete_mean(self):
        # Check correctness, not a wall-clock speed threshold.
        self.assert_complete("delay", 11)

    def test_rank3_nan_raises_on_every_rank_without_hanging(self):
        for report in self.regular:
            result = report["nan"]
            self.assertEqual(result["error_type"], "FloatingPointError")
            self.assertIn("slot 1", result["error"])
            self.assertEqual(result["events"][-1]["phase"], "reduce")
            self.assertNotIn("mean", result)

    def test_prepare_exception_retains_rank3_read_qid(self):
        reports = run_group(["prepare_error"])
        result = reports[3]["prepare_error"]
        self.assertEqual(result["error"], "intentional prepare failure")
        self.assertEqual(result["events"][-1]["phase"], "read")
        self.assertEqual(result["events"][-1]["qid"], "qid-3")
        # Direct exceptions do not use the synchronized nonfinite path.
        self.assertTrue(all("error" in report["prepare_error"] for report in reports))
        self.assertTrue(all("mean" not in report["prepare_error"] for report in reports))

    def test_forward_exception_retains_rank3_forward_qid(self):
        reports = run_group(["forward_error"])
        result = reports[3]["forward_error"]
        self.assertEqual(result["error"], "intentional forward failure")
        self.assertEqual(result["events"][-1]["phase"], "forward")
        self.assertEqual(result["events"][-1]["qid"], "qid-3")
        self.assertTrue(all("error" in report["forward_error"] for report in reports))
        self.assertTrue(all("mean" not in report["forward_error"] for report in reports))


if __name__ == "__main__":
    unittest.main(verbosity=2)
