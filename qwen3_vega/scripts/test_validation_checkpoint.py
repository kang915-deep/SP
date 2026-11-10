"""CPU state-transition tests; simulate an engine with complete model/Adam shards."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_checkpoint import (completed_selection, resume_validation,
                                   save_training_checkpoint, write_validation_receipt)


class FakeEngine:
    def __init__(self, trace, fail=False):
        self.trace = trace
        self.fail = fail
        self.state = None

    def save_checkpoint(self, root, **kwargs):
        self.trace.append("save")
        self.state = json.loads(json.dumps(kwargs["client_state"]))
        target = Path(root) / kwargs["tag"]
        target.mkdir(parents=True)
        (target / "model.pt").write_bytes(b"model-at-current-step")
        if self.fail:
            raise OSError("Observed checkpoint-write failure")
        (target / "adam.pt").write_bytes(b"optimizer-moments-at-current-step")
        assert kwargs["save_latest"] is False
        assert kwargs["exclude_frozen_parameters"] is False
        return True


class ValidationCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.protocol = {"world": 4, "effective_batch": 64}
        self.trace = []
        self.engine = FakeEngine(self.trace)

    def barrier(self):
        self.trace.append("barrier")

    def save_pending(self, step=1165, best=None):
        event = {"step": step, "train_ce": .38, "qa_per_s": .84}
        save_training_checkpoint(self.engine, self.out, self.protocol, step, best,
                                 rank=0, barrier=self.barrier,
                                 pending_validation=True, event=event)
        return event

    def snapshot(self):
        return {str(p.relative_to(self.out)): p.read_bytes()
                for p in self.out.rglob("*") if p.is_file()}

    def test_validation_crash_leaves_full_adam_and_pending_work(self):
        event = self.save_pending()
        self.assertEqual(self.trace, ["barrier", "save", "barrier"])
        marker = json.loads((self.out / "step-001165.json").read_text())
        self.assertTrue((self.out / "checkpoints/step-001165/adam.pt").exists())
        self.assertEqual(self.engine.state["event"], event)
        self.assertEqual(marker["event"], event)
        self.assertEqual(resume_validation(self.out, self.engine.state), (None, True))
        self.assertIsNone(completed_selection(self.out, 1165))

    def test_receipt_resolves_pending_with_new_best_without_rewriting_snapshot(self):
        old_best = {"step": 1165, "tune_ce": .4}
        event = self.save_pending(2330, old_best)
        before = self.snapshot()
        event["tune_ce"] = .3
        best = {"step": 2330, "tune_ce": .3}
        write_validation_receipt(self.out, 2330, event, best)
        self.assertEqual(resume_validation(self.out, self.engine.state), (best, False))
        for name, content in before.items():
            self.assertEqual((self.out / name).read_bytes(), content)
        self.assertEqual(set(self.snapshot()) - set(before), {"validation-002330.json"})
        # Engine client state still carries the old best; the receipt is authoritative.
        self.assertEqual(self.engine.state["best"], old_best)

    def test_final_receipt_selects_prior_best_and_final_pending_never_selects(self):
        best = {"step": 3493, "tune_ce": .25}
        event = self.save_pending(4657, best)
        self.assertIsNone(completed_selection(self.out, 4657))
        event["tune_ce"] = .27
        write_validation_receipt(self.out, 4657, event, best)
        self.assertEqual(completed_selection(self.out, 4657),
                         {"step": 3493, "tune_ce": .25, "tag": "step-003493"})

    def test_old_checkpoints_and_markers_need_no_validation_replay(self):
        best = {"step": 2330, "tune_ce": .3}
        state = {"protocol": self.protocol, "step": 4657, "best": best}
        (self.out / "step-004657.json").write_text(json.dumps({"step": 4657, "best": best}))
        self.assertEqual(resume_validation(self.out, state), (best, False))
        self.assertEqual(completed_selection(self.out, 4657),
                         {"step": 2330, "tune_ce": .3, "tag": "step-002330"})

    def test_failed_save_has_no_marker_and_existing_partial_shards_are_preserved(self):
        self.engine.fail = True
        with self.assertRaises(OSError):
            self.save_pending()
        self.assertFalse((self.out / "step-001165.json").exists())
        before = self.snapshot()
        self.engine.fail = False
        with self.assertRaises(FileExistsError):
            self.save_pending()
        self.assertEqual(self.snapshot(), before)

    def test_duplicate_checkpoint_or_receipt_never_overwrites_files(self):
        event = self.save_pending()
        event["tune_ce"] = .3
        best = {"step": 1165, "tune_ce": .3}
        write_validation_receipt(self.out, 1165, event, best)
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            self.save_pending()
        with self.assertRaises(FileExistsError):
            write_validation_receipt(self.out, 1165, event, best)
        self.assertEqual(self.snapshot(), before)

    def test_mismatched_receipt_is_not_used_for_selection(self):
        self.save_pending()
        (self.out / "validation-001165.json").write_text(json.dumps(
            {"step": 2330, "event": {"step": 2330}, "best": {"step": 2330, "tune_ce": .2}}))
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            resume_validation(self.out, self.engine.state)

    def test_non_validation_checkpoint_keeps_legacy_shape(self):
        save_training_checkpoint(self.engine, self.out, self.protocol, 250, None,
                                 rank=0, barrier=self.barrier)
        marker = json.loads((self.out / "step-000250.json").read_text())
        self.assertEqual(marker, {"step": 250, "best": None})
        self.assertEqual(resume_validation(self.out, self.engine.state), (None, False))


if __name__ == "__main__":
    unittest.main()
