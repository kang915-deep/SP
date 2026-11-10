"""Execute train.py's actual resume/training control flow with CPU stand-ins.

These tests do not initialize a model, DeepSpeed, GPU, or process group. The AST
retains the production ordering, conditions, loop and checkpoint/receipt calls;
only external initialization and media/model computation are substituted.
"""

import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation_checkpoint import (resume_validation, save_training_checkpoint,
                                   write_validation_receipt)


class FakeEngine:
    def __init__(self, trace, state):
        self.trace = trace
        self.state = state
        self.global_steps = 0
        self.optimizer = SimpleNamespace(param_groups=[{}])

    def load_checkpoint(self, root, **kwargs):
        self.global_steps = self.state["step"]
        self.trace.append(("load", self.global_steps))
        return str(Path(root) / kwargs["tag"]), self.state

    def save_checkpoint(self, root, **kwargs):
        state = kwargs["client_state"]
        self.trace.append(("save", state["step"]))
        target = Path(root) / kwargs["tag"]
        target.mkdir(parents=True)
        (target / "client_state.json").write_text(json.dumps(state))
        (target / "adam.pt").write_bytes(b"fake-current-optimizer-state")
        return True

    def eval(self):
        self.trace.append(("eval", self.global_steps))

    def train(self):
        self.trace.append(("train", self.global_steps))

    def backward(self, loss):
        self.trace.append(("backward", self.global_steps + 1))
        assert torch.isfinite(loss)

    def step(self):
        self.global_steps += 1
        self.trace.append(("update", self.global_steps))


class TrainingFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        assignments = {node.targets[0].id: i for i, node in enumerate(tree.body)
                       if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)}
        resume_start = assignments["start"]
        examples = next(i for i, node in enumerate(tree.body)
                        if isinstance(node, ast.ClassDef) and node.name == "Examples")
        # Include the real resume, training order, total and validation schedule.
        body = tree.body[resume_start:examples] + tree.body[assignments["indices"]:]
        cls.code = compile(ast.Module(body=body, type_ignores=[]), str(ROOT / "train.py"), "exec")

    def setUp(self):
        # Ordinary mkdir avoids the Windows sandbox's observed mode-0700 temp
        # directory ACL issue. Retain only these uniquely named test receipts.
        self.out = Path(tempfile.gettempdir()) / ("vega-training-flow-" + uuid.uuid4().hex)
        self.out.mkdir()
        self.trace = []
        self.protocol = {"world": 1, "effective_batch": 1}

    def execute(self, *, state=None, total=4657, stop=None, save_every=250,
                receipt=None, validation_values=None, fail_validation=None):
        if state is not None:
            state = dict(state, protocol=self.protocol)
            marker = {k: v for k, v in state.items() if k != "protocol"}
            (self.out / f"step-{state['step']:06d}.json").write_text(json.dumps(marker))
            if receipt is not None:
                write_validation_receipt(self.out, state["step"], receipt["event"], receipt["best"])
        engine = FakeEngine(self.trace, state)
        args = SimpleNamespace(resume=f"step-{state['step']:06d}" if state else None,
                               smoke=False, benchmark_steps=None, stop_after=stop,
                               save_every=save_every, diagnose_numerics=False,
                               diagnostic_micros=None)
        values = validation_values or {}

        def loader(rows, indices, width):
            self.trace.append(("loader", engine.global_steps))
            return [[(rows[index]["id"], valid, None)] for index, valid in indices]

        def forward(batch, phase):
            self.assertEqual(phase, "train")
            self.trace.append(("forward", engine.global_steps + 1))
            return torch.tensor([2.] * len(batch))

        def validate(*unused, step, **kwargs):
            self.trace.append(("validate", step))
            self.assertEqual(engine.global_steps, step)
            if fail_validation == step:
                raise RuntimeError("intentional validation failure")
            return values.get(step, .4)

        dist = SimpleNamespace(barrier=lambda: self.trace.append(("barrier", engine.global_steps)),
                               all_reduce=lambda value: None,
                               destroy_process_group=lambda: self.trace.append(("destroy", engine.global_steps)))
        cpu_torch = SimpleNamespace(zeros=torch.zeros, isfinite=torch.isfinite,
                                    cuda=SimpleNamespace(max_memory_allocated=lambda device: 0))
        namespace = dict(args=args, out=self.out, engine=engine, protocol=self.protocol,
                         rows=[{"id": f"qid-{i}", "path": f"media-{i}"} for i in range(total)],
                         tune=[{"id": "validation-item", "path": "validation-media"}],
                         data_dir=self.out, rank=0, world=1, width=1, effective_batch=1,
                         accum=1, params=[], processor=None, prepare=None, device="cpu",
                         checks=SimpleNamespace(attach=lambda optimizer: None, check_loss=lambda loss: None),
                         optimizer=engine.optimizer, loader=loader, forward=forward, validate=validate,
                         save_training_checkpoint=save_training_checkpoint,
                         write_validation_receipt=write_validation_receipt,
                         resume_validation=resume_validation, torch=cpu_torch, dist=dist,
                         math=math, hashlib=hashlib, json=json, time=time)
        self.namespace = namespace
        self.engine = engine
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.code, namespace)
        return namespace

    def pending(self, step, best=None):
        return {"step": step, "best": best, "pending_validation": True,
                "event": {"step": step, "train_ce": .38, "qa_per_s": .84}}

    def test_validation_checkpoint_precedes_validation_and_is_saved_only_once_at_stop(self):
        result = self.execute(state={"step": 1164, "best": None}, stop=1165, save_every=1165)
        self.assertEqual([item for item in self.trace if item[0] == "save"], [("save", 1165)])
        self.assertLess(self.trace.index(("update", 1165)), self.trace.index(("save", 1165)))
        self.assertLess(self.trace.index(("save", 1165)), self.trace.index(("validate", 1165)))
        marker = json.loads((self.out / "step-001165.json").read_text())
        self.assertTrue(marker["pending_validation"])
        self.assertIsNone(marker["best"])
        self.assertEqual(result["best"], {"step": 1165, "tune_ce": .4})
        self.assertTrue((self.out / "validation-001165.json").exists())

    def test_validation_exception_keeps_checkpoint_without_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "intentional validation failure"):
            self.execute(state={"step": 1164, "best": None}, stop=1165, fail_validation=1165)
        self.assertTrue((self.out / "checkpoints/step-001165/adam.pt").exists())
        self.assertTrue((self.out / "step-001165.json").exists())
        self.assertFalse((self.out / "validation-001165.json").exists())
        self.assertFalse((self.out / "selected.json").exists())

    def test_pending_resume_validates_before_next_update_without_replaying_saved_step(self):
        result = self.execute(state=self.pending(1165), stop=1166)
        self.assertEqual([item for item in self.trace if item[0] == "update"], [("update", 1166)])
        self.assertEqual([item for item in self.trace if item[0] == "save"], [("save", 1166)])
        self.assertLess(self.trace.index(("validate", 1165)), self.trace.index(("forward", 1166)))
        self.assertEqual(result["best"], {"step": 1165, "tune_ce": .4})

    def test_final_pending_resume_empty_loop_writes_prior_best_selection(self):
        best = {"step": 3493, "tune_ce": .2}
        result = self.execute(state=self.pending(4657, best))
        self.assertEqual(result["completed"], 4657)
        self.assertFalse(any(kind in {"loader", "forward", "update", "save"} for kind, _ in self.trace))
        self.assertEqual([item for item in self.trace if item[0] == "validate"], [("validate", 4657)])
        selected = json.loads((self.out / "selected.json").read_text())
        self.assertEqual(selected, dict(best, tag="step-003493"))

    def test_completed_receipt_restores_best_instead_of_stale_checkpoint_value(self):
        old_best = {"step": 3493, "tune_ce": .3}
        new_best = {"step": 4657, "tune_ce": .2}
        receipt = {"event": {"step": 4657, "tune_ce": .2}, "best": new_best}
        result = self.execute(state=self.pending(4657, old_best), receipt=receipt)
        self.assertEqual(result["best"], new_best)
        self.assertFalse(any(kind in {"validate", "update", "save"} for kind, _ in self.trace))
        self.assertEqual(json.loads((self.out / "selected.json").read_text()),
                         dict(new_best, tag="step-004657"))

    def test_fresh_preflight_has_zero_updates_no_selection_and_existing_train_loader(self):
        result = self.execute(total=8, stop=1, validation_values={0: .001})
        self.assertLess(self.trace.index(("loader", 0)), self.trace.index(("validate", 0)))
        self.assertLess(self.trace.index(("validate", 0)), self.trace.index(("update", 1)))
        preflight = json.loads((self.out / "validation-preflight.json").read_text())
        self.assertEqual(preflight, {"questions": 1, "initial_weight_ce": .001, "optimizer_updates": 0})
        self.assertIsNone(result["best"])
        self.assertIsNone(json.loads((self.out / "step-000001.json").read_text())["best"])
        self.assertFalse((self.out / "selected.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
