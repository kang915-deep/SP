"""CPU numerical equivalence and mocked distributed-prefix control flow."""
import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import checkpoint

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from linear_backward_trace import LinearBackwardTrace, LinearNonfiniteError, linear_mm_node
from numerics import NumericalChecks


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(13)

    def test_checkpoint_bf16_loss_and_gradients_equal_without_tracing(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 7, bias=False), torch.nn.SiLU(),
                                    torch.nn.Linear(7, 4, bias=False)).bfloat16()
        expected = copy.deepcopy(model)
        trace = LinearBackwardTrace({"layers.3.mlp.up_proj": model[0], "layers.3.mlp.down_proj": model[2]},
                                    self.root / "trace")
        for micro in (1, 2):
            x = torch.randn(3, 2, 4, dtype=torch.bfloat16).transpose(0, 1).requires_grad_()
            reference = x.detach().clone().requires_grad_()
            trace.set_context(step=1, micro=micro, qids=[f"qa-{micro}"], rank=2)
            loss = checkpoint(model, x, use_reentrant=False).float().square().sum()
            baseline = checkpoint(expected, reference, use_reentrant=False).float().square().sum()
            torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
            loss.backward()
            baseline.backward()
            torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
            for actual, native in zip(model.parameters(), expected.parameters()):
                torch.testing.assert_close(actual.grad, native.grad, rtol=0, atol=0)
        trace.close()
        events = [json.loads(s) for s in (self.root / "trace" / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["latest_observed_forward"]["call"] != e["forward_call"]
                            for e in events if e["phase"] == "dWeightTranspose_mm"))
        self.assertEqual({e["context"]["micro"] for e in events if e["phase"] == "G_incoming"}, {1, 2})
        self.assertEqual(list((self.root / "trace").glob("*.pt")), [])

    def test_node_posthook_records_specific_projection_before_anomaly(self):
        model = torch.nn.Linear(2, 1, bias=False)
        def inject(module, inputs, output):
            linear_mm_node(output).register_hook(
                lambda gradients, incoming: (gradients[0], torch.full_like(gradients[1], float("nan"))))
        injection = model.register_forward_hook(inject)
        trace = LinearBackwardTrace({"layers.8.mlp.gate_proj": model}, self.root / "trace")
        trace.set_context(step=1, micro=2, qids=["bad-qa"], rank=3)
        with torch.autograd.set_detect_anomaly(True):
            with self.assertRaisesRegex(LinearNonfiniteError, "layers.8.mlp.gate_proj dWeightTranspose_mm"):
                checkpoint(model, torch.ones(2, 2, requires_grad=True), use_reentrant=False).sum().backward()
        trace.close()
        injection.remove()
        record = json.loads((self.root / "trace" / "first_nonfinite.json").read_text())
        self.assertEqual(record["name"], "layers.8.mlp.gate_proj")
        self.assertEqual(record["context"], {"step": 1, "micro": 2, "qids": ["bad-qa"], "rank": 3})
        self.assertEqual(record["stats"]["min"], "nan")

    def test_incoming_infinity_stops_before_local_mm(self):
        model = torch.nn.Linear(2, 1, bias=False)
        trace = LinearBackwardTrace({"layers.0.mlp.up_proj": model}, self.root / "trace")
        trace.set_context(step=1, micro=1)
        output = model(torch.zeros(2, 2, requires_grad=True))
        with self.assertRaisesRegex(LinearNonfiniteError, "G_incoming"):
            output.backward(torch.full_like(output, float("inf")))
        trace.close()
        self.assertIsNone(model.weight.grad)

    def test_finite_operands_and_nonfinite_local_mm_gradient_are_distinct(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        trace = LinearBackwardTrace({"layers.0.mlp.down_proj": model}, self.root / "trace")
        trace.set_context(step=1, micro=1)
        output = model(torch.full((2, 2), 1e30, requires_grad=True))
        with self.assertRaisesRegex(LinearNonfiniteError, "dWeightTranspose_mm"):
            output.backward(torch.full_like(output, 1e30))
        trace.close()
        events = [json.loads(s) for s in (self.root / "trace" / "events.jsonl").read_text().splitlines()]
        self.assertTrue(next(e for e in events if e["phase"] == "G_incoming")["stats"]["finite"])
        self.assertFalse(events[-1]["stats"]["finite"])


class DiagnosticPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((BASE / "train.py").read_text(encoding="utf-8"))
        begin = next(i for i, node in enumerate(cls.tree.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "parser")
        end = next(i for i, node in enumerate(cls.tree.body)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and ast.unparse(node.value.func) == "torch.cuda.set_device")
        cls.entry = compile(ast.Module(body=cls.tree.body[begin:end], type_ignores=[]), str(BASE / "train.py"), "exec")
        cls.micro = next(node for node in ast.walk(cls.tree)
                         if isinstance(node, ast.For) and ast.unparse(node.target) == "micro")
        cls.loop = compile(ast.Module(body=[cls.micro], type_ignores=[]), str(BASE / "train.py"), "exec")

    def parse_entry(self, extra):
        argv = ["train.py", "--qwen", "q", "--wan", "w", "--data", "d", "--output", "o", *extra]
        env = {"argparse": argparse, "Path": Path, "os": os}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, RANK="0", WORLD_SIZE="4", LOCAL_RANK="0", CUDA_VISIBLE_DEVICES="0,1,2,3"), contextlib.redirect_stderr(io.StringIO()):
            exec(self.entry, env)
        return env

    def test_prefix_arguments_are_validated_before_cuda(self):
        valid = self.parse_entry(["--diagnose-numerics", "--benchmark-steps", "2", "--diagnostic-micros", "2"])
        self.assertEqual(valid["accum"], 16)
        self.assertEqual(valid["args"].diagnostic_micros, 2)
        self.assertIsNone(self.parse_entry([])["args"].diagnostic_micros)
        for extra in [
            ["--diagnostic-micros", "2"],
            ["--diagnose-numerics", "--diagnostic-micros", "2"],
            ["--benchmark-steps", "2", "--diagnostic-micros", "2"],
            *[["--diagnose-numerics", "--benchmark-steps", "2", "--diagnostic-micros", str(n)] for n in (0, 16, 17)],
            ["--diagnose-numerics", "--benchmark-steps", "2", "--diagnostic-micros", "2", "--resume", "step-1"],
        ]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as caught:
                self.parse_entry(extra)
            self.assertEqual(caught.exception.code, 2)

    def test_diagnostic_prefix_does_not_create_benchmark_monitor(self):
        monitor_block = next(node for node in self.tree.body if isinstance(node, ast.If)
                             and any(isinstance(child, ast.Import) and any(alias.name == "pynvml" for alias in child.names)
                                     for child in node.body))
        condition = compile(ast.Expression(body=monitor_block.test), str(BASE / "train.py"), "eval")
        self.assertFalse(eval(condition, {"args": SimpleNamespace(benchmark_steps=2, diagnostic_micros=2)}))
        self.assertTrue(eval(condition, {"args": SimpleNamespace(benchmark_steps=2, diagnostic_micros=None)}))

    def test_actual_micro_loop_exits_after_two_backwards_before_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            model = torch.nn.Linear(2, 1, bias=False)
            original = model.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            checks = NumericalChecks(out, 0)
            checks.attach(optimizer)
            trace = LinearBackwardTrace({"layers.0.mlp.up_proj": model}, out / "linear-rank0")
            class Engine:
                global_steps = 0
                step_calls = 0
                backward_calls = 0
                def backward(self, loss):
                    self.backward_calls += 1
                    loss.backward()
                def step(self):
                    self.step_calls += 1
                    if self.step_calls % 16 == 0:
                        optimizer.step()
                        self.global_steps += 1
            engine = Engine()
            def gather(reports, local):
                reports[:] = [dict(local, rank=rank) for rank in range(4)]
            distributed = SimpleNamespace(all_gather_object=Mock(side_effect=gather), destroy_process_group=Mock())
            env = {
                "args": SimpleNamespace(diagnose_numerics=True, diagnostic_micros=2), "accum": 16,
                "batches": iter([[(f"qa-{i}", True, None)] for i in range(16)]), "step_qids": [],
                "checks": checks, "step": 0, "factor": 0., "rank": 0, "world": 4, "n": 64,
                "linear_trace": trace, "json": json, "torch": torch, "device": torch.device("cpu"),
                "engine": engine, "ce": torch.zeros(()), "dist": distributed, "out": out, "protocol": {"test": True},
                "forward": lambda batch, phase: model(torch.ones(1, 2)).square().mean().reshape(1),
            }
            with patch.object(torch.cuda, "synchronize") as synchronize, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                exec(self.loop, env)
            self.assertEqual(caught.exception.code, 0)
            self.assertEqual((engine.backward_calls, engine.step_calls, engine.global_steps, checks.updates), (2, 1, 0, 0))
            self.assertEqual(len(optimizer.state), 0)
            torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
            self.assertGreater(model.weight.grad.abs().sum().item(), 0.)
            synchronize.assert_called_once_with(torch.device("cpu"))
            distributed.destroy_process_group.assert_called_once_with()
            report = json.loads((out / "diagnostic_prefix.json").read_text())
            self.assertEqual(len(report["ranks"]), 4)
            self.assertTrue(all(row["optimizer_updates"] == row["checked_optimizer_updates"] == 0 for row in report["ranks"]))
            self.assertEqual(report["ranks"][0]["context"]["step_qids"], ["qa-0", "qa-1"])
            self.assertFalse((out / "throughput.json").exists())
            self.assertFalse((out / "selected.json").exists())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
