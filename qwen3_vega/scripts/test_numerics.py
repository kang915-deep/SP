"""Real CPU tensor/Adam tests; these do not validate four-card CUDA numerics."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numerics import NumericalChecks


class NumericalChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.checks = NumericalChecks(self.out, rank=2)
        self.checks.context = {
            "step": 1, "micro": 16, "qids": ["qa-last"],
            "step_qids": ["qa-first", "qa-last"], "lr": 0.0,
        }

    def report(self, phase):
        report = json.loads((self.out / "numerics-rank2.json").read_text())
        self.assertEqual(report["phase"], phase)
        self.assertEqual(report["rank"], 2)
        self.assertEqual(report["context"], self.checks.context)
        self.assertFalse(report["stats"]["finite"])
        return report

    def test_finite_adam_updates_and_moments_unchanged(self):
        parameters = [torch.nn.Parameter(torch.tensor([1., -2.])), torch.nn.Parameter(torch.tensor([3.]))]
        baseline = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        optimizer = torch.optim.AdamW([{"params": [p]} for p in parameters], lr=0., eps=1e-8)
        reference = torch.optim.AdamW([{"params": [p]} for p in baseline], lr=0., eps=1e-8)
        self.checks.attach(optimizer)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for step, lr in enumerate((0., 1e-5, 2e-5), 1):
                self.checks.context.update(step=step, lr=lr)
                for opt in (optimizer, reference):
                    opt.zero_grad(set_to_none=True)
                    for group in opt.param_groups:
                        group["lr"] = lr
                loss = sum(p.square().sum() for p in parameters)
                self.checks.check_loss(loss.reshape(1))
                loss.backward()
                sum(p.square().sum() for p in baseline).backward()
                optimizer.step()
                reference.step()
                for actual, expected in zip(parameters, baseline):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for key in optimizer.state[actual]:
                        torch.testing.assert_close(optimizer.state[actual][key], reference.state[expected][key], rtol=0, atol=0)
        self.assertEqual(output.getvalue().count("NUMERICS_OPTIMIZER_PARTITION"), 2)
        self.assertEqual(self.checks.updates, 3)
        self.assertFalse((self.out / "numerics-rank2.json").exists())

    def test_bad_loss_stops_before_backward(self):
        parameter = torch.nn.Parameter(torch.tensor([-1.]))
        values = parameter.sqrt()
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite forward_loss at step 1"):
            self.checks.check_loss(values)
            values.sum().backward()
        self.assertIsNone(parameter.grad)
        report = self.report("forward_loss")
        self.assertEqual(report["stats"]["min"], "nan")

    def test_finite_loss_bad_gradient_stops_before_adam(self):
        parameter = torch.nn.Parameter(torch.tensor([0.]))
        optimizer = torch.optim.AdamW([parameter], lr=0.)
        self.checks.attach(optimizer)
        values = parameter.sqrt()
        self.checks.check_loss(values)
        values.sum().backward()
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite optimizer_gradient at step 1"):
            optimizer.step()
        self.assertEqual(parameter.item(), 0.)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(self.checks.updates, 0)
        report = self.report("optimizer_gradient")
        self.assertEqual(report["stats"]["dtype"], "torch.float32")
        self.assertEqual(report["stats"]["max"], "inf")
        self.assertEqual(report["stats"]["group"], 0)

    def test_bad_parameter_detected_after_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        inactive = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.AdamW([{"params": [inactive]}, {"params": [parameter]}], lr=1e-5)
        # A preceding post-hook injects corruption at the exact checked boundary.
        optimizer.register_step_post_hook(lambda opt, args, kwargs: parameter.data.fill_(float("nan")))
        self.checks.attach(optimizer)
        parameter.square().sum().backward()
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite optimizer_parameter at step 1"):
            optimizer.step()
        self.assertIsNone(inactive.grad)
        self.assertEqual(self.checks.updates, 1)
        report = self.report("optimizer_parameter")
        self.assertEqual(report["stats"]["group"], 1)
        self.assertEqual(report["stats"]["partition"], 0)
        self.assertEqual(report["stats"]["min"], "nan")

    def test_named_frozen_feature_infinity(self):
        features = [
            ("pooler.0", torch.tensor([[1., 2.]], dtype=torch.bfloat16)),
            ("wan", torch.tensor([[3., float("inf")]], dtype=torch.bfloat16)),
        ]
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite frozen_features"):
            self.checks.check_tensors(features, "frozen_features")
        report = self.report("frozen_features")
        self.assertEqual(report["stats"]["name"], "wan")
        self.assertEqual(report["stats"]["dtype"], "torch.bfloat16")
        self.assertEqual(report["stats"]["min"], 3.)
        self.assertEqual(report["stats"]["max"], "inf")


if __name__ == "__main__":
    unittest.main()
