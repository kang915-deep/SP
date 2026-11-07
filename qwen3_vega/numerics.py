"""Fail at the first observed nonfinite training boundary, before contamination."""
import json
import math

import torch


class NumericalChecks:
    def __init__(self, out, rank):
        self.out = out
        self.rank = rank
        self.context = {}
        self.updates = 0
        self.gradient_stats = []

    @staticmethod
    def _stats(tensor):
        low, high = torch.aminmax(tensor.detach())
        low, high = low.item(), high.item()
        return {
            "dtype": str(tensor.dtype), "shape": list(tensor.shape), "numel": tensor.numel(),
            "min": low if math.isfinite(low) else str(low),
            "max": high if math.isfinite(high) else str(high),
            "finite": math.isfinite(low) and math.isfinite(high),
        }

    def _fail(self, phase, stats):
        report = {"rank": self.rank, "phase": phase, "context": dict(self.context), "stats": stats}
        path = self.out / f"numerics-rank{self.rank}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("NUMERICS_FAILURE", json.dumps(report, allow_nan=False), flush=True)
        raise FloatingPointError(
            f"Nonfinite {phase} at step {self.context['step']} "
            f"micro {self.context['micro']} rank {self.rank}; report: {path}; "
            f"qids: {self.context['qids']}; stats: {json.dumps(stats, allow_nan=False)}"
        )

    def check_loss(self, values):
        stats = self._stats(values)
        if not stats["finite"]:
            self._fail("forward_loss", stats)

    def check_tensors(self, named_tensors, phase):
        for name, tensor in named_tensors:
            stats = dict(self._stats(tensor), name=name)
            if not stats["finite"]:
                self._fail(phase, stats)

    def _before_step(self, optimizer, args, kwargs):
        self.gradient_stats = []
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue
                stats = dict(self._stats(parameter.grad), group=group_index, partition=parameter_index)
                self.gradient_stats.append(stats)
                if not stats["finite"]:
                    self._fail("optimizer_gradient", stats)

    def _after_step(self, optimizer, args, kwargs):
        self.updates += 1
        parameter_stats = []
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                stats = dict(self._stats(parameter), group=group_index, partition=parameter_index)
                parameter_stats.append(stats)
                if not stats["finite"]:
                    self._fail("optimizer_parameter", stats)
        if self.context["step"] <= 2:
            print("NUMERICS_OPTIMIZER_PARTITION", json.dumps({
                "rank": self.rank, "context": self.context,
                "gradients": self.gradient_stats, "parameters": parameter_stats,
            }, allow_nan=False), flush=True)

    def attach(self, optimizer):
        self.pre_hook = optimizer.register_step_pre_hook(self._before_step)
        self.post_hook = optimizer.register_step_post_hook(self._after_step)
