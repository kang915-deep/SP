"""Fail at the first observed nonfinite training boundary, before contamination."""
import json
import math

import torch


class NumericalChecks:
    def __init__(self, out, rank, record_changes=False):
        self.out = out
        self.rank = rank
        self.context = {}
        self.updates = 0
        self.gradient_stats = []
        self.record_changes = record_changes
        self.step_groups = {}
        self.nonzero_lr_groups = {}
        self.changed_parameter_steps = set()

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
        self.pending_groups = set()
        self.pending_nonzero_lr_groups = set()
        self.parameter_samples = []
        for group in optimizer.param_groups:
            group_index = self.group_ids[id(group)]
            for parameter_index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue
                stats = dict(self._stats(parameter.grad), group=group_index, partition=parameter_index)
                self.gradient_stats.append(stats)
                if not stats["finite"]:
                    self._fail("optimizer_gradient", stats)
                self.pending_groups.add(group_index)
                if group["lr"] > 0:
                    self.pending_nonzero_lr_groups.add(group_index)
                    if self.record_changes:
                        stride = max(1, parameter.numel() // 1024)
                        sample = parameter.detach().reshape(-1)[::stride][:1024].clone()
                        self.parameter_samples.append((parameter, stride, sample))

    def _after_step(self, optimizer, args, kwargs):
        self.updates += 1
        parameter_stats = []
        for group in optimizer.param_groups:
            group_index = self.group_ids[id(group)]
            for parameter_index, parameter in enumerate(group["params"]):
                stats = dict(self._stats(parameter), group=group_index, partition=parameter_index)
                parameter_stats.append(stats)
                if not stats["finite"]:
                    self._fail("optimizer_parameter", stats)
        step = self.context["step"]
        self.step_groups.setdefault(step, set()).update(self.pending_groups)
        self.nonzero_lr_groups.setdefault(step, set()).update(self.pending_nonzero_lr_groups)
        for parameter, stride, before in self.parameter_samples:
            after = parameter.detach().reshape(-1)[::stride][:1024]
            if not torch.equal(before, after):
                self.changed_parameter_steps.add(step)
        if self.context["step"] <= 2:
            print("NUMERICS_OPTIMIZER_PARTITION", json.dumps({
                "rank": self.rank, "context": self.context,
                "gradients": self.gradient_stats, "parameters": parameter_stats,
            }, allow_nan=False), flush=True)

    def attach(self, optimizer):
        # ZeRO calls this optimizer once per group, temporarily exposing only
        # that group. Preserve identities so those calls count as one full step.
        self.group_ids = {id(group): index for index, group in enumerate(optimizer.param_groups)}
        self.pre_hook = optimizer.register_step_pre_hook(self._before_step)
        self.post_hook = optimizer.register_step_post_hook(self._after_step)

    def acceptance(self, expected_steps, engine_global_steps):
        groups = set(self.group_ids.values())
        completed = sorted(step for step, seen in self.step_groups.items() if seen == groups)
        nonzero = sorted(step for step, seen in self.nonzero_lr_groups.items() if seen == groups)
        changed = sorted(self.changed_parameter_steps)
        accepted = (
            completed == list(range(1, expected_steps + 1))
            and nonzero == list(range(2, expected_steps + 1))
            and bool(set(changed).intersection(nonzero))
            and engine_global_steps == expected_steps
        )
        return {
            "accepted": accepted, "expected_steps": expected_steps,
            "completed_optimizer_steps": completed, "nonzero_lr_optimizer_steps": nonzero,
            "parameter_change_observed": bool(changed), "changed_parameter_steps": changed,
            "optimizer_calls": self.updates, "expected_group_count": len(groups),
            "final_engine_global_steps": engine_global_steps,
            "finite_scope": "Every loss and every actual optimizer partition gradient/parameter boundary",
            "parameter_change_scope": "At most 1024 sampled FP32 master values per optimizer partition at positive LR",
        }
