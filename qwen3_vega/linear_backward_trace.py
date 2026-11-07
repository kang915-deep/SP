"""Diagnostic-only scalar tracing of bias-free Linear's actual MmBackward node.

Hooks leave gradients unchanged but scalar checks synchronize and can hide a
timing-dependent failure. Never read node._saved_self/_saved_mat2: non-reentrant
checkpoint unpack is one-shot. X/W statistics describe observed forward inputs,
including recomputation, not direct inspection of saved Mm operands.
"""
import json
import math
from pathlib import Path

import torch


class LinearNonfiniteError(FloatingPointError):
    pass


def tensor_stats(tensor):
    low, high = torch.aminmax(tensor.detach())
    low, high = low.item(), high.item()
    return {
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype), "device": str(tensor.device),
        "min": low if math.isfinite(low) else str(low),
        "max": high if math.isfinite(high) else str(high),
        "finite": math.isfinite(low) and math.isfinite(high),
    }


def linear_mm_node(output):
    """Follow only Linear's output view chain, never earlier operators."""
    node = output.grad_fn
    views = {"UnsafeViewBackward0", "ViewBackward0", "ReshapeAliasBackward0",
             "CloneBackward0", "SqueezeBackward0", "SqueezeBackward1"}
    while node is not None and node.name() in views:
        edges = [child for child, _ in node.next_functions if child is not None]
        if len(edges) != 1:
            break
        node = edges[0]
    if node is None or node.name() != "MmBackward0":
        raise RuntimeError(f"Expected uncompiled bias-free Linear MmBackward0, got {node}")
    return node


class LinearBackwardTrace:
    def __init__(self, linears, output):
        """Select fully named Linear modules; set context after prior backward."""
        if not linears or any(not isinstance(m, torch.nn.Linear) or m.bias is not None
                              for m in linears.values()):
            raise ValueError("Select nonempty, bias-free nn.Linear modules")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = (self.output / "events.jsonl").open("x", buffering=1)
        self.context = {}
        self.latest = {}
        self.sequence = 0
        self.handles = []
        self.node_handles = []
        for name, module in linears.items():
            self.handles.append(module.register_forward_pre_hook(
                lambda m, args, name=name: self._observe_input(name, m, args)))
            self.handles.append(module.register_forward_hook(
                lambda m, args, output, name=name: self._observe_output(name, output)))

    def set_context(self, **context):
        # Previous microbatch backward must finish before advancing context.
        for handle in self.node_handles:
            handle.remove()
        self.node_handles.clear()
        self.latest.clear()
        self.context = dict(context)

    def _observe_input(self, name, module, inputs):
        self.sequence += 1
        state = {"name": name, "call": self.sequence, "context": dict(self.context),
                 "x": tensor_stats(inputs[0]), "weight": tensor_stats(module.weight)}
        self.latest[name] = state
        self._check("X_forward_observed", state["x"], state)
        self._check("W_forward_observed", state["weight"], state)

    def _observe_output(self, name, output):
        node = linear_mm_node(output)
        state = self.latest[name]

        def before(grad_outputs):
            self._check("G_incoming", tensor_stats(grad_outputs[0]), state)

        def after(grad_inputs, grad_outputs):
            # Mm input 1 is W.T; this is d(W.T), not dW's layout.
            for label, gradient in zip(("dX_mm", "dWeightTranspose_mm"), grad_inputs):
                if gradient is not None:
                    self._check(label, tensor_stats(gradient), state)

        self.node_handles.extend((node.register_prehook(before), node.register_hook(after)))

    def _check(self, phase, stats, state):
        latest = self.latest[state["name"]]
        record = {
            "phase": phase, "name": state["name"], "forward_call": state["call"],
            "context": state["context"], "stats": stats,
            "original_forward": {k: state[k] for k in ("call", "x", "weight")},
            "latest_observed_forward": {k: latest[k] for k in ("call", "x", "weight")},
            "x_evidence": "Observed forward inputs; saved Mm operands are not unpacked by tracer",
        }
        self.events.write(json.dumps(record, allow_nan=False) + "\n")
        if stats["finite"]:
            return
        (self.output / "first_nonfinite.json").write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        raise LinearNonfiniteError(
            f"Nonfinite {state['name']} {phase}; context={state['context']}; "
            f"report={self.output / 'first_nonfinite.json'}")

    def close(self):
        for handle in self.handles + self.node_handles:
            handle.remove()
        self.handles.clear()
        self.node_handles.clear()
        self.latest.clear()
        self.events.close()
