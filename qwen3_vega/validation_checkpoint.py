"""Append-only training checkpoints and receipts for validation at a saved step."""

import json
from pathlib import Path


def save_training_checkpoint(engine, out, protocol, step, best, *, rank, barrier,
                             pending_validation=False, event=None,
                             exclude_frozen_parameters=False):
    """All ranks call this; rank zero commits the marker after all shards save."""
    out = Path(out)
    tag = f"step-{step:06d}"
    state = {"protocol": protocol, "step": step, "best": best}
    if pending_validation:
        if event is None or event["step"] != step:
            raise ValueError("Pending validation needs the completed training-step event")
        state.update(pending_validation=True, event=event)
    marker = out / (tag + ".json")
    if rank == 0 and (marker.exists() or (out / "checkpoints" / tag).exists()):
        raise FileExistsError(f"Checkpoint already exists; preserve it and resume its committed marker: {tag}")
    # Nobody may create a shard until rank zero has checked the new tag.
    barrier()
    engine.save_checkpoint(str(out / "checkpoints"), tag=tag, save_latest=False,
                           exclude_frozen_parameters=exclude_frozen_parameters,
                           client_state=state)
    barrier()
    if rank == 0:
        summary = {k: v for k, v in state.items() if k != "protocol"}
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, allow_nan=False)
    return tag


def write_validation_receipt(out, step, event, best):
    """Rank zero records success without rewriting the pre-validation snapshot."""
    out = Path(out)
    marker = json.loads((out / f"step-{step:06d}.json").read_text(encoding="utf-8"))
    if marker["step"] != step or not marker.get("pending_validation") or event["step"] != step:
        raise ValueError("Validation receipt must match a committed pending checkpoint")
    receipt = {"step": step, "event": event, "best": best}
    with (out / f"validation-{step:06d}.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, allow_nan=False)


def resume_validation(out, state):
    """Return (best, needs_validation) for loaded client state or its marker."""
    if not state.get("pending_validation", False):
        return state["best"], False
    step = state["step"]
    path = Path(out) / f"validation-{step:06d}.json"
    if not path.exists():
        return state["best"], True
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt["step"] != step or receipt["event"]["step"] != step:
        raise ValueError("Validation receipt belongs to another checkpoint")
    return receipt["best"], False


def completed_selection(out, step):
    """Recover final selection, or None while that checkpoint awaits validation."""
    marker = json.loads((Path(out) / f"step-{step:06d}.json").read_text(encoding="utf-8"))
    best, pending = resume_validation(out, marker)
    if pending:
        return None
    if best is None:
        raise ValueError("Completed checkpoint has no selected validation result")
    return dict(best, tag=f"step-{best['step']:06d}")
