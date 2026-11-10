"""Synchronous validation rounds with visible per-rank media/compute progress."""
import json
import math
import time

import torch
import torch.distributed as dist


def validate(tune, processor, prepare, forward, *, rank, world, device, step):
    # The old loop let an early rank wait in NCCL for the entire remaining
    # validation workload of the slowest rank. Reduce after every equal slot.
    # Read on this process: do not fork another loader after CUDA/threads start.
    rounds = math.ceil(len(tune) / world)
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    begin = time.perf_counter()
    with torch.no_grad():
        for slot in range(rounds):
            index = slot * world + rank
            valid = index < len(tune)
            row = tune[min(index, len(tune) - 1)]

            def progress(phase):
                print("VALIDATION_PROGRESS", json.dumps({
                    "step": step, "rank": rank, "slot": slot + 1,
                    "slots": rounds, "qid": row["id"], "valid": valid,
                    "phase": phase, "seconds": time.perf_counter() - begin,
                }), flush=True)

            progress("read")
            example = prepare(row, processor)
            values = forward([(row["id"], valid, example)], "validation", progress=progress)
            finite = torch.isfinite(values).all()
            stats = torch.zeros(3, device=device, dtype=torch.float64)
            # Signal nonfinite values to all ranks before throwing, rather than
            # leaving peers in a collective. A padding row never changes CE.
            stats[0] = torch.where(finite, values.double().sum(), 0.) if valid else 0.
            stats[1] = int(valid)
            stats[2] = (~finite).double()
            progress("reduce")
            dist.all_reduce(stats)
            if stats[2].item():
                raise FloatingPointError(f"Nonfinite validation loss at step {step}, slot {slot + 1}; see per-rank progress")
            totals += stats[:2]
            progress("done")
    if totals[1].item() != len(tune) or not torch.isfinite(totals[0]):
        raise ValueError(f"Invalid validation totals at step {step}: {totals.tolist()}")
    return (totals[0] / totals[1]).item()
