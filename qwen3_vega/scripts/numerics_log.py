"""Read existing failure evidence; no model imports, downloads, or training."""
import argparse
from pathlib import Path
import re


RANK = re.compile(r"^\s*\[rank(\d+)\]:")
ERROR = re.compile(r"^\s*(?:\[rank\d+\]:\s*)?([\w.]*(?:Error|Exception))\s*:")
FORWARD = "Traceback of forward call that caused the error"


def failure_summary(content):
    lines = content.splitlines()
    errors = [i for i, line in enumerate(lines)
              if (match := ERROR.match(line)) and not match[1].endswith("ChildFailedError")]
    if not errors:
        return content[-3000:]
    ranked = [i for i in errors if RANK.match(lines[i])]
    numerical = [i for i in ranked or errors if "Nonfinite" in lines[i] or "returned nan" in lines[i]]
    index = (numerical or ranked or errors)[-1]
    rank_match = RANK.match(lines[index])
    rank = rank_match[1] if rank_match else None
    sections = []

    contexts = []
    for i, line in enumerate(lines[:index]):
        if "NUMERICS_CONTEXT" not in line:
            continue
        context_rank = re.search(r'"rank"\s*:\s*(\d+)', line)
        if rank is None or (context_rank and context_rank[1] == rank):
            contexts.append(i)
    context_index = contexts[-1] if contexts else 0
    if contexts:
        sections.append(f"NUMERICS_CONTEXT (rank {rank if rank is not None else 'unattributed'}):\n"
                        f"L{context_index + 1}: {lines[context_index]}")

    headers = [i for i in range(context_index, index) if FORWARD in lines[i]]
    for start in headers:
        header_rank = RANK.match(lines[start])
        if rank is not None and header_rank and header_rank[1] != rank:
            continue
        end = start + 1
        while end < index:
            line = lines[end]
            other = RANK.match(line)
            context_rank = re.search(r'"rank"\s*:\s*(\d+)', line) if "NUMERICS_CONTEXT" in line else None
            if rank is not None and ((other and other[1] != rank) or (context_rank and context_rank[1] != rank)):
                end += 1
                continue
            if FORWARD in line or "Traceback (most recent call last):" in line or "NUMERICS_CONTEXT" in line:
                break
            end += 1
            if "Triggered internally at" in line:
                break
        evidence = [f"L{i + 1}: {lines[i]}" for i in range(start, end)
                    if "NUMERICS_CONTEXT" not in lines[i]
                    and not (rank is not None and (other := RANK.match(lines[i])) and other[1] != rank)]
        attribution = f"rank {header_rank[1]}" if header_rank else "unattributed; shared log"
        sections.append(f"FORWARD TRACEBACK ({attribution}):\n" + "\n".join(evidence))

    # A complete backward traceback is useful for numerical failures. For other
    # errors retain the established short context, without the torchrun wrapper.
    start = max(0, index - 14)
    if numerical:
        traceback_starts = [i for i in range(context_index, index)
                            if "Traceback (most recent call last):" in lines[i]
                            and (rank is None or not RANK.match(lines[i]) or RANK.match(lines[i])[1] == rank)]
        if traceback_starts:
            start = traceback_starts[-1]
    evidence = []
    for i in range(start, min(len(lines), index + 5)):
        line = lines[i]
        if "ChildFailedError" in line or "NCCL" in line:
            continue
        other = RANK.match(line)
        if rank is not None and other and other[1] != rank:
            continue
        evidence.append(f"L{i + 1}: {line}")
    sections.append("ACTUAL EXCEPTION:\n" + "\n".join(evidence))
    return "\n\n".join(sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the latest failed numerical diagnostic log; never rerun training.")
    parser.add_argument("workspace", type=Path, help="Data/workspace directory passed to start.sh, or its qwen3-vega subdirectory")
    args = parser.parse_args()
    roots = (args.workspace / "logs", args.workspace / "qwen3-vega" / "logs")
    paths = sorted({path for root in roots for path in root.glob("sweep-*/*-numerics.log")},
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in paths:
        content = path.read_text(errors="replace")
        if any((match := ERROR.match(line)) and not match[1].endswith("ChildFailedError")
               for line in content.splitlines()):
            print(f"Existing failed diagnostic log: {path.resolve()}")
            print(failure_summary(content))
            break
    else:
        raise SystemExit(f"No existing failed *-numerics.log found under {args.workspace}. No jobs were started.")
