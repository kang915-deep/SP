"""Stages for start.sh. One process owns the workspace lock for the whole run."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from numerics_log import failure_summary

SCRIPTS = Path(__file__).resolve().parent
BASE = SCRIPTS.parent
MIN_FREE_GIB = 8
CUDA_OOM = re.compile(r"CUDA out of memory|CUDA error: out of memory|cudaErrorMemoryAllocation", re.I)


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def benchmark(work, inputs, no_overlap=False):
    sweep = Path(tempfile.mkdtemp(prefix="sweep-", dir=work / "logs"))
    trials = []
    for width in (1, 2):
        for stride in (1, 2, 0):
            name = f"m{width}-c{stride}"
            log = sweep / (name + ".log")
            out = sweep / name
            print(f"Benchmark {name}: {log}", flush=True)
            cmd = [
                "bash",
                str(SCRIPTS / "train_4gpu.sh"),
                *inputs,
                "--output",
                str(out),
                "--micro-batch",
                str(width),
                "--checkpoint-every",
                str(stride),
                "--benchmark-steps",
                "5",
            ]
            if no_overlap:
                cmd.append("--no-overlap")
            with log.open("w") as stream:
                rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
            trial = {"micro_batch": width, "checkpoint_every": stride, "log": str(log)}
            if rc:
                message = log.read_text(errors="replace")
                if not CUDA_OOM.search(message):
                    failures = [(f"Benchmark failed ({log})", message)]
                    if "FloatingPointError: Nonfinite" in message:
                        diagnostic_log = sweep / f"{name}-numerics.log"
                        diagnostic_cmd = cmd.copy()
                        diagnostic_cmd[cmd.index("--output") + 1] = str(sweep / f"{name}-numerics")
                        failed_step = re.search(r"FloatingPointError: Nonfinite[^\n]*?\bstep\s+(\d+)", message)
                        steps = max(2, min(5, int(failed_step[1]))) if failed_step else 5
                        diagnostic_cmd[cmd.index("--benchmark-steps") + 1] = str(steps)
                        diagnostic_cmd.append("--diagnose-numerics")
                        print(f"Numerical failure; one diagnostic replay: {diagnostic_log}", flush=True)
                        with diagnostic_log.open("w") as stream:
                            diagnostic_rc = subprocess.run(
                                diagnostic_cmd, stdout=stream, stderr=subprocess.STDOUT
                            ).returncode
                        failures.append((
                            f"Diagnostic replay ({diagnostic_log}), exit {diagnostic_rc}; training remains stopped",
                            diagnostic_log.read_text(errors="replace"),
                        ))
                    summaries = []
                    for heading, content in failures:
                        summaries.append(f"{heading}:\n{failure_summary(content)}")
                    raise RuntimeError("\n\n".join(summaries))
                trial["status"] = "cuda_oom"
            else:
                report = json.loads((out / "throughput.json").read_text())
                ranks = report["ranks"]
                speed = report["qa_per_s"]
                if (
                    report["measured_steps"] != 4
                    or report["qa"] != 256
                    or sorted(r["rank"] for r in ranks) != list(range(4))
                    or not math.isfinite(speed)
                    or speed <= 0
                ):
                    raise ValueError(f"Incomplete four-GPU measurement in {out}")
                protocol = report["protocol"]
                if (protocol.get("diagnose_numerics", False)
                        or protocol.get("overlap_comm", True) != (not no_overlap)
                        or any(r.get("training_acceptance", {}).get("accepted") is not True
                               or r["training_acceptance"]["expected_steps"] != 5
                               or r["training_acceptance"]["final_engine_global_steps"] != 5
                               for r in ranks)):
                    raise ValueError(f"Five real optimizer updates were not accepted on every GPU: {out}")
                free = min(r["min_device_free_gib"] for r in ranks)
                trial.update(
                    qa_per_s=speed,
                    min_free_gib=free,
                    report=str(out / "throughput.json"),
                    status="eligible" if free >= MIN_FREE_GIB else "low_headroom",
                )
            trials.append(trial)
            write_json(sweep / "trials.json", trials)
            print(trial, flush=True)
    eligible = [t for t in trials if t["status"] == "eligible"]
    if not eligible:
        raise RuntimeError(
            f"No measured configuration left {MIN_FREE_GIB} GiB on every GPU. See {sweep}"
        )
    winner = max(eligible, key=lambda t: (t["qa_per_s"], t["min_free_gib"]))
    return {
        "winner": winner,
        "min_free_gib": MIN_FREE_GIB,
        "trials": trials,
        "gpus": os.environ["CUDA_VISIBLE_DEVICES"],
        "effective_batch": 64,
        "overlap_comm": not no_overlap,
        "acceptance_version": 1,
    }


def launch(work):
    result = work / "results.json"
    if result.is_file():
        print(f"Already complete: {result}")
        return
    data = work / "data"
    assets = work / "assets"
    runs = work / "runs"
    runs.mkdir(exist_ok=True)
    selection = work / "speed_selection.json"
    attempts = sorted(runs.glob("full-*"))
    run = attempts[-1] if attempts else None
    markers = sorted(run.glob("step-*.json")) if run else []
    has_progress = run and (markers or (run / "selected.json").exists())
    choice = json.loads(selection.read_text()) if selection.exists() else None
    setting = os.environ.get("VEGA_NO_OVERLAP")
    if setting not in (None, "0", "1"):
        raise ValueError("VEGA_NO_OVERLAP must be 0 or 1")
    no_overlap = setting == "1" if setting is not None else not (choice or {}).get("overlap_comm", True)
    if markers and not (run / "selected.json").exists() and markers[-1].stem != "step-004657":
        checkpoint_protocol = json.loads((run / "protocol.json").read_text())
        recorded = not checkpoint_protocol.get("overlap_comm", True)
        if setting is not None and no_overlap != recorded:
            raise ValueError("Communication overlap differs from the existing checkpoint; leave VEGA_NO_OVERLAP unset to resume its recorded setting")
        no_overlap = recorded
    print(f"Training communication overlap: {not no_overlap}", flush=True)
    inputs = ["--qwen", str(assets / "qwen"), "--wan", str(assets / "wan"), "--data", str(data)]
    py = sys.executable

    print("Preparing verified weights and datasets.", flush=True)
    subprocess.run(
        [
            py,
            str(SCRIPTS / "download_models.py"),
            "--source",
            "modelscope",
            "--sp-text",
            "--output",
            str(assets),
        ],
        check=True,
    )
    if not (data / "media_ready.json").is_file():
        subprocess.run([py, str(SCRIPTS / "prepare_data.py"), "--output", str(data)], check=True)
    vsi = Path(os.environ["VSI_BENCH_DIR"])
    if not (vsi / "benchmark.json").is_file():
        subprocess.run([py, str(BASE / "sp" / "prepare.py"), "--output", str(vsi)], check=True)

    if not has_progress and (not choice or choice["gpus"] != os.environ["CUDA_VISIBLE_DEVICES"]
                             or choice.get("overlap_comm", True) != (not no_overlap)
                             or choice.get("acceptance_version") != 1):
        choice = benchmark(work, inputs, no_overlap=no_overlap)
        if selection.exists():
            with tempfile.NamedTemporaryFile(prefix="speed-selection-prior-", suffix=".json",
                                             dir=work / "logs", delete=False) as previous:
                previous.write(selection.read_bytes())
        write_json(selection, choice)
    if not choice:
        raise RuntimeError("Missing speed_selection.json for existing training checkpoints")
    if not has_progress:
        run = runs / f"full-{len(attempts)+1:04d}"
    print(f"Selected {choice['winner']}; training run: {run}", flush=True)
    last = markers[-1] if markers else None
    # The final step marker is durable before train.py writes selected.json.
    if last and last.stem == "step-004657" and not (run / "selected.json").exists():
        best = json.loads(last.read_text())["best"]
        write_json(run / "selected.json", dict(best, tag=f"step-{best['step']:06d}"))
    if (run / "selected.json").exists():
        subprocess.run(
            [py, str(BASE / "sp" / "pipeline.py"), *inputs, "--output", str(run)], check=True
        )
    else:
        winner = choice["winner"]
        cmd = [
            "bash",
            str(SCRIPTS / "train_4gpu.sh"),
            *inputs,
            "--output",
            str(run),
            "--micro-batch",
            str(winner["micro_batch"]),
            "--checkpoint-every",
            str(winner["checkpoint_every"]),
            "--save-every",
            "4657",
        ]
        if no_overlap:
            cmd.append("--no-overlap")
        if last:
            cmd.extend(["--resume", last.stem])
        subprocess.run(cmd, check=True)
    final = run / "sp16k_vsi" / "results.json"
    summary = json.loads(final.read_text())
    if summary["questions"] != 5130 or set(summary["overall"]) != {"baseline", "sp"}:
        raise ValueError(f"Incomplete paired evaluation: {final}")
    write_json(result, dict(summary, detailed_results=str(final.relative_to(work))))
    print(f"Complete: {result}\n{summary['overall']}", flush=True)


if __name__ == "__main__":
    launch(Path(sys.argv[1]).resolve())
