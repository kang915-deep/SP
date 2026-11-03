"""Automatic post-training stages, resumable on the GPUs explicitly assigned by the caller."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from common import digest, read, save

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def main():
    p = argparse.ArgumentParser()
    for k in ("qwen", "wan", "data", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--after-train", action="store_true")
    if "--after-train" in sys.argv:
        args, _ = p.parse_known_args()
    else:
        args = p.parse_args()
    run = args.output.resolve()
    if not (run / "selected.json").exists() and args.after_train:
        print("VEGA budget unfinished. Downstream stages start after selection.", flush=True)
        return
    chosen = read(run / "selected.json")
    protocol = read(run / "protocol.json")
    if (
        protocol["smoke"]
        or protocol["train_sha256"] != digest(args.data / "train.json")
        or protocol["tune_sha256"] != digest(args.data / "tune.json")
    ):
        raise ValueError("A completed full VEGA run and its original data directory are required")
    cards = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(cards) != protocol["world"] or len(set(cards)) != len(cards):
        raise ValueError("Use the explicitly assigned training GPUs")
    root = run / "sp16k_vsi"
    root.mkdir(exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export = root / "export"
        if not export.exists():
            # A unique staging directory allows retry after an interrupted large export.
            import uuid

            staging = root / ("export.incomplete-" + uuid.uuid4().hex)
            subprocess.run(
                [sys.executable, str(BASE / "export.py"), "--run", str(run), "--output", str(staging)],
                check=True,
            )
            staging.rename(export)
        identity = dict(
            protocol="vega-sp16k-v1",
            selected=chosen["tag"],
            export_sha256=digest(export / "vega.safetensors"),
            train=protocol["train_sha256"],
            tune=protocol["tune_sha256"],
            confirm=digest(args.data / "confirm.json"),
        )
        if read(export / "config.json")["checkpoint"] != chosen["tag"]:
            raise ValueError("Selected checkpoint differs from the export")
        if (root / "identity.json").exists() and read(root / "identity.json") != identity:
            raise ValueError("Post-training cache belongs to another experiment")
        save(root / "identity.json", identity)
        if (root / "results.json").exists():
            print(root / "results.json")
            return
        src = os.environ["VEGA_SRC"]
        env = dict(
            os.environ, PYTHONPATH=os.pathsep.join((str(BASE), src, os.environ.get("PYTHONPATH", "")))
        )
        env.pop("HF_HUB_OFFLINE", None)
        from prepare import prepare_sp, benchmark

        prepare_sp(args.data.resolve(), root)
        if not (root / "context.pt").exists():
            cmd = [sys.executable, str(HERE / "make_context.py"), str(root / "context.pt")]
            if (args.wan / "models_t5_umt5-xxl-enc-bf16.pth").exists():
                cmd.extend(["--source-dir", str(args.wan)])
            subprocess.run(cmd, check=True, env=env)

        def extract(parts):
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={len(cards)}",
                str(HERE / "extract.py"),
                "--qwen",
                str(args.qwen),
                "--wan",
                str(args.wan),
                "--export",
                str(export),
                "--feature-dir",
                str(root),
                "--parts",
                *parts,
            ]
            subprocess.run(cmd, check=True, env=env)

        if not (root / "sp_selection.json").exists():
            extract(["train", "tune"])
            subprocess.run(
                [sys.executable, str(HERE / "head.py"), "train", str(root)], check=True, env=env
            )
        if not (root / "confirm_results.json").exists():
            extract(["confirm"])
            subprocess.run(
                [sys.executable, str(HERE / "head.py"), "confirm", str(root)], check=True, env=env
            )
        if not (root / "benchmark.json").exists():
            cache = Path(os.environ.get("VSI_BENCH_DIR", root)).resolve()
            benchmark(cache)
            if cache != root:
                save(root / "benchmark_answers.json", read(cache / "benchmark_answers.json"))
                save(root / "benchmark.json", read(cache / "benchmark.json"))
        extract(["benchmark"])
        from head import predict
        from score_vsi import score_results

        predict(root, "benchmark")
        score_results(root)
        print(root / "results.json", flush=True)


if __name__ == "__main__":
    main()
