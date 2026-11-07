"""Replay the first two microbatches using an existing four-GPU installation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from numerics_log import failure_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_disk", type=Path, help="Same DATA_DISK passed to start.sh")
    parser.add_argument("gpus", help="Four assigned physical GPU indices, e.g. 0,1,2,3")
    args = parser.parse_args()
    cards = args.gpus.split(",")
    if len(cards) != 4 or any(not card.isdecimal() for card in cards) or len(set(map(int, cards))) != 4:
        parser.error("Specify four distinct nonnegative GPU indices")
    work = (args.data_disk / "qwen3-vega").resolve()
    key = hashlib.sha256(str(work).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    runtime = Path(os.environ["VEGA_RUNTIME_DIR"]) if "VEGA_RUNTIME_DIR" in os.environ else (
        work if (work / "env/bin/python").is_file() else cache / "qwen3-vega" / key)
    python = runtime / "env/bin/python"
    upstream = runtime / "upstream/VEGA-3D/src"
    required = [python, upstream, work / "assets/qwen", work / "assets/wan", work / "data/media_ready.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Existing installation/data required; this command downloads nothing: " + ", ".join(missing))

    # Same workspace lock as start.sh; never wait for cards or an active job.
    with (work / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuids = [subprocess.check_output(
            ["nvidia-smi", "-i", card, "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip() for card in cards]
        if len(set(uuids)) != 4 or any(not uuid.startswith("GPU-") for uuid in uuids):
            raise ValueError("GPU indices did not resolve to four distinct device UUIDs")
        base = Path(__file__).resolve().parents[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                   PYTHONPATH=str(base) + ":" + str(upstream) + (
                       ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
                   TORCH_EXTENSIONS_DIR=str(runtime / "cache/torch-extensions"),
                   TRITON_CACHE_DIR=str(runtime / "cache/triton"), HF_HOME=str(runtime / "cache/huggingface"))
        log_root = runtime / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        diagnostic = Path(tempfile.mkdtemp(prefix="linear-diagnostic-", dir=log_root))
        out = diagnostic / "result"
        log = diagnostic / "diagnostic.log"
        command = [str(python), "-B", "-u", "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=4", "--max-restarts=0", str(base / "train.py"),
                   "--qwen", str(work / "assets/qwen"), "--wan", str(work / "assets/wan"),
                   "--data", str(work / "data"), "--output", str(out),
                   "--micro-batch", "1", "--checkpoint-every", "1", "--benchmark-steps", "2",
                   "--diagnose-numerics", "--diagnostic-micros", "2"]
        (diagnostic / "command.json").write_text(json.dumps(command, indent=2))
        print(f"Two microbatches per rank, zero optimizer updates. Log: {log}", flush=True)
        with log.open("x") as stream:
            rc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        reports = sorted(out.glob("linear-rank*/first_nonfinite.json"))
        for path in reports:
            report = json.loads(path.read_text())
            print("FIRST_NONFINITE", json.dumps({"report": str(path), **report}, allow_nan=False), flush=True)
        if rc and not reports:
            print(failure_summary(log.read_text(errors="replace")), flush=True)
        if not rc:
            print("Prefix completed. Synchronized diagnostics may hide a timing-dependent failure; "
                  "this is not training acceptance.", flush=True)
        print(f"Diagnostics: {diagnostic}. Formal training remains stopped.", flush=True)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
