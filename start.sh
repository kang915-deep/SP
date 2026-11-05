#!/usr/bin/env bash
# Launch the complete experiment on four explicitly assigned physical GPUs.
set -euo pipefail
repo="$(cd "$(dirname "$0")" && pwd)"
if [[ ${1:-} != --worker ]]; then
  if [[ $# != 2 || ! $2 =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "Usage: bash start.sh DATA_DISK GPU0,GPU1,GPU2,GPU3" >&2
    exit 2
  fi
  ids="$2"
  for cmd in nvidia-smi git curl flock nohup; do command -v "$cmd" >/dev/null || { echo "Missing $cmd" >&2; exit 2; }; done
  IFS=, read -ra cards <<< "$ids"
  uuids=()
  for card in "${cards[@]}"; do
    uuid="$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)"
    [[ $uuid == GPU-* ]] || { echo "Invalid GPU $card" >&2; exit 2; }
    uuids+=("$uuid")
  done
  [[ $(printf '%s\n' "${uuids[@]}" | sort -u | wc -l) -eq 4 ]] || { echo "Four distinct GPUs required" >&2; exit 2; }
  [[ -d "$1" && -w "$1" ]] || { echo "Data disk must be an existing writable directory: $1" >&2; exit 2; }
  work="$1/qwen3-vega"
  mkdir -p "$work/logs"
  work="$(cd "$work" && pwd)"
  exec 9>>"$work/launch.lock"
  flock -n 9 || { echo "A launcher is already active in $work" >&2; exit 1; }
  if [[ -s "$work/results.json" ]]; then echo "Already complete: $work/results.json"; exit 0; fi
  gpu_list="$(IFS=,; echo "${uuids[*]}")"
  # Keep live stdout on local storage. BOS/FUSE may reject tail/read on an open object.
  key="$(printf '%s' "$work" | sha256sum | cut -c1-16)"
  logdir="${VEGA_LOG_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/qwen3-vega/$key/logs}"
  mkdir -p "$logdir"
  logdir="$(cd "$logdir" && pwd)"
  log="$(mktemp "$logdir/launch-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
  printf '%s\n' "$log" > "$work/logs/$(basename "$log").location.txt"
  nohup bash "$repo/start.sh" --worker "$work" "$gpu_list" 9>&9 >"$log" 2>&1 </dev/null &
  echo "Started PID $! on GPUs $ids"
  printf 'Follow: tail -f %q\n' "$log"
  echo "Results: $work/results.json"
  echo "Live log is on local storage; data and checkpoints remain in $work"
  exit 0
fi

shift
flock -n 9 || { echo "Use the public entry point: bash start.sh DATA_DISK GPU_IDS" >&2; exit 2; }
work="$1"
export CUDA_VISIBLE_DEVICES="$2" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
runtime="${VEGA_RUNTIME_DIR:-$work}"
# venv creates lib64 -> lib even in --copies mode on 64-bit Linux.
if [[ -z ${VEGA_RUNTIME_DIR:-} && ! -L "$work/.runtime-link-check" ]] &&
   ! ln -s launch.lock "$work/.runtime-link-check" 2>/dev/null; then
  key="$(printf '%s' "$work" | sha256sum | cut -c1-16)"
  runtime="${XDG_CACHE_HOME:-$HOME/.cache}/qwen3-vega/$key"
fi
mkdir -p "$runtime"
runtime="$(cd "$runtime" && pwd -P)"
if [[ "$runtime" != "$work" ]]; then export VEGA_RANGE_ARCHIVES=1; fi
if [[ ! -L "$runtime/.runtime-link-check" ]] && ! ln -s .runtime-link-target "$runtime/.runtime-link-check"; then
  echo "Set VEGA_RUNTIME_DIR to a writable local Linux directory that supports symlinks." >&2
  exit 1
fi
export HF_HOME="$runtime/cache/huggingface" PIP_CACHE_DIR="$runtime/cache/pip"
export UV_CACHE_DIR="$runtime/cache/uv" UV_PYTHON_INSTALL_DIR="$runtime/python"
export TORCH_EXTENSIONS_DIR="$runtime/cache/torch-extensions" TRITON_CACHE_DIR="$runtime/cache/triton"
export TMPDIR="$runtime/tmp" VSI_BENCH_DIR="$work/vsi-bench"
if [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
mkdir -p "$TMPDIR"
trap 'echo "Launch failed at line $LINENO (exit $?). See preceding error."' ERR
echo "Workspace: $work"
echo "Software runtime: $runtime"
df -h "$work" "$runtime"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
cd "$repo"
envdir="$runtime/env"
if [[ ! -f "$envdir/bin/activate" ]]; then
  # Reuse system wheels only for the tested torch build.
  py=""
  for name in python3.12 python3.11 python3; do
    if command -v "$name" >/dev/null && "$name" -c 'import ensurepip, sys, venv; sys.exit(sys.version_info[:2] not in [(3,11),(3,12)])' 2>/dev/null; then py="$name"; break; fi
  done
  if [[ -z "$py" ]]; then
    python3 -m pip install --target "$runtime/bootstrap" 'uv==0.8.22'
    "$runtime/bootstrap/bin/uv" python install 3.12
    py="$("$runtime/bootstrap/bin/uv" python find --managed-python 3.12)"
  fi
  flags=()
  if "$py" -c 'import torch,torchvision; assert torch.__version__=="2.8.0+cu128" and torchvision.__version__.split("+")[0]=="0.23.0"' >/dev/null 2>&1; then flags+=(--system-site-packages); fi
  "$py" -m venv "${flags[@]}" "$envdir"
fi
source "$envdir/bin/activate"
requirements="$repo/qwen3_vega/requirements.txt"
stamp="$(sha256sum "$requirements" | cut -d' ' -f1)"
if [[ ! -e "$envdir/.ready-$stamp" ]]; then
  python -m ensurepip --upgrade
  python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
  python -m pip install setuptools wheel packaging
  DS_BUILD_OPS=0 python -m pip install -r "$requirements" --no-build-isolation
fi
python - <<'PY'
import torch, torchvision
if torch.version.cuda != "12.8" or torch.cuda.device_count() != 4:
    raise SystemExit("Expected cu128 PyTorch and exactly four visible CUDA devices")
for i in range(4):
    print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
PY
upstream="$runtime/upstream/VEGA-3D"
rev=ca64583b3f961849a6a0a282a3c16918321adf87
if [[ ! -d "$upstream" ]]; then
  mkdir -p "$runtime/upstream"
  staging="$(mktemp -d "$runtime/upstream/clone-XXXXXX")"
  git -C "$staging" init -q
  git -C "$staging" fetch --depth 1 --no-tags https://github.com/H-EmbodVis/VEGA-3D.git "$rev"
  git -C "$staging" checkout --detach FETCH_HEAD
  mv -T "$staging" "$upstream"
fi
[[ $(git -C "$upstream" rev-parse HEAD) == "$rev" ]] || { echo "Unexpected official VEGA revision"; exit 1; }
export VEGA_SRC="$upstream/src"
export PYTHONPATH="$repo/qwen3_vega:$VEGA_SRC${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import deepspeed, decord, cv2, pynvml; from batched_vega import BatchedVega; from qwen_vl.model.generative_encoders.wan_vace_encoder import WanModel'
touch "$envdir/.ready-$stamp"
if [[ -z ${HF_ENDPOINT:-} ]]; then
  if curl -fsSL --max-time 15 -o /dev/null https://huggingface.co/api/datasets/nyu-visionx/VSI-Bench; then
    export HF_ENDPOINT=https://huggingface.co
  else
    export HF_ENDPOINT=https://hf-mirror.com
  fi
fi
python qwen3_vega/scripts/one_click.py "$work"
