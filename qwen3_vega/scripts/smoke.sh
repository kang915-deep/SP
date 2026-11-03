#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set the single GPU ID assigned to you}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'One assigned GPU required'; exit 2; }
: "${VEGA_SRC:?Official VEGA src directory required}"
export PYTHONPATH="$VEGA_SRC${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
root="$(cd "$(dirname "$0")/.." && pwd)"
assets="${1:?model directory}"; data="${2:?smoke data directory}"; out="${3:?new output directory}"
common=(--qwen "$assets/qwen" --wan "$assets/wan" --data "$data" --output "$out" --smoke --workers 1)
python -m torch.distributed.run --standalone --nproc-per-node=1 "$root/train.py" "${common[@]}" --stop-after 2
python -m torch.distributed.run --standalone --nproc-per-node=1 "$root/train.py" "${common[@]}" --resume step-000002
python "$root/export.py" --run "$out" --output "$out/export"
python "$root/check_export.py" --qwen "$assets/qwen" --wan "$assets/wan" --data "$data" --export "$out/export"
