#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set the four GPU IDs assigned to you}"
IFS=, read -ra cards <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#cards[@]} == 4 && "$(printf '%s\n' "${cards[@]}" | sort -u | wc -l)" == 4 ]] || { echo 'Four assigned GPUs required'; exit 2; }
: "${VEGA_SRC:?Set VEGA_SRC to the official spatial-reasoning checkout src directory}"
export PYTHONPATH="$VEGA_SRC${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec python -m torch.distributed.run --standalone --nproc-per-node=4 "$(dirname "$0")/../train.py" "$@"
