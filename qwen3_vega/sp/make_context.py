"""Prepare the independent SP Wan prompt once before loading the VLM."""

import argparse
from pathlib import Path
import os
import torch
from huggingface_hub import snapshot_download
from common import digest
from qwen_vl.model.generative_encoders.wan.modules.t5 import T5EncoderModel

REV = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument(
        "--source-dir",
        type=Path,
        help="Optional local official Wan directory containing T5 and tokenizer",
    )
    args = p.parse_args()
    root = args.source_dir or Path(
        snapshot_download(
            "Wan-AI/Wan2.1-T2V-1.3B",
            revision=REV,
            allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"],
        )
    )
    checkpoint = root / "models_t5_umt5-xxl-enc-bf16.pth"
    if digest(checkpoint) != "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d":
        raise ValueError("Unexpected Wan UMT5 checkpoint")
    device = torch.device("cuda", 0)
    encoder = T5EncoderModel(
        512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(root / "google/umt5-xxl"),
    )
    with torch.inference_mode():
        context = encoder(["An indoor visual scene."], device)[0].cpu()
    part = args.output.with_suffix(".tmp")
    torch.save(context, part)
    os.replace(part, args.output)
