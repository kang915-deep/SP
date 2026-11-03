"""Fetch only required Qwen/Wan files and verify contents against the training run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

def download(row):
    url, dest, size, expected, git_blob = row
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    path = dest if dest.exists() else part
    if path == part:
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "4", "--connect-timeout", "20", "-C", "-", "-o", str(part), url], check=True)
    digest = hashlib.sha1(f"blob {size}\0".encode()) if git_blob else hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    if path.stat().st_size != size or digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {path}")
    if path == part:
        os.replace(part, dest)
    print("verified", dest, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--sp-text", action="store_true", help="Also prepare the independent SP Wan text encoder")
    args = parser.parse_args()
    public = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    manifest = json.loads(Path(__file__).with_name("qwen_manifest.json").read_text())
    def prefix(repo, rev):
        if args.source == "modelscope":
            return f"https://www.modelscope.cn/models/{repo}/resolve/master"
        return f"{public}/{repo}/resolve/{rev}"
    jobs = []
    for row in manifest["siblings"]:
        name = row["rfilename"]
        if name.startswith(".") or name == "README.md":
            continue
        jobs.append((f"{prefix(manifest['id'],manifest['sha'])}/{name}", args.output/"qwen"/name,
                     row["size"], row["lfs"]["sha256"] if "lfs" in row else row["blobId"], "lfs" not in row))
    wan = {"config.json": (249,"ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce"),
           "Wan2.1_VAE.pth": (507609880,"38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"),
           "diffusion_pytorch_model.safetensors": (5676070424,"96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f")}
    if args.sp_text:
        wan.update({
            "models_t5_umt5-xxl-enc-bf16.pth": (11361920418,"7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"),
            "google/umt5-xxl/special_tokens_map.json": (6623,"14855e7052ffbb595057dfd791d293c1c940db2c"),
            "google/umt5-xxl/spiece.model": (4548313,"e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
            "google/umt5-xxl/tokenizer.json": (16837417,"6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
            "google/umt5-xxl/tokenizer_config.json": (61728,"4e1cc1cd85599ce0b47fd0a746af188fe4043ff2"),
        })
    wan_url=prefix("Wan-AI/Wan2.1-T2V-1.3B","37ec512624d61f7aa208f7ea8140a131f93afc9a")
    for name,(size,digest) in wan.items():
        jobs.append((f"{wan_url}/{name}",args.output/"wan"/name,size,digest,len(digest)==40))
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(download,jobs))
