"""Rebuild a fresh source/type-stratified 16K and download the official full benchmark."""

from collections import defaultdict
from pathlib import Path
import hashlib
import re
import shutil
import os
import subprocess
import zipfile
from common import digest, key, read, save

VSI_REV = "3a6a1cf9ac54c403509b15260897cc6ddc41b408"
TYPE = {
    "object_counting": "absolute_count",
    "object_abs_distance": "absolute_distance_object",
    "object_size_estimation": "absolute_size_object",
    "room_size_estimation": "absolute_size_room",
    "object_rel_distance": "relative_distance_object",
    "obj_appearance_order": "appearance_order",
    "route_planning": "route_planning",
    **{"object_rel_direction_" + k: "relative_direction_object" for k in ("easy", "medium", "hard")},
}


def select(rows, count=16000):
    cells = defaultdict(list)
    for r in rows:
        if r["id"].startswith("vsi-") and r["kind"] == "video":
            cells[(r["media"].split("/")[0], r["type"])].append(r)
    total = sum(map(len, cells.values()))
    if total < count:
        raise ValueError("Insufficient VSI training rows")
    quotas = {c: len(v) * count // total for c, v in cells.items()}
    order = sorted(cells, key=lambda c: (-(len(cells[c]) * count % total), c))
    for c in order[: count - sum(quotas.values())]:
        quotas[c] += 1
    selected = []
    for c, members in sorted(cells.items()):
        members.sort(key=lambda r: key("sp16k-seed42:" + r["id"]))
        selected.extend(members[: quotas[c]])
    return selected


def normalize(r, data):
    prompt = r["conversations"][0]["value"].removeprefix("<video>").strip()
    options = re.findall(r"^[A-H]\.\s+(.+)$", prompt, re.M)
    question = (
        prompt.removeprefix("These are frames of a video.")
        .strip()
        .split("\nOptions:")[0]
        .split("\nPlease answer")[0]
    )
    return {
        "id": r["id"],
        "media": r["media"],
        "group": r.get("group", r["media"]),
        "path": str((data / r["path"]).resolve()),
        "type": r["type"],
        "question": question,
        "prompt": prompt,
        "options": options,
        "answer": r["conversations"][-1]["value"],
    }


def prepare_sp(data, out):
    from legacy_candidates import build_banks

    train = select(read(data / "train.json"))
    splits = {"train": train, "tune": read(data / "tune.json"), "confirm": read(data / "confirm.json")}
    for part, rows in splits.items():
        splits[part] = [normalize(r, data) for r in rows]
    ids = [r["id"] for rows in splits.values() for r in rows]
    if len(set(ids)) != len(ids) or [len(rows) for rows in splits.values()] != [16000, 2036, 2049]:
        raise ValueError("Invalid SP partitions")
    media = [set(r["media"] for r in rows) for rows in splits.values()]
    if any(a & b for i, a in enumerate(media) for b in media[i + 1 :]):
        raise ValueError("Overlapping SP media")
    train = splits["train"]
    id_hash = hashlib.sha256(("\n".join(sorted(r["id"] for r in train)) + "\n").encode()).hexdigest()
    if id_hash != "f2ba5018a37c056a885d356266a55749be06af63e0c5593efdbd98d18524622c":
        raise ValueError("New 16K sample differs from the released selection")
    save(out / "banks.json", build_banks(train, {r["id"]: r["answer"] for r in train}))
    for part, rows in splits.items():
        save(out / (part + ".json"), rows)
    save(
        out / "sp_split_manifest.json",
        {
            "train_id_sha256": id_hash,
            "splits": {
                part: {"questions": len(rows), "sha256": digest(out / (part + ".json"))}
                for part, rows in splits.items()
            },
        },
    )


def benchmark(out):
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    with (out / "download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prepare_benchmark(out)


def _prepare_benchmark(out):
    import json

    def fetch(name):
        checks = {
            "test.jsonl": "aa3ff8cc2010e555f4f865ff96483812ec0b275005b68562f53bb114ad4db9db",
            "scannet.zip": "787b0c061bde5c1f5e076012c1239340fdb1330787c644977c7cad5cdbe1d548",
            "scannetpp.zip": "164b2314107e070c7d8a652897404904adf36a8868c2293be04382727d9a19be",
            "arkitscenes.zip": "005232fa20ccfa287255ca96c4d0c0c0863c24bdc1a40a89165b75f509bf4907",
        }
        path = out / "downloads" / name
        path.parent.mkdir(exist_ok=True, parents=True)
        part = path.with_suffix(".part")
        source = path if path.exists() else part
        if source == part:
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
            url = f"{endpoint}/datasets/nyu-visionx/VSI-Bench/resolve/{VSI_REV}/{name}"
            subprocess.run(
                ["curl", "-fsSL", "--retry", "4", "-C", "-", "-o", str(part), url], check=True
            )
        if digest(source) != checks[name]:
            raise ValueError(f"Invalid official file: {name}")
        if source == part:
            part.replace(path)
        return path

    source = fetch("test.jsonl")
    docs = [json.loads(line) for line in Path(source).read_text().splitlines()]
    if len({d["id"] for d in docs}) != len(docs) or len(docs) != 5130:
        raise ValueError("Expected the complete 5130-question benchmark")
    rows = []
    for dataset in sorted({d["dataset"] for d in docs}):
        dest = out / "videos" / dataset
        dest.mkdir(parents=True, exist_ok=True)
        names = {str(d["scene_name"]) + ".mp4" for d in docs if d["dataset"] == dataset}
        with zipfile.ZipFile(fetch(dataset + ".zip")) as z:
            files = {
                Path(m.filename).name: m for m in z.infolist() if not m.filename.startswith("__MACOSX/")
            }
            for name in sorted(names):
                m = files[name]
                path = dest / name
                if path.exists() and path.stat().st_size == m.file_size:
                    continue
                tmp = path.with_suffix(".part")
                with z.open(m) as src, tmp.open("wb") as f:
                    shutil.copyfileobj(src, f)
                tmp.replace(path)
    for d in docs:
        group = d["dataset"] + "/" + str(d["scene_name"])
        options = d.get("options") or []
        text = ["These are frames of a video.", d["question"]]
        if options:
            text.extend(
                [
                    "Options:",
                    *options,
                    "Answer with the option's letter from the given choices directly.",
                ]
            )
        else:
            text.append("Please answer the question using a single word or phrase.")
        rows.append(
            {
                "id": "bench-" + str(d["id"]),
                "media": group,
                "group": group,
                "path": str(out / "videos" / (group + ".mp4")),
                "question": d["question"],
                "type": TYPE[d["question_type"]],
                "benchmark_type": d["question_type"],
                "prompt": "\n".join(text),
                "options": options,
            }
        )
    save(out / "benchmark_answers.json", {str(d["id"]): d["ground_truth"] for d in docs})
    save(out / "benchmark.json", rows)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Download and check all official VSI-Bench videos and questions"
    )
    p.add_argument("--output", type=Path, required=True)
    benchmark(p.parse_args().output.resolve())
