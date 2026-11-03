"""Rebuild the project's exact 234277 VSI + 63750 Hound training set from public sources."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from archive_download import ArchiveStream, download, download_pieces
from hound_archive_manifest import ITEMS
from vsi_media_archives import VSI_ARCHIVES

HERE = Path(__file__).resolve().parent
SPLITS = ("train","tune","confirm")
TYPES = {"absolute_count","absolute_distance_object","absolute_size_object","absolute_size_room",
         "relative_distance_object","relative_direction_object","appearance_order"}
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--stage",choices=["annotations","media","all"],default="all")
parser.add_argument("--archives-dir",type=Path)
args=parser.parse_args()
out=args.output.resolve()
out.mkdir(exist_ok=True,parents=True)
archives=(args.archives_dir or out/"downloads").resolve()
archives.mkdir(exist_ok=True,parents=True)

def fetch(repo,rev,name,digest,size=None):
    target=archives/Path(name).name
    part=target.with_suffix(target.suffix+".part")
    path=target if target.is_file() else part
    if path==part:
        endpoint=os.environ.get("HF_ENDPOINT","https://huggingface.co").rstrip("/")
        url=f"{endpoint}/datasets/{repo}/resolve/{rev}/{name}"
        fallback=None
        hound=repo=="ShareGPTVideo/train_video_and_instruction"
        if hound:
            mirror=f"https://modelscope.cn/datasets/wzywuan/train_video_and_instruction/resolve/dd8f96756c1c2b407adc78673acc96e5ec93f16f/{name}"
            if os.environ.get("HOUND_SOURCE")=="modelscope":
                url=mirror
            else:
                fallback=mirror
        if size is not None and (hound or part.exists() or os.environ.get("VEGA_RANGE_ARCHIVES")=="1"
                                 or part.with_name(part.name+".ranges").is_dir()):
            pieces=download_pieces(url,part,size,fallback_url=fallback)
            with ArchiveStream(pieces) as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest()!=digest:
                    raise ValueError(f"Archive SHA-256 mismatch before extraction: {name}")
            return pieces
        download(url,part,size)
    with path.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=digest:
            raise ValueError(f"Checksum mismatch: {path}")
    if path==part:os.replace(part,target)
    return [target] if size is not None else target

def dataset_digest(rows):
    digest=hashlib.sha256()
    for row in sorted(rows,key=lambda r:r["id"]):
        cols=[row[key] for key in ("id","media","type","conversations")]
        digest.update((json.dumps(cols,ensure_ascii=False,separators=(",",":"))+"\n").encode())
    return digest.hexdigest()

if args.stage!="media":
    vsi=fetch("nyu-visionx/VSI-590K",VSI_ARCHIVES[0]["revision"],"vsi_590k.jsonl",
              "025355ea328c0f1f73ba997fd52070a1f487248acd1fb9d8ac727886b319d231")
    hound=fetch("zd11024/VG-LLM-Data","4dc7eb0e9a5b481c3683f2af525f4aa4f08a7b8e","train/llava_hound_64k.json",
                "cf7c4ac7a825d38cba89aac709497e65f1c3d6537fa3e9f5f4a266348f352c2a")
    membership=gzip.decompress((HERE/"vsi_split_ids.gz").read_bytes())
    result={s:[] for s in SPLITS}
    with vsi.open(encoding="utf8") as f:
        for idx,line in enumerate(f,1):
            code=membership[idx]
            if code==0:continue
            row=json.loads(line)
            media=row["video"];typ=row["question_type"]
            assert typ in TYPES and media
            prompt=row["conversations"][0]["value"].replace("<image>","").replace("\\n","\n").strip()
            answer=row["conversations"][1]["value"].strip()
            if re.search(r"^A\.\s*.+$",prompt,re.M):
                answer=re.fullmatch(r"([A-H])(?:\.\s*(.*))?",answer)[1]
            result[SPLITS[code-1]].append({"id":f"vsi-{idx}","kind":"video","type":typ,"media":media,
                "path":"media/vsi/"+media,"conversations":[{"from":"human","value":"<video>\n"+prompt},{"from":"gpt","value":answer}]})
    for row in json.loads(hound.read_text()):
        media=row["video"]
        result["train"].append({"id":"hound-"+row["id"],"kind":"frames","type":"general_video","media":media,
            "path":"media/hound/"+Path(media).name,"conversations":row["conversations"]})
    checks=json.loads((HERE/"split_checksums.json").read_text())
    for s,rows in result.items():
        if len(rows)!=checks[s]["rows"] or dataset_digest(rows)!=checks[s]["qa_sha256"]:
            raise ValueError(f"{s} differs from the original selected IDs, media, types, questions or answers")
    for s,rows in result.items():
        part=out/(s+".json.part");part.write_text(json.dumps(rows,ensure_ascii=False));os.replace(part,out/(s+".json"))
    print("Verified original QA splits:",{s:len(rr) for s,rr in result.items()},flush=True)

if args.stage!="annotations":
    wanted={"video":set(),"frames":set()}
    for split in SPLITS:
        for r in json.loads((out/(split+".json")).read_text()):
            wanted[r["kind"]].add(r["media"] if r["kind"]=="video" else Path(r["media"]).name)
    covered={"video":set(),"frames":set()}
    for kind,items in (("video",VSI_ARCHIVES),("frames",ITEMS)):
        # Resume the interrupted archive before rechecking earlier multi-GB archives.
        resumed = {}
        for item in items:
            if (archives/(Path(item["name"]).name+".part")).is_file():
                resumed[item["name"]] = fetch(item["repo"],item["revision"],item["name"],item["sha256"],item["bytes"])
        for item in items:
            path=resumed.get(item["name"]) or fetch(item["repo"],item["revision"],item["name"],item["sha256"],item["bytes"])
            with ArchiveStream(path) as stream, tarfile.open(fileobj=stream,mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():continue
                    rel=PurePosixPath(member.name)
                    if rel.is_absolute() or ".." in rel.parts:raise ValueError(member.name)
                    if kind=="frames" and (len(rel.parts)!=2 or rel.suffix.lower() not in (".jpg",".jpeg",".png")):continue
                    key=str(rel) if kind=="video" else rel.parts[0]
                    if key not in wanted[kind]:continue
                    dest=out/"media"/("vsi" if kind=="video" else "hound")/str(rel)
                    if dest.is_file() and dest.stat().st_size==member.size:
                        covered[kind].add(key)
                        continue
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    part=dest.with_suffix(dest.suffix+".part")
                    with tar.extractfile(member) as stream,part.open("wb") as dst:shutil.copyfileobj(stream,dst,4<<20)
                    if part.stat().st_size!=member.size:raise IOError(f"Incomplete {dest}")
                    os.replace(part,dest);covered[kind].add(key)
            print("Verified and extracted",item["name"],flush=True)
    if covered!=wanted:raise ValueError({k:len(wanted[k]-covered[k]) for k in wanted})
    (out/"media_ready.json").write_text(json.dumps({k:len(v) for k,v in covered.items()}))
