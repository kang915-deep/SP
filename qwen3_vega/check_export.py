"""Reload a selected export and recompute its validation CE with original frozen towers."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from qwen_vl.model.generative_encoders.encoder import GenerativeVisionEncoder,GenerativeEncoderConfig
from data import prepare,sample_seed
from train_model import FrozenEncoders,TrainableVega
from model_buffers import snapshot_buffers, restore_buffers
from wan_sdpa import configure_attention

p=argparse.ArgumentParser()
for key in ("data","export","qwen","wan"):p.add_argument("--"+key,type=Path,required=True)
args=p.parse_args()
if not os.environ.get("CUDA_VISIBLE_DEVICES"):p.error("Assign a GPU explicitly")
cfg=json.loads((args.export/"config.json").read_text());raw=(args.data/"tune.json").read_bytes()
assert hashlib.sha256(raw).hexdigest()==cfg["tune_sha256"]
torch.manual_seed(0)
device=torch.device("cuda:0")
proc=AutoProcessor.from_pretrained(args.qwen,local_files_only=True)
qwen=Qwen3VLForConditionalGeneration.from_pretrained(args.qwen,dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True)
configure_attention();wan=GenerativeVisionEncoder(GenerativeEncoderConfig(tower_type="wan_t2v_online",checkpoint=str(args.wan.resolve())))
enc=FrozenEncoders(qwen.model.visual,wan);qwen.model.visual=None
model=TrainableVega(qwen)
buffers=snapshot_buffers(model)
model.bfloat16()
weights=load_file(str(args.export/"vega.safetensors"))
assert any(k.startswith("generative_fusion.") for k in weights) and any(k.startswith("generative_merger.") for k in weights)
incompatible=model.load_state_dict(weights,strict=not cfg["smoke"])
assert not incompatible.unexpected_keys and all(k.startswith("qwen.") for k in incompatible.missing_keys)
del weights
model.to(device).eval()
restore_buffers(model,buffers,device)
enc.activate(device)
losses=[]
with torch.no_grad():
    for r in json.loads(raw):
        r["path"]=str(args.data/r["path"]);b=prepare(r,proc)
        torch.manual_seed(sample_seed(r["id"],0,"validation"))
        losses.append(float(model(**enc.encode(b,device))))
ce=sum(losses)/len(losses)
print(json.dumps({"rows":len(losses),"export_ce":ce,"selected_ce":cfg["tune_ce"]}),flush=True)
if abs(ce-cfg["tune_ce"])>0.005:raise RuntimeError("Reloaded validation loss differs from the selected checkpoint")
