"""Export the selected ZeRO-2 model weights. Optimizer shards stay in the run."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import save_file

p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
args=p.parse_args()
choice=json.loads((args.run/"selected.json").read_text())
tag=choice["tag"]
protocol=json.loads((args.run/"protocol.json").read_text())
assert (args.run/(tag+".json")).is_file()
checkpoint=args.run/"checkpoints"/tag/"mp_rank_00_model_states.pt"
state=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
assert state["step"]==choice["step"] and state["protocol"]==protocol
weights={k:v.detach().contiguous() for k,v in state["module"].items()}
args.output.mkdir(parents=True,exist_ok=False)
save_file(weights,str(args.output/"vega.safetensors"))
(args.output/"config.json").write_text(json.dumps({"checkpoint":tag,"tune_ce":choice["tune_ce"],
    "smoke":protocol["smoke"],"base":"Qwen/Qwen3-VL-8B-Instruct","wan":"Wan-AI/Wan2.1-T2V-1.3B",
    "fusion":"token_gated_residual","tune_sha256":protocol["tune_sha256"]},indent=2))
print("EXPORTED",tag,len(weights),flush=True)
