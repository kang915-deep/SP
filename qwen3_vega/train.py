"""Portable data-parallel VEGA training on explicitly selected A800 GPUs."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl.model.generative_encoders.encoder import GenerativeEncoderConfig, GenerativeVisionEncoder
from batched_vega import BatchedVega
from data import prepare, sample_seed
from train_model import FrozenEncoders
from model_buffers import snapshot_buffers, restore_buffers
from wan_sdpa import configure_attention
from numerics import NumericalChecks

parser = argparse.ArgumentParser()
for name in ("qwen", "wan", "data", "output"):
    parser.add_argument("--" + name, required=True, type=Path)
parser.add_argument("--resume", help="Committed step tag in the same output directory")
parser.add_argument("--smoke", action="store_true", help="Real 8B+Wan forward; update final decoder layer and fusion only")
parser.add_argument("--micro-batch", type=int, default=1)
parser.add_argument("--checkpoint-every", type=int, choices=(0, 1, 2), default=1,
                    help="Recompute every decoder layer (1), alternate layers (2), or none (0)")
parser.add_argument("--benchmark-steps", type=int, help="Measure full optimizer updates in a fresh run, without checkpoint I/O or downstream stages")
parser.add_argument("--diagnose-numerics", action="store_true", help="Trace nonfinite operations in a separate benchmark replay; never use its timing for configuration selection")
parser.add_argument("--diagnostic-micros", type=int, help="Stop a numerical replay after this many first-step microbatches, before any optimizer update")
parser.add_argument("--cpu-adam", action="store_true", help="Offload FP32 optimizer states to host RAM (slower, less VRAM)")
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--stop-after", type=int, help="Commit an optimizer checkpoint and exit after this global step")
parser.add_argument("--save-every", type=int, default=250)
args = parser.parse_args()
if args.benchmark_steps is not None and (args.benchmark_steps < 2 or args.resume or args.stop_after):
    parser.error("Use at least two benchmark steps in a fresh run without --resume or --stop-after")
if args.diagnose_numerics and args.benchmark_steps is None:
    parser.error("Numerical tracing requires a fresh --benchmark-steps run")
rank, world, local = (int(os.environ[k]) for k in ("RANK","WORLD_SIZE","LOCAL_RANK"))
if args.workers < 0 or args.save_every < 1:
    parser.error('workers must be nonnegative and save-every must be positive')
width = args.micro_batch
effective_batch = world * width if args.smoke else 64
if not os.environ.get("CUDA_VISIBLE_DEVICES") or width < 1 or effective_batch % (world * width):
    parser.error("Set CUDA_VISIBLE_DEVICES explicitly; global batch must be divisible by world size * micro batch")
accum = effective_batch//(world*width)
if args.diagnostic_micros is not None and (
        not args.diagnose_numerics or args.benchmark_steps is None or not 1 <= args.diagnostic_micros < accum):
    parser.error("--diagnostic-micros requires a fresh numerical benchmark and 1 <= N < gradient accumulation steps")
torch.cuda.set_device(local)
# DeepSpeed 0.18.9's NVML memory query casts UUID visibility entries to int.
# Use CUDA's logical device mapping and actual free bytes for this process.
deepspeed.accelerator.get_accelerator().available_memory = (
    lambda device_index=None: torch.cuda.mem_get_info(device_index)[0]
)
device = torch.device("cuda",local)
dist.init_process_group("nccl",device_id=device)
torch.manual_seed(0)
data_dir = args.data.resolve()
raw = (data_dir/"train.json").read_bytes()
rows = json.loads(raw)
tune = json.loads((data_dir/"tune.json").read_text())
if not rows or not tune or (not args.smoke and (len(rows),len(tune)) != (298027,2036)):
    raise ValueError("Unexpected train/tune size")
if not args.smoke and not (data_dir/"media_ready.json").is_file():
    raise ValueError("Finish the verified media preparation before full training")
protocol = {"train_sha256":hashlib.sha256(raw).hexdigest(),"tune_sha256":hashlib.sha256((data_dir/"tune.json").read_bytes()).hexdigest(),
            "world":world,"micro_batch":width,"effective_batch":effective_batch,"smoke":args.smoke,"cpu_adam":args.cpu_adam}
if args.checkpoint_every != 1:
    protocol["checkpoint_every"] = args.checkpoint_every
if args.benchmark_steps is not None:
    protocol["benchmark_steps"] = args.benchmark_steps
if args.diagnose_numerics:
    protocol["diagnose_numerics"] = True
if args.diagnostic_micros is not None:
    protocol["diagnostic_micros"] = args.diagnostic_micros
out = args.output.resolve()
if rank == 0:
    if args.resume:
        assert json.loads((out/"protocol.json").read_text()) == protocol
        if (out/"selected.json").exists() or sorted(out.glob("step-*.json"))[-1].stem!=args.resume:
            raise ValueError("Resume the latest committed step of an unfinished run")
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/"protocol.json").write_text(json.dumps(protocol))
dist.barrier()
checks = NumericalChecks(out, rank)
if args.diagnose_numerics:
    torch.autograd.set_detect_anomaly(True, check_nan=True)
processor = AutoProcessor.from_pretrained(args.qwen,local_files_only=True)
qwen = Qwen3VLForConditionalGeneration.from_pretrained(args.qwen,dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True)
configure_attention()
wan = GenerativeVisionEncoder(GenerativeEncoderConfig(tower_type="wan_t2v_online",checkpoint=str(args.wan.resolve())))
enc = FrozenEncoders(qwen.model.visual,wan)
qwen.model.visual = None
model = BatchedVega(qwen)
qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
for index,layer in enumerate(qwen.model.language_model.layers):
    layer.gradient_checkpointing = bool(args.checkpoint_every and index % args.checkpoint_every == 0)
if args.smoke:
    qwen.requires_grad_(False)
    qwen.model.language_model.layers[-1].requires_grad_(True)
buffers = snapshot_buffers(model)
model.bfloat16()
enc.activate(device)
params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
norm_ids={id(p) for m in model.modules() if isinstance(m,torch.nn.LayerNorm) or "RMSNorm" in type(m).__name__
          for p in m.parameters(recurse=False)}
decay=[];no_decay=[]
for name,p in params:
    (no_decay if "bias" in name or id(p) in norm_ids or name.endswith("norm_weight") else decay).append(p)
groups=[{"params":decay,"weight_decay":0.01},{"params":no_decay,"weight_decay":0.0}]
config = {"train_batch_size":effective_batch,"train_micro_batch_size_per_gpu":width,"gradient_accumulation_steps":accum,
          "gradient_clipping":1.0,"bf16":{"enabled":True},"zero_allow_untested_optimizer":True,
          "zero_optimization":{"stage":2,"overlap_comm":True,"contiguous_gradients":True,
                              "reduce_bucket_size":50000000,"allgather_bucket_size":50000000},"steps_per_print":10000}
if args.cpu_adam:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    config["zero_optimization"]["offload_optimizer"]={"device":"cpu","pin_memory":True}
    optimizer=DeepSpeedCPUAdam(groups,lr=1e-5,betas=(0.9,0.999),eps=1e-8)
else:
    optimizer=torch.optim.AdamW(groups,lr=1e-5,betas=(0.9,0.999),eps=1e-8,fused=True)
engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,config=config)
checks.attach(optimizer)
restore_buffers(engine.module,buffers,device)
if args.diagnose_numerics:
    from linear_backward_trace import LinearBackwardTrace
    linears = {f"qwen.model.language_model.layers.{index}.mlp.{name}": getattr(layer.mlp, name)
               for index, layer in enumerate(qwen.model.language_model.layers)
               for name in ("gate_proj", "up_proj", "down_proj")}
    linear_trace = LinearBackwardTrace(linears, out/f"linear-rank{rank}")
    # Only the replay traces intermediate features; normal training checks the
    # loss and the real optimizer partitions at their update boundaries.
    for name, module in (("merger", model.generative_merger),
                         ("fusion", model.generative_fusion),
                         ("language_model", qwen.model.language_model),
                         ("lm_head", qwen.lm_head)):
        def trace_output(module, inputs, output, name=name):
            value = output.last_hidden_state if name == "language_model" else output
            checks.check_tensors([(name, value)], "forward_activation")
        module.register_forward_hook(trace_output)
start=0
best=None
if args.resume:
    loaded,state=engine.load_checkpoint(str(out/"checkpoints"),tag=args.resume,load_module_strict=not args.smoke)
    if not loaded or state["protocol"]!=protocol:
        raise ValueError("Checkpoint does not match the run")
    start=state["step"];best=state["best"]
    assert engine.global_steps==start
for row in rows+tune:
    row["path"]=str(data_dir/row["path"])
order=sorted(range(len(rows)),key=lambda i:hashlib.sha256(("train-order-0:"+rows[i]["id"]).encode()).digest())
total=math.ceil(len(rows)/effective_batch)
if args.benchmark_steps is not None and args.benchmark_steps > total:
    raise ValueError("Benchmark exceeds available training steps")
validation_steps={math.ceil(total*f) for f in (.25,.5,.75,1.)}
if args.stop_after is not None and not start<args.stop_after<=total:raise ValueError("stop-after outside remaining budget")

class Examples:
    def __init__(self,split,indices):
        self.split=split;self.indices=indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,i):
        index,valid=self.indices[i]
        row=self.split[index]
        return row["id"],valid,prepare(row,processor)

def as_list(items):
    return items

def loader(split,indices,batch_size):
    return DataLoader(Examples(split,indices),batch_size=batch_size,num_workers=args.workers,
                      pin_memory=True,collate_fn=as_list)

def forward(batch,phase):
    features=[]
    for qid,_,example in batch:
        torch.manual_seed(sample_seed(qid,0,phase))
        feature = enc.encode(example,device)
        if args.diagnose_numerics:
            tensors = [(qid + "/wan", feature["wan"])]
            tensors.extend((f"{qid}/visual/{i}", v) for i, v in enumerate(feature["visual"].pooler_output))
            tensors.extend((f"{qid}/deepstack/{i}/{j}", v)
                           for i, layer in enumerate(feature["visual"].deepstack_features)
                           for j, v in enumerate(layer))
            checks.check_tensors(tensors, "frozen_features")
        features.append(feature)
    return engine(**features[0]).reshape(1) if len(features)==1 else engine(examples=features)

indices=[]
for pos in range(start*effective_batch,total*effective_batch,world*width):
    for i in range(pos+rank*width,pos+(rank+1)*width):
        indices.append((order[min(i,len(rows)-1)],i<len(rows)))
batches=iter(loader(rows,indices,width))
if rank==0:print("READY",json.dumps(dict(protocol,trainable=sum(p.numel() for _,p in params),start=start)),flush=True)
measured_seconds=[]
measured_examples=0
if args.benchmark_steps is not None and args.diagnostic_micros is None:
    import threading
    import pynvml
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByUUID("GPU-"+str(torch.cuda.get_device_properties(device).uuid))
    sampling_done=threading.Event()
    utilization=[]
    free_bytes=[]
    def sample_gpu():
        while not sampling_done.is_set():
            utilization.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            free_bytes.append(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
            sampling_done.wait(.5)
    monitor=threading.Thread(target=sample_gpu,daemon=True)
for step in range(start,args.benchmark_steps or total):
    if args.benchmark_steps is not None and step==1:
        monitor.start()
    begin=time.perf_counter()
    n=min(effective_batch,len(rows)-step*effective_batch)
    warmup=math.ceil(total*.03)
    factor=step/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))
    for group in engine.optimizer.param_groups:group["lr"]=1e-5*factor
    ce=torch.zeros((),device=device)
    step_qids=[]
    for micro in range(accum):
        batch=next(batches)
        qids=[item[0] for item in batch]
        step_qids.extend(qids)
        checks.context={"step":step+1,"micro":micro+1,"qids":qids,
                        "step_qids":list(step_qids),"lr":1e-5*factor}
        if args.diagnose_numerics:
            linear_trace.set_context(**checks.context, rank=rank)
            print("NUMERICS_CONTEXT",json.dumps(dict(checks.context,rank=rank)),flush=True)
        values=forward(batch,"train")
        checks.check_loss(values)
        valid=sum(item[1] for item in batch)
        loss=values[:valid].sum()
        engine.backward(loss*(world*accum/n))
        if args.diagnostic_micros is not None and micro+1 == args.diagnostic_micros:
            torch.cuda.synchronize(device)
            if engine.global_steps != 0 or checks.updates != 0:
                raise RuntimeError("Numerical prefix unexpectedly performed an optimizer update")
            local_report = {"rank":rank,"finite":True,"optimizer_updates":engine.global_steps,
                            "checked_optimizer_updates":checks.updates,"context":dict(checks.context),
                            "finite_scope":"loss, observed features, MLP Mm node gradients, anomaly checking",
                            "diagnostic_only":True}
            reports = [None]*world
            dist.all_gather_object(reports, local_report)
            linear_trace.close()
            if rank == 0:
                report = {"protocol":protocol,"ranks":reports,"diagnostic_only":True,
                          "optimizer_updates":0,"training_completed":False}
                (out/"diagnostic_prefix.json").write_text(json.dumps(report,indent=2,allow_nan=False))
                print("NUMERICS_PREFIX_FINITE",json.dumps(report,allow_nan=False),flush=True)
            dist.destroy_process_group()
            raise SystemExit(0)
        engine.step()
        ce+=loss.detach()
    dist.all_reduce(ce)
    completed=step+1
    if not torch.isfinite(ce):raise FloatingPointError(f"Nonfinite loss at step {completed}")
    elapsed=time.perf_counter()-begin
    event={"step":completed,"train_ce":ce.item()/n,"qa_per_s":n/elapsed,"seconds":elapsed,
           "peak_gpu_gib":torch.cuda.max_memory_allocated(device)/2**30,"lr":1e-5*factor}
    if args.benchmark_steps is not None and completed > 1:
        measured_seconds.append(elapsed)
        measured_examples+=n
    if args.benchmark_steps is None and completed in validation_steps:
        engine.eval()
        # Match standalone export verification even when training uses padded batches.
        count=math.ceil(len(tune)/world)*world
        slots=[(min(i,len(tune)-1),i<len(tune)) for i in range(rank,count,world)]
        stats=torch.zeros(2,device=device,dtype=torch.float64)
        with torch.no_grad():
            for batch in loader(tune,slots,1):
                values=forward(batch,"validation")
                valid=sum(item[1] for item in batch)
                stats[0]+=values[:valid].double().sum();stats[1]+=valid
        dist.all_reduce(stats)
        if stats[1].item()!=len(tune) or not torch.isfinite(stats[0]):raise ValueError("Invalid validation reduction")
        event["tune_ce"]=(stats[0]/stats[1]).item()
        if best is None or event["tune_ce"]<best["tune_ce"]:
            best={"step":completed,"tune_ce":event["tune_ce"]}
        engine.train()
    if rank==0:
        print("STEP",json.dumps(event),flush=True)
        with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
    if args.benchmark_steps is None and (completed in validation_steps or completed % args.save_every==0 or completed==args.stop_after):
        tag=f"step-{completed:06d}"
        engine.save_checkpoint(str(out/"checkpoints"),tag=tag,save_latest=False,exclude_frozen_parameters=args.smoke,
                               client_state={"protocol":protocol,"step":completed,"best":best})
        dist.barrier()
        if rank==0:
            with (out/(tag+".json")).open("x") as f:json.dump({"step":completed,"best":best},f)
    if completed==args.stop_after:break
if args.benchmark_steps is not None:
    sampling_done.set()
    monitor.join()
    pynvml.nvmlShutdown()
    local_stats={"rank":rank,"seconds":sum(measured_seconds),"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30,
                 "peak_reserved_gib":torch.cuda.max_memory_reserved(device)/2**30,
                 "nvml_samples":len(utilization),"mean_device_util_pct":sum(utilization)/len(utilization),
                 "min_device_free_gib":min(free_bytes)/2**30}
    ranks=[None]*world
    dist.all_gather_object(ranks,local_stats)
    if rank==0:
        report={"measured_steps":len(measured_seconds),"warmup_steps":1,"qa":measured_examples,
                "qa_per_s":measured_examples/max(r["seconds"] for r in ranks),"ranks":ranks,"protocol":protocol}
        if args.diagnose_numerics:
            (out/"numerics_replay.json").write_text(json.dumps(report,indent=2))
            print("NUMERICS_REPLAY_FINITE",json.dumps(report),flush=True)
        else:
            (out/"throughput.json").write_text(json.dumps(report,indent=2))
            print("BENCHMARK",json.dumps(report),flush=True)
elif completed==total and rank==0:
    (out/"selected.json").write_text(json.dumps(dict(best,tag=f"step-{best['step']:06d}")))
if args.diagnose_numerics:
    linear_trace.close()
dist.destroy_process_group()
