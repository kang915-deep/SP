"""Frozen selected Qwen3-VEGA. Each assigned GPU owns a fixed subset of videos."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from decord import VideoReader, DECORDError
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl.model.generative_encoders.encoder import GenerativeEncoderConfig, GenerativeVisionEncoder
from qwen_vl.data.utils import load_and_preprocess_images
from common import key, read, save
from candidates import actions, candidate_indices, number
from model_buffers import snapshot_buffers, restore_buffers
from train_model import TrainableVega, FrozenEncoders
from wan_sdpa import configure_attention


class Extractor:
    def __init__(self, qwen_path, wan_path, export, context, device):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        configure_attention()
        wan = GenerativeVisionEncoder(
            GenerativeEncoderConfig(tower_type="wan_t2v_online", checkpoint=str(wan_path.resolve()))
        )
        self.enc = FrozenEncoders(qwen.model.visual, wan)
        qwen.model.visual = None
        self.net = TrainableVega(qwen)
        buffers = snapshot_buffers(self.net)
        self.net.bfloat16()
        weights = load_file(str(export / "vega.safetensors"))
        self.net.load_state_dict(weights, strict=not read(export / "config.json")["smoke"])
        del weights
        self.net.to(device).eval().requires_grad_(False)
        restore_buffers(self.net, buffers, device)
        self.enc.activate(device)
        self.context = context.to(device)

    def media(self, row, bench):
        reader = VideoReader(row["path"], num_threads=1)

        def frames(idx):
            try:
                return [Image.fromarray(a) for a in reader.get_batch(idx).asnumpy()]
            except DECORDError as error:
                if "Unable to handle EOF" not in str(error):
                    raise
                import cv2

                cap = cv2.VideoCapture(row["path"])
                wanted = set(map(int, idx))
                found = {}
                for index in range(max(wanted) + 1):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index in wanted:
                        found[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if set(found) != wanted:
                    raise ValueError(f"Missing sampled frames in {row['path']}")
                return [found[int(i)] for i in idx]

        images = frames(np.linspace(0, len(reader) - 1, 16).round().astype(int))
        w, h = images[0].size
        scale = min(1.0, (112896 / (w * h)) ** 0.5)
        small = [
            im.resize(
                (max(32, round(w * scale / 16) * 16), max(32, round(h * scale / 16) * 16)),
                Image.Resampling.BICUBIC,
            )
            for im in images
        ]
        # Independent video-level Wan memory: 17 frames -> 5 latent time slots, block 20, 6x6 grid.
        tower = self.enc.wan.tower
        x = (
            torch.from_numpy(np.stack(small + [small[-1]])).permute(3, 0, 1, 2).to(self.device).float()
            / 127.5
            - 1
        )
        z = tower.vae.encode([x])[0].float()
        t, h, w = z.shape[1], z.shape[2] // 2, z.shape[3] // 2
        rng = torch.Generator(device=self.device).manual_seed(int(key(row["media"])[:8], 16))
        noise = torch.randn(z.shape, device=self.device, generator=rng)
        tapped = []
        hook = tower.model.blocks[20].register_forward_hook(lambda m, i, o: tapped.append(o.detach()))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tower.model(
                    [0.7 * z + 0.3 * noise],
                    t=torch.tensor([300.0], device=self.device),
                    context=[self.context],
                    seq_len=t * h * w,
                )
        finally:
            hook.remove()
        mem = tapped[0][0].reshape(t, h, w, 1536).permute(0, 3, 1, 2)
        mem = F.adaptive_avg_pool2d(mem.float(), 6).permute(0, 2, 3, 1).reshape(-1, 1536)
        tt, yy, xx = np.meshgrid(
            np.linspace(0, 1, t), np.linspace(-1, 1, 6), np.linspace(-1, 1, 6), indexing="ij"
        )
        pos = np.stack([xx, yy, tt], axis=-1).reshape(-1, 3)
        rgb = (
            frames(np.linspace(0, len(reader) - 1, min(32, len(reader))).astype(int))
            if bench
            else small
        )
        geometry = load_and_preprocess_images(rgb)
        geometry = geometry[:, :, : geometry.shape[-2] // 32 * 32, : geometry.shape[-1] // 32 * 32]
        vision = self.proc.image_processor(
            geometry, return_tensors="pt", do_rescale=False, do_resize=False
        )
        grid = vision.image_grid_thw
        blank = torch.empty((1, 0), dtype=torch.long)
        batch = dict(
            pixel_values=vision.pixel_values,
            image_grid_thw=grid,
            wan_images=geometry,
            input_ids=blank,
            labels=blank,
            attention_mask=blank,
            mm_token_type_ids=blank,
        )
        torch.manual_seed(int(key("vega_sp16k:" + row["media"])[:8], 16))
        encoded = self.enc.encode(batch, self.device)
        self.grid = encoded["image_grid_thw"]
        self.visual = self.net.fused_visual(encoded["visual"], encoded["wan"], self.grid)
        return mem.half().cpu().numpy(), pos.astype(np.float32)

    def question(self, row, banks):
        tok = self.proc.tokenizer
        image_tokens = [
            "<|vision_start|>" + "<|image_pad|>" * (int(n) // 4) + "<|vision_end|>"
            for n in self.grid.prod(-1)
        ]
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "".join(image_tokens) + row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        n = ids.shape[1]
        if n + 144 > 12800:
            raise ValueError(f"Context too long: {row['id']} {n}")
        q = self.net.qwen
        prefix = q.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=(ids == tok.convert_tokens_to_ids("<|image_pad|>")).long(),
            image_grid_thw=self.grid,
            mm_encoder_outputs={"image": self.visual},
            use_cache=True,
        )
        cache = prefix.past_key_values
        context = prefix.last_hidden_state[:, -1]
        first = q.lm_head(context).float().log_softmax(-1)[0]
        delta = int(prefix.rope_deltas.item())

        def continuation(tokens):
            length = tokens.shape[1]
            positions = (torch.arange(length, device=self.device) + cache.get_seq_length() + delta)[
                None, None, :
            ].expand(3, 1, -1)
            return q.model(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=torch.ones(
                    (1, cache.get_seq_length() + length), device=self.device, dtype=torch.long
                ),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state

        generated = []
        next_token = first.argmax().item()
        eos = q.generation_config.eos_token_id
        stop = {eos} if isinstance(eos, int) else set(eos)
        for _ in range(16):
            if next_token in stop:
                break
            generated.append(next_token)
            next_token = q.lm_head(continuation(ids.new_tensor([[next_token]]))[:, -1]).argmax().item()
        cache.crop(n)
        baseline = tok.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        cs = actions(row, banks, baseline)
        kept = candidate_indices(dict(row, actions=cs), number)
        hs = []
        vals = []
        for answer in cs:
            pieces = tok.encode(answer or " ", add_special_tokens=False) + [
                tok.convert_tokens_to_ids("<|im_end|>")
            ]
            tokens = ids.new_tensor([pieces])
            hidden = continuation(tokens)
            lp = (
                q.lm_head(hidden[:, :-1])
                .float()
                .log_softmax(-1)
                .gather(-1, tokens[:, 1:, None])
                .flatten()
            )
            lp = torch.cat([first[tokens[0, :1]], lp])
            hs.append(hidden[0, -2].half().cpu().numpy())
            vals.append([lp.sum().item(), lp.mean().item(), len(pieces)])
            cache.crop(n)
        a = np.asarray(vals, np.float32)
        prob = np.exp(a[:, 1] - max(a[:, 1]))
        prob /= prob.sum()
        native = np.stack(
            [
                a[:, 1] / 10,
                a[:, 0] / 30,
                a[:, 2] / 16,
                prob,
                (a[:, 1] - a[0, 1]) / 10,
                (a[:, 0] - a[0, 0]) / 30,
                np.arange(len(cs)) == 0,
                np.full(len(cs), bool(row["options"])),
            ],
            axis=1,
        )
        return [cs[j] for j in kept], dict(
            h=np.stack(hs)[kept],
            context=context[0].half().cpu().numpy(),
            native=native[kept].astype(np.float32),
        )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    for k in ("qwen", "wan", "export", "feature-dir"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--parts", nargs="+", required=True)
    args = p.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    enc = Extractor(
        args.qwen,
        args.wan,
        args.export,
        torch.load(args.feature_dir / "context.pt", weights_only=True),
        torch.device("cuda", rank),
    )
    banks = read(args.feature_dir / "banks.json")
    for part in args.parts:
        folder = args.feature_dir / "features" / part
        folder.mkdir(parents=True, exist_ok=True)
        groups = defaultdict(list)
        for row in read(args.feature_dir / (part + ".json")):
            groups[row["media"]].append(row)
        # Assign a whole video to one worker to reuse both visual encoders.
        shards = [[] for _ in range(world)]
        loads = [0] * world
        for media in sorted(groups, key=lambda m: (-len(groups[m]), m)):
            owner = min(range(world), key=loads.__getitem__)
            shards[owner].append(media)
            loads[owner] += len(groups[media])
        print("SP_EXTRACT", part, rank, "assigned_qa", loads[rank], flush=True)
        completed = 0
        for i, media in enumerate(shards[rank], 1):
            target = folder / (key(media) + ".pt")
            completed += len(groups[media])
            if target.exists():
                continue
            rows = groups[media]
            memory, pos = enc.media(rows[0], part == "benchmark")
            records = []
            for row in rows:
                answers, arr = enc.question(row, banks)
                records.append(dict(id=row["id"], actions=answers, **arr))
            temp = target.with_suffix(".tmp")
            torch.save(dict(memory=memory, pos=pos, records=records), temp)
            os.replace(temp, target)
            if i % 25 == 0 or i == len(shards[rank]):
                print("SP_EXTRACT", part, rank, "qa", completed, "of", loads[rank], flush=True)


if __name__ == "__main__":
    main()
