"""Train the frozen-feature head, select on tune, then apply the fixed head and policy."""

from pathlib import Path
import os
import numpy as np
import torch
from torch.nn import functional as F
from common import key, read, save
from candidates import score
from numeric_features import features
from numeric_model import NumericCorrector
from metrics import calibrate, choices, metrics
from convergence import Plateau


class Split:
    def __init__(self, root, part):
        self.rows = read(root / (part + ".json"))
        self.records = {}
        self.memory = {}
        for media in sorted({r["media"] for r in self.rows}):
            state = torch.load(
                root / "features" / part / (key(media) + ".pt"), weights_only=False, map_location="cpu"
            )
            self.memory[media] = (torch.from_numpy(state["memory"]), torch.from_numpy(state["pos"]))
            for record in state["records"]:
                self.records[record["id"]] = record
        if set(self.records) != {r["id"] for r in self.rows}:
            raise ValueError(f"Incomplete {part} cache")
        self.quality = []
        for r in self.rows:
            record = self.records[r["id"]]
            r["actions"] = record["actions"]
            record["numeric"] = features(r, r["actions"])
            if "answer" in r:
                self.quality.append([score(r, a, r["answer"]) for a in r["actions"]])

    def batch(self, indices, device):
        rows = [self.rows[i] for i in indices]
        records = [self.records[r["id"]] for r in rows]

        def pad(name):
            values = [torch.from_numpy(v[name]) for v in records]
            return torch.nn.utils.rnn.pad_sequence(values, batch_first=True).float().to(device)

        h = pad("h")
        lengths = torch.tensor([len(v["actions"]) for v in records], device=device)
        padding = torch.arange(h.shape[1], device=device)[None] >= lengths[:, None]
        memories = [self.memory[r["media"]] for r in rows]
        x = dict(
            h=h,
            context=torch.stack([torch.from_numpy(v["context"]) for v in records]).float().to(device),
            native=pad("native"),
            numeric=pad("numeric"),
            w=torch.stack([v[0] for v in memories]).float().to(device),
            wp=torch.stack([v[1] for v in memories]).to(device),
        )
        return x, padding


def utility(z, q, pad):
    teacher = (q / 0.25).masked_fill(pad, -torch.inf).log_softmax(1)
    student = (z / 0.25).masked_fill(pad, -torch.inf).log_softmax(1)
    return (teacher.exp() * (teacher.masked_fill(pad, 0) - student.masked_fill(pad, 0))).sum(
        1
    ) * 0.25**2


def targets(split, indices, device):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(split.quality[i], dtype=torch.float32) for i in indices], batch_first=True
    ).to(device)


@torch.no_grad()
def evaluate(model, split, device):
    model.eval()
    scores = []
    total = 0.0
    for start in range(0, len(split.rows), 32):
        ids = range(start, min(start + 32, len(split.rows)))
        x, pad = split.batch(ids, device)
        z = model(**x)
        if split.quality:
            total += utility(z, targets(split, ids, device), pad).sum().item()
        scores.extend([v[: len(split.rows[i]["actions"])].tolist() for i, v in zip(ids, z)])
    return scores, total / len(scores)


def train(root, limit=160):
    train = Split(root, "train")
    tune = Split(root, "tune")
    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    model = NumericCorrector(True, 128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    plateau = Plateau()
    statefile = root / "head_state.pt"
    epoch = 1
    best = None
    history = []
    if statefile.exists():
        state = torch.load(statefile, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        plateau.__dict__.update(state["plateau"])
        epoch = state["epoch"] + 1
        best = state["best"]
        history = state["history"]
    while epoch <= limit and (
        not history or history[-1]["stop"]["action"] != "sufficient_validation_plateau"
    ):
        order = np.random.default_rng(20261022 + epoch).permutation(len(train.rows))
        model.train()
        for j in range(0, len(order), 32):
            ids = order[j : j + 32]
            x, pad = train.batch(ids, device)
            z = model(**x)
            loss = utility(z, targets(train, ids, device), pad).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        _, train_loss = evaluate(model, train, device)
        scores, val_loss = evaluate(model, tune, device)
        policy = calibrate(tune.rows, scores, tune.quality)
        result = metrics(tune.rows, tune.quality, choices(scores, tune.rows, policy))
        rank = (result["score_points_group"], -result["harmed_count"], -result["changed_rate"])
        if best is None or rank > best["rank"]:
            best = dict(
                rank=rank,
                epoch=epoch,
                policy=policy,
                weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        stop = plateau.update(epoch, val_loss, result["net_gain_points_group"])
        for group in opt.param_groups:
            group["lr"] = stop["learning_rate"]
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=val_loss,
                validation=result,
                stop=stop,
            )
        )
        tmp = statefile.with_suffix(".tmp")
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                plateau=plateau.__dict__,
                epoch=epoch,
                best=best,
                history=history,
            ),
            tmp,
        )
        os.replace(tmp, statefile)
        print("SP_EPOCH", epoch, train_loss, val_loss, flush=True)
        epoch += 1
    tmp = root / "sp.tmp"
    torch.save(best["weights"], tmp)
    os.replace(tmp, root / "sp.pt")
    save(
        root / "sp_selection.json",
        dict(
            epoch=best["epoch"],
            policy=best["policy"],
            history=history,
            adequacy_passed=history[-1]["stop"]["action"] == "sufficient_validation_plateau",
        ),
    )


def predict(root, part):
    split = Split(root, part)
    device = torch.device("cuda", 0)
    model = NumericCorrector(True, 128).to(device)
    model.load_state_dict(torch.load(root / "sp.pt", map_location=device, weights_only=True))
    scores, loss = evaluate(model, split, device)
    indices = choices(scores, split.rows, read(root / "sp_selection.json")["policy"])
    preds = [
        {
            "id": r["id"],
            "type": r.get("benchmark_type", r["type"]),
            "baseline": r["actions"][0],
            "sp": r["actions"][i],
        }
        for r, i in zip(split.rows, indices)
    ]
    save(root / (part + "_predictions.json"), preds)
    if part == "confirm":
        save(
            root / "confirm_results.json",
            dict(loss=loss, **metrics(split.rows, split.quality, indices)),
        )
    return preds


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["train", "confirm", "benchmark"])
    p.add_argument("run", type=Path)
    args = p.parse_args()
    if args.stage == "train":
        train(args.run)
    else:
        predict(args.run, args.stage)
