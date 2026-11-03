"""Original full VSI eight-task macro. Score only after the two predictions are sealed."""

from collections import defaultdict
import numpy as np
from common import digest, read, save
from prepare import TYPE

NUMERIC = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}


def official_score(typ, prediction, answer):
    first = prediction.split(" ")[0].rstrip(".").strip()
    if typ not in NUMERIC:
        return float(first.lower() == str(answer).lower())
    try:
        value = float(first)
    except ValueError:
        return 0.0
    target = float(answer)
    thresholds = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))
    return float((abs(value - target) / target <= 1 - thresholds).mean())


def score_results(root):
    preds = read(root / "benchmark_predictions.json")
    answers = read(root / "benchmark_answers.json")
    if len(preds) != 5130 or {p["id"] for p in preds} != {"bench-" + i for i in answers}:
        raise ValueError("Incomplete paired benchmark")
    groups = defaultdict(list)
    for p in preds:
        gt = answers[p["id"].removeprefix("bench-")]
        b, s = [official_score(p["type"], p[arm], gt) for arm in ("baseline", "sp")]
        p.update(baseline_score=b, sp_score=s, delta=s - b)
        groups[p["type"]].append(p)
    if set(groups) != set(TYPE):
        raise ValueError("Missing official question types")

    def summarize(items):
        b = np.array([v["baseline_score"] for v in items])
        s = np.array([v["sp_score"] for v in items])
        return dict(
            questions=len(items),
            baseline=float(b.mean() * 100),
            sp=float(s.mean() * 100),
            improved=int((s > b).sum()),
            harmed=int((s < b).sum()),
            changed=sum(v["baseline"] != v["sp"] for v in items),
            net_points=float((s - b).mean() * 100),
            improved_rate=float((s > b).mean()),
            harmed_rate=float((s < b).mean()),
        )

    by_type = {t: summarize(v) for t, v in groups.items()}
    macro = {}
    for arm in ("baseline", "sp"):
        tasks = defaultdict(list)
        for t, v in by_type.items():
            tasks[TYPE[t]].append(v[arm])
        macro[arm] = float(np.mean([np.mean(values) for values in tasks.values()]))
    total = summarize(preds)
    selection = read(root / "sp_selection.json")
    save(root / "per_question_scores.json", preds)
    save(
        root / "results.json",
        dict(
            questions=len(preds),
            overall=macro,
            gain_pp=macro["sp"] - macro["baseline"],
            improved=total["improved"],
            harmed=total["harmed"],
            changed=total["changed"],
            by_type=by_type,
            improved_rate=total["improved_rate"],
            harmed_rate=total["harmed_rate"],
            prediction_sha256=digest(root / "benchmark_predictions.json"),
            export=read(root / "identity.json"),
            selected_sp_epoch=selection["epoch"],
            adequacy_passed=selection["adequacy_passed"],
            sp_sha256=digest(root / "sp.pt"),
            train_ids_sha256=read(root / "sp_split_manifest.json")["train_id_sha256"],
            vega_frames=32,
            sp_memory_frames=16,
        ),
    )
