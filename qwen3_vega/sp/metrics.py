"""Rates use all questions as denominator; continuous gains retain score magnitude."""

from collections import defaultdict
import numpy as np


def group_mean(rows, values):
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[row["group"]].append(float(value))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def choices(scores, rows, policy=None):
    result = []
    for row, z in zip(rows, scores):
        j = int(np.argmax(z))
        th = 0.0 if policy is None else policy.get(row["type"], "never")
        result.append(j if j != 0 and th != "never" and z[j] - z[0] > th else 0)
    return result


def metrics(rows, quality, selected):
    base = np.array([q[0] for q in quality])
    final = np.array([q[j] for q, j in zip(quality, selected)])
    delta = final - base
    up, down = delta > 0, delta < 0
    opportunity = np.array([max(q) > q[0] for q in quality])
    perfect = base >= 1.0 - 1e-9
    return {
        "questions": len(rows),
        "improved_count": int(up.sum()),
        "harmed_count": int(down.sum()),
        "improved_rate": float(up.mean()),
        "harmed_rate": float(down.mean()),
        "changed_rate": float(np.mean(np.array(selected) != 0)),
        "net_count_rate": float(up.mean() - down.mean()),
        "gain_points_micro": float(np.maximum(delta, 0).mean() * 100),
        "harm_points_micro": float(np.maximum(-delta, 0).mean() * 100),
        "net_gain_points_micro": float(delta.mean() * 100),
        "baseline_points_group": group_mean(rows, base) * 100,
        "score_points_group": group_mean(rows, final) * 100,
        "net_gain_points_group": group_mean(rows, delta) * 100,
        "gain_points_group": group_mean(rows, np.maximum(delta, 0)) * 100,
        "harm_points_group": group_mean(rows, np.maximum(-delta, 0)) * 100,
        "opportunity_count": int(opportunity.sum()),
        "opportunity_used_rate": float(up.sum() / opportunity.sum()) if opportunity.any() else None,
        "perfect_baseline_count": int(perfect.sum()),
        "perfect_baseline_harmed_rate": (
            float((down & perfect).sum() / perfect.sum()) if perfect.any() else None
        ),
        "oracle_net_gain_points_group": group_mean(rows, np.array([max(q) for q in quality]) - base)
        * 100,
    }


def calibrate(rows, scores, quality):
    """Same 9 candidates per type/arm: zero, 7 positive-margin quantiles, never.

    Quantiles depend on predictions, not labels; accommodates differing logit scales.
    """
    types = defaultdict(list)
    for i, row in enumerate(rows):
        types[row["type"]].append(i)
    policy = {}
    for typ, ids in sorted(types.items()):
        rr = [rows[i] for i in ids]
        zz = [scores[i] for i in ids]
        qq = [quality[i] for i in ids]
        positive = [max(z) - z[0] for z in zz if max(z) > z[0]]
        thresholds = (
            [0.0]
            + (
                np.quantile(positive, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist()
                if positive
                else [0.0] * 7
            )
            + ["never"]
        )
        best = None
        for th in thresholds:
            selected = choices(zz, rr, {typ: th})
            m = metrics(rr, qq, selected)
            rank = (
                m["score_points_group"],
                -m["harmed_count"],
                -sum(j != 0 for j in selected),
                th == "never",
            )
            if best is None or rank > best[0]:
                best = (rank, th)
        policy[typ] = best[1]
    return policy
