import numpy as np
from common import key
from legacy_candidates import candidates, kind, number, bank_key


def actions(row, banks, baseline):
    # Deliberately no reference-answer argument, including on training inputs.
    if row["options"]:
        others = [chr(65 + i) for i in range(len(row["options"]))]
    elif bank_key(row) not in banks["numeric"]:
        others = banks["words"].get(kind(row), [])[:12]
    else:
        others = candidates(row, banks, baseline=baseline)
    return [baseline] + sorted(
        {str(x) for x in others if str(x) != baseline}, key=lambda s: key(row["id"] + ":action:" + s)
    )


def score(row, pred, gt):
    if row["options"]:
        return float(str(pred).split(" ")[0].rstrip(".").strip().lower() == str(gt).lower())
    g = number(gt)
    if g is None:
        return float(str(pred).strip().lower() == str(gt).strip().lower())
    p = number(pred)
    if p is None:
        return 0.0
    if kind(row) == "absolute_direction_object":
        return max(0.0, 1 - min(abs(p - g) % 360, 360 - abs(p - g) % 360) / 180)
    if g == 0:
        return float(p == 0)
    # Exact threshold construction used by the existing official metric source.
    n = int((0.95 - 0.5) / 0.05 + 2)
    return float(np.mean(abs(p - g) / g <= 1 - np.linspace(0.5, 0.95, n)))


def candidate_indices(row, number):
    """Label-free canonicalization; retain the first representation, baseline first."""
    seen, kept = set(), []
    for i, action in enumerate(row["actions"]):
        text = str(action)
        if row["options"]:
            key = ("choice", text.split(" ")[0].rstrip(".").strip().lower())
        else:
            value = number(action)
            key = ("number", float(value)) if value is not None else ("text", text.strip().lower())
        if key not in seen:
            seen.add(key)
            kept.append(i)
    assert kept[0] == 0
    return kept
