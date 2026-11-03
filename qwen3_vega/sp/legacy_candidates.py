"""Training-only numeric banks and reference-free development/test candidates."""

import math, re
from collections import defaultdict, Counter
import numpy as np
from common import key

CANON = {
    "object_abs_distance": "absolute_distance_object",
    "object_counting": "absolute_count",
    "object_size_estimation": "absolute_size_object",
    "room_size_estimation": "absolute_size_room",
}


def kind(row):
    return CANON.get(row["type"], row["type"])


def unit(row):
    s = row["question"].lower()
    k = kind(row)
    if k == "absolute_count":
        return "count"
    if k == "absolute_direction_object":
        return "degree"
    for word, u in [
        ("centimeter", "cm"),
        ("centimetre", "cm"),
        ("millimeter", "mm"),
        ("millimetre", "mm"),
        ("inches", "inch"),
        ("inch", "inch"),
        ("feet", "ft"),
        ("foot", "ft"),
        ("meter", "m"),
        ("metre", "m"),
    ]:
        if word in s:
            return u + ("2" if "square" in s else "")
    return "unspecified"


def bank_key(row):
    return kind(row) + "|" + unit(row)


def number(s):
    # Parse only an entire numeric response with an optional known unit suffix.
    m = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:meters?|metres?|cm|m|feet|ft|inches|degrees?|square meters?|m²|m2)?\.?\s*",
        str(s),
        re.I,
    )
    if not m:
        return None
    x = float(m[1])
    return x if math.isfinite(x) and x >= 0 else None


def fmt(v):
    return f"{v:.6g}"


def build_banks(rows, labels):
    nums = defaultdict(list)
    words = defaultdict(Counter)
    for r in rows:
        a = labels[r["id"]]
        if r["options"]:
            continue
        v = number(a)
        if v is None:
            words[kind(r)][a] += 1
        else:
            nums[bank_key(r)].append(v)
    return {
        "numeric": {
            k: [float(v) for v in np.quantile(vs, [0.05, 0.25, 0.5, 0.75, 0.95])]
            for k, vs in nums.items()
        },
        "words": {k: [s for s, _ in c.most_common(64)] for k, c in words.items()},
        "fit_rows": len(rows),
        "fit_split": "train_only",
    }


def candidates(row, banks, reference=None, baseline=None):
    assert reference is None or row["split"] == "train", "Reference answers prohibited outside train"
    if row["options"]:
        return list(row["options"])
    k = bank_key(row)
    bank = banks["numeric"].get(k)
    if bank is not None:
        center = number(reference if reference is not None else baseline)
        if center is None:
            center = bank[2]
        if kind(row) == "absolute_direction_object":
            vs = [
                center,
                (center - 90) % 360,
                (center - 45) % 360,
                (center - 15) % 360,
                (center + 15) % 360,
                (center + 45) % 360,
                (center + 90) % 360,
            ] + bank
        else:
            vs = [center] + [center * f for f in [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]] + bank
        if kind(row) == "absolute_count":
            vs = [round(x) for x in vs] + [max(0, round(center) - 1), round(center) + 1]
        vals = list(dict.fromkeys(fmt(x) for x in vs))[:12]
        if len(vals) < 2:
            vals.append(fmt(center + 1))
    else:
        vals = list(
            dict.fromkeys(
                ([reference] if reference else [baseline]) + banks["words"].get(kind(row), [])
            )
        )
        vals = [x for x in vals if x][:12]
        assert len(vals) >= 2, ("Missing free-answer candidate bank", row["id"])
    # A content-independent hash permutes candidates; no center/correct-slot index.
    return sorted(vals, key=lambda s: key(row["id"] + ":candidate:" + s))
