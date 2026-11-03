"""Reference-free numerical relations, shared by training and inference."""

import math
import numpy as np
from legacy_candidates import kind, unit, number

KINDS = (
    "absolute_count",
    "absolute_distance_object",
    "absolute_size_object",
    "absolute_size_room",
    "absolute_direction_object",
)
FEATURE_NAMES = (
    "both_valid",
    "candidate_valid",
    "baseline_valid",
    "log_ratio",
    "relative_delta",
    "absolute_relative_delta",
    "candidate_log_physical",
    "baseline_log_physical",
    "candidate_zero",
    "baseline_zero",
    "angular_delta",
    "same_value",
    "is_count",
    "is_distance",
    "is_object_size",
    "is_room_size",
    "is_angle",
    "unit_known",
)
FEATURE_DIM = len(FEATURE_NAMES)
LENGTH_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "inch": 0.0254}


def physical_scale(typ, u):
    if typ == "absolute_count":
        return 1.0
    if typ == "absolute_direction_object":
        return 1 / 360
    if typ == "absolute_size_room":
        return LENGTH_TO_METERS[u[:-1]] ** 2 if u.endswith("2") and u[:-1] in LENGTH_TO_METERS else None
    return LENGTH_TO_METERS.get(u)


def clip(value, limit):
    return max(-limit, min(limit, value)) / limit


def features(row, answers):
    """No label, quality, target, model scores, fitted statistics or test tuning."""
    assert answers
    typ = kind(row)
    result = np.zeros((len(answers), FEATURE_DIM), dtype=np.float32)
    if typ not in KINDS or row.get("options"):
        return result
    scale = physical_scale(typ, unit(row))
    baseline = number(answers[0])
    for j, answer in enumerate(answers):
        candidate = number(answer)
        cv, bv = candidate is not None, baseline is not None
        both = cv and bv
        # Absolute features use standard physical units. Unknown units retain
        # only unit-free ratios, with an explicit unit_known=0 indicator.
        c = (candidate % 360 if typ == "absolute_direction_object" else candidate) if cv else 0.0
        b = (baseline % 360 if typ == "absolute_direction_object" else baseline) if bv else 0.0
        c *= scale if scale is not None else 1.0
        b *= scale if scale is not None else 1.0
        logratio = math.log(c + 0.001) - math.log(b + 0.001) if both else 0.0
        relative = c / (b + 0.001) - b / (b + 0.001) if both else 0.0
        angle = (
            ((candidate - baseline + 180) % 360 - 180) / 180
            if both and typ == "absolute_direction_object"
            else 0.0
        )
        result[j] = [
            both,
            cv,
            bv,
            clip(logratio, 6.0),
            clip(relative, 4.0),
            abs(clip(relative, 4.0)),
            min(math.log1p(c), 100.0) / 10 if cv and scale is not None else 0.0,
            min(math.log1p(b), 100.0) / 10 if bv and scale is not None else 0.0,
            cv and c == 0,
            bv and b == 0,
            angle,
            both and c == b,
            *[typ == k for k in KINDS],
            scale is not None,
        ]
    assert result.shape == (len(answers), FEATURE_DIM) and np.isfinite(result).all()
    return result
