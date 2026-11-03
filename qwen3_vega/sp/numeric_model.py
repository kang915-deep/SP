"""Preserve the original head initialization; add a zero-initialized input path."""

import torch
from torch import nn
from model import Corrector
from numeric_features import FEATURE_DIM


class NumericCorrector(Corrector):
    def __init__(self, wan=True, width=128):
        super().__init__(wan, width)
        old = self.fuse[0]
        extended = nn.Linear(old.in_features + FEATURE_DIM, old.out_features)
        with torch.no_grad():
            extended.weight[:, : old.in_features].copy_(old.weight)
            extended.weight[:, old.in_features :].zero_()
            extended.bias.copy_(old.bias)
        self.fuse[0] = extended

    def forward(self, h, context, native, w=None, wp=None, wmask=None, numeric=None):
        assert numeric is not None and numeric.shape == (*native.shape[:-1], FEATURE_DIM)
        assert torch.isfinite(numeric).all()
        return super().forward(
            h, context, torch.cat([native, numeric.to(native.dtype)], dim=-1), w, wp, wmask
        )
