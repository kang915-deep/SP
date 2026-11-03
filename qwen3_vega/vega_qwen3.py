"""VEGA token-gated residual fusion inside native Qwen3-VL.

Requires the original VEGA src directory on PYTHONPATH. Original Qwen2.5
weights are not loaded: the changed-width merger/fusion are initialized anew.
The native Qwen3 DeepStack branches and position encoding remain intact.
"""
import torch
from torch import nn


class WanPatchMerger(nn.Module):
    """Official VEGA MLP merger, with explicit RMSNorm (no LayerNorm fallback)."""
    def __init__(self, context_dim, hidden_dim, merge_size):
        super().__init__()
        self.merge_size = merge_size
        self.context_dim = context_dim
        self.norm_weight = nn.Parameter(torch.ones(context_dim))
        self.mlp = nn.Sequential(
            nn.Linear(context_dim * merge_size**2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features):
        n, h, w, c = features.shape
        m = self.merge_size
        x = features.reshape(n, h//m, m, w//m, m, c).permute(0,1,3,2,4,5).contiguous()
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        x = self.norm_weight * normalized.to(x.dtype)
        return self.mlp(x.reshape(-1, c*m*m)).reshape(n,h//m,w//m,-1)
