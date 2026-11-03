import torch
from torch import nn


class Corrector(nn.Module):
    def __init__(self, wan=False, width=128):
        super().__init__()
        self.wan = wan
        self.project = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, width), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(width * 3 + 8, width), nn.GELU(), nn.LayerNorm(width))
        self.body = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Created after core so core initialization is identical in both arms.
        if wan:
            self.memory = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, width))
            self.position = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)

    def forward(self, h, context, native, w=None, wp=None, wmask=None):
        z = self.project(h)
        b = z[:, :1].expand_as(z)
        q = self.project(context)[:, None].expand_as(z)
        u = self.fuse(torch.cat([z, b, q, native], dim=-1))
        if self.wan:
            assert w is not None and wp is not None
            memory = self.memory(w) + self.position(wp)
            v = self.attention(u, memory, memory, key_padding_mask=wmask, need_weights=False)[0]
        else:
            v = torch.zeros_like(u)
        raw = self.output(self.body(torch.cat([u, v], dim=-1))).squeeze(-1)
        return raw - raw[:, :1]
