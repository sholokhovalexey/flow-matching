"""RNNFormer-style streaming backbone inspired by FastEnhancer (ICASSP 2026).

Reference: https://github.com/aask1357/fastenhancer

This is a self-contained approximation of the RNNFormer stack (freq attention + causal
time GRU) for flow matching; it does not depend on the upstream ``functional`` / STFT helpers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone, TimeCondMLP


class _FreqAttention(nn.Module):
    """Lightweight frequency attention (per time step)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F, D)
        b, t, f, d = x.shape
        h = x.reshape(b * t, f, d)
        qkv = self.qkv(h).reshape(b * t, f, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b * t, f, d)
        o = self.proj(o).reshape(b, t, f, d)
        return o


class _RNNFormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, causal_time: bool):
        super().__init__()
        self.causal_time = causal_time
        self.ln1 = nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=not causal_time)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = _FreqAttention(dim, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F, D) - GRU must step along time; cannot use reshape alone (would scramble T/F).
        b, t, f, d = x.shape
        h = self.ln1(x)
        g = h.permute(0, 2, 1, 3).contiguous().reshape(b * f, t, d)
        g, _ = self.gru(g)
        g = g.reshape(b, f, t, d).transpose(1, 2)
        x = x + g

        h = self.ln2(x)
        a = self.attn(h)
        return x + a


class FastEnhancerRNNFormerBackbone(SpeechFlowBackbone):
    """Streaming-oriented backbone: causal GRU on time when ``causal=True``."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 48,
        num_blocks: int = 4,
        num_heads: int = 4,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.stem = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _RNNFormerBlock(hidden, num_heads, causal_time=causal)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.time_in = TimeCondMLP(hidden)
        self.time_r = TimeCondMLP(hidden)
        self.time_fuse = nn.Linear(hidden * 2, hidden)

    def _temb(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        a = self.time_in(t)
        if r is None:
            return a
        return self.time_fuse(torch.cat([a, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        # (B, C, F, T) -> (B, T, F, H)
        c = self._temb(t, r)
        h = self.stem(sample)
        h = h + c[:, :, None, None]
        b, ch, f, tm = h.shape
        x = h.permute(0, 3, 2, 1)  # (B, T, F, C)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 2, 1)  # (B, C, F, T)
        return self.head(x)
