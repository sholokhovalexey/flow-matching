"""SpatialNet-style T-F backbone (simplified OnlineSpatialNet-style stack).

Reference: https://github.com/Audio-WestlakeU/NBSS/blob/main/models/arch/OnlineSpatialNet.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import (
    CausalConv2d,
    SpeechFlowBackbone,
    TimeCondMLP,
    time_causal_sdpa,
)


class _MHSATime(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        # (B*F, T, D)
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if causal:
            o = time_causal_sdpa(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        else:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(b, t, d)
        return self.proj(o)


class _SpatialNetLayer(nn.Module):
    """Frequency-time block: time attention + depthwise T-F conv."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kf: int,
        kt: int,
        dropout: float,
        causal: bool,
    ):
        super().__init__()
        self.causal = causal
        self.n1 = nn.LayerNorm(dim)
        self.attn = _MHSATime(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        self.dw = (
            CausalConv2d(dim, dim, (kf, kt), groups=dim)
            if causal
            else nn.Conv2d(
                dim,
                dim,
                (kf, kt),
                padding=(kf // 2, kt // 2),
                groups=dim,
            )
        )
        self.pw = nn.Conv2d(dim, dim, 1)
        self.n3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,D,F,T
        b, d, f, tm = x.shape
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ht = h.permute(0, 2, 3, 1).reshape(b * f, tm, d)
        ht = self.attn(ht, causal=self.causal)
        ht = ht.reshape(b, f, tm, d).permute(0, 3, 1, 2)
        x = x + ht

        h = self.n2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.gelu(self.pw(self.dw(h)))
        x = x + h

        h = self.n3(x.permute(0, 2, 3, 1))
        x = x + self.ff(h).permute(0, 3, 1, 2)
        return x


class SpatialNetBackbone(SpeechFlowBackbone):
    """SpatialNet-inspired encoder for STFT / latent maps."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 128,
        num_layers: int = 6,
        heads: int = 4,
        kernel_freq: int = 5,
        kernel_time: int = 3,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.stem = nn.Conv2d(in_channels, dim, 1)
        self.layers = nn.ModuleList(
            [
                _SpatialNetLayer(dim, heads, kernel_freq, kernel_time, dropout, causal)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Conv2d(dim, out_channels, 1)
        self.time_in = TimeCondMLP(dim)
        self.time_r = TimeCondMLP(dim)
        self.time_fuse = nn.Linear(dim * 2, dim)

    def _temb(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        a = self.time_in(t)
        if r is None:
            return a
        b = self.time_r(r)
        return self.time_fuse(torch.cat([a, b], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        c = self._temb(t, r)
        x = self.stem(sample) + c[:, :, None, None]
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
