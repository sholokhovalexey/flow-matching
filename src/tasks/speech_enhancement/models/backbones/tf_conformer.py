"""Time-frequency Conformer stack (unifies CMGAN-style and music-enhancement-style presets).

References (architectural lineage; implementation is self-contained):

* CMGAN / Conformer speech enhancement - https://github.com/ruizhecao96/CMGAN
* Music audio enhancement Conformer - https://github.com/yoongi43/music_audio_enhancement_conformer
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


class _FFN(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hid = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MultiHeadTimeAttention(nn.Module):
    """Self-attention along time (fixed frequency slice). Causal optional."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        # x: (B*F, T, D)
        bft, t, d = x.shape
        qkv = self.qkv(x).reshape(bft, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if causal:
            out = time_causal_sdpa(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop.p if self.training else 0.0, is_causal=False
            )
        out = out.transpose(1, 2).reshape(bft, t, d)
        return self.proj(out)


class TFConformerLayer(nn.Module):
    """One T-F Conformer layer: macaron FFN, time attention, depthwise T-F conv, FFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kernel_freq: int,
        kernel_time: int,
        dropout: float,
        causal: bool,
    ):
        super().__init__()
        self.causal = causal
        self.ffn1 = _FFN(dim, dropout=dropout)
        self.norm_att = nn.LayerNorm(dim)
        self.mhsa_t = _MultiHeadTimeAttention(dim, heads, dropout=dropout)
        self.norm_conv = nn.LayerNorm(dim)
        self.dw_conv = (
            CausalConv2d(dim, dim, (kernel_freq, kernel_time), groups=dim)
            if causal
            else nn.Conv2d(
                dim,
                dim,
                (kernel_freq, kernel_time),
                padding=(kernel_freq // 2, kernel_time // 2),
                groups=dim,
            )
        )
        self.pw_conv = nn.Conv2d(dim, dim, 1)
        self.ffn2 = _FFN(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, D, F, T
        b, d, f, tm = x.shape
        x = x + 0.5 * self.ffn1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Time MHSA per frequency bin
        h = x.permute(0, 2, 3, 1).reshape(b * f, tm, d)
        h = self.norm_att(h)
        h = self.mhsa_t(h, causal=self.causal)
        h = h.reshape(b, f, tm, d).permute(0, 3, 1, 2)
        x = x + h

        # Depthwise T-F conv
        h = self.norm_conv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.gelu(self.dw_conv(h))
        h = self.pw_conv(h)
        x = x + h

        x = x + 0.5 * self.ffn2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


_PRESETS = {
    # Wider convs / fewer heads - closer to music enhancement recipes.
    "music": dict(depth=4, heads=4, kernel_freq=9, kernel_time=5, ffn_mult=4.0),
    # Tighter kernels - closer to CMGAN speech defaults.
    "cmgan": dict(depth=6, heads=8, kernel_freq=15, kernel_time=3, ffn_mult=4.0),
}


class TFConformerBackbone(SpeechFlowBackbone):
    """STFT-grid Conformer stack with shared interface for flow matching."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 128,
        depth: Optional[int] = None,
        heads: Optional[int] = None,
        kernel_freq: Optional[int] = None,
        kernel_time: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
        variant: str = "cmgan",
    ):
        super().__init__()
        if variant not in _PRESETS:
            raise ValueError(f"variant must be one of {list(_PRESETS)}, got {variant}")
        pr = _PRESETS[variant]
        depth = depth if depth is not None else pr["depth"]
        heads = heads if heads is not None else pr["heads"]
        kernel_freq = kernel_freq if kernel_freq is not None else pr["kernel_freq"]
        kernel_time = kernel_time if kernel_time is not None else pr["kernel_time"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.variant = variant

        self.stem = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                TFConformerLayer(hidden, heads, kernel_freq, kernel_time, dropout, causal)
                for _ in range(depth)
            ]
        )
        self.head = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.time_in = TimeCondMLP(hidden)
        self.time_r_in = TimeCondMLP(hidden)
        self.time_fuse = nn.Linear(hidden * 2, hidden)

    def cond_vec(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        te = self.time_in(t)
        if r is None:
            return te
        re = self.time_r_in(r)
        return self.time_fuse(torch.cat([te, re], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        c = self.cond_vec(t, r)
        x = self.stem(sample)
        x = x + c[:, :, None, None]

        for layer in self.layers:
            x = layer(x)

        x = self.head(x)
        return x
