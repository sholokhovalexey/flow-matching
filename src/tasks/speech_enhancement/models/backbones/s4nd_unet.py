"""S4ND-U-Net inspired STFT backbone with optional causal inference.

This is a lightweight adaptation for flow-matching speech enhancement:
- U-Net over (freq, time) STFT grids
- Long temporal mixing via depthwise convs (S4-like surrogate)
- Causal mode uses left-only temporal padding via :class:`CausalConv2d`
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import CausalConv2d, SpeechFlowBackbone, TimeCondMLP


class _CumLN2d(nn.Module):
    """Cumulative LayerNorm surrogate for causal streams (time-prefix stats)."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        m = x.mean(dim=(1, 2), keepdim=True)
        v = (x - m).pow(2).mean(dim=(1, 2), keepdim=True)
        return (x - m) / torch.sqrt(v + 1e-5) * self.weight + self.bias


class _S4LikeBlock(nn.Module):
    """Residual long-range temporal mixer with depthwise conv."""

    def __init__(self, channels: int, kernel_t: int, causal: bool, dropout: float):
        super().__init__()
        self.causal = causal
        self.norm = _CumLN2d(channels)
        if causal:
            self.dw = CausalConv2d(channels, channels, kernel_size=(1, kernel_t), groups=channels, bias=True)
        else:
            self.dw = nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, kernel_t),
                padding=(0, kernel_t // 2),
                groups=channels,
                bias=True,
            )
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.pw2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h)
        h = F.gelu(self.pw1(h))
        h = self.drop(self.pw2(h))
        return x + h


class S4NDUNetBackbone(SpeechFlowBackbone):
    """S4ND-inspired U-Net backbone for STFT flow matching."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 80,
        depth: int = 6,
        kernel_time: int = 31,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.causal = bool(causal)

        self.stem = nn.Conv2d(self.in_channels, hidden, kernel_size=1)
        self.time_t = TimeCondMLP(hidden)
        self.time_r = TimeCondMLP(hidden)
        self.time_fuse = nn.Linear(hidden * 2, hidden)

        self.enc = nn.ModuleList([_S4LikeBlock(hidden, kernel_time, self.causal, dropout) for _ in range(depth // 2)])
        self.down = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.mid = _S4LikeBlock(hidden, kernel_time, self.causal, dropout)
        self.up = nn.ConvTranspose2d(hidden, hidden, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.dec = nn.ModuleList([_S4LikeBlock(hidden, kernel_time, self.causal, dropout) for _ in range(depth - depth // 2)])
        self.head = nn.Conv2d(hidden, self.out_channels, kernel_size=1)

    def _cond(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        te = self.time_t(t)
        if r is None:
            return te
        return self.time_fuse(torch.cat([te, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        del cond
        c = self._cond(t, r)
        x = self.stem(sample)
        x = x + c[:, :, None, None]
        for blk in self.enc:
            x = blk(x)
        skip = x
        x = self.down(x)
        x = self.mid(x)
        x = self.up(x)
        # frequency-size safe merge
        if x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=(skip.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        x = x + skip
        for blk in self.dec:
            x = blk(x)
        return self.head(x)

