"""Lightweight strictly-causal STFT backbone (alternative when UNet cannot be made causal)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import CausalConv2d, SpeechFlowBackbone, TimeCondMLP


class CausalSTFTStackBackbone(SpeechFlowBackbone):
    """Encoder stack using :class:`CausalConv2d` along time (width)."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 64,
        num_blocks: int = 8,
        kernel: tuple = (3, 5),
        causal: bool = True,
    ):
        super().__init__()
        if not causal:
            raise ValueError("CausalSTFTStackBackbone requires causal=True")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = True
        kf, kt = kernel
        self.stem = CausalConv2d(in_channels, hidden, (kf, kt))
        self.blocks = nn.ModuleList(
            [CausalConv2d(hidden, hidden, (kf, kt)) for _ in range(num_blocks - 1)]
        )
        self.head = nn.Conv2d(hidden, out_channels, 1)
        self.time_in = TimeCondMLP(hidden)
        self.time_r = TimeCondMLP(hidden)
        self.time_fuse = nn.Linear(hidden * 2, hidden)

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        te = self.time_in(t)
        if r is None:
            tc = te
        else:
            tc = self.time_fuse(torch.cat([te, self.time_r(r)], dim=-1))
        x = F.gelu(self.stem(sample))
        x = x + tc[:, :, None, None]
        for b in self.blocks:
            x = F.gelu(b(x))
        return self.head(x)
