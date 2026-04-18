"""Optional SpeechBrain :class:`ConformerEncoder` adapter (install ``speechbrain``).

Uses ``attention_type="regularMHA"`` so positional embeddings are not required. When
``causal=True``, convolution modules use causal padding and a causal attention mask
is applied on the time sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.tasks.speech_enhancement.compat import apply_torchaudio_speechbrain_compat
from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone, TimeCondMLP

apply_torchaudio_speechbrain_compat()

try:
    from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
except ImportError:
    ConformerEncoder = None


class SpeechBrainConformerBackbone(SpeechFlowBackbone):
    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_freq_bins: int,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.0,
        causal: bool = False,
        d_ffn: Optional[int] = None,
    ):
        super().__init__()
        if ConformerEncoder is None:
            raise ImportError(
                "SpeechBrainConformerBackbone requires the `speechbrain` package. "
                "Install with: pip install speechbrain"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_freq_bins = num_freq_bins
        self.causal = causal
        d_ffn = d_ffn or d_model * 4
        self.in_proj = nn.Linear(in_channels * num_freq_bins, d_model)
        self.encoder = ConformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            attention_type="regularMHA",
        )
        self.out_proj = nn.Linear(d_model, out_channels * num_freq_bins)
        self.time_in = TimeCondMLP(d_model)
        self.time_r = TimeCondMLP(d_model)
        self.time_fuse = nn.Linear(d_model * 2, d_model)

    def _mask(self, tlen: int, device: torch.device) -> torch.Tensor:
        # float mask: (T,T) with -inf above diagonal - SpeechBrain passes to MHA
        m = torch.triu(torch.ones(tlen, tlen, device=device, dtype=torch.bool), diagonal=1)
        fill = torch.zeros(tlen, tlen, device=device, dtype=torch.float32)
        fill = fill.masked_fill(m, float("-inf"))
        return fill

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        # sample: B, C, F, T  -  flatten frequency into features per time step
        b, c, f, tm = sample.shape
        assert f == self.num_freq_bins, (f, self.num_freq_bins)
        x = sample.permute(0, 3, 2, 1).reshape(b, tm, f * c)
        x = self.in_proj(x)
        te = self.time_in(t)
        if r is None:
            tc = te
        else:
            tc = self.time_fuse(torch.cat([te, self.time_r(r)], dim=-1))
        x = x + tc.unsqueeze(1)

        src_mask = self._mask(tm, sample.device) if self.causal else None
        y, _ = self.encoder(x, src_mask=src_mask, pos_embs=None)
        y = self.out_proj(y)
        y = y.reshape(b, tm, f, self.out_channels).permute(0, 3, 2, 1)
        return y
