"""Unified interface and shared utilities for speech enhancement flow backbones."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn_profile import (
    STFT_SAMPLE_RATE_16K,
    STFT_WIN_MS_20,
    default_rfft_num_freq_bins,
    estimate_backbone_forward_flops,
    print_backbone_flop_line,
    stft_hop_samples_half_win,
    stft_time_frames_one_second_16k_20ms,
    stft_win_samples_16k_20ms,
)
from ..nn_profile import count_parameters as count_backbone_parameters


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """``t``: scalar, ``(B,)``, or any shape that flattens to a batch of times -> ``(N, dim)``.

    ODE solvers (e.g. ``torchdiffeq``) often pass ``t`` as a 0-dim tensor; training passes ``(B,)``.
    """
    t = t.detach().float().reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class TimeCondMLP(nn.Module):
    """Maps scalar flow time(s) to a conditioning vector (same role as UNet time embedding)."""

    def __init__(self, cond_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.net = nn.Sequential(
            nn.Linear(frequency_embedding_size, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_time_embedding(t, self.frequency_embedding_size)
        return self.net(emb)


class SpeechFlowBackbone(nn.Module, ABC):
    """Contract for velocity fields used with :class:`SpeechCondWrapper`.

    Inputs follow the diffusers-style UNet convention: ``sample`` is ``(B, C, H, W)`` with
    ``H`` frequency bins and ``W`` STFT time frames (or latent width).

    Subclasses should set ``in_channels`` / ``out_channels`` to match the wrapped tensor
    (concatenated trajectory + condition).
    """

    in_channels: int
    out_channels: int
    causal: bool
    #: If ``True``, this backbone enforces time causality (no future STFT frames).
    supports_causal: bool = True

    @abstractmethod
    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        """Return tensor shaped like the flow target (typically ``out_channels`` on first dim)."""

    def extra_repr(self) -> str:
        return f"causal={self.causal}, supports_causal={self.supports_causal}"


class CausalConv2d(nn.Module):
    """2D conv with **causal** padding along the last axis (STFT time / width ``W`` only)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad frequency (dim -2) symmetrically; time (dim -1) left-only for causality.
        kf, kt = self.kernel_size
        df, dt = self.dilation
        pad_f = ((kf - 1) * df) // 2
        pad_t = (kt - 1) * dt
        x = F.pad(x, (pad_t, 0, pad_f, pad_f))
        return self.conv(x)


def time_causal_attn_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Lower-triangular mask for ``scaled_dot_product_attention`` (additive, 0 / -inf)."""
    m = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(m, float("-inf"))
    return mask


def time_causal_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """``q,k,v``: (B, heads, T, D). Uses PyTorch SDPA causal flag when available."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
