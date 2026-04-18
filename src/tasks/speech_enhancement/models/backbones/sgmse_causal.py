"""Causal SGMSE-inspired backbones for streaming-capable flow matching.

Two variants are provided for empirical comparison:

1) ``CausalSGMSEDilatedBackbone``:
   Purely convolutional temporal modeling with dilated causal depthwise convs.
2) ``CausalSGMSEAttentionBackbone``:
   Hybrid of local causal conv + causal temporal self-attention.

Both variants:
- keep a parallel full-sequence ``forward(sample, t, r, cond)`` API for training;
- use strictly causal time operations, so they can be wrapped with
  :class:`StatefulBackboneAdapter` for frame-by-frame streaming inference.
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
)


class _CumLN2d(nn.Module):
    """Causal-safe channel/frequency normalization surrogate."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2), keepdim=True)
        v = (x - m).pow(2).mean(dim=(1, 2), keepdim=True)
        return (x - m) / torch.sqrt(v + 1e-5) * self.weight + self.bias


class _FiLM2d(nn.Module):
    """Feature-wise affine modulation from time embeddings."""

    def __init__(self, channels: int):
        super().__init__()
        self.to_scale = nn.Linear(channels, channels)
        self.to_shift = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T), cond: (B, C)
        scale = self.to_scale(cond)[:, :, None, None]
        shift = self.to_shift(cond)[:, :, None, None]
        return x * (1.0 + scale) + shift


class _DilatedCausalBlock(nn.Module):
    """Residual block with causal dilated depthwise temporal conv."""

    def __init__(
        self,
        channels: int,
        kernel_time: int,
        dilation_time: int,
        kernel_freq: int,
        dropout: float,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.norm = _CumLN2d(channels)
        self.film = _FiLM2d(channels)
        cdim = int(cond_dim) if cond_dim is not None else self.channels
        self.cond_proj = nn.Identity() if cdim == self.channels else nn.Linear(cdim, self.channels)
        self.dw = CausalConv2d(
            channels,
            channels,
            kernel_size=(1, kernel_time),
            dilation=(1, int(dilation_time)),
            groups=channels,
            bias=True,
        )
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.pw2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        # Cheap frequency mixer: depthwise F-conv + 1x1 projection.
        self.freq_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(int(kernel_freq), 1),
            padding=(int(kernel_freq) // 2, 0),
            groups=channels,
            bias=True,
        )
        self.freq_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(cond)
        h = self.norm(x)
        h = self.film(h, cond)
        h = self.dw(h)
        h = F.gelu(self.pw1(h))
        h = self.drop(self.pw2(h))
        h = h + self.freq_pw(F.gelu(self.freq_dw(h)))
        return x + h


class _CausalTimeAttention(nn.Module):
    """Causal temporal MHSA applied independently per frequency bin."""

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float,
        context_frames: int,
        rel_pos_max_distance: int,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads}).")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = channels // heads
        self.drop_p = float(dropout)
        self.context_frames = int(context_frames)
        self.rel_pos_max_distance = int(max(1, rel_pos_max_distance))
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.rel_pos_bias = nn.Parameter(torch.zeros(self.heads, self.rel_pos_max_distance))

    def _causal_band_mask(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q_idx = torch.arange(t, device=device)[:, None]
        k_idx = torch.arange(t, device=device)[None, :]
        dist = q_idx - k_idx
        invalid = dist < 0
        if self.context_frames > 0:
            invalid = invalid | (dist >= self.context_frames)
        mask = torch.zeros(t, t, device=device, dtype=dtype)
        return mask.masked_fill(invalid, float("-inf"))

    def _relative_bias(self, t: int, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(t, device=device)[:, None]
        k_idx = torch.arange(t, device=device)[None, :]
        dist = (q_idx - k_idx).clamp(min=0, max=self.rel_pos_max_distance - 1)
        # (heads, T, T)
        return self.rel_pos_bias[:, dist]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> process (B*F, T, C)
        b, c, f, t = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        qkv = self.qkv(h).reshape(b * f, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B*F, T, H, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + self._causal_band_mask(t, device=attn.device, dtype=attn.dtype)[None, None, :, :]
        attn = attn + self._relative_bias(t, device=attn.device)[None, :, :, :]
        attn = F.softmax(attn, dim=-1)
        if self.training and self.drop_p > 0:
            attn = F.dropout(attn, p=self.drop_p)
        y = torch.matmul(attn, v)

        y = y.permute(0, 2, 1, 3).reshape(b * f, t, c)
        y = self.proj(y)
        y = y.reshape(b, f, t, c).permute(0, 3, 1, 2).contiguous()
        return y


class _ConvHeavyBlock(nn.Module):
    """Conv-heavy residual block to preserve local detail."""

    def __init__(self, channels: int, kernel_time: int, dropout: float):
        super().__init__()
        self.norm1 = _CumLN2d(channels)
        self.film1 = _FiLM2d(channels)
        self.local1 = CausalConv2d(channels, channels, kernel_size=(3, kernel_time), bias=True)
        self.local2 = CausalConv2d(channels, channels, kernel_size=(3, kernel_time), bias=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 3, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 3, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.film1(h, cond)
        h = F.gelu(self.local1(h))
        h = F.gelu(self.local2(h))
        x = x + h
        x = x + self.ffn(h)
        return x


class _AttnLightBlock(nn.Module):
    """Attention-light residual block for long-range continuity."""

    def __init__(
        self,
        channels: int,
        kernel_time: int,
        heads: int,
        dropout: float,
        context_frames: int,
        rel_pos_max_distance: int,
    ):
        super().__init__()
        self.norm1 = _CumLN2d(channels)
        self.film1 = _FiLM2d(channels)
        self.local = CausalConv2d(channels, channels, kernel_size=(3, kernel_time), bias=True)
        self.norm2 = _CumLN2d(channels)
        self.film2 = _FiLM2d(channels)
        self.attn = _CausalTimeAttention(
            channels,
            heads=heads,
            dropout=dropout,
            context_frames=context_frames,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.film1(h, cond)
        x = x + F.gelu(self.local(h))
        h = self.norm2(x)
        h = self.film2(h, cond)
        x = x + self.attn(h)
        x = x + self.ffn(h)
        return x


class CausalSGMSEDilatedBackbone(SpeechFlowBackbone):
    """Causal SGMSE-style backbone using dilated temporal conv stacks.

    Design choice: prioritize stable low-latency behavior with local-to-mid receptive
    field growth from dilation cycles, while keeping a parallel training path.
    """

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 72,
        depth: int = 6,
        kernel_time: int = 7,
        kernel_freq: int = 5,
        freq_scales: int = 3,
        channel_mults: Optional[list[int]] = None,
        blocks_per_scale: Optional[list[int]] = None,
        dilation_cycle: int = 4,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        if not causal:
            raise ValueError("CausalSGMSEDilatedBackbone requires causal=True.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.causal = True
        self.freq_scales = int(max(1, freq_scales))
        if channel_mults is None:
            channel_mults = [1 + i for i in range(self.freq_scales)]
        if len(channel_mults) != self.freq_scales:
            raise ValueError(f"channel_mults must have length={self.freq_scales}, got {len(channel_mults)}")
        self.stage_channels = [int(hidden) * int(m) for m in channel_mults]

        if blocks_per_scale is None:
            # Memory-efficient default: fewer blocks at high-resolution scale.
            blocks_per_scale = [1] + [max(1, int(depth) // self.freq_scales)] * (self.freq_scales - 1)
        if len(blocks_per_scale) != self.freq_scales:
            raise ValueError(f"blocks_per_scale must have length={self.freq_scales}, got {len(blocks_per_scale)}")
        self.blocks_per_scale = [int(max(1, b)) for b in blocks_per_scale]

        # History depends on temporal kernel/dilations and total number of residual blocks
        # traversed along encoder + bottleneck + decoder.
        n_res_blocks_total = (
            sum(self.blocks_per_scale)  # encoder
            + 1  # bottleneck
            + sum(self.blocks_per_scale[:-1])  # decoder
        )
        self.stream_history = int(
            (kernel_time - 1) * sum(2 ** (k % max(1, int(dilation_cycle))) for k in range(n_res_blocks_total))
            + 8
        )

        self.stem = nn.Conv2d(self.in_channels, self.stage_channels[0], kernel_size=1, bias=True)
        cond_dim = int(self.stage_channels[-1])
        self.time_t = TimeCondMLP(cond_dim)
        self.time_r = TimeCondMLP(cond_dim)
        self.time_fuse = nn.Linear(cond_dim * 2, cond_dim)

        # Encoder: per-scale residual stacks + frequency downsampling only (stride=(2,1)).
        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        d_idx = 0
        for s in range(self.freq_scales):
            stage = nn.ModuleList()
            ch_s = self.stage_channels[s]
            for _ in range(self.blocks_per_scale[s]):
                stage.append(
                    _DilatedCausalBlock(
                        channels=ch_s,
                        kernel_time=kernel_time,
                        dilation_time=2 ** (d_idx % max(1, int(dilation_cycle))),
                        kernel_freq=kernel_freq,
                        dropout=dropout,
                        cond_dim=cond_dim,
                    )
                )
                d_idx += 1
            self.enc_blocks.append(stage)
            if s < self.freq_scales - 1:
                self.down.append(
                    nn.Conv2d(
                        self.stage_channels[s],
                        self.stage_channels[s + 1],
                        kernel_size=(3, 1),
                        stride=(2, 1),
                        padding=(1, 0),
                        bias=True,
                    )
                )

        # Bottleneck at lowest frequency resolution.
        self.mid = _DilatedCausalBlock(
            channels=self.stage_channels[-1],
            kernel_time=kernel_time,
            dilation_time=2 ** (d_idx % max(1, int(dilation_cycle))),
            kernel_freq=kernel_freq,
            dropout=dropout,
            cond_dim=cond_dim,
        )
        d_idx += 1

        # Decoder: frequency upsampling + residual stacks.
        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for s in range(self.freq_scales - 1):
            ch_in = self.stage_channels[-(s + 1)]
            ch_out = self.stage_channels[-(s + 2)]
            self.up.append(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=True)
            )
            stage = nn.ModuleList()
            dec_scale_idx = self.freq_scales - s - 2
            for _ in range(self.blocks_per_scale[dec_scale_idx]):
                stage.append(
                    _DilatedCausalBlock(
                        channels=ch_out,
                        kernel_time=kernel_time,
                        dilation_time=2 ** (d_idx % max(1, int(dilation_cycle))),
                        kernel_freq=kernel_freq,
                        dropout=dropout,
                        cond_dim=cond_dim,
                    )
                )
                d_idx += 1
            self.dec_blocks.append(stage)

        self.head = nn.Conv2d(self.stage_channels[0], self.out_channels, kernel_size=1)

    def _cond(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        te = self.time_t(t)
        if r is None:
            return te
        return self.time_fuse(torch.cat([te, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        del cond
        tc = self._cond(t, r)
        x = self.stem(sample)

        skips: list[torch.Tensor] = []
        for s, stage in enumerate(self.enc_blocks):
            for blk in stage:
                x = blk(x, tc)
            if s < self.freq_scales - 1:
                skips.append(x)
                x = self.down[s](x)

        x = self.mid(x, tc)

        for s, stage in enumerate(self.dec_blocks):
            x = self.up[s](x)
            skip = skips[-(s + 1)]
            if x.shape[-2] != skip.shape[-2]:
                x = F.interpolate(x, size=(skip.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
            x = x + skip
            for blk in stage:
                x = blk(x, tc)

        return self.head(x)


class CausalSGMSEAttentionBackbone(SpeechFlowBackbone):
    """Causal SGMSE-style hybrid backbone with temporal attention.

    Design choice: combine local causal convs with causal long-range temporal attention
    for improved global consistency while preserving streamability.
    """

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 128,
        depth: int = 8,
        heads: int = 8,
        kernel_time: int = 7,
        attention_context_frames: int = 256,
        rel_pos_max_distance: int = 256,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        if not causal:
            raise ValueError("CausalSGMSEAttentionBackbone requires causal=True.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.causal = True
        self.attention_context_frames = int(attention_context_frames)
        # Keep default historical context around 2s at 16k/128-hop.
        self.stream_history = int(max(64, self.attention_context_frames))

        self.stem = nn.Conv2d(self.in_channels, hidden, kernel_size=1)
        self.time_t = TimeCondMLP(hidden)
        self.time_r = TimeCondMLP(hidden)
        self.time_fuse = nn.Linear(hidden * 2, hidden)
        blocks: list[nn.Module] = []
        for i in range(int(depth)):
            # Alternate pattern: conv-heavy -> attn-light -> conv-heavy
            if i % 3 == 1:
                blocks.append(
                    _AttnLightBlock(
                        channels=hidden,
                        kernel_time=kernel_time,
                        heads=heads,
                        dropout=dropout,
                        context_frames=self.attention_context_frames,
                        rel_pos_max_distance=rel_pos_max_distance,
                    )
                )
            else:
                blocks.append(_ConvHeavyBlock(channels=hidden, kernel_time=kernel_time, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = _CumLN2d(hidden)
        self.head = nn.Conv2d(hidden, self.out_channels, kernel_size=1)

    def _cond(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        te = self.time_t(t)
        if r is None:
            return te
        return self.time_fuse(torch.cat([te, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        del cond
        tc = self._cond(t, r)
        x = self.stem(sample)
        for blk in self.blocks:
            x = blk(x, tc)
        x = self.norm_out(x)
        return self.head(x)
