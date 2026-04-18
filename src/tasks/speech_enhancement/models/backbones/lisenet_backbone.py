"""LiSenNet-inspired encoder / dual-path stack for flow matching.

Reference: https://github.com/hyyan2k/LiSenNet

The original model uses mag / GD / IFD features and Griffin-Lim; here we use a learned
``stem`` from flow channels into the LiSenNet-style encoder and predict the velocity
field on the STFT grid. Frequency size defaults to ``n_fft//2+1`` (512-point FFT → 257 bins).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone, TimeCondMLP


class _CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps: float = 1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = nn.Parameter(torch.ones(*param_size))
        self.beta = nn.Parameter(torch.zeros(*param_size))
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=self.stat_dims, keepdim=True)
        std = torch.sqrt(x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps)
        return (x - mu) / std * self.gamma + self.beta


class _RNN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout_p: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(hidden_dim * (2 if bidirectional else 1), emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(x)
        return self.dense(x)


class _DualPathRNN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, n_freqs: int, dropout_p: float = 0.1, intra_causal: bool = False):
        super().__init__()
        # Intra: along frequency - bidirectional unless streaming-only freq context is required
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = _RNN(emb_dim, hidden_dim // 2, dropout_p, bidirectional=not intra_causal)
        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = _RNN(emb_dim, hidden_dim, dropout_p, bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T, F)
        b, d, t, f = x.shape
        z = x.permute(0, 2, 3, 1)  # (B,T,F,D)
        res = z
        z = self.intra_norm(z)
        z = z.reshape(b * t, f, d)
        z = self.intra_rnn_attn(z).reshape(b, t, f, d)
        z = z + res

        res = z
        z = self.inter_norm(z)
        z = z.permute(0, 2, 1, 3).reshape(b * f, t, d)
        z = self.inter_rnn_attn(z).reshape(b, f, t, d).permute(0, 2, 1, 3)
        z = z + res
        return z.permute(0, 3, 1, 2)


class _ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim: int, n_freqs: int, expansion_factor: int = 2, dropout_p: float = 0.1, causal: bool = True):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.causal = causal
        pad = (1, 1, 2, 0) if causal else (1, 1, 1, 1)
        self.norm = _CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d(pad, 0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        return x + res


class _DPR(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, n_freqs: int, dropout_p: float = 0.1, causal: bool = True):
        super().__init__()
        self.dp_rnn = _DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p, intra_causal=causal)
        self.conv_glu = _ConvolutionalGLU(emb_dim, n_freqs=n_freqs, dropout_p=dropout_p, causal=causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_glu(self.dp_rnn(x))


class _DSConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_freqs: int, causal: bool = True):
        super().__init__()
        self.low_freqs = n_freqs // 4
        p = (1, 1, 1, 0) if causal else (1, 1, 1, 1)
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d(p, 0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d(p, 0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3)),
        )
        self.norm = _CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x[..., : self.low_freqs]
        x_high = x[..., self.low_freqs :]
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        return self.act(self.norm(x))


class _Encoder(nn.Module):
    def __init__(self, in_ch: int, num_channels: int = 16, n_freqs: int = 257):
        super().__init__()
        self.n_freqs = n_freqs
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, num_channels // 4, (1, 1)),
            _CustomLayerNorm((1, n_freqs), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        self.conv_2 = _DSConv(num_channels // 4, num_channels // 2, n_freqs=n_freqs)
        self.conv_3 = _DSConv(num_channels // 2, num_channels // 4 * 3, n_freqs=n_freqs // 2)
        self.conv_4 = _DSConv(num_channels // 4 * 3, num_channels, n_freqs=n_freqs // 4)

    def forward(self, x: torch.Tensor):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x)
        out_list.append(x)
        x = self.conv_3(x)
        out_list.append(x)
        x = self.conv_4(x)
        out_list.append(x)
        return out_list


class _LiSenNetDecoderHead(nn.Module):
    """Upsample narrow-band features back to full frequency and project to ``out_channels``."""

    def __init__(self, num_channels: int, n_freqs: int, out_channels: int):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (1, 1)),
            nn.PReLU(num_channels),
            nn.Conv2d(num_channels, out_channels, (1, 1)),
        )
        self.n_target = n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T, F_small) - interpolate to full STFT width
        x = F.interpolate(x, size=(x.shape[2], self.n_target), mode="bilinear", align_corners=False)
        return self.out(x)


class LiSenNetFlowBackbone(SpeechFlowBackbone):
    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_fft: int = 512,
        num_channels: int = 16,
        n_blocks: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1

        self.stem = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.encoder = _Encoder(3, num_channels=num_channels, n_freqs=self.n_freqs)
        nf = self.n_freqs // (2**3)
        self.blocks = nn.Sequential(
            *[_DPR(num_channels, num_channels // 2 * 3, n_freqs=nf, causal=causal) for _ in range(n_blocks)]
        )
        self.decoder = _LiSenNetDecoderHead(num_channels, self.n_freqs, out_channels)
        self.time_in = TimeCondMLP(num_channels)
        self.time_r = TimeCondMLP(num_channels)
        self.time_fuse = nn.Linear(num_channels * 2, num_channels)

    def _temb(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        a = self.time_in(t)
        if r is None:
            return a
        return self.time_fuse(torch.cat([a, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        # Backbone API: (B, C, F, T). Encoder / LiSenNet blocks use (B, C, T, F) - frequency last.
        c = self._temb(t, r)
        x = self.stem(sample)
        x = x.permute(0, 1, 3, 2).contiguous()
        if x.shape[-1] != self.n_freqs:
            x = F.interpolate(x, size=(x.shape[2], self.n_freqs), mode="bilinear", align_corners=False)
        enc = self.encoder(x)
        h = enc[-1] + c[:, :, None, None]
        h = self.blocks(h)
        out = self.decoder(h)
        return out.permute(0, 1, 3, 2).contiguous()
