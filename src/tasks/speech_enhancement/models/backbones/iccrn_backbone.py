"""ICCRN-inspired backbone adapted to flow-matching STFT tensors.

Reference implementation source: https://github.com/JinjiangLiu/ICCRN

This adapts the core ICCRN ideas (channel-wise LSTMs, CFB gating, cepstral unit)
to the unified speech-flow interface used in this project:
``forward(sample, t, r, cond)`` with ``sample`` shaped ``(B, C, F, T)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone, TimeCondMLP


class _LayerNormCF(nn.Module):
    """Instance-like norm over channel/frequency with learnable (C,F)-affine terms."""

    def __init__(self, channels: int, freq_bins: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, freq_bins, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, freq_bins, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + 1e-8) * self.weight + self.bias


class _ChannelLSTMF(nn.Module):
    """Frequency-axis LSTM per time frame: (B,C,F,T) -> (B,C',F,T)."""

    def __init__(self, in_ch: int, feat_ch: int, out_ch: int, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, feat_ch, batch_first=True, bidirectional=bidirectional)
        proj_in = feat_ch * (2 if bidirectional else 1)
        self.linear = nn.Linear(proj_in, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        b, c, f, t = x.shape
        y = x.permute(0, 3, 2, 1).reshape(b * t, f, c)  # (B*T, F, C)
        # MeanFlow/ImprovedMeanFlow JVP may require higher-order grads; cuDNN RNN
        # kernels do not support double-backward, so disable cuDNN for this call.
        with torch.backends.cudnn.flags(enabled=False):
            y, _ = self.lstm(y.float())
        y = self.linear(y)
        y = y.reshape(b, t, f, -1).permute(0, 3, 2, 1).contiguous()
        return y


class _ChannelLSTMT(nn.Module):
    """Time-axis LSTM per frequency bin: (B,C,F,T) -> (B,C',F,T)."""

    def __init__(self, in_ch: int, feat_ch: int, out_ch: int, bidirectional: bool = False, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_ch, feat_ch, num_layers=num_layers, batch_first=True, bidirectional=bidirectional
        )
        proj_in = feat_ch * (2 if bidirectional else 1)
        self.linear = nn.Linear(proj_in, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        b, c, f, t = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b * f, t, c)  # (B*F, T, C)
        with torch.backends.cudnn.flags(enabled=False):
            y, _ = self.lstm(y.float())
        y = self.linear(y)
        y = y.reshape(b, f, t, -1).permute(0, 3, 1, 2).contiguous()
        return y


class _CepsUnit(nn.Module):
    """Cepstral branch from ICCRN: rFFT over freq, channel LSTM, iFFT reconstruction."""

    def __init__(self, channels: int, freq_bins: int):
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins
        self.norm = _LayerNormCF(channels * 2, freq_bins // 2 + 1)
        self.ch_lstm_f = _ChannelLSTMF(channels * 2, channels, channels * 2, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) real
        x_fft = torch.fft.rfft(x, n=self.freq_bins, dim=2)
        z = torch.cat([x_fft.real, x_fft.imag], dim=1)
        z = self.ch_lstm_f(self.norm(z))
        z = torch.complex(z[:, : self.channels], z[:, self.channels :])
        z = z * x_fft
        y = torch.fft.irfft(z, n=self.freq_bins, dim=2)
        return y


class _CFB(nn.Module):
    """ICCRN CFB block with gate, local conv and cepstral branch."""

    def __init__(self, in_channels: int, out_channels: int, freq_bins: int):
        super().__init__()
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.norm0 = _LayerNormCF(in_channels, freq_bins)
        self.norm1 = _LayerNormCF(out_channels, freq_bins)
        self.norm2 = _LayerNormCF(out_channels, freq_bins)
        self.ceps = _CepsUnit(out_channels, freq_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.conv_gate(self.norm0(x)))
        xin = self.conv_input(x)
        y = self.conv(self.norm1(g * xin))
        y = y + self.ceps(self.norm2((1.0 - g) * xin))
        return y


class ICCRNFlowBackbone(SpeechFlowBackbone):
    """ICCRN-style spectrogram backbone for flow matching."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_freq_bins: int = 256,
        channels: int = 20,
        num_blocks: int = 5,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_freq_bins = num_freq_bins
        self.channels = channels
        self.causal = bool(causal)

        self.time_in = TimeCondMLP(channels)
        self.time_r = TimeCondMLP(channels)
        self.time_fuse = nn.Linear(channels * 2, channels)

        self.in_ch_lstm = _ChannelLSTMF(in_channels, channels, channels, bidirectional=True)
        self.in_conv = nn.Conv2d(in_channels + channels, channels, kernel_size=1)
        self.enc = nn.ModuleList([_CFB(channels, channels, num_freq_bins) for _ in range(num_blocks)])

        self.norm_mid = _LayerNormCF(channels, num_freq_bins)
        self.ch_lstm_t = _ChannelLSTMT(channels, channels * 2, channels, bidirectional=False, num_layers=2)

        self.dec0 = _CFB(channels, channels, num_freq_bins)
        self.dec_rest = nn.ModuleList([_CFB(channels * 2, channels, num_freq_bins) for _ in range(num_blocks - 1)])

        self.out_ch_lstm = _ChannelLSTMT(channels * 2, channels, channels * 2, bidirectional=False, num_layers=1)
        self.out_conv = nn.Conv2d(channels * 3, out_channels, kernel_size=1)

    def _cond_vec(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        # ODE solvers may pass scalar t/r; expand to batch so TimeCondMLP sees (B,).
        if t.dim() == 0:
            t = t.repeat(1)
        if r is not None and r.dim() == 0:
            r = r.repeat(t.shape[0])
        te = self.time_in(t)
        if r is None:
            return te
        return self.time_fuse(torch.cat([te, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        b, _, f, _ = sample.shape
        if t.dim() == 0:
            t = t.repeat(b)
        if r is not None and r.dim() == 0:
            r = r.repeat(b)
        if f != self.num_freq_bins:
            raise ValueError(
                f"ICCRNFlowBackbone expected num_freq_bins={self.num_freq_bins}, got input F={f}."
            )

        e0_l = self.in_ch_lstm(sample)
        e0 = self.in_conv(torch.cat([e0_l, sample], dim=1))
        e0 = e0 + self._cond_vec(t, r).view(b, self.channels, 1, 1)

        skips = []
        x = e0
        for blk in self.enc:
            x = blk(x)
            skips.append(x)

        x = self.ch_lstm_t(self.norm_mid(x))

        d = self.dec0(skips[-1] * x)
        for i, blk in enumerate(self.dec_rest):
            d = blk(torch.cat([skips[-2 - i], d], dim=1))

        y0 = self.out_ch_lstm(torch.cat([e0, d], dim=1))
        y = self.out_conv(torch.cat([y0, d], dim=1))
        return y
