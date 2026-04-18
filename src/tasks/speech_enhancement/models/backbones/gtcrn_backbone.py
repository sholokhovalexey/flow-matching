"""H-GTCRN core (GTCRN + ERB + dual-path GRU) adapted for flow matching.

Reference: https://github.com/Max1Wz/H-GTCRN

Vendored network blocks follow the upstream implementation (MIT); the IVA/WPE front-end
is omitted - we map flow tensors onto the GTCRN feature path and predict a 2-channel field.
``einops`` is not required (``rearrange`` replaced with reshape).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone, TimeCondMLP


class ERB(nn.Module):
    def __init__(self, erb_subband_1: int, erb_subband_2: int, nfft: int = 512, high_lim: int = 8000, fs: int = 16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz: float) -> float:
        return 24.7 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f: float) -> float:
        return (10 ** (erb_f / 24.7) - 1) / 0.00437

    def erb_filter_banks(
        self, erb_subband_1: int, erb_subband_2: int, nfft: int = 512, high_lim: int = 8000, fs: int = 16000
    ):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
        erb_filters[0, bins[0] : bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i] : bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) / (
                bins[i + 1] - bins[i] + 1e-12
            )
            erb_filters[i + 1, bins[i + 1] : bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) / (
                bins[i + 2] - bins[i + 1] + 1e-12
            )
        erb_filters[-1, bins[-2] : bins[-1] + 1] = 1 - erb_filters[-2, bins[-2] : bins[-1] + 1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x[..., : self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1 :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb: torch.Tensor) -> torch.Tensor:
        x_erb_low = x_erb[..., : self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1 :])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    def __init__(self, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zt = torch.mean(x.pow(2), dim=-1)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)[..., None]
        return x * at


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
    ):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        self.sfe = SFE(kernel_size=3, stride=1)
        self.point_conv1 = ConvBlock(in_channels // 2 * 3, hidden_channels, (1, 1))
        self.depth_conv = ConvBlock(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.point_conv2 = ConvBlock(hidden_channels, in_channels // 2, (1, 1))
        self.point_conv2.act = nn.Identity()
        self.tra = TRA(in_channels // 2)

    def shuffle(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, _, t, f = x1.shape
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(b, -1, t, f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.sfe(x1)
        h1 = self.point_conv1(x1)
        h1 = F.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_conv(h1)
        h1 = self.point_conv2(h1)
        h1 = self.tra(h1)
        return self.shuffle(h1, x2)


class GRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        if h is None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    def __init__(self, input_size: int, width: int, hidden_size: int):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)
        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)
        return inter_out.permute(0, 3, 1, 2)


class _GTCRNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(6 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
                GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
                GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
                GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
            ]
        )

    def forward(self, x: torch.Tensor):
        en_outs = []
        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class _GTCRNDecoder(nn.Module):
    def __init__(self, out_channels: int = 2):
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
                GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
                GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
                ConvBlock(16, out_channels, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True),
            ]
        )
        self.de_convs[-1].act = nn.Tanh()

    def forward(self, x: torch.Tensor, en_outs):
        n = len(self.de_convs)
        for i in range(n):
            x = self.de_convs[i](x + en_outs[n - 1 - i])
        return x


class GTCRNFlowBackbone(SpeechFlowBackbone):
    """GTCRN feature path with causal temporal padding in :class:`GTConvBlock`."""

    supports_causal = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_fft: int = 512,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.n_fft = n_fft
        self.n_rfft = n_fft // 2 + 1
        # Upstream fusion uses 6 raw channels × SFE(3) → 18 encoder input channels.
        self.stem = nn.Conv2d(in_channels, 6, kernel_size=1)
        self.erb = ERB(65, 64, nfft=n_fft)
        self.sfe = SFE(3, 1)
        self.encoder = _GTCRNEncoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = _GTCRNDecoder(out_channels=out_channels)
        self.time_in = TimeCondMLP(16)
        self.time_r = TimeCondMLP(16)
        self.time_fuse = nn.Linear(32, 16)

    def _temb(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        a = self.time_in(t)
        if r is None:
            return a
        return self.time_fuse(torch.cat([a, self.time_r(r)], dim=-1))

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        # Flow tensors may use ``n_freq_bins < n_fft//2+1`` (cropped STFT). Remember F for output alignment.
        f_in = int(sample.shape[2])
        # (B, C, F, T) → (B, C, T, F) as in H-GTCRN Conv2d layout (time = height).
        x = sample.permute(0, 1, 3, 2)
        if x.shape[-1] != self.n_rfft:
            x = F.interpolate(x, size=(x.shape[2], self.n_rfft), mode="bilinear", align_corners=False)
        x = self.stem(x)
        x = self.erb.bm(x)
        x = self.sfe(x)
        x, en_outs = self.encoder(x)
        c = self._temb(t, r)
        x = x + c[:, :, None, None]
        x = self.dpgrnn1(x)
        x = self.dpgrnn2(x)
        x = self.decoder(x, en_outs)
        x = self.erb.bs(x)
        x = x.permute(0, 1, 3, 2)
        # Match flow MSE target ``v = x1 - x0`` (same F as input); internal path uses full ``n_rfft`` bins.
        if x.shape[2] > f_in:
            x = x[:, :, :f_in, :]
        return x
