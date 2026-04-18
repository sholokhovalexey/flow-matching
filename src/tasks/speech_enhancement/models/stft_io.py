"""Shared STFT waveform ↔ 2-channel (Re/Im) spec used by flow and baseline SE modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.stft_compress import (
    compress_complex_stft_magnitude,
    decompress_complex_stft_magnitude,
)


def wav_to_stft_spec(
    wav: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    n_freq_bins: int,
    mag_compression_gamma: float,
) -> torch.Tensor:
    """``wav`` (B, 1, T) → (B, 2, F, T_frames) with optional magnitude compression."""
    w = torch.hann_window(n_fft, device=wav.device)
    z = torch.stft(
        wav.squeeze(1),
        n_fft,
        hop_length,
        window=w,
        return_complex=True,
        center=True,
    )
    z = compress_complex_stft_magnitude(z, float(mag_compression_gamma))
    spec = torch.stack([z.real, z.imag], dim=1)
    if spec.shape[2] > n_freq_bins:
        spec = spec[:, :, :n_freq_bins, :]
    return spec


def stft_spec_to_wav(
    spec: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    length: int,
    mag_compression_gamma: float,
) -> torch.Tensor:
    """Inverse of :func:`wav_to_stft_spec` → ``(B, 1, length)``."""
    n_bins = n_fft // 2 + 1
    if spec.shape[2] < n_bins:
        spec = F.pad(spec, (0, 0, 0, n_bins - spec.shape[2]))
    w = torch.hann_window(n_fft, device=spec.device)
    c = torch.complex(spec[:, 0], spec[:, 1])
    c = decompress_complex_stft_magnitude(c, float(mag_compression_gamma))
    wav = torch.istft(
        c,
        n_fft,
        hop_length,
        window=w,
        center=True,
        length=length,
    )
    return wav.unsqueeze(1)
