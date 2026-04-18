"""Crop length and STFT grid sizes (must match ``torch.stft`` / ``torch.istft`` used in training).

We use ``center=True`` so Hann-window STFT/iSTFT round-trips under PyTorch 2.10+ (NOLA checks fail for
``center=False`` with typical Hann + hop settings).
"""

from __future__ import annotations

import random
from typing import Tuple

import torch.nn.functional as F


def resolve_segment_samples(
    sample_rate: int,
    segment_length: int,
    segment_duration_sec: float | None = None,
) -> int:
    """Return crop length in samples. If ``segment_duration_sec`` is set, it overrides ``segment_length``."""
    if segment_duration_sec is not None:
        return max(1, int(round(float(segment_duration_sec) * sample_rate)))
    return int(segment_length)


def stft_num_time_frames(
    num_samples: int,
    n_fft: int,
    hop_length: int,
) -> int:
    """Number of STFT frames for ``torch.stft(..., center=True)`` (matches PyTorch 2.10+)."""
    if num_samples < 1:
        return 0
    return 1 + num_samples // hop_length


def stft_unet_spatial_size(
    segment_samples: int,
    n_fft: int,
    hop_length: int,
    n_freq_bins: int,
) -> Tuple[int, int]:
    """``(height, width)`` for a real/imag spectrogram tensor ``(B, 2, F, T)`` after freq crop.

    ``F = min(n_freq_bins, n_fft // 2 + 1)``, ``T = stft_num_time_frames(...)``.
    """
    n_rfft = n_fft // 2 + 1
    f_use = min(int(n_freq_bins), n_rfft)
    t_frames = stft_num_time_frames(segment_samples, n_fft, hop_length)
    return f_use, t_frames


def normalize_segment_crop_mode(mode: str) -> str:
    """``random``: uniform start in ``[0, T - L]``; ``start``: first ``L`` samples (or full clip if shorter)."""
    m = str(mode).lower().strip()
    if m in ("random", "start"):
        return m
    raise ValueError(f"segment_crop_mode must be 'random' or 'start', got {mode!r}")


def normalize_batch_time_align(align: str) -> str:
    """How to align time lengths across a batch (see :func:`speech_waveform_pair_collate_fn`)."""
    a = str(align).lower().strip()
    if a in ("pad_to_segment_length", "truncate_to_min"):
        return a
    raise ValueError(
        "batch_time_align must be 'pad_to_segment_length' or 'truncate_to_min', "
        f"got {align!r}"
    )


def crop_mono_waveform_to_segment(
    wav: torch.Tensor,
    segment_length: int,
    *,
    crop_mode: str,
    batch_time_align: str,
    rng: random.Random,
) -> torch.Tensor:
    """Crop or pad mono ``wav`` ``(1, T)`` to a segment for one dataset item.

    * If ``T >= segment_length``: take a ``segment_length`` window (random or from start).
    * If ``T < segment_length``: either right-pad to ``segment_length`` (pad mode) or return ``(1, T)`` unchanged
      (truncate mode; collate will align batch time to ``min`` lengths).
    """
    crop_mode = normalize_segment_crop_mode(crop_mode)
    batch_time_align = normalize_batch_time_align(batch_time_align)
    t = int(wav.shape[1])
    L = int(segment_length)
    if t < L:
        if batch_time_align == "pad_to_segment_length":
            return F.pad(wav, (0, L - t))
        return wav
    if crop_mode == "start":
        start = 0
    else:
        start = rng.randint(0, t - L)
    return wav[:, start : start + L]


def crop_clean_noisy_waveform_pair(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    segment_length: int,
    *,
    crop_mode: str,
    batch_time_align: str,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align parallel clean/noisy ``(1, T)`` to ``min(Tc, Tn)``, then same rules as :func:`crop_mono_waveform_to_segment`."""
    crop_mode = normalize_segment_crop_mode(crop_mode)
    batch_time_align = normalize_batch_time_align(batch_time_align)
    t = min(int(clean.shape[1]), int(noisy.shape[1]))
    clean = clean[:, :t]
    noisy = noisy[:, :t]
    L = int(segment_length)
    if t < L:
        if batch_time_align == "pad_to_segment_length":
            clean = F.pad(clean, (0, L - t))
            noisy = F.pad(noisy, (0, L - t))
        return clean, noisy
    if crop_mode == "start":
        start = 0
    else:
        start = rng.randint(0, t - L)
    return clean[:, start : start + L], noisy[:, start : start + L]


def format_stft_shape_mismatch_message(
    expected_hw: Tuple[int, int],
    net_shape: tuple,
) -> str:
    f_e, t_e = expected_hw
    return (
        f"STFT spatial size mismatch: data crop implies spectrogram shape (2, {f_e}, {t_e}) "
        f"but ``net.shape`` is {tuple(net_shape)}. Update the model config "
        f"``net.shape: [2, {f_e}, {t_e}]`` and backbone ``sample_size: [{f_e}, {t_e}]`` "
        f"to match ``data.segment_length`` / ``data.segment_duration_sec`` and model "
        f"``n_fft``, ``hop_length``, ``n_freq_bins``."
    )
