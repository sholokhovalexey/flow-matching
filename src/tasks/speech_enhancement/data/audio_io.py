"""Waveform file I/O via ``soundfile`` (libsndfile).

Avoids ``torchaudio.load`` / ``torchaudio.save`` on recent torchaudio builds that route file
decoding through **torchcodec** (FFmpeg / libtorchcodec), which is brittle on Windows.

Training-time resampling and STFT still use ``torchaudio.functional`` / transforms - those do not
use the same loading path.

Install: ``pip install soundfile`` (wheels usually bundle libsndfile on Windows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

PathLike = Union[str, Path]

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "soundfile is required for audio file I/O. Install: pip install soundfile\n"
        "See https://pypi.org/project/soundfile/"
    ) from e


class AudioMeta:
    """Torchaudio-compatible metadata subset for scripts that used ``torchaudio.info``."""

    __slots__ = ("sample_rate", "num_frames", "num_channels")

    def __init__(self, sample_rate: int, num_frames: int, num_channels: int) -> None:
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels


def audio_info(path: PathLike) -> AudioMeta:
    info = sf.info(str(path))
    return AudioMeta(int(info.samplerate), int(info.frames), int(info.channels))


def load_audio(path: PathLike) -> tuple[torch.Tensor, int]:
    """Load a sound file as float32 tensor ``(num_channels, num_samples)``.

    Supported formats depend on libsndfile (WAV, FLAC, OGG, …).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    sr = int(sr)
    if data.size == 0:
        return torch.zeros(1, 0, dtype=torch.float32), sr
    # (frames, channels) -> (channels, frames)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return wav, sr


def save_audio(
    path: PathLike,
    wav: torch.Tensor,
    sample_rate: int,
    *,
    subtype: str = "PCM_16",
) -> None:
    """Save ``wav`` as ``(C, T)`` or ``(T,)`` float tensor. Clips to [-1, 1] before PCM writes."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = wav.detach().cpu()
    if x.dim() == 1:
        arr = x.numpy()
    elif x.dim() == 2:
        arr = x.transpose(0, 1).numpy()
    else:
        raise ValueError(f"wav must be 1D or 2D, got shape {tuple(x.shape)}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if subtype.upper().startswith("PCM"):
        arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(out), arr, sample_rate, subtype=subtype)
