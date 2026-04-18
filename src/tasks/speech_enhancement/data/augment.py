"""Augmentations with real noise (e.g. MUSAN) and measured RIRs (OpenSLR-style corpora).

Probabilities ``p_musan`` and ``p_rir`` in :class:`AugmentConfig` are independent:
each augmentation is drawn with its own Bernoulli trial per clip (see ``degrade_clean``).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from src.tasks.speech_enhancement.data.audio_io import load_audio


def _list_wavs(root: str) -> List[str]:
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".wav", ".flac")):
                out.append(os.path.join(dirpath, f))
    return out


def coerce_augment_mapping(a: Any) -> Dict[str, Any]:
    """Convert Hydra DictConfig / plain dict to a plain ``dict`` for probability fields."""
    if a is None:
        return {}
    if isinstance(a, dict):
        return dict(a)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(a, resolve=True)
    except Exception:
        return dict(a)


def augment_config_from_dict(a: Mapping[str, Any]) -> AugmentConfig:
    """Build :class:`AugmentConfig` from a Hydra/OmegaConf-style dict (see data YAML ``augment:`` blocks).

    ``p_musan`` and ``p_rir`` are independent. For VoiceBank-only extra MUSAN, ``p_musan_extra`` is an
    alias for ``p_musan`` when ``p_musan`` is omitted.
    """
    snr = a.get("snr_range_db", (5.0, 15.0))
    rir_g = a.get("rir_gain_range_db", (-5.0, 5.0))
    if not isinstance(snr, tuple):
        snr = tuple(snr)
    if not isinstance(rir_g, tuple):
        rir_g = tuple(rir_g)
    pm = a.get("p_musan", None)
    if pm is None:
        pm = a.get("p_musan_extra", 0.0)
    return AugmentConfig(
        musan_root=a.get("musan_root"),
        rir_root=a.get("rir_root"),
        p_musan=float(pm),
        p_rir=float(a.get("p_rir", 0.0)),
        snr_range_db=snr,
        rir_gain_range_db=rir_g,
        max_rir_samples=int(a.get("max_rir_samples", 8000)),
    )


def load_audio_random_segment(
    path: str,
    length: int,
    sample_rate: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    """Load mono segment of exactly ``length`` samples; pad if file shorter."""
    wav, sr = load_audio(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[1] <= length:
        return F.pad(wav, (0, length - wav.shape[1])).to(device)
    start = rng.randint(0, wav.shape[1] - length)
    return wav[:, start : start + length].to(device)


def fft_convolve_full(x: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Full convolution (x * rir) for mono (1, T) tensors."""
    x = x.squeeze(0)
    h = rir.squeeze(0)
    n = x.numel() + h.numel() - 1
    X = torch.fft.rfft(x, n=n)
    H = torch.fft.rfft(h, n=n)
    y = torch.fft.irfft(X * H, n=n)
    return y[: x.numel()].unsqueeze(0)


def snr_to_noise_scale(clean: torch.Tensor, noise: torch.Tensor, snr_db: float, eps: float = 1e-12) -> float:
    p_sig = clean.pow(2).mean().clamp(min=eps)
    p_noise = noise.pow(2).mean().clamp(min=eps)
    snr = 10.0 ** (snr_db / 10.0)
    return float(torch.sqrt(p_sig / (snr * p_noise)))


@dataclass
class AugmentConfig:
    musan_root: Optional[str] = None
    rir_root: Optional[str] = None
    p_musan: float = 0.0
    p_rir: float = 0.0
    snr_range_db: Tuple[float, float] = (5.0, 15.0)
    rir_gain_range_db: Tuple[float, float] = (-5.0, 5.0)
    max_rir_samples: int = 8000


class MusanRirAugment:
    """Caches file lists under ``musan_root`` / ``rir_root``; applies augmentations per configured probabilities."""

    def __init__(self, cfg: AugmentConfig, sample_rate: int, seed: int = 0):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)
        self._musan_files = _list_wavs(cfg.musan_root) if cfg.musan_root else []
        self._rir_files = _list_wavs(cfg.rir_root) if cfg.rir_root else []

    @property
    def has_musan(self) -> bool:
        return len(self._musan_files) > 0

    @property
    def has_rir(self) -> bool:
        return len(self._rir_files) > 0

    def reverb_fixed_len(self, dry: torch.Tensor) -> torch.Tensor:
        """Apply a random RIR; output length matches ``dry`` (``dry``: (1, T)). No random skip - caller gates with ``p_rir``."""
        if not self.has_rir:
            return dry
        path = self._rir_files[self.rng.randint(0, len(self._rir_files) - 1)]
        rir, sr = load_audio(path)
        if rir.shape[0] > 1:
            rir = rir.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            rir = torchaudio.functional.resample(rir, sr, self.sample_rate)
        if rir.shape[1] > self.cfg.max_rir_samples:
            rir = rir[:, : self.cfg.max_rir_samples]
        rir = rir.to(dry.device)
        db = self.rng.uniform(self.cfg.rir_gain_range_db[0], self.cfg.rir_gain_range_db[1])
        g = 10.0 ** (db / 20.0)
        full = fft_convolve_full(dry, g * rir)
        T = dry.shape[1]
        if full.shape[1] >= T:
            return full[:, :T]
        return F.pad(full, (0, T - full.shape[1]))

    def add_musan_noise(self, speech: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
        """speech: (1, T). Returns noisy speech."""
        if not self.has_musan:
            return speech
        if snr_db is None:
            snr_db = self.rng.uniform(self.cfg.snr_range_db[0], self.cfg.snr_range_db[1])
        path = self._musan_files[self.rng.randint(0, len(self._musan_files) - 1)]
        n = load_audio_random_segment(
            path, speech.shape[1], self.sample_rate, self.rng, speech.device
        )
        scale = snr_to_noise_scale(speech, n, snr_db)
        return speech + scale * n

    def degrade_clean(
        self,
        clean: torch.Tensor,
        apply_rir: bool = True,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        """Build degraded clip from clean speech (1, T). RIR and MUSAN use **independent** Bernoulli draws ``p_rir`` and ``p_musan``."""
        x = clean
        if apply_rir and self.has_rir and self.rng.random() < self.cfg.p_rir:
            x = self.reverb_fixed_len(x)
        if apply_noise and self.has_musan and self.rng.random() < self.cfg.p_musan:
            x = self.add_musan_noise(x)
        return x

    def extra_degrade_noisy(self, noisy: torch.Tensor) -> torch.Tensor:
        """Optional extra corruption on an already-noisy clip (e.g. VoiceBank). Same independent ``p_rir``, ``p_musan``."""
        x = noisy
        if self.has_rir and self.rng.random() < self.cfg.p_rir:
            x = self.reverb_fixed_len(x)
        if self.has_musan and self.rng.random() < self.cfg.p_musan:
            x = self.add_musan_noise(x)
        return x
