"""Collate for ``(clean, noisy)`` waveform batches with optional per-batch time alignment."""

from __future__ import annotations

from functools import partial
from typing import List, Tuple

import torch
from torch.utils.data._utils.collate import default_collate

from src.tasks.speech_enhancement.data.audio_crop import normalize_batch_time_align


def speech_waveform_pair_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_time_align: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack ``(B, 1, T)`` clean/noisy pairs.

    * ``pad_to_segment_length``: every sample should already have the same ``T`` (dataset pads short clips).
    * ``truncate_to_min``: samples may have different ``T``; truncate all to ``min(T)`` in the batch.
    """
    align = normalize_batch_time_align(batch_time_align)
    if not batch:
        raise ValueError("empty batch")
    if align == "pad_to_segment_length":
        return default_collate(batch)
    cleans, noisies = zip(*batch)
    lengths = [int(c.shape[-1]) for c in cleans]
    m = min(lengths)
    if m < 1:
        raise ValueError(f"invalid waveform lengths in batch: {lengths}")
    c_stacked = torch.stack([x[..., :m] for x in cleans], dim=0)
    n_stacked = torch.stack([x[..., :m] for x in noisies], dim=0)
    return c_stacked, n_stacked


def make_speech_waveform_pair_collate_fn(batch_time_align: str):
    """Pickle-safe collate (works with ``num_workers`` > 0)."""
    align = normalize_batch_time_align(batch_time_align)
    return partial(speech_waveform_pair_collate_fn, batch_time_align=align)
