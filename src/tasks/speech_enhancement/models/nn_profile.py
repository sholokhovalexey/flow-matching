"""FLOP and parameter-count helpers for speech backbones (PyTorch ``FlopCounterMode``).

Centralizes STFT grid assumptions used for “~1 s of audio” profiling so new backbones can
reuse the same constants. For production metrics, prefer measuring your exact ``n_fft`` / hop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

# --- STFT reference grid: 1 s @ 16 kHz, 20 ms window, 50 % hop (10 ms) ---

STFT_SAMPLE_RATE_16K = 16_000
STFT_WIN_MS_20 = 20


def stft_win_samples_16k_20ms() -> int:
    """Analysis window length in samples (``20 ms`` at ``16 kHz`` → 320)."""
    return int(STFT_SAMPLE_RATE_16K * STFT_WIN_MS_20 / 1000)


def stft_hop_samples_half_win() -> int:
    """Hop in samples (half the window → ``10 ms`` at ``16 kHz``, 50 % overlap)."""
    return stft_win_samples_16k_20ms() // 2


def stft_time_frames_one_second_16k_20ms() -> int:
    """STFT time frames for **1 s** of audio: ``(N - win) // hop + 1`` with ``N = 16_000``."""
    sr = STFT_SAMPLE_RATE_16K
    win = stft_win_samples_16k_20ms()
    hop = stft_hop_samples_half_win()
    return (sr - win) // hop + 1


def default_rfft_num_freq_bins(n_fft: int = 512) -> int:
    """Bins for an ``n_fft``-point real FFT (``n_fft // 2 + 1``)."""
    return n_fft // 2 + 1


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Scalar parameter count (``sum(numel)``)."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def estimate_flops_callable(forward_fn: Callable[[], None]) -> int:
    """Run ``forward_fn()`` under :class:`torch.utils.flop_counter.FlopCounterMode`."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:
        raise ImportError("``torch.utils.flop_counter`` requires PyTorch >= 2.1.") from e

    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            forward_fn()
    total = fcm.get_total_flops()
    if total is None:
        raise RuntimeError("FlopCounterMode returned no total; see PyTorch version / operator coverage.")
    return int(total)


def estimate_backbone_forward_flops(
    model: nn.Module,
    *,
    in_channels: int,
    n_freq_bins: int,
    n_time_frames: int,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> int:
    """One ``model(sample, t, None)`` forward for a speech backbone on an STFT-shaped tensor."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:
        raise ImportError("FLOP counting needs ``torch.utils.flop_counter`` (PyTorch >= 2.1).") from e

    device = torch.device(device)
    model = model.to(device).eval()
    sample = torch.randn(
        batch_size, in_channels, n_freq_bins, n_time_frames, device=device, dtype=torch.float32
    )
    t = torch.zeros(batch_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            model(sample, t, None)
    total = fcm.get_total_flops()
    if total is None:
        raise RuntimeError("FlopCounterMode returned no total for this backbone.")
    return int(total)


def print_backbone_flop_line(
    backbone_name: str,
    flops: int,
    *,
    n_freq_bins: int,
    n_time_frames: int,
    n_parameters: Optional[int] = None,
) -> None:
    """CLI one-liner for backbone ``__main__`` blocks."""
    line = (
        f"{backbone_name}: {flops:,} FLOPs ({flops / 1e9:.3f} GFLOPs) - "
        f"one forward, STFT F×T = {n_freq_bins}×{n_time_frames} "
        f"(~1 s @ 16 kHz, 20 ms win, hop = win/2)"
    )
    if n_parameters is not None:
        line += f" | {n_parameters:,} parameters ({n_parameters / 1e6:.3f} M params)"
    print(line)
