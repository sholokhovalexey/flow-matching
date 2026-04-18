"""Complex STFT magnitude compression (SGMSE / score-based speech enhancement style).

The flow sees a 2-channel tensor ``[Re(z_c), Im(z_c)]`` where

    z_c = |STFT(x)|^gamma * exp(j * arg(STFT(x)))

so phase is unchanged and only magnitude is warped. ``gamma == 1`` is the identity (raw STFT).
Typical values in the literature are in ``(0, 1]``, e.g. ``0.3``-``0.5``, to reduce dynamic range.
"""

from __future__ import annotations

import torch


def compress_complex_stft_magnitude(z: torch.Tensor, gamma: float, eps: float = 1e-10) -> torch.Tensor:
    """Apply magnitude power law: ``z * |z|^(gamma - 1)`` (equivalently ``|z|^gamma * e^{j arg z}``)."""
    if gamma == 1.0:
        return z
    mag = z.abs().clamp_min(eps)
    return z * mag.pow(gamma - 1.0)


def decompress_complex_stft_magnitude(z_c: torch.Tensor, gamma: float, eps: float = 1e-10) -> torch.Tensor:
    """Inverse of :func:`compress_complex_stft_magnitude`: recover uncompressed complex STFT."""
    if gamma == 1.0:
        return z_c
    mag_c = z_c.abs().clamp_min(eps)
    unit = z_c / mag_c
    return unit * mag_c.pow(1.0 / gamma)
