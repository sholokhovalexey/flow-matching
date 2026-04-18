"""Scale-invariant SDR (Le Roux et al., 2019)."""

from __future__ import annotations

import torch


def _si_sdr_fallback(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable SI-SDR fallback when torchmetrics audio functional is unavailable."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target = dot * target / (target.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps))
    e_noise = estimate - s_target
    ratio = (s_target.pow(2).sum(dim=-1) + eps) / (e_noise.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio.clamp(min=eps))


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB, higher is better.

    Prefers ``torchmetrics.functional.audio.scale_invariant_signal_distortion_ratio``.
    Input supports ``(B, T)`` or ``(B, 1, T)``.
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    try:
        from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

        return scale_invariant_signal_distortion_ratio(estimate, target)
    except Exception:
        return _si_sdr_fallback(estimate, target, eps=eps)
