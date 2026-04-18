"""SGMSE NCSN++ wrapper backbone for speech flow matching.

Adapts SGMSE's complex-STFT score network interface to this project's
real-channel flow interface:

- input ``sample``: ``(B, 4, F, T)`` = ``[x_real, x_imag, cond_real, cond_imag]``
- output: ``(B, 2, F, T)`` = predicted flow field in real/imag channels
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone


class SGMSEBackbone(SpeechFlowBackbone):
    """Wrapper around SGMSE NCSN++ backbones (``ncsnpp`` / ``ncsnpp_v2`` / ``ncsnpp_48k``)."""

    supports_causal = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone_variant: str = "ncsnpp_v2",
        sigma_min: float = 0.03,
        sigma_max: float = 1.0,
        causal: bool = False,
        **sgmse_kwargs,
    ):
        super().__init__()
        if in_channels != 4:
            raise ValueError(
                f"SGMSEBackbone expects in_channels=4 ([x_r, x_i, cond_r, cond_i]); got {in_channels}"
            )
        if out_channels != 2:
            raise ValueError(f"SGMSEBackbone expects out_channels=2 (real/imag); got {out_channels}")
        if sigma_min <= 0.0 or sigma_max <= 0.0:
            raise ValueError("sigma_min and sigma_max must be > 0 for log-time mapping.")
        if sigma_max < sigma_min:
            raise ValueError(f"sigma_max must be >= sigma_min (got {sigma_max} < {sigma_min}).")
        if causal:
            raise ValueError("SGMSEBackbone is non-causal; set causal=false.")

        try:
            from sgmse.backbones.ncsnpp import NCSNpp
            from sgmse.backbones.ncsnpp_48k import NCSNpp_48k
            from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2
        except Exception as e:
            raise ImportError(
                "SGMSEBackbone requires the `sgmse` package. Install it in the active environment, "
                "e.g. `python -m pip install /path/to/sgmse`."
            ) from e

        variants = {
            "ncsnpp": NCSNpp,
            "ncsnpp_v2": NCSNpp_v2,
            "ncsnpp_48k": NCSNpp_48k,
        }
        if backbone_variant not in variants:
            raise ValueError(f"backbone_variant must be one of {list(variants)}, got {backbone_variant}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = False
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.backbone_variant = backbone_variant
        self.net = variants[backbone_variant](**sgmse_kwargs)

    def _time_to_sigma(self, t: torch.Tensor, r: Optional[torch.Tensor]) -> torch.Tensor:
        # Keep compatibility with flows that optionally pass r by using their midpoint.
        tau = t if r is None else 0.5 * (t + r)
        tau = tau.float().reshape(-1).clamp(0.0, 1.0)
        log_ratio = torch.log(
            torch.tensor(self.sigma_max / self.sigma_min, device=tau.device, dtype=tau.dtype)
        )
        return self.sigma_min * torch.exp(tau * log_ratio)

    def forward(self, sample: torch.Tensor, t: torch.Tensor, r: Optional[torch.Tensor] = None, cond=None):
        if sample.ndim != 4 or sample.shape[1] != 4:
            raise ValueError(
                f"SGMSEBackbone expects sample shape (B, 4, F, T), got {tuple(sample.shape)}"
            )

        # SGMSE backbones contain multiple stride-2 stages and can hit shape mismatches
        # on non-multiple-of-8 time lengths (e.g. 2 s crops -> T=251). Pad to a
        # multiple of 8 and crop back to preserve caller-visible shape.
        t_len = int(sample.shape[-1])
        pad_t = (-t_len) % 8
        if pad_t:
            sample = F.pad(sample, (0, pad_t, 0, 0))

        x_complex = torch.complex(sample[:, 0], sample[:, 1])[:, None, :, :]
        cond_complex = torch.complex(sample[:, 2], sample[:, 3])[:, None, :, :]
        sigma = self._time_to_sigma(t, r)

        if self.backbone_variant == "ncsnpp_v2":
            out_complex = self.net(x_complex, cond_complex, sigma)
        else:
            sgmse_in = torch.cat([x_complex, cond_complex], dim=1)
            out_complex = self.net(sgmse_in, sigma)
        if out_complex.ndim != 4 or out_complex.shape[1] != 1:
            raise RuntimeError(
                "Unexpected SGMSE output shape; expected (B, 1, F, T) complex, "
                f"got {tuple(out_complex.shape)}."
            )
        out_complex = out_complex[:, 0]
        out = torch.stack([out_complex.real, out_complex.imag], dim=1).to(sample.dtype)
        if pad_t:
            out = out[..., :t_len]
        return out
