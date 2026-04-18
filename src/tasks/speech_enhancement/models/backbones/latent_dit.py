"""DiT on a fixed spatial grid for codec latents (DiTSE-style: transformer in latent space)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones.dit import DiT
from src.tasks.speech_enhancement.models.backbones.base import SpeechFlowBackbone


class LatentDiTBackbone(SpeechFlowBackbone):
    """Resize latent maps to ``spatial_size`` for DiT, then map velocities back to the input grid.

    CFM builds ``x_t`` and ``v_tgt = x1 - x0`` at codec resolution (e.g. ``H=1``, ``W=T``). The DiT
    runs on a fixed square grid ``spatial_size``; outputs are interpolated back so ``v_pred`` matches
    ``v_tgt`` in :class:`~src.flows.base.CFM`.

    :class:`~src.tasks.speech_enhancement.models.wrappers.SpeechCondWrapper` feeds **concat** ``[x, cond]``,
    so DiT ``in_channels`` is ``2 * C_x``, while DiT ``out_channels`` defaults to the same as ``in_channels``.
    Set ``velocity_out_channels`` to **``C_x``** (same as ``net.shape[0]``) so the predicted velocity
    matches the flow state dimension.

    Pass ``causal: true`` into the nested ``dit`` config to use causal self-attention
    (sequence order follows timm patch rasterisation - prefer thin latent maps that
    advance primarily along time).
    """

    supports_causal = True

    def __init__(
        self,
        dit: DiT,
        spatial_size: tuple = (32, 32),
        velocity_out_channels: int | None = None,
    ):
        super().__init__()
        self.dit = dit
        # Hydra may pass ``ListConfig``; ``F.interpolate`` requires ``tuple[int, int]``.
        self.spatial_size = tuple(int(x) for x in spatial_size)
        self.in_channels = dit.in_channels
        voc = int(dit.out_channels if velocity_out_channels is None else velocity_out_channels)
        if voc != dit.out_channels:
            self.velocity_head = nn.Conv2d(dit.out_channels, voc, kernel_size=1)
        else:
            self.velocity_head = None
        self.out_channels = voc
        self.causal = getattr(dit, "causal", False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, r=None, cond=None):
        # x: (B, C, H, W); typically H=1 and W=L for encoder features
        h_in, w_in = int(x.shape[2]), int(x.shape[3])
        x = F.interpolate(x, size=self.spatial_size, mode="bilinear", align_corners=False)
        y = self.dit(x, t, r=r, cond=None)
        if self.velocity_head is not None:
            y = self.velocity_head(y)
        if y.shape[2] != h_in or y.shape[3] != w_in:
            y = F.interpolate(y, size=(h_in, w_in), mode="bilinear", align_corners=False)
        return y
