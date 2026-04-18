"""Conditioning wrapper: concat flow state with noisy STFT / latent (FlowSE / MeanFlowSE / DiTSE style)."""

import torch
import torch.nn as nn


class SpeechCondWrapper(nn.Module):
    """Concatenate ``x`` (flow trajectory) and noisy-condition tensor along the channel axis."""

    def __init__(self, backbone: nn.Module, shape: tuple):
        """
        Args:
            backbone: Maps concatenated input to velocity / average velocity. Called with ``cond=None``.
            shape: Shape of the flow variable ``x`` alone: ``(C, H, W)``.
        """
        super().__init__()
        self.backbone = backbone
        self.shape = tuple(shape)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_null_condition(self, batch_size: int) -> torch.Tensor:
        dt = next(self.parameters()).dtype
        return torch.zeros(batch_size, *self.shape, device=self.device, dtype=dt)

    def forward(self, x: torch.Tensor, t: torch.Tensor, r=None, cond=None):
        if cond is None:
            cond = torch.zeros(x.shape[0], *self.shape, device=x.device, dtype=x.dtype)
        x_in = torch.cat([x, cond], dim=1)
        return self.backbone(x_in, t, r=r, cond=None)
