"""Stateful frame-by-frame adapter for causal speech backbones.

Training uses parallel full-sequence inference; deployment can use this wrapper to
process one STFT frame at a time while maintaining a rolling state buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StreamState:
    """Rolling input buffer for stateful inference."""

    x_hist: torch.Tensor


class StatefulBackboneAdapter(nn.Module):
    """Wrap a causal backbone with frame-by-frame ``(frame, state) -> (frame, state)`` API.

    Notes:
    - This is an exact functional adapter: each step re-runs the original parallel
      backbone on a bounded history buffer and returns the newest frame.
    - It preserves parameter loading by reusing the same backbone module/weights.
    """

    def __init__(self, backbone: nn.Module, max_history: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.max_history = int(getattr(backbone, "stream_history", max_history))

    @property
    def causal(self) -> bool:
        return bool(getattr(self.backbone, "causal", False))

    def init_state(self, batch: int, channels: int, freq: int, device: torch.device, dtype: torch.dtype) -> StreamState:
        return StreamState(x_hist=torch.zeros(batch, channels, freq, 0, device=device, dtype=dtype))

    def forward(
        self,
        x_frame: torch.Tensor,
        state: Optional[StreamState],
        t: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        cond=None,
    ):
        """``x_frame`` is ``(B, C, F, 1)``; returns ``(B, C_out, F, 1)``, new state."""
        if state is None:
            state = self.init_state(
                batch=x_frame.shape[0],
                channels=x_frame.shape[1],
                freq=x_frame.shape[2],
                device=x_frame.device,
                dtype=x_frame.dtype,
            )
        x_hist = torch.cat([state.x_hist, x_frame], dim=-1)
        if x_hist.shape[-1] > self.max_history:
            x_hist = x_hist[..., -self.max_history :]
        y = self.backbone(x_hist, t=t, r=r, cond=cond)
        y_last = y[..., -1:].contiguous()
        return y_last, StreamState(x_hist=x_hist)


class CausalSTFTStackStateful(StatefulBackboneAdapter):
    """Low-latency preset for :class:`CausalSTFTStackBackbone`."""

    def __init__(self, backbone: nn.Module):
        # Small local receptive field in time (conv stack); keep short history.
        super().__init__(backbone, max_history=64)


class S4NDUNetStateful(StatefulBackboneAdapter):
    """Low-latency preset for :class:`S4NDUNetBackbone`."""

    def __init__(self, backbone: nn.Module):
        # Temporal depthwise kernels + U-Net skip path: moderate history.
        super().__init__(backbone, max_history=192)


class TFConformerStateful(StatefulBackboneAdapter):
    """Streaming preset for :class:`TFConformerBackbone`."""

    def __init__(self, backbone: nn.Module):
        # Attention needs long context for parity; keep a larger rolling buffer.
        super().__init__(backbone, max_history=1024)


class SpatialNetStateful(StatefulBackboneAdapter):
    """Streaming preset for :class:`SpatialNetBackbone`."""

    def __init__(self, backbone: nn.Module):
        super().__init__(backbone, max_history=512)

