"""Speech enhancement flow backbones (unified ``forward(sample, t, r, cond)``)."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SpeechFlowBackbone
    from .causal_stft_net import CausalSTFTStackBackbone
    from .fastenhancer_rnnformer import FastEnhancerRNNFormerBackbone
    from .gtcrn_backbone import GTCRNFlowBackbone
    from .iccrn_backbone import ICCRNFlowBackbone
    from .latent_dit import LatentDiTBackbone
    from .lisenet_backbone import LiSenNetFlowBackbone
    from .s4nd_unet import S4NDUNetBackbone
    from .sgmse_causal import CausalSGMSEAttentionBackbone, CausalSGMSEDilatedBackbone
    from .sgmse_ncsnpp import SGMSEBackbone
    from .spatialnet import SpatialNetBackbone
    from .speechbrain_conformer import SpeechBrainConformerBackbone
    from .streaming import (
        CausalSTFTStackStateful,
        S4NDUNetStateful,
        SpatialNetStateful,
        StatefulBackboneAdapter,
        StreamState,
        TFConformerStateful,
    )
    from .tf_conformer import TFConformerBackbone

_SYMBOL_TO_MODULE = {
    "SpeechFlowBackbone": ".base",
    "LatentDiTBackbone": ".latent_dit",
    "TFConformerBackbone": ".tf_conformer",
    "SpatialNetBackbone": ".spatialnet",
    "SpeechBrainConformerBackbone": ".speechbrain_conformer",
    "S4NDUNetBackbone": ".s4nd_unet",
    "StatefulBackboneAdapter": ".streaming",
    "TFConformerStateful": ".streaming",
    "SpatialNetStateful": ".streaming",
    "CausalSTFTStackStateful": ".streaming",
    "S4NDUNetStateful": ".streaming",
    "StreamState": ".streaming",
    "CausalSTFTStackBackbone": ".causal_stft_net",
    "FastEnhancerRNNFormerBackbone": ".fastenhancer_rnnformer",
    "LiSenNetFlowBackbone": ".lisenet_backbone",
    "GTCRNFlowBackbone": ".gtcrn_backbone",
    "ICCRNFlowBackbone": ".iccrn_backbone",
    "SGMSEBackbone": ".sgmse_ncsnpp",
    "CausalSGMSEDilatedBackbone": ".sgmse_causal",
    "CausalSGMSEAttentionBackbone": ".sgmse_causal",
}

__all__ = [
    "SpeechFlowBackbone",
    "LatentDiTBackbone",
    "TFConformerBackbone",
    "SpatialNetBackbone",
    "SpeechBrainConformerBackbone",
    "S4NDUNetBackbone",
    "StatefulBackboneAdapter",
    "TFConformerStateful",
    "SpatialNetStateful",
    "CausalSTFTStackStateful",
    "S4NDUNetStateful",
    "StreamState",
    "CausalSTFTStackBackbone",
    "FastEnhancerRNNFormerBackbone",
    "LiSenNetFlowBackbone",
    "GTCRNFlowBackbone",
    "ICCRNFlowBackbone",
    "SGMSEBackbone",
    "CausalSGMSEDilatedBackbone",
    "CausalSGMSEAttentionBackbone",
]


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
