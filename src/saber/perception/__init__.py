"""Perception front end.

Ref: Sec. 2.2 -- the encoder ``E_theta`` maps a frame set to instance masks
``M_t``, trajectories over epochs and one morphology embedding ``z_t`` per
instance. Two pretrained backbones are adapted jointly through low-rank
adapters of rank ``r``: the vision tower of a vision-language-action model, and
a morphology foundation model trained on histopathology texture. Adaptation is
confined to the adapters, so both backbones stay static.

Bright-field and phase-contrast frames are the inference modalities; fluorescence
is an auxiliary signal used during adaptation and in the modality ablation only
(Sec. 2.2).
"""

from saber.perception.adapters import LoRAConfig, LoRALinear, count_adapter_parameters
from saber.perception.backbones import (
    MorphologyBackbone,
    VisionTowerBackbone,
    build_backbones,
)
from saber.perception.embedder import MorphologyEmbedder
from saber.perception.head import (
    INFERENCE_CHANNELS,
    FrameBundle,
    PerceptionFrontend,
    PerceptionOutput,
)
from saber.perception.segmenter import InstanceSegmenter, connected_instances
from saber.perception.tracker import Trajectory, associate

__all__ = [
    "INFERENCE_CHANNELS",
    "FrameBundle",
    "InstanceSegmenter",
    "LoRAConfig",
    "LoRALinear",
    "MorphologyBackbone",
    "MorphologyEmbedder",
    "PerceptionFrontend",
    "PerceptionOutput",
    "Trajectory",
    "VisionTowerBackbone",
    "associate",
    "build_backbones",
    "connected_instances",
    "count_adapter_parameters",
]
