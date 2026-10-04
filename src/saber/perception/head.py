"""The perception front end ``E_theta``.

Ref: Sec. 2.2, and Algorithm 1 line 3 -- ``(M_t, z_t) <- E_theta(frames at epoch t)``.

The front end is presented as "a mechanism and not a mechanism": its job is to
supply the accuracy the modality supports while making the cost of dropping
fluorescence explicit. The release encodes that cost literally -- fluorescence is
an inference channel only in the modality ablation, and the fluorescence frames
are already present in :class:`FrameBundle`, so enabling them changes the cost of
the decision rather than the architecture of the encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from saber.perception.adapters import LoRAConfig, count_adapter_parameters
from saber.perception.backbones import (
    BackboneConfig,
    MorphologyBackbone,
    VisionTowerBackbone,
    build_backbones,
)
from saber.perception.embedder import MorphologyEmbedder
from saber.perception.segmenter import InstanceSegmenter
from saber.perception.tracker import TrackerState, Trajectory, associate

Array = NDArray[np.float64]
LabelArray = NDArray[np.int32]

INFERENCE_CHANNELS: tuple[str, ...] = ("brightfield", "phase_contrast")
AUXILIARY_CHANNELS: tuple[str, ...] = ("fluorescence",)


@dataclass(frozen=True)
class FrameBundle:
    """One epoch's frames for one specimen."""

    frames: Tensor
    channel_names: tuple[str, ...]
    specimen: str
    epoch: int

    @property
    def channels(self) -> int:
        return int(self.frames.shape[1])

    def select(self, wanted: Sequence[str]) -> FrameBundle:
        """Restrict the bundle to a subset of channels, in the given order."""
        index = [self.channel_names.index(name) for name in wanted if name in self.channel_names]
        if not index:
            raise ValueError("no channel of the request is present in the bundle")
        chosen = self.frames[:, index]
        return FrameBundle(
            frames=chosen,
            channel_names=tuple(self.channel_names[i] for i in index),
            specimen=self.specimen,
            epoch=self.epoch,
        )


@dataclass(frozen=True)
class PerceptionOutput:
    """What one epoch of the front end produces for one specimen."""

    masks: tuple[LabelArray, ...]
    trajectories: tuple[Trajectory, ...]
    embeddings: tuple[Tensor, ...]

    def instance_count(self) -> int:
        return len(self.embeddings)

    def label(self) -> dict[str, int]:
        return {
            "epochs": len(self.masks),
            "instances": self.instance_count(),
            "trajectories": len(self.trajectories),
        }


class PerceptionFrontend(nn.Module):
    """The dual-backbone, adapter-adapted encoder with its mask head and pooling."""

    vision: VisionTowerBackbone
    morphology: MorphologyBackbone

    def __init__(
        self,
        in_channels: int,
        backbone: BackboneConfig,
        adapter: LoRAConfig,
        embedding_dim: int,
        hidden: int = 32,
        use_adapters: bool = True,
    ) -> None:
        super().__init__()
        vision, morphology = build_backbones(
            in_channels, backbone, adapter if use_adapters else None
        )
        self.vision = vision
        self.morphology = morphology
        self.use_adapters = use_adapters
        self.segmenter = InstanceSegmenter(int(self.vision.out_width), hidden)
        self.embedder = MorphologyEmbedder(
            int(self.vision.out_width), int(self.morphology.out_width), embedding_dim
        )

    def trainable_parameters(self) -> int:
        return count_adapter_parameters(self)

    def features(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        return self.vision(frames), self.morphology(frames)

    def forward(self, bundle: FrameBundle, gate: float) -> PerceptionOutput:
        vision_features, morphology_features = self.features(bundle.frames)
        labels = self.segmenter.masks(vision_features)
        state = TrackerState(gate=gate)
        completed: list[Trajectory] = list(associate(state, labels))
        embeddings = self.embedder.embed_instances(vision_features, morphology_features, labels)
        return PerceptionOutput(
            masks=(labels,),
            trajectories=tuple(completed),
            embeddings=tuple(embeddings[identity] for identity in sorted(embeddings)),
        )

    def run_epochs(self, bundles: Sequence[FrameBundle], gate: float) -> PerceptionOutput:
        """Run the front end over a specimen's epochs, carrying the tracker.

        This is the loop Algorithm 1's line 3 sits inside, so the released stream
        is per specimen and per epoch rather than per frame.
        """
        state = TrackerState(gate=gate)
        masks: list[LabelArray] = []
        completed: list[Trajectory] = []
        stream: list[Tensor] = []
        for bundle in bundles:
            vision_features, morphology_features = self.features(bundle.frames)
            labels = self.segmenter.masks(vision_features)
            masks.append(labels)
            completed.extend(associate(state, labels))
            embeddings = self.embedder.embed_instances(vision_features, morphology_features, labels)
            stream.append(
                torch.stack([embeddings[key] for key in sorted(embeddings)], dim=0).mean(dim=0)
            )
        return PerceptionOutput(
            masks=tuple(masks),
            trajectories=tuple(completed),
            embeddings=tuple(stream),
        )
