"""Morphology embeddings.

Ref: Sec. 2.2 -- ``E_theta`` outputs "the morphology embedding for each instance
``z_t``"; Sec. 2.3 consumes that stream in the variational estimator; the
Introduction cites the image-based-profiling literature for morphology carrying
functional rather than purely geometric information.

An instance's embedding is the mask-pooled concatenation of the two backbones'
features, projected to a common width. Pooling over the instance rather than over
the frame is what makes the embedding an instance property, which the estimator's
per-specimen belief requires.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

Array = NDArray[np.float64]
LabelArray = NDArray[np.int32]


def mask_pool(features: Tensor, labels: LabelArray, identity: int) -> Tensor:
    """Mean of a feature map over the pixels of one instance.

    A zero-area instance returns the frame mean rather than a zero vector, so a
    degenerate mask cannot masquerade as evidence of a zero state.
    """
    mask = torch.as_tensor(np.asarray(labels) == identity, dtype=features.dtype)
    weights = mask.unsqueeze(0).unsqueeze(0)
    total = float(mask.sum())
    if total <= 0.0:
        pooled: Tensor = torch.mean(features, dim=(2, 3))
        return pooled
    summed: Tensor = torch.sum(features * weights, dim=(2, 3)) / total
    return summed


class MorphologyEmbedder(nn.Module):
    """Projects pooled dual-backbone features to the embedding width."""

    def __init__(self, vision_width: int, morphology_width: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(vision_width + morphology_width, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        vision_features: Tensor,
        morphology_features: Tensor,
        labels: LabelArray,
        identity: int,
    ) -> Tensor:
        pooled_vision = mask_pool(vision_features, labels, identity)
        pooled_morphology = mask_pool(morphology_features, labels, identity)
        joined: Tensor = torch.cat([pooled_vision, pooled_morphology], dim=-1)
        projected: Tensor = self.projection(joined)
        normalised: Tensor = self.norm(projected)
        return normalised.squeeze(0)

    def embed_instances(
        self, vision_features: Tensor, morphology_features: Tensor, labels: LabelArray
    ) -> dict[int, Tensor]:
        """One embedding per labelled instance, keyed by label."""
        embeddings: dict[int, Tensor] = {}
        for identity in sorted({int(value) for value in np.unique(labels)} - {0}):
            embeddings[identity] = self.forward(
                vision_features, morphology_features, labels, identity
            )
        if not embeddings:
            embeddings[0] = self.forward(
                vision_features, morphology_features, np.zeros_like(labels), 0
            )
        return embeddings
