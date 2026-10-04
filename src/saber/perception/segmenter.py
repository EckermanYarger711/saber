"""Instance segmentation of organoids.

Ref: Sec. 2.2 -- the encoder outputs "instance masks ``M_t``"; the Introduction
records that "general-purpose and base segmentation models achieve Dice
coefficient values close to the practically achievable maximum", so the front
end's job is to reach the modality's accuracy rather than to improve it.

The head is deliberately a mask-proposal head with connected-component
extraction: the article's Dice claim is a property of the modality, and the
release needs an instance decomposition whose components are exactly the objects
of the frame rather than a soft attention map.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage
from torch import Tensor, nn

Array = NDArray[np.float64]
LabelArray = NDArray[np.int32]


def connected_instances(logits: Array, threshold: float = 0.5) -> LabelArray:
    """Threshold a logit map and label its connected components in 2-D.

    Returns an integer label image in which 0 is background; the foreground
    components are the instances, which is what the tracker associates across
    epochs.
    """
    mask = np.asarray(logits, dtype=np.float64) > threshold
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(labels, dtype=np.int32)
    return np.asarray(labels, dtype=np.int32)


def component_properties(labels: LabelArray, spacing: float = 1.0) -> list[dict[str, float]]:
    """Area, centroid and radius of every labelled component."""
    properties: list[dict[str, float]] = []
    count = int(labels.max())
    for index in range(1, count + 1):
        rows, columns = np.nonzero(labels == index)
        if rows.size == 0:
            continue
        area = float(rows.size) * spacing * spacing
        properties.append(
            {
                "label": float(index),
                "area": area,
                "centroid_row": float(rows.mean()),
                "centroid_column": float(columns.mean()),
                "radius": float(np.sqrt(area / np.pi)),
            }
        )
    return properties


def dice_coefficient(predicted: LabelArray, truth: LabelArray) -> float:
    """Aggregate Dice over the foreground, the coefficient the article quotes."""
    predicted_mask = np.asarray(predicted) > 0
    truth_mask = np.asarray(truth) > 0
    denominator = int(predicted_mask.sum() + truth_mask.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(predicted_mask, truth_mask).sum())
    return float(2.0 * intersection / denominator)


class InstanceSegmenter(nn.Module):
    """A mask-proposal head over the pooled backbone features."""

    def __init__(self, in_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.proposal = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        logits: Tensor = self.proposal(features)
        return logits

    def masks(self, features: Tensor, threshold: float = 0.5) -> LabelArray:
        """Instance label image for one frame, without gradients."""
        with torch.no_grad():
            logits = self.forward(features)
        return connected_instances(logits[0, 0].detach().cpu().numpy(), threshold)
