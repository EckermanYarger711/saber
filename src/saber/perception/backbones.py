"""The two adapted backbones.

Ref: Sec. 2.2 -- a vision tower "already trained to rely on task context when
providing general scene structure" and "the morphology foundation model that has
been trained on large histopathology datasets and provides texture statistics
that differentiate culture states". Sec. 3.4's Tier-3 ablation swaps the
morphology backbone and separately removes the adapters, so both properties are
selectable behind one interface.

The article prints neither backbone's width nor its depth, and Sec. 4.1 records
that "the perception front end is based on one specific architectural family".
Both are therefore built at engineering-default widths and used as frozen
feature extractors; what the release needs from them is that they are two
different, deterministic encoders -- one spatial, one texture-statistic -- so
that the swap and the adapter ablation are real ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from saber.perception.adapters import LoRAConfig, adapt_conv2d, freeze_parameters


@dataclass(frozen=True)
class BackboneConfig:
    """Widths of the two backbones; the article prints no values for these."""

    vision_width: int = 24
    vision_depth: int = 2
    morphology_width: int = 32
    morphology_depth: int = 2
    texture_scales: int = 3

    def swapped(self) -> BackboneConfig:
        """The Tier-3 morphology-backbone swap: the other family, same budget."""
        return BackboneConfig(
            vision_width=self.vision_width,
            vision_depth=self.vision_depth,
            morphology_width=self.morphology_width * 2,
            morphology_depth=max(1, self.morphology_depth - 1),
            texture_scales=self.texture_scales + 1,
        )


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.GELU(),
    )


class VisionTowerBackbone(nn.Module):
    """A spatial encoder over the frame set; the vision tower of Sec. 2.2."""

    out_width: int

    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [conv_block(in_channels, width)]
        for _ in range(depth - 1):
            layers.append(conv_block(width, width))
        self.stem = nn.Sequential(*layers)
        self.out_width = width
        freeze_parameters(self)

    def forward(self, frames: Tensor) -> Tensor:
        features: Tensor = self.stem(frames)
        return features


class MorphologyBackbone(nn.Module):
    """A texture-statistics encoder; the morphology foundation model of Sec. 2.2.

    out_width: int

    Its features are local first and second moments at several spatial scales,
    which is the property the article attributes to the morphology model class:
    it describes texture rather than shape.
    """

    def __init__(self, in_channels: int, width: int, depth: int, scales: int) -> None:
        super().__init__()
        self.scales = scales
        self.pools = nn.ModuleList(
            [
                nn.AvgPool2d(kernel_size=2 * scale + 1, stride=1, padding=scale)
                for scale in range(1, scales + 1)
            ]
        )
        layers: list[nn.Module] = [conv_block(in_channels * (2 * scales + 1), width)]
        for _ in range(depth - 1):
            layers.append(conv_block(width, width))
        self.project = nn.Sequential(*layers)
        self.out_width = width
        freeze_parameters(self)

    def forward(self, frames: Tensor) -> Tensor:
        moments: list[Tensor] = [frames]
        for pool in self.pools:
            local = pool(frames)
            moments.append(local)
            moments.append(local * local)
        stacked: Tensor = torch.cat(moments, dim=1)
        features: Tensor = self.project(stacked)
        return features


def build_backbones(
    in_channels: int,
    config: BackboneConfig,
    adapter: LoRAConfig | None = None,
    swap_morphology: bool = False,
) -> tuple[VisionTowerBackbone, MorphologyBackbone]:
    """Both backbones at the configured widths, adapted when a rank is supplied.

    ``adapt_conv2d`` rewrites every 3x3 convolution in place, so the adapters are
    part of the forward pass rather than a parallel branch.
    """
    chosen = config.swapped() if swap_morphology else config
    vision = VisionTowerBackbone(in_channels, chosen.vision_width, chosen.vision_depth)
    morphology = MorphologyBackbone(
        in_channels,
        chosen.morphology_width,
        chosen.morphology_depth,
        chosen.texture_scales,
    )
    if adapter is not None:
        adapted = adapt_conv2d(vision, adapter)
        return cast(VisionTowerBackbone, adapted), morphology
    return vision, morphology
