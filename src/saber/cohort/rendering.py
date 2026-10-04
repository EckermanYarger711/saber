"""Frame rendering for the cohort's specimens.

Ref: Sec. 2.2 -- the front end consumes "bright-field and phase-contrast frames"
and, when the modality is enabled, fluorescence frames; Sec. 3.1 -- the
evaluation is a secondary analysis of logged trajectories, so the frames are a
presentation of a logged record rather than a new acquisition.

The renderer turns a specimen's axis trajectory into a frame set with a
deterministic geometry: a handful of discs whose count and radius follow the
specimen's biomass proxy and whose per-channel brightness follows the axes that
channel reports. The geometry is fixed by the specimen's identity, so two runs
render the same frames and a check can compare the front end's output across
runs.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from saber.cohort.schema import SpecimenRecord

Array = NDArray[np.float64]

MIN_INSTANCES = 3
MAX_INSTANCES = 9
BASE_RADIUS = 0.06
RADIUS_SPAN = 0.10

CHANNEL_MIX: dict[str, tuple[int, float, float]] = {
    "brightfield": (0, 1.4, 0.15),
    "phase_contrast": (1, 1.1, 1.0),
    "fluorescence": (2, 1.6, 0.25),
}


def _identity_seed(identity: str) -> int:
    digest = hashlib.blake2b(identity.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def instance_geometry(specimen: SpecimenRecord, frame_size: int) -> Array:
    """Disc centres and radii, in pixels, for one specimen.

    The count grows with the specimen's mean axis level, which is the quantity
    the tracking benchmark's instance counts are reported against.
    """
    generator = np.random.default_rng(_identity_seed(specimen.identity))
    level = float(np.mean(specimen.axes[-1]))
    count = int(round(MIN_INSTANCES + (MAX_INSTANCES - MIN_INSTANCES) * np.clip(level, 0.0, 1.0)))
    centres = generator.uniform(0.2, 0.8, size=(count, 2)) * frame_size
    radii = (
        (BASE_RADIUS + RADIUS_SPAN * np.clip(level, 0.0, 1.0))
        * frame_size
        * generator.uniform(0.7, 1.3, size=count)
    )
    return np.asarray(np.concatenate([centres, radii[:, None]], axis=1), dtype=np.float64)


def render_channel(
    specimen: SpecimenRecord,
    epoch: int,
    channel: str,
    frame_size: int,
    noise: float = 0.01,
) -> Array:
    """One frame for one channel at one epoch."""
    geometry = instance_geometry(specimen, frame_size)
    axis_index, gain, offset = CHANNEL_MIX.get(channel, (0, 1.0, 0.2))
    axes = specimen.axes[min(epoch, specimen.axes.shape[0] - 1)]
    level = float(axes[axis_index])
    axis = np.arange(frame_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    frame = np.full((frame_size, frame_size), offset, dtype=np.float64)
    for centre_x, centre_y, radius in geometry:
        distance = np.sqrt((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2)
        frame += gain * level * np.clip(1.0 - distance / max(radius, 1e-6), 0.0, 1.0)
    generator = np.random.default_rng(_identity_seed(specimen.identity) + epoch * 131 + axis_index)
    frame = frame + generator.normal(scale=noise, size=frame.shape)
    return np.asarray(np.clip(frame, 0.0, None), dtype=np.float64)


def frame_bundle(
    specimen: SpecimenRecord, epoch: int, channels: tuple[str, ...], frame_size: int
) -> Tensor:
    """A ``(1, C, H, W)`` tensor over the requested channels."""
    frames = [render_channel(specimen, epoch, channel, frame_size) for channel in channels]
    stacked = np.stack(frames, axis=0)[None, :, :, :]
    return torch.as_tensor(stacked, dtype=torch.float32)
