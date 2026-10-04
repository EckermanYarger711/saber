"""Trajectories over epochs.

Ref: Sec. 2.2 -- ``E_theta`` outputs "trajectories over epochs" alongside the
masks and the embeddings; Sec. 3.1 splits the corpus by specimen "such that all
the frames of a single specimen are on the same side of the split", which is only
meaningful once instances carry identity across epochs.

Association is greedy on centroid distance with a gate, which is the standard
tracking-by-detection step; the gate is what keeps two neighbouring organoids
from being merged into one trajectory when they touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from saber.perception.segmenter import component_properties

Array = NDArray[np.float64]
LabelArray = NDArray[np.int32]


@dataclass(frozen=True)
class Trajectory:
    """One tracked instance: its id and the centroids it occupied."""

    identity: int
    centroids: tuple[tuple[float, float], ...]
    areas: tuple[float, ...]

    @property
    def length(self) -> int:
        return len(self.centroids)

    def displacement(self) -> float:
        if self.length < 2:
            return 0.0
        start = np.asarray(self.centroids[0])
        end = np.asarray(self.centroids[-1])
        return float(np.linalg.norm(end - start))

    def label(self) -> dict[str, float | int]:
        return {
            "identity": self.identity,
            "length": self.length,
            "displacement": round(self.displacement(), 9),
            "final_area": round(self.areas[-1], 9) if self.areas else 0.0,
        }


@dataclass
class TrackerState:
    """The open trajectories and the identity counter."""

    gate: float
    open_tracks: list[Trajectory] = field(default_factory=list)
    next_identity: int = 1


def associate(
    state: TrackerState,
    labels: LabelArray,
    spacing: float = 1.0,
) -> list[Trajectory]:
    """Associate one epoch's components with the open trajectories.

    The product is a list of trajectories closed at this epoch, which is what
    Algorithm 1 consumes: an embedding stream indexed by identity.
    """
    components = component_properties(labels, spacing)
    observations = [
        (item["centroid_row"], item["centroid_column"], item["area"]) for item in components
    ]
    unmatched = set(range(len(observations)))
    completed: list[Trajectory] = []
    for track in state.open_tracks:
        if not unmatched:
            state.open_tracks = []
            break
        last = np.asarray(track.centroids[-1])
        candidates = sorted(
            unmatched,
            key=lambda index: float(np.linalg.norm(last - np.asarray(observations[index][:2]))),
        )
        best = candidates[0]
        distance = float(np.linalg.norm(last - np.asarray(observations[best][:2])))
        if distance <= state.gate:
            unmatched.discard(best)
            row, column, area = observations[best]
            completed.append(
                Trajectory(
                    identity=track.identity,
                    centroids=(*track.centroids, (float(row), float(column))),
                    areas=(*track.areas, float(area)),
                )
            )
        else:
            completed.append(track)
    state.open_tracks = []
    for index in sorted(unmatched):
        row, column, area = observations[index]
        state.open_tracks.append(
            Trajectory(
                identity=state.next_identity,
                centroids=((float(row), float(column)),),
                areas=(float(area),),
            )
        )
        state.next_identity += 1
    return completed
