"""Recalibration of drifting axes.

Ref: Sec. 2.3 -- "In cases when the uncertainty pertaining to an axis surpasses a
certain recalibration threshold value, it gets re-attached to its atlas prior,
instead of having the freedom to drift." Algorithm 1 lines 6-10.

The re-tie replaces the axis's posterior mean and variance with the prior's. It
is a hard constraint rather than a shrinkage, because the stated purpose is to
prevent *silent* drift: a shrunk axis would still move, only more slowly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.belief.estimator import Belief
from saber.belief.priors import AtlasPrior

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RecalibrationEvent:
    """One axis re-tied to its prior at one epoch."""

    epoch: int
    axis: int
    axis_name: str
    observed_uncertainty: float
    threshold: float

    def label(self) -> dict[str, float | int | str]:
        return {
            "epoch": self.epoch,
            "axis": self.axis_name,
            "observed_uncertainty": round(self.observed_uncertainty, 9),
            "threshold": round(self.threshold, 9),
        }


def recalibrate(
    belief: Belief, prior: AtlasPrior, threshold: float, epoch: int
) -> tuple[Belief, tuple[RecalibrationEvent, ...]]:
    """Re-tie every axis whose uncertainty exceeds the threshold.

    The operation is applied per axis, so a specimen can be recalibrated on its
    co-culture coordinate while its oxygen coordinate keeps integrating evidence.
    """
    mean = np.array(belief.mean, dtype=np.float64, copy=True)
    deviation = np.array(belief.standard_deviation, dtype=np.float64, copy=True)
    events: list[RecalibrationEvent] = []
    for axis in range(deviation.size):
        if deviation[axis] > threshold:
            events.append(
                RecalibrationEvent(
                    epoch=epoch,
                    axis=axis,
                    axis_name=prior.names[axis],
                    observed_uncertainty=float(deviation[axis]),
                    threshold=float(threshold),
                )
            )
            mean[axis] = float(prior.means[axis])
            deviation[axis] = float(prior.standard_deviations[axis])
    return Belief(mean=mean, standard_deviation=deviation), tuple(events)


def drifting_axes(belief: Belief, prior: AtlasPrior, z_score: float = 3.0) -> tuple[int, ...]:
    """Axes whose mean has moved more than ``z_score`` prior deviations.

    This is the diagnostic the recalibration threshold is protecting against: an
    axis can sit inside the uncertainty threshold while its mean has already left
    the atlas's range.
    """
    shifted = np.abs(prior.standardise(belief.mean))
    return tuple(int(index) for index in np.flatnonzero(shifted > z_score))


def converged(history: list[float], patience: int, epsilon: float) -> bool:
    """Algorithm 1 lines 11-13: total uncertainty has stopped falling.

    The test is on the decrease per epoch, so a belief that is still moving
    cannot terminate the stream even if its total uncertainty is already small.
    """
    if len(history) < patience + 1:
        return False
    recent = history[-(patience + 1) :]
    decreases = [recent[index] - recent[index + 1] for index in range(patience)]
    return all(decrease < epsilon for decrease in decreases)
