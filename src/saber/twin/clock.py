"""The monitoring clock.

Ref: Sec. 3.1 -- the reference protocol images "every 6 h, 3 channels"; Sec. 2.4
defines the budget as the impact of that protocol "during the same wall-clock
time frame", so the session length is an epoch count and the epoch length is the
reference cadence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

SUBSTEPS_PER_EPOCH = 8


@dataclass(frozen=True)
class EpochClock:
    """A session of equal-length epochs at the reference cadence."""

    epochs: int
    epoch_hours: float

    @property
    def duration_hours(self) -> float:
        return float(self.epochs) * self.epoch_hours

    def times_hours(self) -> Array:
        return np.arange(1.0, float(self.epochs) + 1.0) * self.epoch_hours

    def substep_hours(self) -> float:
        return self.epoch_hours / float(SUBSTEPS_PER_EPOCH)

    def exchange_epochs(self) -> tuple[int, ...]:
        """Epochs at which the reference protocol performs a medium exchange."""
        return tuple(range(self.epochs))

    def label(self) -> dict[str, float | int]:
        return {
            "epochs": self.epochs,
            "epoch_hours": self.epoch_hours,
            "duration_hours": self.duration_hours,
        }


@dataclass(frozen=True)
class SessionSchedule:
    """The clock plus the cadence agreement the two configs must satisfy.

    The dish block carries the reference cadence and the twin block carries the
    epoch length; they describe the same clock, so a mismatch is a configuration
    error rather than a modelling choice.
    """

    clock: EpochClock
    reference_epoch_hours: float

    @classmethod
    def build(
        cls, epochs: int, epoch_hours: float, reference_epoch_hours: float
    ) -> SessionSchedule:
        return cls(
            clock=EpochClock(epochs=epochs, epoch_hours=epoch_hours),
            reference_epoch_hours=reference_epoch_hours,
        )

    def cadence_agrees(self, tolerance: float = 1e-9) -> bool:
        return abs(self.clock.epoch_hours - self.reference_epoch_hours) <= tolerance

    def label(self) -> dict[str, float | int | bool]:
        return {
            **self.clock.label(),
            "reference_epoch_hours": self.reference_epoch_hours,
            "cadence_agrees": self.cadence_agrees(),
        }
