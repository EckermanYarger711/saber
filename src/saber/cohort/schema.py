"""Record shapes for the evaluation substrate.

Ref: Sec. 3.1 -- the evaluation is a retrospective secondary analysis of logged
trajectories; Sec. 2.2 -- the per-epoch product of the front end is a set of
instances with embeddings; Sec. 2.6 -- the response head's target is a
dose-response summary.

A specimen record therefore carries three things: its microenvironment axis
trajectory (what the estimator is trying to recover), the logged observation
features (what the front end produced), and the dose-response summary (what the
response head predicts). The corpus and the stratum are carried as fields rather
than derived, because the article reports every number per corpus and per
regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SpecimenRecord:
    """One tracked specimen over a session."""

    identity: str
    corpus: str
    stratum: str
    transition_rate: float
    axes: Array
    observations: Array
    dose_response: Array

    @property
    def epochs(self) -> int:
        return int(self.axes.shape[0])

    def final_axes(self) -> Array:
        return np.asarray(self.axes[-1], dtype=np.float64)

    def label(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "corpus": self.corpus,
            "stratum": self.stratum,
            "transition_rate": round(self.transition_rate, 6),
            "epochs": self.epochs,
        }


@dataclass(frozen=True)
class CohortRecord:
    """The whole substrate: every specimen, with the session's shape."""

    specimens: tuple[SpecimenRecord, ...]
    epoch_hours: float
    dose_points: int

    def size(self) -> int:
        return len(self.specimens)

    def corpora(self) -> tuple[str, ...]:
        seen: list[str] = []
        for specimen in self.specimens:
            if specimen.corpus not in seen:
                seen.append(specimen.corpus)
        return tuple(sorted(seen))

    def by_corpus(self, corpus: str) -> tuple[SpecimenRecord, ...]:
        return tuple(specimen for specimen in self.specimens if specimen.corpus == corpus)

    def strata(self) -> tuple[str, ...]:
        return tuple(sorted({specimen.stratum for specimen in self.specimens}))

    def transition_rates(self) -> Array:
        return np.asarray(
            [specimen.transition_rate for specimen in self.specimens], dtype=np.float64
        )

    def summary(self) -> dict[str, object]:
        counts = {corpus: len(self.by_corpus(corpus)) for corpus in self.corpora()}
        return {
            "specimens": self.size(),
            "corpora": counts,
            "strata": list(self.strata()),
            "epochs": self.specimens[0].epochs if self.specimens else 0,
            "epoch_hours": self.epoch_hours,
            "dose_points": self.dose_points,
        }

    def axes_tensor(self) -> Array:
        return np.asarray(np.stack([s.axes for s in self.specimens], axis=0), dtype=np.float64)
