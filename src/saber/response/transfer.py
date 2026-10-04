"""Directed cross-corpus transfer of the response head.

Ref: Sec. 3.6 -- "The coefficients of determination from within the corpus ...
The contributions lie in the conditioned off-diagonal entries. When
microenvironment state is conditioned on, the value of mean directed transfer ...
increases in the determination coefficient ... Two reporting decisions are
critical to interpreting the table. Each entry comes with information about the
eligible retention rate and the effective sample size of the target policy,
as the reference we rely on warns about the fact that the eligible retention
metric gives an impression of a well-performing model if the baseline indicator
is poor".

The table is directed: entry ``(s, t)`` is a head fitted on corpus ``s`` and
evaluated on corpus ``t``, so the diagonal is the within-corpus coefficient and
the off-diagonal entries are the transfer. Every entry carries its retention and
its effective sample size, which is why the record is a dataclass rather than a
matrix of floats.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from saber.belief.estimator import Belief
from saber.evaluation.metrics import effective_sample_size, normalised_weights, r_squared, retention
from saber.response.fingerprint import Molecule
from saber.response.head import ResponseHead, predict_summaries

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CorpusArm:
    """One corpus's evaluation arm: beliefs, drugs, observed summaries."""

    name: str
    beliefs: tuple[Belief, ...]
    fingerprints: Array
    molecules: tuple[Molecule, ...]
    observed: Array
    baseline: Array
    weights: Array


@dataclass(frozen=True)
class CorpusPrediction:
    """A fitted head's predictions on one corpus arm."""

    source: str
    target: str
    predicted: Array
    observed: Array
    baseline: Array
    weights: Array

    def coefficient(self) -> float:
        return r_squared(self.predicted, self.observed)

    def within_corpus(self) -> bool:
        return self.source == self.target

    def label(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "r_squared": round(self.coefficient(), 9),
            "retention": round(self.retention_statistic(), 9),
            "effective_sample_size": round(self.effective_sample_size(), 3),
        }

    def retention_statistic(self) -> float:
        return retention(self.predicted, self.observed, self.baseline)

    def effective_sample_size(self) -> float:
        return effective_sample_size(self.weights)


@dataclass(frozen=True)
class TransferMatrix:
    """The directed table of Sec. 3.6, one prediction per (source, target) pair."""

    predictions: tuple[CorpusPrediction, ...]
    floor: float

    def sources(self) -> tuple[str, ...]:
        seen: list[str] = []
        for entry in self.predictions:
            if entry.source not in seen:
                seen.append(entry.source)
        return tuple(seen)

    def directed(self) -> tuple[CorpusPrediction, ...]:
        return tuple(entry for entry in self.predictions if not entry.within_corpus())

    def mean_directed(self) -> float:
        entries = self.directed()
        if not entries:
            return 0.0
        return float(np.mean([entry.coefficient() for entry in entries]))

    def within_corpus_range(self) -> tuple[float, float]:
        values = [entry.coefficient() for entry in self.predictions if entry.within_corpus()]
        if not values:
            return (0.0, 0.0)
        return (float(min(values)), float(max(values)))

    def weakest_directed(self) -> CorpusPrediction | None:
        entries = self.directed()
        if not entries:
            return None
        return min(entries, key=lambda entry: entry.coefficient())

    def below_floor(self) -> tuple[CorpusPrediction, ...]:
        return tuple(
            entry for entry in self.predictions if entry.effective_sample_size() < self.floor
        )

    def label(self) -> dict[str, object]:
        rows = [entry.label() for entry in self.predictions]
        low, high = self.within_corpus_range()
        return {
            "rows": rows,
            "mean_directed": round(self.mean_directed(), 9),
            "within_corpus_min": round(low, 9),
            "within_corpus_max": round(high, 9),
            "below_effective_sample_floor": [
                {"source": entry.source, "target": entry.target} for entry in self.below_floor()
            ],
        }


def fit_head(
    head: ResponseHead,
    beliefs: tuple[Belief, ...],
    fingerprints: Array,
    molecules: tuple[Molecule, ...],
    targets: Array,
    *,
    epochs: int = 60,
    learning_rate: float = 5e-3,
    seed: int = 0,
) -> float:
    """Fit the head on one corpus and return the final training loss.

    The fit is what makes the transfer directed: the head's weights come from the
    source corpus only, and the target corpus is never seen during fitting.
    """
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(head.parameters(), lr=learning_rate)
    desired = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    final = 0.0
    for _ in range(epochs):
        optimiser.zero_grad()
        rows = [
            head.predict_tensor(belief, fingerprint, molecule)
            for belief, fingerprint, molecule in zip(beliefs, fingerprints, molecules)
        ]
        predicted = torch.stack(rows, dim=0)
        loss = torch.mean((predicted - desired) ** 2)
        torch.autograd.backward(loss)
        optimiser.step()
        final = float(loss.detach())
    return final


def directed_transfer(
    head_factory: Callable[[], ResponseHead],
    sources: tuple[CorpusArm, ...],
    targets: tuple[CorpusArm, ...],
    *,
    epochs: int = 60,
    learning_rate: float = 5e-3,
    seed: int = 0,
    floor: float = 400.0,
) -> TransferMatrix:
    """Fit on every source and evaluate on every target.

    ``head_factory`` is a callable taking nothing and returning a fresh head, so
    no source's fitted weights can leak into another source's entry.
    """
    rows: list[CorpusPrediction] = []
    for source in sources:
        head = head_factory()
        fit_head(
            head,
            source.beliefs,
            source.fingerprints,
            source.molecules,
            source.observed,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
        )
        for target in targets:
            predicted = predict_summaries(
                head, target.beliefs, target.fingerprints, target.molecules
            )
            rows.append(
                CorpusPrediction(
                    source=source.name,
                    target=target.name,
                    predicted=predicted,
                    observed=target.observed,
                    baseline=target.baseline,
                    weights=target.weights,
                )
            )
    return TransferMatrix(predictions=tuple(rows), floor=floor)


def retention_table(matrix: TransferMatrix) -> list[dict[str, object]]:
    """Rows of the transfer table with the retention statistic beside the absolute."""
    return [
        {
            "source": entry.source,
            "target": entry.target,
            "r_squared": round(entry.coefficient(), 9),
            "retention": round(entry.retention_statistic(), 9),
            "neff": round(entry.effective_sample_size(), 3),
        }
        for entry in matrix.predictions
    ]


def importance_weights(target_probabilities: Array, logging_probabilities: Array) -> Array:
    """The support-mismatch diagnostic's input: the ratio of the two policies.

    Ref: Sec. 3.7 -- "inverse of the propensity scoring method performs worse as
    we move further away from the action distribution of the target policy from
    that of the logging policy's predetermined pattern, thus highlighting the
    very occurrence of the support mismatch which is why neff is reported
    for every estimation result obtained."
    """
    target = np.asarray(target_probabilities, dtype=np.float64)
    logging = np.asarray(logging_probabilities, dtype=np.float64)
    ratio = np.where(logging > 0.0, target / np.maximum(logging, 1e-12), 0.0)
    return normalised_weights(ratio)
