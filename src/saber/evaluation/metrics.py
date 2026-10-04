"""Metrics and the statistical protocol.

Ref: Sec. 3.1 -- "Every result is represented based on five runs done by ``m = 5``,
and uncertainty is expressed as the standard deviation between experiments. The
noise band for all headline comparisons equals twice the value of the
aforementioned standard deviation"; Sec. 3.6 -- per-corpus reporting with the
retention statistic and the effective sample size, "because no pair is below the
lower limit set beforehand".

Everything here is a definition, not an estimator choice: balanced accuracy is
per-class recall averaged over classes, R-squared is the coefficient of
determination against the identity, and the effective sample size is the
importance-weight diagnostic ``(sum w)^2 / sum w^2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.belief.axes import regime_label

Array = NDArray[np.float64]


def balanced_accuracy(predicted: NDArray[np.str_], truth: NDArray[np.str_]) -> float:
    """Mean per-class recall over the strata that occur in the truth.

    Ref: the "balanced accuracy" column of Table 1, which the article uses
    because the regime strata are not equally populated.
    """
    predicted_labels = np.asarray(predicted, dtype=object)
    truth_labels = np.asarray(truth, dtype=object)
    if predicted_labels.size == 0:
        return 0.0
    recalls: list[float] = []
    for label in sorted(set(truth_labels.tolist())):
        selected = truth_labels == label
        total = int(selected.sum())
        if total == 0:
            continue
        recalls.append(float(np.sum(predicted_labels[selected] == label)) / total)
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def regime_sequence(axes: Array) -> NDArray[np.str_]:
    """The stratum label of each row of an axis trajectory."""
    values = np.asarray(axes, dtype=np.float64)
    labels = [regime_label(tuple(float(value) for value in row)) for row in values]
    return np.asarray(labels, dtype=object)


def r_squared(predicted: Array, observed: Array) -> float:
    """Coefficient of determination against the identity line.

    The transfer table of Sec. 3.6 reports a coefficient of determination between
    a predicted response and a held-out corpus's observed response, so the
    reference is the observed values rather than their mean.
    """
    predicted_values = np.asarray(predicted, dtype=np.float64)
    observed_values = np.asarray(observed, dtype=np.float64)
    residual = float(np.sum((predicted_values - observed_values) ** 2))
    total = float(np.sum((observed_values - np.mean(observed_values)) ** 2))
    if total <= 0.0:
        return 0.0
    return 1.0 - residual / total


def retention(predicted: Array, observed: Array, baseline: Array) -> float:
    """The retention statistic the article warns about.

    Ref: the cited benchmark -- "a normalized retention statistic will benefit
    models with weak baseline element in the corpus", which is why the release
    reports the absolute coefficient beside it.
    """
    reference = r_squared(baseline, observed)
    transfer = r_squared(predicted, observed)
    if abs(reference) <= 1e-12:
        return 0.0
    return transfer / reference


def effective_sample_size(weights: Array) -> float:
    """``(sum w)^2 / sum w^2`` for a vector of non-negative importance weights."""
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or float(np.sum(values**2)) <= 0.0:
        return 0.0
    return float(np.sum(values) ** 2 / np.sum(values**2))


def normalised_weights(weights: Array) -> Array:
    values = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.zeros_like(values)
    return np.asarray(values / total, dtype=np.float64)


def noise_band(values: Array) -> float:
    """Twice the between-run standard deviation, the article's headline band."""
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return 0.0
    return float(2.0 * np.std(series, ddof=1))


def within_band(difference: float, band: float) -> bool:
    return abs(difference) <= band


@dataclass(frozen=True)
class RunSummary:
    """Means over the article's five runs with their between-run dispersion."""

    mean: float
    standard_deviation: float
    runs: tuple[float, ...]

    @classmethod
    def of(cls, values: Array) -> RunSummary:
        series = np.asarray(values, dtype=np.float64)
        return cls(
            mean=float(np.mean(series)),
            standard_deviation=float(np.std(series, ddof=1)) if series.size > 1 else 0.0,
            runs=tuple(float(value) for value in series),
        )

    def label(self) -> dict[str, float | int]:
        return {
            "mean": round(self.mean, 9),
            "standard_deviation": round(self.standard_deviation, 9),
            "runs": len(self.runs),
        }
