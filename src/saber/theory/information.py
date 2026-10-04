"""Gaussian belief algebra and information gain, from the definitions.

Ref: Sec. 1.1.3 (Theorem 1 assumes Gaussian beliefs with precision additive in
the number of exposures, and conditionally independent observations given the
latent state); Eq. (4) (``G(a | b) = E_o[H[b] - H[b' | a, o]]``).

The routines are written against the closed form so the verification layer can
compare a planner's estimate of ``G`` with a value that never went through the
planner. Entropies are in nats, which is the unit the article prints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _logdet(matrix: Array) -> float:
    """Log-determinant through the Cholesky factor; the covariance is SPD here."""
    factor = np.linalg.cholesky(matrix)
    return float(2.0 * np.sum(np.log(np.diag(factor))))


@dataclass(frozen=True)
class GaussianBelief:
    """A Gaussian belief over a latent microenvironment state.

    ``covariance`` is the belief covariance, so its inverse is the precision
    that Theorem 1 requires to be additive.
    """

    mean: Array
    covariance: Array

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def entropy(self) -> float:
        """Differential entropy of the belief, in nats."""
        return entropy(self.covariance)

    def precision(self) -> Array:
        return np.asarray(np.linalg.inv(self.covariance), dtype=np.float64)

    def with_covariance(self, covariance: Array) -> GaussianBelief:
        return GaussianBelief(mean=self.mean, covariance=covariance)


def entropy(covariance: Array) -> float:
    """``H = 0.5 * (d * (1 + ln 2pi) + ln det Sigma)`` in nats."""
    dimension = int(covariance.shape[0])
    value = 0.5 * (dimension * (1.0 + np.log(2.0 * np.pi)) + _logdet(covariance))
    return float(value)


def observe(belief: GaussianBelief, design: Array, noise_variance: float) -> GaussianBelief:
    """Kalman-style Gaussian update under one linear observation.

    Precision additivity is the same arithmetic Theorem 1 assumes: for repeated
    observations with the same design the posterior precision is the prior
    precision plus ``n`` copies of ``design^T design / noise_variance``.
    """
    prior_precision = belief.precision()
    posterior_precision = prior_precision + np.outer(design, design) / noise_variance
    posterior_covariance = np.asarray(np.linalg.inv(posterior_precision), dtype=np.float64)
    gain = posterior_covariance @ design / noise_variance
    residual = float(design @ belief.mean)
    mean = belief.mean + gain * (0.0 - residual)
    return GaussianBelief(mean=mean, covariance=posterior_covariance)


def information_gain(belief: GaussianBelief, design: Array, noise_variance: float) -> float:
    """``E_o[H[b] - H[b' | a, o]]`` for a linear-Gaussian observation."""
    posterior = observe(belief, design, noise_variance)
    return belief.entropy() - posterior.entropy()


def mutual_information(belief: GaussianBelief, design: Array, noise_variance: float) -> float:
    """The same quantity read as ``I(s; o | b, a)``, Eq. (3)."""
    return information_gain(belief, design, noise_variance)


def repeated_precision_information(
    prior_precision: Array, per_exposure_precision: Array, counts: Array
) -> Array:
    """``G(n)`` for a vector of exposure counts, by accumulated precision.

    This is the function Theorem 1 differentiates: cumulative information is
    concave in the accumulated precision, hence in the number of exposures when
    the per-exposure precision is constant.
    """
    prior_logdet = _logdet(prior_precision)
    values = np.empty_like(counts, dtype=np.float64)
    for index, count in enumerate(np.atleast_1d(counts)):
        posterior = prior_precision + float(count) * per_exposure_precision
        values[index] = 0.5 * (_logdet(posterior) - prior_logdet)
    return values


def marginal_information(prior_precision: Array, per_exposure_precision: Array, n: int) -> float:
    """``g(n) = G(n + 1) - G(n)``: the information the next exposure adds."""
    values = repeated_precision_information(
        prior_precision, per_exposure_precision, np.array([float(n), float(n + 1)])
    )
    return float(values[1] - values[0])


def posterior_uncertainty(covariance: Array) -> Array:
    """Axis standard deviations ``sigma_t``, the second moment of the posterior.

    Ref: Sec. 2.3 -- "the second moment of the posterior provides a proper
    uncertainty value sigma_t for each axis".
    """
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    return np.asarray(np.sqrt(diagonal), dtype=np.float64)
