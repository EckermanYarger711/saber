"""The structure of the optimum on a fixed field.

Ref: Definition 1 and Theorem 1, Sec. 1.1.3.

The theorem's three assumptions are the ones modelled here: observations at a
fixed location are conditionally independent given the state, the belief is
Gaussian with precision additive in the number of exposures, and the marginal
information ``g(n) = G(n+1) - G(n)`` is unimodal -- non-decreasing up to an
adaptation point ``n_p`` and non-increasing after it. Cumulative damage is
increasing and convex because phototoxicity and shear reinforce rather than
saturate. Under those three the article asserts the currency
``rho(n) = G(n)/C(n)`` is maximised at an interior exposure count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.theory.information import repeated_precision_information

Array = NDArray[np.float64]


def is_unimodal(values: Array) -> bool:
    """True when the sequence is non-decreasing and then non-increasing.

    A single interior maximum is what the assumption asks for; plateaux are
    allowed on either side of it.
    """
    sequence = np.asarray(values, dtype=np.float64)
    if sequence.size == 0:
        return True
    differences = np.diff(sequence)
    signs = np.sign(differences)
    transitions = np.flatnonzero(signs[1:] < signs[:-1])
    if transitions.size == 0:
        return True
    turn = int(transitions[0]) + 1
    return bool(np.all(differences[:turn] >= -1e-12) and np.all(differences[turn:] <= 1e-12))


def is_convex(values: Array, atol: float = 1e-9) -> bool:
    """True when the discrete second differences are non-negative."""
    sequence = np.asarray(values, dtype=np.float64)
    if sequence.size < 3:
        return True
    second = np.diff(sequence, n=2)
    return bool(np.all(second >= -atol))


def is_increasing(values: Array, atol: float = 1e-12) -> bool:
    sequence = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.diff(sequence) >= -atol))


@dataclass(frozen=True)
class ExposureLadder:
    """``g``, ``G``, ``C`` and ``rho`` on a ladder of exposure counts."""

    counts: Array
    marginal: Array
    information: Array
    damage: Array

    @property
    def currency(self) -> Array:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(self.damage > 0.0, self.information / self.damage, 0.0)
        return np.asarray(ratio, dtype=np.float64)

    def argmax_currency(self) -> int:
        """Index into ``counts`` of the maximiser of ``rho``."""
        return int(np.argmax(self.currency))

    def interior(self) -> bool:
        """Definition 1: the maximiser is strictly inside the ladder."""
        index = self.argmax_currency()
        return 0 < index < self.counts.size - 1

    def fixed_cadence_index(self) -> int:
        """The ladder index the reference protocol occupies, one exposure."""
        return 0

    def label(self) -> dict[str, float | int | bool]:
        index = self.argmax_currency()
        return {
            "n_star": int(self.counts[index]),
            "rho_star": round(float(self.currency[index]), 9),
            "rho_first": round(float(self.currency[0]), 9),
            "rho_last": round(float(self.currency[-1]), 9),
            "interior": self.interior(),
        }


def adapt_then_redundant(
    counts: Array, amplitude: float, rise: float, decay: float, floor: float = 0.0
) -> Array:
    """A unimodal marginal-information shape with an adaptation phase.

    Ref: the theorem's justification -- "the primary experiments undertaken in a
    newly studied domain incur a constant cost of adaptation, which renders them
    less information-rich than the following experiments conducted after initial
    adaptation has been reached and the autofluorescent interference is
    eliminated; after the ``n_p`` stage the subsequent frames are characterized
    by an increasing redundancy."
    """
    n = np.asarray(counts, dtype=np.float64)
    values = amplitude * (1.0 - np.exp(-(n + 1.0) / rise)) * np.exp(-n / decay)
    return np.asarray(values + floor, dtype=np.float64)


def convex_damage_curve(
    counts: Array, base: float, curvature: float, exchange: float = 0.0
) -> Array:
    """Cumulative damage with non-decreasing per-exposure increments.

    ``C(n) = sum_{k=1..n} base * (1 + curvature * (k - 1)) + exchange``; the
    per-exposure increment grows with ``n`` because phototoxic and shear effects
    accumulate instead of saturating.
    """
    n = np.asarray(counts, dtype=np.float64)
    total = np.zeros_like(n)
    for index, count in enumerate(n):
        steps = np.arange(1.0, float(count) + 1.0)
        total[index] = float(np.sum(base * (1.0 + curvature * (steps - 1.0)))) + exchange
    return total


def gaussian_precision_ladder(
    prior_precision: Array, per_exposure_precision: Array, maximum: int
) -> Array:
    """``G(n)`` for ``n = 1..maximum`` under additive-precision observations."""
    counts = np.arange(1.0, float(maximum) + 1.0)
    return repeated_precision_information(prior_precision, per_exposure_precision, counts)


def build_ladder(marginal: Array, damage: Array) -> ExposureLadder:
    """Assemble a ladder from a marginal-information sequence and its costs."""
    counts = np.arange(1.0, float(marginal.size) + 1.0)
    information = np.cumsum(marginal)
    return ExposureLadder(
        counts=counts,
        marginal=np.asarray(marginal, dtype=np.float64),
        information=np.asarray(information, dtype=np.float64),
        damage=np.asarray(damage, dtype=np.float64),
    )


def rho_curve(information: Array, damage: Array) -> Array:
    """``rho(n) = G(n) / C(n)``, Eq. (4) restricted to a fixed field."""
    gains = np.asarray(information, dtype=np.float64)
    costs = np.asarray(damage, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(costs > 0.0, gains / costs, 0.0), dtype=np.float64)


def interior_optimum(ladder: ExposureLadder) -> int:
    """Index of the maximiser of ``rho`` on the ladder."""
    return ladder.argmax_currency()


@dataclass(frozen=True)
class DamageCurve:
    """A named convex cumulative-damage curve, evaluated on a count grid."""

    counts: Array
    cumulative: Array
    per_exposure: Array

    @classmethod
    def build(
        cls, maximum: int, base: float, curvature: float, exchange: float = 0.0
    ) -> DamageCurve:
        counts = np.arange(1.0, float(maximum) + 1.0)
        cumulative = convex_damage_curve(counts, base, curvature, exchange)
        per_exposure = np.diff(np.concatenate([[exchange], cumulative]))
        return cls(counts=counts, cumulative=cumulative, per_exposure=per_exposure)

    def increasing(self) -> bool:
        return is_increasing(self.cumulative)

    def convex(self) -> bool:
        return is_convex(self.cumulative)
