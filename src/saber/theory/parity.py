"""Parity under an unbinding budget, and the separation once it binds.

Ref: Theorem 2, Eq. (5) --

    E[I^{pi_BIG}] <= E[I^{pi_unc}], with equality when the constraint is not
    active.

The two policies are built here as the article defines them: ``pi_unc``
maximises ``E[sum_t gamma^t G(a_t | b_t)]`` with no damage constraint, and
``pi_BIG`` maximises the Lagrangian objective of Eq. (7) subject to a running
damage budget. The test sweeps the budget from large to small and records the
deficit, which must be zero while the budget does not bind and must stay
non-negative once it does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.theory.information import GaussianBelief, information_gain, observe

Array = NDArray[np.float64]


@dataclass(frozen=True)
class EpisodeResult:
    """What one rollout of a policy realised."""

    info_gain: float
    damage: float
    steps: int
    actions: tuple[int, ...]
    uncertainty: tuple[float, ...]

    def label(self) -> dict[str, float | int]:
        return {
            "info_gain": round(self.info_gain, 9),
            "damage": round(self.damage, 9),
            "steps": self.steps,
        }


@dataclass(frozen=True)
class ParityOutcome:
    """The Theorem 2 sweep: one row per budget level."""

    budgets: tuple[float, ...]
    budgeted: tuple[float, ...]
    unconstrained: tuple[float, ...]

    @property
    def deficits(self) -> tuple[float, ...]:
        return tuple(
            unconstrained_value - budgeted_value
            for budgeted_value, unconstrained_value in zip(self.budgeted, self.unconstrained)
        )

    def unbinding_levels(self, tolerance: float = 1e-9) -> tuple[float, ...]:
        return tuple(
            budget
            for budget, deficit in zip(self.budgets, self.deficits)
            if abs(deficit) <= tolerance
        )

    def row_labels(self) -> list[dict[str, float]]:
        return [
            {
                "budget": round(budget, 6),
                "budgeted": round(budgeted, 6),
                "unconstrained": round(unconstrained, 6),
                "deficit": round(unconstrained - budgeted, 6),
            }
            for budget, budgeted, unconstrained in zip(
                self.budgets, self.budgeted, self.unconstrained
            )
        ]


def episode_belief(prior_precision: Array, dimension: int) -> GaussianBelief:
    """A belief whose covariance is the inverse of the supplied precision."""
    covariance = np.asarray(np.linalg.inv(prior_precision), dtype=np.float64)
    return GaussianBelief(mean=np.zeros(dimension, dtype=np.float64), covariance=covariance)


def design_pool(seed: int, dimension: int = 4, n_designs: int = 12) -> tuple[Array, Array]:
    """A seeded pool of observation designs with their per-action damage.

    A design's length is its aperture and its cost is the delivered dose, so the
    pool spans a range of information-per-unit-damage; a pool in which every
    action costs the same would make Theorem 2's conditional statement vacuous.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_designs, dimension))
    directions = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)
    aperture = rng.uniform(0.3, 1.2, size=n_designs)
    designs = np.asarray(directions * aperture[:, None], dtype=np.float64)
    costs = np.asarray(0.5 + 1.5 * aperture, dtype=np.float64)
    return designs, costs


def rollout_episode(
    designs: Array,
    costs: Array,
    prior_precision: Array,
    noise_variance: float,
    horizon: int,
    discount: float,
    budget: float,
    *,
    constrained: bool,
    dual_step: float = 0.05,
    saturation_threshold: float = 1e-4,
) -> EpisodeResult:
    """Run one policy over a fixed horizon.

    The budgeted branch keeps the running dual variable of Eq. (7); the
    unconstrained branch is the same loop with the feasible set widened to the
    whole pool, which is the only difference Theorem 2 turns on.
    """
    dimension = prior_precision.shape[0]
    belief = episode_belief(prior_precision, dimension)
    remaining = budget
    multiplier = 0.0
    spent = 0.0
    total = 0.0
    chosen: list[int] = []
    uncertainties: list[float] = []
    for step in range(horizon):
        gains = np.array(
            [
                information_gain(belief, designs[index], noise_variance)
                for index in range(designs.shape[0])
            ],
            dtype=np.float64,
        )
        if constrained:
            feasible = np.flatnonzero(costs <= remaining + 1e-12)
            if feasible.size == 0:
                break
            scores = gains[feasible] - multiplier * costs[feasible]
            if float(np.max(gains[feasible])) < saturation_threshold:
                break
            best = int(feasible[int(np.argmax(scores))])
        else:
            if float(np.max(gains)) < saturation_threshold:
                break
            best = int(np.argmax(gains))
        gain = float(gains[best])
        total += (discount**step) * gain
        spent += float(costs[best])
        remaining = max(0.0, remaining - float(costs[best]))
        chosen.append(best)
        if constrained:
            multiplier = max(0.0, multiplier + dual_step * (spent - budget))
        belief = observe(belief, designs[best], noise_variance)
        uncertainties.append(float(np.sqrt(np.trace(belief.covariance))))
    return EpisodeResult(
        info_gain=total,
        damage=spent,
        steps=len(chosen),
        actions=tuple(chosen),
        uncertainty=tuple(uncertainties),
    )


def unconstrained_value(
    designs: Array,
    costs: Array,
    prior_precision: Array,
    noise_variance: float,
    horizon: int,
    discount: float,
) -> EpisodeResult:
    """``pi_unc``: information-greedy with no damage constraint."""
    return rollout_episode(
        designs,
        costs,
        prior_precision,
        noise_variance,
        horizon,
        discount,
        float("inf"),
        constrained=False,
    )


def budgeted_value(
    designs: Array,
    costs: Array,
    prior_precision: Array,
    noise_variance: float,
    horizon: int,
    discount: float,
    budget: float,
    dual_step: float = 0.05,
) -> EpisodeResult:
    """``pi_BIG``: the Lagrangian-relaxed planner of Eq. (7) and Algorithm 2."""
    return rollout_episode(
        designs,
        costs,
        prior_precision,
        noise_variance,
        horizon,
        discount,
        budget,
        constrained=True,
        dual_step=dual_step,
    )


def parity_test(
    seed: int,
    budgets: Sequence[float],
    dimension: int = 4,
    n_designs: int = 12,
    horizon: int = 12,
    discount: float = 0.95,
    noise_variance: float = 0.35,
    unbinding_headroom: float = 1.05,
) -> ParityOutcome:
    """Evaluate both policies across the budget sweep of Theorem 2.

    The sweep always begins with the budget level at which the constraint cannot
    bind -- the free policy's own realised damage plus a margin -- because that
    level is where the theorem asserts equality and it is not knowable in
    advance from the sweeps alone.
    """
    designs, costs = design_pool(seed, dimension=dimension, n_designs=n_designs)
    prior_precision = np.eye(dimension, dtype=np.float64) * 0.25
    free = unconstrained_value(designs, costs, prior_precision, noise_variance, horizon, discount)
    levels = [free.damage * unbinding_headroom, *[float(budget) for budget in budgets]]
    budgeted: list[float] = []
    for budget in levels:
        outcome = budgeted_value(
            designs,
            costs,
            prior_precision,
            noise_variance,
            horizon,
            discount,
            float(budget),
        )
        budgeted.append(outcome.info_gain)
    return ParityOutcome(
        budgets=tuple(float(level) for level in levels),
        budgeted=tuple(budgeted),
        unconstrained=tuple(free.info_gain for _ in levels),
    )
