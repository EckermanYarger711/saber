"""The constant-factor guarantee of the greedy selection.

Ref: Proposition 1 -- "If the expected information gain is monotone submodular
with respect to the action multiset chosen by the agent and there is a linear
damage constraint, then the greedy action selection based on Lagrangian
relaxation guarantees the agent has an expected value of at least
``(1 - 1/e)`` of the optimum solution to the relaxed problem." Corollary 1 adds
that with a constant ``C(a)`` the bound holds with a factor independent of the
horizon ``T``.

The article's gain over a set of facets is modelled as a weighted coverage
function: it is monotone, it is submodular, and its marginal gain of adding one
more exposure to an already-observed facet is zero, which is exactly the
"increasing redundancy" of Theorem 1's justification. The relaxed optimum is
computed with an LP relaxation rather than with a second greedy, so the
comparison does not rest on the same approximation twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SubmodularInstance:
    """A monotone submodular coverage instance with per-action damage costs."""

    coverage: tuple[frozenset[int], ...]
    costs: Array
    facets: int

    @property
    def n_actions(self) -> int:
        return len(self.coverage)

    def value(self, selected: tuple[int, ...]) -> float:
        covered: set[int] = set()
        for index in selected:
            covered |= self.coverage[index]
        return float(len(covered))

    def marginal(self, selected: tuple[int, ...], candidate: int) -> float:
        covered: set[int] = set()
        for index in selected:
            covered |= self.coverage[index]
        return float(len(self.coverage[candidate] - covered))

    def is_monotone(self) -> bool:
        """Adding an action never lowers the value, checked over all subsets."""
        for size in range(self.n_actions + 1):
            for subset in combinations(range(self.n_actions), size):
                for candidate in range(self.n_actions):
                    if candidate in subset:
                        continue
                    if self.marginal(subset, candidate) < -1e-12:
                        return False
        return True

    def is_submodular(self) -> bool:
        """Check ``f(S + a) - f(S) >= f(T + a) - f(T)`` for ``S subset T``."""
        indices = range(self.n_actions)
        for candidate in indices:
            for size in range(self.n_actions):
                for small in combinations(indices, size):
                    for extra in indices:
                        if extra in small or extra == candidate:
                            continue
                        larger = tuple(sorted({*small, extra}))
                        if (
                            self.marginal(tuple(small), candidate)
                            < self.marginal(larger, candidate) - 1e-12
                        ):
                            return False
        return True


@dataclass(frozen=True)
class ExactSearch:
    """Exact and relaxed optima of one instance, with the greedy values."""

    budget: float
    greedy_value: float
    exact_value: float
    relaxed_value: float
    greedy_actions: tuple[int, ...]

    def ratio_exact(self) -> float:
        return self.greedy_value / self.exact_value if self.exact_value > 0 else 0.0

    def ratio_relaxed(self) -> float:
        return self.greedy_value / self.relaxed_value if self.relaxed_value > 0 else 0.0

    def label(self) -> dict[str, float]:
        return {
            "budget": round(self.budget, 9),
            "greedy": round(self.greedy_value, 9),
            "exact": round(self.exact_value, 9),
            "relaxed": round(self.relaxed_value, 9),
            "ratio_relaxed": round(self.ratio_relaxed(), 9),
        }


def coverage_instance(
    seed: int,
    n_actions: int = 14,
    n_facets: int = 24,
    min_cost: float = 0.5,
    max_cost: float = 3.0,
    support: int = 6,
) -> SubmodularInstance:
    """A seeded coverage instance: each action covers a small facet set."""
    rng = np.random.default_rng(seed)
    coverage = tuple(
        frozenset(int(facet) for facet in rng.choice(n_facets, size=support, replace=False))
        for _ in range(n_actions)
    )
    costs = rng.uniform(min_cost, max_cost, size=n_actions)
    return SubmodularInstance(coverage=coverage, costs=np.asarray(costs), facets=n_facets)


def uniform_instance(instance: SubmodularInstance, cost: float = 1.0) -> SubmodularInstance:
    """The same coverage with a constant per-action damage, as in Corollary 1."""
    return SubmodularInstance(
        coverage=instance.coverage,
        costs=np.full(instance.n_actions, cost, dtype=np.float64),
        facets=instance.facets,
    )


def greedy_uniform(instance: SubmodularInstance, budget: float) -> tuple[int, ...]:
    """Cardinality greedy: repeatedly take the largest absolute marginal gain."""
    remaining = budget
    selected: list[int] = []
    available = list(range(instance.n_actions))
    while available:
        feasible = [index for index in available if instance.costs[index] <= remaining + 1e-12]
        if not feasible:
            break
        gains = [(instance.marginal(tuple(selected), index), index) for index in feasible]
        gain, choice = max(gains, key=lambda pair: (pair[0], -pair[1]))
        if gain <= 0.0:
            break
        selected.append(choice)
        remaining -= float(instance.costs[choice])
        available.remove(choice)
    return tuple(selected)


def greedy_ratio_order(instance: SubmodularInstance, budget: float) -> tuple[int, ...]:
    """Ratio greedy: repeatedly take the largest marginal gain per unit damage."""
    remaining = budget
    selected: list[int] = []
    available = list(range(instance.n_actions))
    while available:
        feasible = [
            index
            for index in available
            if instance.costs[index] <= remaining + 1e-12 and instance.costs[index] > 0.0
        ]
        if not feasible:
            break
        ratios = [
            (instance.marginal(tuple(selected), index) / float(instance.costs[index]), index)
            for index in feasible
        ]
        ratio, choice = max(ratios, key=lambda pair: (pair[0], -pair[1]))
        if ratio <= 0.0:
            break
        selected.append(choice)
        remaining -= float(instance.costs[choice])
        available.remove(choice)
    return tuple(selected)


def lagrangian_greedy_value(
    instance: SubmodularInstance, budget: float, multiplier: float
) -> float:
    """Value of the greedy run on the Lagrangian objective ``f - lambda * C``.

    The selection order is the ratio order, which is the Lagrangian order for a
    linear damage constraint; ``multiplier`` only decides when to stop.
    """
    remaining = budget
    selected: list[int] = []
    available = list(range(instance.n_actions))
    while available:
        feasible = [
            index
            for index in available
            if instance.costs[index] <= remaining + 1e-12 and instance.costs[index] > 0.0
        ]
        if not feasible:
            break
        scored = [
            (
                instance.marginal(tuple(selected), index) / float(instance.costs[index])
                - multiplier,
                index,
            )
            for index in feasible
        ]
        score, choice = max(scored, key=lambda pair: (pair[0], -pair[1]))
        if score <= 0.0:
            break
        selected.append(choice)
        remaining -= float(instance.costs[choice])
        available.remove(choice)
    return instance.value(tuple(selected))


def best_lagrangian_value(
    instance: SubmodularInstance, budget: float, multipliers: Array | None = None
) -> tuple[float, float]:
    """Best greedy value over a multiplier grid, with the multiplier that won.

    The multiplier is the Lagrangian dual variable of the damage constraint, so
    scanning it is the selection step of Proposition 1 rather than a second
    heuristic.
    """
    grid = (
        np.asarray(multipliers, dtype=np.float64)
        if multipliers is not None
        else np.linspace(0.0, 0.5, 26)
    )
    scored = [
        (lagrangian_greedy_value(instance, budget, float(multiplier)), float(multiplier))
        for multiplier in grid
    ]
    value, multiplier = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return value, multiplier


def exact_optimum(
    instance: SubmodularInstance, budget: float, max_size: int | None = None
) -> tuple[int, ...]:
    """Brute-force best feasible subset; the instance sizes here are small."""
    limit = max_size if max_size is not None else instance.n_actions
    best: tuple[int, ...] = ()
    best_value = 0.0
    for size in range(1, min(limit, instance.n_actions) + 1):
        for subset in combinations(range(instance.n_actions), size):
            cost = float(sum(instance.costs[index] for index in subset))
            if cost > budget + 1e-9:
                continue
            value = instance.value(subset)
            if value > best_value:
                best_value = value
                best = subset
    return best


def fractional_relaxation(instance: SubmodularInstance, budget: float) -> float:
    """LP relaxation of maximum coverage under a linear damage constraint.

    ``max sum_j y_j`` subject to ``y_j <= sum_{a covers j} x_a``, ``y_j <= 1``,
    ``sum_a c_a x_a <= budget``, ``x, y >= 0``. The polytope contains every
    integral feasible solution, so its optimum bounds the relaxed problem of
    Proposition 1 from above.
    """
    from scipy.optimize import linprog

    n = instance.n_actions
    m = instance.facets
    objective = -np.concatenate([np.zeros(n), np.ones(m)])
    rows: list[Array] = []
    bounds_rhs: list[float] = []
    for facet in range(m):
        row = np.zeros(n + m)
        row[:n] = [-1.0 if facet in instance.coverage[index] else 0.0 for index in range(n)]
        row[n + facet] = 1.0
        rows.append(row)
        bounds_rhs.append(0.0)
    for facet in range(m):
        row = np.zeros(n + m)
        row[n + facet] = 1.0
        rows.append(row)
        bounds_rhs.append(1.0)
    constraint = np.vstack(rows)
    damage_row = np.concatenate([instance.costs, np.zeros(m)])
    constraint = np.vstack([constraint, damage_row])
    bounds_rhs.append(float(budget))
    result = linprog(
        objective,
        A_ub=constraint,
        b_ub=np.asarray(bounds_rhs, dtype=np.float64),
        bounds=[(0.0, None)] * (n + m),
        method="highs",
    )
    if not result.success:
        return float("inf")
    return float(-result.fun)


def uniform_bound(
    seed: int, horizons: tuple[int, ...] = (4, 8, 16), n_actions: int = 14
) -> dict[int, float]:
    """Greedy-to-optimum ratios at increasing horizons under uniform damage.

    Ref: Corollary 1 -- with a constant ``C(a)`` the factor is independent of the
    horizon ``T``. The action set grows with the horizon here (each epoch adds
    fresh facets), so a ratio that stays put is the corollary's content.
    """
    ratios: dict[int, float] = {}
    for horizon in horizons:
        instance = coverage_instance(seed + horizon, n_actions=n_actions, n_facets=6 * horizon)
        uniform = uniform_instance(instance)
        budget = float(horizon - 1) * 1.0
        greedy = instance.value(greedy_uniform(uniform, budget))
        optimum = instance.value(exact_optimum(uniform, budget))
        ratios[horizon] = greedy / optimum if optimum > 0 else 0.0
    return ratios
