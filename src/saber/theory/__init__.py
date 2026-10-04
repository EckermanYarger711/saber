"""Closed-form analysis layer for the article's statements and its planner.

Nothing here calls the estimator or the planner; each routine recomputes its
quantity from the definition, which is what lets the checks in
``saber.verification.execution`` stand apart from the modules they verify.
"""

from saber.theory.certificate import (
    ExactSearch,
    SubmodularInstance,
    best_lagrangian_value,
    coverage_instance,
    exact_optimum,
    fractional_relaxation,
    greedy_ratio_order,
    greedy_uniform,
    lagrangian_greedy_value,
    uniform_bound,
    uniform_instance,
)
from saber.theory.information import (
    GaussianBelief,
    entropy,
    information_gain,
    marginal_information,
    mutual_information,
    observe,
)
from saber.theory.interior import (
    DamageCurve,
    ExposureLadder,
    convex_damage_curve,
    interior_optimum,
    is_convex,
    is_unimodal,
    rho_curve,
)
from saber.theory.parity import (
    EpisodeResult,
    ParityOutcome,
    budgeted_value,
    parity_test,
    rollout_episode,
    unconstrained_value,
)

__all__ = [
    "DamageCurve",
    "EpisodeResult",
    "ExactSearch",
    "ExposureLadder",
    "GaussianBelief",
    "ParityOutcome",
    "SubmodularInstance",
    "best_lagrangian_value",
    "budgeted_value",
    "convex_damage_curve",
    "coverage_instance",
    "entropy",
    "exact_optimum",
    "fractional_relaxation",
    "greedy_ratio_order",
    "greedy_uniform",
    "information_gain",
    "interior_optimum",
    "is_convex",
    "is_unimodal",
    "lagrangian_greedy",
    "lagrangian_greedy_value",
    "marginal_information",
    "mutual_information",
    "observe",
    "parity_test",
    "rho_curve",
    "rollout_episode",
    "uniform_bound",
    "uniform_instance",
    "unconstrained_value",
]
