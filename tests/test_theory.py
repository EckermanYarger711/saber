"""The theory layer: Theorem 1, Proposition 1, Corollary 1 and Theorem 2."""

from __future__ import annotations

import numpy as np

from saber.theory import certificate, interior, parity
from saber.theory.information import (
    GaussianBelief,
    information_gain,
    marginal_information,
    repeated_precision_information,
)

GUARANTEE = 1.0 - 1.0 / np.e


def test_gaussian_information_matches_the_scalar_expression() -> None:
    variance = np.asarray([0.04, 0.09, 0.16, 0.01])
    design = np.asarray([0.3, 0.5, 0.2, 0.4])
    noise = 0.25
    belief = GaussianBelief(mean=np.zeros(4), covariance=np.diag(variance))
    closed_form = 0.5 * np.log(1.0 + float(np.sum(design**2 * variance)) / noise)
    assert abs(information_gain(belief, design, noise) - closed_form) <= 1e-12


def test_marginal_information_falls_once_the_design_saturates() -> None:
    precision = np.diag([4.0, 4.0, 4.0, 4.0])
    per_exposure = np.diag([0.5, 0.5, 0.5, 0.5])
    ladder = [marginal_information(precision, per_exposure, index) for index in range(8)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(ladder, ladder[1:]))


def test_accumulated_information_is_concave_in_the_precision() -> None:
    precision = np.diag([2.0, 3.0, 4.0, 5.0])
    per_exposure = np.diag([0.7, 0.4, 0.9, 0.2])
    curve = repeated_precision_information(precision, per_exposure, np.arange(1.0, 12.0))
    assert not interior.is_convex(curve)
    assert interior.is_increasing(curve)


def test_adaptation_phase_makes_the_marginal_unimodal() -> None:
    counts = np.arange(1.0, 25.0)
    marginal = interior.adapt_then_redundant(counts, 1.0, 2.0, 8.0)
    assert interior.is_unimodal(marginal)
    assert marginal[0] < np.max(marginal)


def test_convex_damage_curve_is_increasing_and_convex() -> None:
    curve = interior.DamageCurve.build(20, base=0.3, curvature=0.05)
    assert curve.increasing()
    assert curve.convex()


def test_currency_is_maximised_inside_the_ladder() -> None:
    counts = np.arange(1.0, 25.0)
    marginal = interior.adapt_then_redundant(counts, 1.0, 2.0, 8.0)
    damage = interior.convex_damage_curve(counts, 0.3, 0.05)
    ladder = interior.build_ladder(marginal, damage)
    assert ladder.interior()
    assert ladder.label()["n_star"] > 1


def test_monotone_marginal_pushes_the_optimum_to_the_boundary() -> None:
    counts = np.arange(1.0, 25.0)
    marginal = np.linspace(1.0, 2.0, counts.size)
    damage = interior.convex_damage_curve(counts, 0.3, 0.05)
    ladder = interior.build_ladder(marginal, damage)
    assert not ladder.interior()


def test_optimal_cadence_varies_with_the_regime() -> None:
    counts = np.arange(1.0, 25.0)
    damage = interior.convex_damage_curve(counts, 0.3, 0.05)
    optima = set()
    for rise, decay in ((1.0, 12.0), (3.0, 4.0), (1.5, 9.0)):
        marginal = interior.adapt_then_redundant(counts, 1.0, rise, decay)
        optima.add(interior.build_ladder(marginal, damage).argmax_currency())
    assert len(optima) > 1


def test_coverage_instance_is_monotone_and_submodular() -> None:
    instance = certificate.coverage_instance(1, n_actions=10, n_facets=16)
    assert instance.is_monotone()
    assert instance.is_submodular()


def test_cardinality_greedy_meets_the_guarantee() -> None:
    for seed in range(6):
        instance = certificate.coverage_instance(seed, n_actions=12, n_facets=20)
        uniform = certificate.uniform_instance(instance, 1.0)
        selected = certificate.greedy_uniform(uniform, 4.0)
        best = certificate.exact_optimum(uniform, 4.0)
        assert instance.value(selected) >= GUARANTEE * instance.value(best) - 1e-9


def test_lagrangian_greedy_meets_the_relaxed_bound() -> None:
    for seed in range(6):
        instance = certificate.coverage_instance(seed, n_actions=12, n_facets=20)
        budget = 5.0
        value, _multiplier = certificate.best_lagrangian_value(instance, budget)
        ratio_greedy = instance.value(certificate.greedy_ratio_order(instance, budget))
        relaxed = certificate.fractional_relaxation(instance, budget)
        assert max(value, ratio_greedy) >= GUARANTEE * relaxed - 1e-9


def test_uniform_damage_guarantee_is_horizon_independent() -> None:
    ratios = certificate.uniform_bound(31)
    assert all(ratio >= GUARANTEE - 1e-9 for ratio in ratios.values())


def test_theorem_two_parity_and_separation() -> None:
    outcome = parity.parity_test(7, [10.0, 6.0, 4.0, 2.0])
    assert all(deficit >= -1e-9 for deficit in outcome.deficits)
    assert len(outcome.unbinding_levels()) >= 1
    assert abs(outcome.deficits[0]) <= 1e-9
