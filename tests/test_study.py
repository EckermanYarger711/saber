"""The session-level studies: perception, belief, acquisition, frontier, twin, transfer."""

from __future__ import annotations

import numpy as np

from saber.cli.context import ReleaseContext
from saber.study import (
    COMPARED_POLICIES,
    fit_value_function,
    linear_fit,
    run_frontier_study,
    run_perception_study,
)


def test_perception_study_reports_instances(context: ReleaseContext) -> None:
    study = run_perception_study(context)
    assert study.specimens == context.cohort.size()
    assert study.instances > 0
    assert study.trainable_parameters > 0


def test_belief_study_scores_every_specimen(context: ReleaseContext, belief_study: object) -> None:
    study = belief_study
    assert study.specimens == context.cohort.size()
    assert 0.0 <= study.accuracy <= 1.0
    assert len(study.streams) == context.cohort.size()


def test_acquisition_study_covers_every_policy(
    context: ReleaseContext, acquisition: tuple[object, ...]
) -> None:
    outcomes = acquisition
    assert tuple(outcome.policy for outcome in outcomes) == COMPARED_POLICIES
    pooled_budget = context.budget.total * context.cohort.size()
    for outcome in outcomes:
        assert outcome.damage <= pooled_budget + 1e-6
        assert len(outcome.per_specimen_accuracy) == context.cohort.size()


def test_frontier_regresses_the_strata(
    context: ReleaseContext, acquisition: tuple[object, ...]
) -> None:
    frontier = run_frontier_study(acquisition, context)
    assert len(frontier.strata) == len(context.cohort.strata())
    rates = [entry.transition_rate for entry in frontier.strata]
    assert rates == sorted(rates)


def test_linear_fit_recovers_a_known_line() -> None:
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    y = 3.0 * x + 1.0
    slope, intercept, r2, interval = linear_fit(x, y)
    assert abs(slope - 3.0) <= 1e-9
    assert abs(intercept - 1.0) <= 1e-9
    assert abs(r2 - 1.0) <= 1e-9
    assert interval[0] <= slope <= interval[1]


def test_twin_study_reports_fidelity(context: ReleaseContext, twin_study: object) -> None:
    study = twin_study
    assert study.fidelity.biomass_monotone
    assert study.fidelity.oxygen_non_negative
    assert set(study.suite.values) == {"ips", "doubly_robust", "fitted_q"}


def test_value_function_fit_improves(context: ReleaseContext, twin_study: object) -> None:
    fit = fit_value_function(context, twin_study)
    assert fit.epochs > 0
    assert fit.final_loss >= 0.0
