"""The planner: the gain table, the currency, the dual update and the session loop."""

from __future__ import annotations

import numpy as np

from saber.belief.estimator import Belief
from saber.belief.pipeline import initial_belief
from saber.cli.context import ReleaseContext
from saber.planner.big import BigPlanner, candidates_for, plan_session
from saber.planner.currency import currency, currency_per_unit_damage, currency_vector
from saber.planner.info_gain import design_vector, observation_noise
from saber.planner.lagrangian import DualState, charge, dual_update, lagrangian_score
from saber.planner.policies import POLICY_NAMES, build_policy
from saber.planner.rollout import logged_observer, run_policy_session
from saber.theory.information import GaussianBelief, information_gain


def _reference_gain(belief: Belief, design: np.ndarray, noise: float) -> float:
    gaussian = GaussianBelief(mean=belief.mean.copy(), covariance=np.diag(belief.variance()))
    return float(information_gain(gaussian, design, noise))


def test_gain_table_matches_the_matrix_form(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    for index in np.linspace(0, context.action_set.size - 1, 6).astype(int):
        design = context.action_set.encodings.designs[index]
        noise = float(context.action_set.encodings.noises[index])
        assert (
            abs(
                context.action_set.encodings.gain(int(index), belief)
                - _reference_gain(belief, design, noise)
            )
            <= 1e-12
        )


def test_posterior_contraction_matches_the_matrix_form(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    for index in np.linspace(0, context.action_set.size - 1, 6).astype(int):
        design = context.action_set.encodings.designs[index]
        noise = float(context.action_set.encodings.noises[index])
        gaussian = GaussianBelief(mean=belief.mean.copy(), covariance=np.diag(belief.variance()))
        from saber.theory.information import observe

        expected = np.diag(observe(gaussian, design, noise).covariance)
        observed = context.action_set.encodings.posterior(belief, int(index)).variance()
        assert float(np.max(np.abs(expected - observed))) <= 1e-12


def test_short_exposures_are_noisier(context: ReleaseContext) -> None:
    short = context.space.actions[0]
    long_one = next(
        action
        for action in context.space.actions
        if action.channel == 0
        and action.exposure_ms == context.dish.exposure_levels_ms[-1]
        and action.depth_um == 0.0
    )
    assert observation_noise(short, context.dish) > observation_noise(long_one, context.dish)


def test_design_vector_is_bounded_by_the_modality(context: ReleaseContext) -> None:
    action = context.space.actions[0]
    design = design_vector(action, context.dish)
    assert design.shape == (4,)
    assert float(np.max(design)) < 1.0


def test_currency_is_gain_over_damage(context: ReleaseContext) -> None:
    action = context.functional.actions[64]
    gain = 0.25
    assert (
        abs(currency(gain, action, context.functional) - gain / context.functional.of(action))
        <= 1e-12
    )
    assert currency_per_unit_damage(gain, 0.0) == 0.0


def test_currency_vector_marks_a_free_action(context: ReleaseContext) -> None:
    gains = np.asarray([0.1, 0.2])
    costs = np.asarray([0.0, 0.5])
    values = currency_vector(gains, costs)
    assert np.isinf(values[0])
    assert abs(values[1] - 0.4) <= 1e-12


def test_dual_update_projects_onto_the_non_negative_orthant() -> None:
    state = DualState.initial(0.0)
    raised = dual_update(charge(state, 2.0), budget=1.0, step=0.5)
    assert raised.multiplier > 0.0
    lowered = dual_update(charge(DualState.initial(1.0), 0.0), budget=5.0, step=0.5)
    assert lowered.multiplier == 0.0


def test_lagrangian_score_subtracts_the_price() -> None:
    assert lagrangian_score(1.0, 2.0, 0.25) == 0.5
    assert lagrangian_score(1.0, 2.0, 0.25, continuation=0.5) == 1.0


def test_candidates_are_restricted_to_one_well(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    candidates = candidates_for(context.action_set, belief, context.budget.total, 32, well=0)
    wells = {context.space.actions[int(index)].well for index in candidates.indices}
    assert wells == {0}
    assert candidates.size <= context.action_set.well_slices[0].size


def test_every_named_policy_builds() -> None:
    for name in POLICY_NAMES:
        assert callable(build_policy(name))


def test_every_named_policy_runs_a_session(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    observe = logged_observer(context.specimens()[0].axes, context.dish, 20260929)
    for name in POLICY_NAMES:
        outcome = run_policy_session(
            name,
            context.planner,
            context.action_set,
            context.space.actions,
            belief,
            context.budget.total,
            observe,
            seed=20260929,
            update_dual=name == "big",
        )
        assert outcome.damage <= context.budget.total + 1e-6
        assert np.isfinite(outcome.information)


def test_planner_reports_an_algorithm_two_trace(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    observe = logged_observer(context.specimens()[0].axes, context.dish, 20260929)
    trace = plan_session(context.planner, context.action_set, belief, context.budget.total, observe)
    assert len(trace.steps) <= context.planner.horizon
    assert trace.damage <= context.budget.total + 1e-6
    assert trace.stopped in {"horizon", "budget", "saturation", "infeasible"}


def test_planner_records_the_feasible_set_size(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    planner = BigPlanner.build(
        context.planner, context.action_set, context.budget.total, None, 20260929
    )
    index, _score, feasible = planner.select(belief, context.budget.total, 0, 0)
    assert index >= 0
    assert feasible > 0


def test_session_is_reproducible(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    specimen = context.specimens()[0]
    first = run_policy_session(
        "big",
        context.planner,
        context.action_set,
        context.space.actions,
        belief,
        context.budget.total,
        logged_observer(specimen.axes, context.dish, 1),
        seed=1,
    )
    second = run_policy_session(
        "big",
        context.planner,
        context.action_set,
        context.space.actions,
        belief,
        context.budget.total,
        logged_observer(specimen.axes, context.dish, 1),
        seed=1,
    )
    assert first.information == second.information
    assert first.damage == second.damage
