"""The minimal end-to-end loops: a fit that moves the parameters and a session that runs."""

from __future__ import annotations

import numpy as np
import torch

from saber.belief.elbo import BeliefDecoder, elbo_terms
from saber.belief.pipeline import initial_belief
from saber.cli.context import ReleaseContext
from saber.planner.rollout import logged_observer, run_policy_session
from saber.study import fit_estimator, front_end_output


def test_single_specimen_overfit_decreases_the_bound(context: ReleaseContext) -> None:
    fit = fit_estimator(context, context.specimens()[:1], steps=25, seed=11)
    assert fit.final_loss < fit.initial_loss
    assert fit.final_loss <= fit.initial_loss


def test_one_optimiser_step_moves_every_head(context: ReleaseContext) -> None:
    torch.manual_seed(11)
    estimator = context.build_estimator()
    decoder = BeliefDecoder(
        embedding_dim=context.perception_spec.embedding_dim,
        latent_dim=4,
        hidden_dim=context.belief_spec.hidden_dim,
    )
    before = [parameter.detach().clone() for parameter in estimator.parameters()]
    specimen = context.specimens()[0]
    with torch.no_grad():
        _output, stream = front_end_output(context, specimen)
    prior = context.prior
    mean, log_variance, _memory = estimator.forward(stream[0], initial_belief(prior), None)
    loss, _terms = elbo_terms(
        decoder,
        stream[0],
        stream[1],
        mean,
        log_variance,
        prior,
        kl_weight=context.belief_spec.kl_weight,
        generator=torch.Generator().manual_seed(11),
    )
    optimiser = torch.optim.Adam(list(estimator.parameters()) + list(decoder.parameters()), lr=5e-3)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, estimator.parameters())
    )
    assert all(parameter.grad is not None for parameter in estimator.parameters())


def test_session_smoke_runs_the_planner(context: ReleaseContext) -> None:
    observe = logged_observer(context.specimens()[0].axes, context.dish, 20260929)
    outcome = run_policy_session(
        "big",
        context.planner,
        context.action_set,
        context.space.actions,
        initial_belief(context.prior),
        context.budget.total,
        observe,
        seed=20260929,
    )
    assert outcome.exposures() >= 1
    assert outcome.damage > 0.0
    assert np.isfinite(outcome.information)


def test_smoke_session_is_cheaper_than_the_full_horizon(context: ReleaseContext) -> None:
    observe = logged_observer(context.specimens()[0].axes, context.dish, 20260929)
    outcome = run_policy_session(
        "entropy_greedy",
        context.planner,
        context.action_set,
        context.space.actions,
        initial_belief(context.prior),
        context.budget.total * 0.05,
        observe,
        seed=20260929,
    )
    assert outcome.damage <= context.budget.total * 0.05 + 1e-6
