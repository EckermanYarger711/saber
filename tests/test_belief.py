"""The estimator, the variational bound, the drift guard and Algorithm 1."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from saber.belief.elbo import BeliefDecoder, elbo_terms, kl_to_prior, reparameterise
from saber.belief.estimator import Belief
from saber.belief.pipeline import initial_belief, run_tme_belief
from saber.belief.priors import default_atlas_prior, prior_from_means
from saber.belief.recalibrate import converged, drifting_axes, recalibrate
from saber.cli.context import ReleaseContext
from saber.study import fit_estimator, front_end_output


def test_initial_belief_is_the_atlas_prior(context: ReleaseContext) -> None:
    prior = default_atlas_prior()
    belief = initial_belief(prior)
    assert np.array_equal(belief.mean, prior.means)
    assert belief.dimension == 4


def test_recalibration_re_ties_only_the_uncertain_axis() -> None:
    prior = default_atlas_prior()
    belief = Belief(mean=prior.means.copy(), standard_deviation=prior.standard_deviations.copy())
    belief = Belief(mean=belief.mean, standard_deviation=np.asarray([9.0, 0.1, 0.1, 0.1]))
    re_tied, events = recalibrate(belief, prior, 1.0, epoch=3)
    assert len(events) == 1
    assert events[0].axis_name == "oxygen"
    assert re_tied.standard_deviation[0] == prior.standard_deviations[0]


def test_drifting_axis_is_detected_by_its_mean() -> None:
    prior = default_atlas_prior()
    belief = Belief(
        mean=prior.means + np.asarray([4.0, 0.0, 0.0, 0.0]) * prior.standard_deviations,
        standard_deviation=prior.standard_deviations.copy(),
    )
    assert 0 in drifting_axes(belief, prior)


def test_convergence_test_needs_the_patience() -> None:
    history = [1.0, 0.999, 0.998, 0.997, 0.996]
    assert not converged(history, patience=3, epsilon=1e-9)
    assert converged(history, patience=3, epsilon=0.01)


def test_estimator_shapes(context: ReleaseContext) -> None:
    estimator = context.build_estimator()
    prior = default_atlas_prior()
    embedding = torch.zeros(context.perception_spec.embedding_dim)
    mean, log_variance, hidden = estimator.forward(embedding, initial_belief(prior), None)
    assert mean.shape == (4,)
    assert log_variance.shape == (4,)
    assert hidden.shape == (context.belief_spec.hidden_dim,)


def test_kl_divergence_is_zero_at_the_prior() -> None:
    prior = default_atlas_prior()
    mean = torch.as_tensor(prior.means, dtype=torch.float32)
    log_variance = torch.as_tensor(np.log(prior.variances()), dtype=torch.float32)
    assert float(kl_to_prior(mean, log_variance, prior)) <= 1e-5


def test_reparameterisation_is_unbiased_in_the_mean() -> None:
    generator = torch.Generator().manual_seed(0)
    mean = torch.zeros(4)
    log_variance = torch.zeros(4)
    draws = torch.stack([reparameterise(mean, log_variance, generator) for _ in range(400)])
    assert float(torch.abs(draws.mean(dim=0)).max()) < 0.2


def test_variational_bound_decomposes(context: ReleaseContext) -> None:
    estimator = context.build_estimator()
    decoder = BeliefDecoder(
        embedding_dim=context.perception_spec.embedding_dim,
        latent_dim=4,
        hidden_dim=context.belief_spec.hidden_dim,
    )
    prior = default_atlas_prior()
    embedding = torch.zeros(context.perception_spec.embedding_dim)
    mean, log_variance, _ = estimator.forward(embedding, initial_belief(prior), None)
    total, terms = elbo_terms(
        decoder,
        embedding,
        embedding,
        mean,
        log_variance,
        prior,
        kl_weight=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert np.isfinite(float(total.detach()))
    assert terms.reconstruction >= 0.0
    assert abs(terms.total - float(total.detach())) <= 1e-9


def test_belief_loop_runs_over_the_stream(context: ReleaseContext) -> None:
    estimator = context.build_estimator()
    specimen = context.specimens()[0]
    with torch.no_grad():
        _output, stream = front_end_output(context, specimen)
    beliefs, events = run_tme_belief(
        estimator,
        list(torch.unbind(stream, dim=0)),
        default_atlas_prior(),
        context.belief_spec,
    )
    assert len(beliefs.updates) <= specimen.epochs
    assert beliefs.updates[0].recalibrated.dimension == 4
    assert isinstance(events, tuple)


def test_an_unreachable_epsilon_runs_the_whole_stream(context: ReleaseContext) -> None:
    estimator = context.build_estimator()
    specimen = context.specimens()[0]
    with torch.no_grad():
        _output, stream = front_end_output(context, specimen)
    strict = replace(context.belief_spec, convergence_epsilon=-1e9)
    loose = replace(context.belief_spec, convergence_epsilon=1e6)
    strict_stream, _ = run_tme_belief(
        estimator, list(torch.unbind(stream, dim=0)), default_atlas_prior(), strict
    )
    loose_stream, _ = run_tme_belief(
        estimator, list(torch.unbind(stream, dim=0)), default_atlas_prior(), loose
    )
    assert strict_stream.converged_at is None
    assert loose_stream.converged_at is not None
    assert len(loose_stream.updates) < len(strict_stream.updates)


def test_fit_lowers_the_variational_bound(context: ReleaseContext) -> None:
    fit = fit_estimator(context, context.specimens()[:1], steps=12, seed=20260929)
    assert fit.improved()


def test_prior_from_means_keeps_the_axis_count() -> None:
    prior = prior_from_means(np.asarray([0.1, 0.2, 0.3, 0.4]))
    assert prior.dimension == 4
    assert prior.variances().shape == (4,)
