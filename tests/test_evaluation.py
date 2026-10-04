"""The metrics, the statistical protocol and the off-policy family."""

from __future__ import annotations

import numpy as np

from saber.evaluation.metrics import (
    RunSummary,
    balanced_accuracy,
    effective_sample_size,
    noise_band,
    normalised_weights,
    r_squared,
    regime_sequence,
    retention,
    within_band,
)
from saber.evaluation.offpolicy import (
    ESTIMATOR_NAMES,
    LoggedBatch,
    doubly_robust,
    evaluate,
    fit_q,
    fitted_q,
    importance_ratio,
    ips,
    q_values,
    relative_percentage_error,
    target_probabilities,
)


def _batch(seed: int = 0, records: int = 60) -> LoggedBatch:
    generator = np.random.default_rng(seed)
    states = generator.normal(size=(records, 4))
    actions = generator.integers(0, 3, size=records)
    behaviour = np.full(records, 1.0 / 3.0)
    logits = np.full((records, 3), 0.1)
    logits[np.arange(records), actions] += 0.8
    target = np.exp(logits - logits.max(axis=1, keepdims=True))
    target = target / target.sum(axis=1, keepdims=True)
    rewards = 1.0 - 0.3 * np.abs(states[:, 0]) + generator.normal(scale=0.05, size=records)
    return LoggedBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        behaviour_propensity=behaviour,
        target_propensity=target[np.arange(records), actions],
        n_actions=3,
    )


def test_balanced_accuracy_is_mean_recall() -> None:
    predicted = np.asarray(["a", "a", "b", "b"], dtype=object)
    truth = np.asarray(["a", "b", "b", "b"], dtype=object)
    assert abs(balanced_accuracy(predicted, truth) - (1.0 + 2.0 / 3.0) / 2.0) <= 1e-12


def test_r_squared_is_one_for_a_perfect_prediction() -> None:
    observed = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert abs(r_squared(observed, observed) - 1.0) <= 1e-12


def test_r_squared_is_zero_for_the_mean_predictor() -> None:
    observed = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert abs(r_squared(np.full(4, float(observed.mean())), observed)) <= 1e-12


def test_retention_is_the_ratio_of_coefficients() -> None:
    observed = np.asarray([1.0, 2.0, 3.0])
    baseline = np.full(3, 2.0)
    assert retention(observed, observed, baseline) == 0.0


def test_effective_sample_size_bounds() -> None:
    assert abs(effective_sample_size(np.ones(5)) - 5.0) <= 1e-12
    dominated = np.asarray([1.0, 1.0, 1.0, 1.0, 1000.0])
    assert effective_sample_size(dominated) < 2.0


def test_normalised_weights_sum_to_one() -> None:
    weights = normalised_weights(np.asarray([1.0, 3.0]))
    assert abs(float(weights.sum()) - 1.0) <= 1e-12


def test_noise_band_is_twice_the_sample_deviation() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    assert abs(noise_band(values) - 2.0 * float(np.std(values, ddof=1))) <= 1e-12
    assert within_band(0.1, noise_band(values))


def test_run_summary_reports_the_runs() -> None:
    summary = RunSummary.of(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert summary.label()["runs"] == 5
    assert summary.mean == 3.0


def test_regime_sequence_maps_the_axes() -> None:
    axes = np.asarray([[0.2, 0.5, 0.2, 0.1], [0.9, 0.5, 0.1, 0.1]])
    labels = regime_sequence(axes)
    assert list(labels) == ["hypoxic", "normoxic-mono-culture"]


def test_importance_ratio_clips_an_impossible_record() -> None:
    batch = LoggedBatch(
        states=np.zeros((2, 4)),
        actions=np.asarray([0, 1]),
        rewards=np.asarray([1.0, 1.0]),
        behaviour_propensity=np.asarray([1.0, 0.0]),
        target_propensity=np.asarray([0.5, 0.5]),
        n_actions=2,
    )
    ratio = importance_ratio(batch)
    assert ratio[1] == 0.0
    assert ratio[0] == 0.5


def test_estimators_are_finite_and_named() -> None:
    batch = _batch()
    suite = evaluate(batch, 1.0)
    assert set(suite.values) == set(ESTIMATOR_NAMES)
    assert all(np.isfinite(value) for value in suite.values.values())
    assert suite.best() in ESTIMATOR_NAMES


def test_q_values_have_one_row_per_state() -> None:
    batch = _batch(records=12)
    coefficients = fit_q(batch)
    values = q_values(coefficients, batch.states, batch.n_actions)
    assert values.shape == (12, 3)


def test_target_probabilities_are_a_distribution() -> None:
    batch = _batch(records=8)
    probabilities = target_probabilities(batch)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_every_estimator_returns_a_finite_value_on_one_batch() -> None:
    batch = _batch(seed=3, records=40)
    coefficients = fit_q(batch)
    assert np.isfinite(doubly_robust(batch, coefficients))
    assert np.isfinite(fitted_q(batch, coefficients))
    assert np.isfinite(ips(batch))


def test_relative_percentage_error() -> None:
    assert abs(relative_percentage_error(1.1, 1.0) - 10.0) <= 1e-9
    assert relative_percentage_error(1.0, 0.0) == 0.0
