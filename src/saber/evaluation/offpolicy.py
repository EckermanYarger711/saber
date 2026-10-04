"""Off-policy evaluation of the closed-loop policy.

Ref: Sec. 3.7 -- "the closed-loop assessment is based on the use of previously
collected data regarding the trajectory and of the twin -- the evaluated policy
is tested in the twin, and the resulting value is estimated retrospectively on the
basis of data previously collected and utilizing a family of off-policy
estimators whose conformity with the value measured in the twin is in itself
obtained ... The successful categories of estimating are the doubly robust and
fitted-Q classes; the inverse of the propensity scoring method performs worse as
we move further away from the action distribution of the target policy from that
of the logging policy's predetermined pattern".

The estimators are implemented from their definitions on a logged batch with
behaviour and target propensities. The value model for the doubly-robust and
fitted-Q estimators is a linear least-squares fit, solved in closed form, so the
family is deterministic and its disagreement is a property of the estimators
rather than of an optimiser's schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

ESTIMATOR_NAMES: tuple[str, ...] = ("ips", "doubly_robust", "fitted_q")


@dataclass(frozen=True)
class LoggedBatch:
    """A logged batch: states, chosen actions, rewards and both propensities."""

    states: Array
    actions: IntArray
    rewards: Array
    behaviour_propensity: Array
    target_propensity: Array
    n_actions: int

    @property
    def size(self) -> int:
        return int(self.states.shape[0])

    def label(self) -> dict[str, int]:
        return {
            "records": self.size,
            "actions": self.n_actions,
            "state_dim": int(self.states.shape[1]),
        }


def importance_ratio(batch: LoggedBatch) -> Array:
    """``pi_t(a | s) / pi_b(a | s)`` per logged record.

    A record the behaviour policy could not have produced carries an infinite
    ratio in theory; it is clipped to zero here because a zero-propensity logged
    action contributes no evidence, which is the support mismatch the article
    attributes the propensity method's failure to.
    """
    ratio = np.where(
        batch.behaviour_propensity > 0.0,
        batch.target_propensity / np.maximum(batch.behaviour_propensity, 1e-12),
        0.0,
    )
    return np.asarray(ratio, dtype=np.float64)


def ips(batch: LoggedBatch) -> float:
    """Inverse propensity scoring: the importance-weighted mean reward."""
    weights = importance_ratio(batch)
    return float(np.mean(weights * batch.rewards))


def fit_q(batch: LoggedBatch, ridge: float = 1e-6) -> Array:
    """Closed-form ridge least squares for ``Q(s, a)`` with a one-hot action block.

    The design is ``[s, one_hot(a)]`` with an interaction between the state and
    the action block, so the fit can represent a different linear response per
    action rather than only an additive action offset.
    """
    states = np.asarray(batch.states, dtype=np.float64)
    one_hot = np.zeros((batch.size, batch.n_actions), dtype=np.float64)
    one_hot[np.arange(batch.size), np.asarray(batch.actions, dtype=int)] = 1.0
    interaction = states[:, :, None] * one_hot[:, None, :]
    design = np.concatenate([states, one_hot, interaction.reshape(batch.size, -1)], axis=1)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    ridge_coefficients = np.linalg.solve(gram, design.T @ batch.rewards)
    return np.asarray(ridge_coefficients, dtype=np.float64)


def q_values(coefficients: Array, states: Array, n_actions: int) -> Array:
    """``Q(s, a)`` for every action, one row per state."""
    states = np.asarray(states, dtype=np.float64)
    rows = states.shape[0]
    one_hot = np.eye(n_actions, dtype=np.float64)
    tiled = np.repeat(states, n_actions, axis=0)
    hot = np.tile(one_hot, (rows, 1))
    interaction = tiled[:, :, None] * hot[:, None, :]
    design = np.concatenate([tiled, hot, interaction.reshape(tiled.shape[0], -1)], axis=1)
    values = design @ coefficients
    return np.asarray(values.reshape(rows, n_actions), dtype=np.float64)


def doubly_robust(batch: LoggedBatch, coefficients: Array | None = None) -> float:
    """Doubly-robust estimate: importance-weighted residual plus the model's value."""
    model = fit_q(batch) if coefficients is None else coefficients
    values = q_values(model, batch.states, batch.n_actions)
    chosen = values[np.arange(batch.size), np.asarray(batch.actions, dtype=int)]
    weights = importance_ratio(batch)
    residual = batch.rewards - chosen
    model_value = np.sum(values * target_probabilities(batch), axis=1)
    return float(np.mean(weights * residual + model_value))


def fitted_q(batch: LoggedBatch, coefficients: Array | None = None) -> float:
    """Fitted-Q: the value model's own estimate under the target policy."""
    model = fit_q(batch) if coefficients is None else coefficients
    values = q_values(model, batch.states, batch.n_actions)
    return float(np.mean(np.sum(values * target_probabilities(batch), axis=1)))


def target_probabilities(batch: LoggedBatch) -> Array:
    """The target policy's action distribution per state, read from the batch.

    The logged target propensity is per record; the copy is to the state's row so
    the model-based estimators can weight every action by the target policy.
    """
    rows = batch.size
    distribution = np.zeros((rows, batch.n_actions), dtype=np.float64)
    chosen = np.asarray(batch.actions, dtype=int)
    distribution[np.arange(rows), chosen] = batch.target_propensity
    remainder = 1.0 - distribution.sum(axis=1, keepdims=True)
    uniform = np.full((rows, batch.n_actions), 1.0 / float(batch.n_actions))
    distribution = distribution + np.maximum(remainder, 0.0) * uniform
    return np.asarray(distribution / distribution.sum(axis=1, keepdims=True), dtype=np.float64)


@dataclass(frozen=True)
class EstimatorSuite:
    """Every estimator's value on one batch, plus the twin's measured value."""

    values: dict[str, float]
    twin_value: float

    def errors(self) -> dict[str, float]:
        return {
            name: relative_percentage_error(value, self.twin_value)
            for name, value in self.values.items()
        }

    def best(self) -> str:
        errors = self.errors()
        return min(errors, key=lambda name: errors[name])

    def label(self) -> dict[str, object]:
        return {
            "values": {name: round(value, 9) for name, value in self.values.items()},
            "twin_value": round(self.twin_value, 9),
            "relative_percentage_error": {
                name: round(value, 6) for name, value in self.errors().items()
            },
            "best": self.best(),
        }


def evaluate(batch: LoggedBatch, twin_value: float) -> EstimatorSuite:
    """Run the estimator family the article's Sec. 3.7 reports."""
    coefficients = fit_q(batch)
    return EstimatorSuite(
        values={
            "ips": ips(batch),
            "doubly_robust": doubly_robust(batch, coefficients),
            "fitted_q": fitted_q(batch, coefficients),
        },
        twin_value=twin_value,
    )


def relative_percentage_error(estimate: float, reference: float) -> float:
    """The article's read-out of estimator conformity, as a percentage."""
    if abs(reference) <= 1e-12:
        return 0.0
    return 100.0 * abs(estimate - reference) / abs(reference)
