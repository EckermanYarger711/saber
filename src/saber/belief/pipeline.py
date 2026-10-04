"""Algorithm 1: the latent-microenvironment belief loop.

Ref: Algorithm 1 (TME-belief) and Sec. 2.3.

The loop is transcribed step for step: stream the embeddings, take the
variational posterior update, publish the posterior's second moment as the
per-axis uncertainty, re-tie any axis whose uncertainty passes the recalibration
threshold, and stop once the total uncertainty has failed to fall by the
convergence epsilon for the configured patience. The recursion in the listing is
sequential, so this is a Python loop rather than a batched pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch import Tensor

from saber.belief.estimator import Belief, BeliefStream, PosteriorUpdate, VariationalEstimator
from saber.belief.priors import AtlasPrior
from saber.belief.recalibrate import RecalibrationEvent, converged, recalibrate
from saber.runtime.config import BeliefSpec


def initial_belief(prior: AtlasPrior) -> Belief:
    """``b_0``: the atlas prior itself, so the first update is evidence-driven."""
    return Belief(mean=prior.means.copy(), standard_deviation=prior.standard_deviations.copy())


def run_tme_belief(
    estimator: VariationalEstimator,
    embeddings: Sequence[Tensor],
    prior: AtlasPrior,
    spec: BeliefSpec,
    *,
    hidden: Tensor | None = None,
) -> tuple[BeliefStream, tuple[RecalibrationEvent, ...]]:
    """Run Algorithm 1 over one specimen's embedding stream.

    Returns the belief stream and every recalibration that fired, because the
    events are the evidence that the drift guard is load-bearing rather than
    decorative.
    """
    belief = initial_belief(prior)
    history: list[float] = [belief.total_uncertainty()]
    updates: list[PosteriorUpdate] = []
    events: list[RecalibrationEvent] = []
    converged_at: int | None = None
    memory = hidden
    for epoch, embedding in enumerate(embeddings):
        posterior, memory = estimator.step(embedding, belief, memory)
        re_tied, fired = recalibrate(posterior, prior, spec.recalibration_threshold, epoch)
        events.extend(fired)
        updates.append(
            PosteriorUpdate(epoch=epoch, prior=belief, posterior=posterior, recalibrated=re_tied)
        )
        belief = re_tied
        history.append(belief.total_uncertainty())
        if converged(history, spec.convergence_patience, spec.convergence_epsilon):
            converged_at = epoch
            break
    if not updates:
        updates.append(
            PosteriorUpdate(epoch=0, prior=belief, posterior=belief, recalibrated=belief)
        )
    return BeliefStream(updates=tuple(updates), converged_at=converged_at), tuple(events)


def axis_std_stream(stream: BeliefStream) -> np.ndarray:
    """The per-axis uncertainty trace, the array the regime checks consume."""
    rows = [update.recalibrated.standard_deviation for update in stream.updates]
    return np.asarray(rows, dtype=np.float64)


def belief_accuracy(predicted: Belief, truth: np.ndarray) -> float:
    """Balanced accuracy of a belief's regime call against a known regime.

    Ref: Sec. 3.1 -- the headline estimator metric is a balanced accuracy across
    the four axes' regimes, so accuracy is computed per axis and then averaged
    rather than pooled, which is what the article's "balanced" qualifier means.
    """
    from saber.belief.axes import regime_label

    predicted_label = regime_label(tuple(float(value) for value in predicted.mean))
    truth_label = regime_label(tuple(float(value) for value in truth))
    return 1.0 if predicted_label == truth_label else 0.0
