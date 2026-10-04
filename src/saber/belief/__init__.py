"""Latent microenvironment state estimation.

Ref: Sec. 2.3 -- a repeating variational state-space model whose posterior
``q_psi(s_t | b_{t-1}, z_t)`` is updated into the belief ``b_t``, whose second
moment gives the per-axis uncertainty ``sigma_t``, with a recalibration threshold
that re-ties a drifting axis to its atlas prior. Algorithm 1 spells the loop out.
"""

from saber.belief.axes import AXIS_COUNT, AXIS_NAMES, axis_index, describe
from saber.belief.belief_value import BeliefValue, fit_belief_value
from saber.belief.elbo import VariationalLoss, elbo_terms, next_embedding_loss
from saber.belief.estimator import (
    Belief,
    BeliefStream,
    PosteriorUpdate,
    VariationalEstimator,
)
from saber.belief.pipeline import run_tme_belief
from saber.belief.priors import AtlasPrior, default_atlas_prior
from saber.belief.recalibrate import RecalibrationEvent, recalibrate

__all__ = [
    "AXIS_COUNT",
    "AXIS_NAMES",
    "AtlasPrior",
    "Belief",
    "BeliefStream",
    "BeliefValue",
    "PosteriorUpdate",
    "RecalibrationEvent",
    "VariationalEstimator",
    "VariationalLoss",
    "axis_index",
    "default_atlas_prior",
    "describe",
    "elbo_terms",
    "fit_belief_value",
    "next_embedding_loss",
    "recalibrate",
    "run_tme_belief",
]
