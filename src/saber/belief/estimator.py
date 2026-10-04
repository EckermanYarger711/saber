"""The variational state-space estimator.

Ref: Sec. 2.3 -- "The estimator can be viewed as a repeating variational
state-space model. We can calculate the posterior approximation
``q_psi(s_t | b_{t-1}, z_t)``, which then gets modified to the belief ``b_t`` in
view of the beliefs ``b_{t-1}`` and the embedding ``z_t``, then the second moment
of the posterior provides a proper uncertainty value ``sigma_t`` for each axis."

The posterior is diagonal Gaussian over the four axes, with the recurrent state
carrying the previous belief forward: the update takes the previous belief's mean
and standard deviation, concatenates them with the current embedding, and emits
the new posterior. Making the previous belief an explicit input is what gives the
"repeating" state-space structure rather than a per-epoch regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from saber.belief.axes import AXIS_COUNT

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Belief:
    """Belief mean and axis standard deviations at one epoch."""

    mean: Array
    standard_deviation: Array

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def variance(self) -> Array:
        return np.asarray(self.standard_deviation**2, dtype=np.float64)

    def total_uncertainty(self) -> float:
        return float(np.sum(self.standard_deviation))

    def to_tensor(self) -> Tensor:
        return torch.as_tensor(
            np.concatenate([self.mean, self.standard_deviation]), dtype=torch.float32
        )

    def label(self) -> dict[str, list[float] | float]:
        return {
            "mean": [round(float(value), 9) for value in self.mean],
            "standard_deviation": [round(float(value), 9) for value in self.standard_deviation],
            "total_uncertainty": round(self.total_uncertainty(), 9),
        }


@dataclass(frozen=True)
class PosteriorUpdate:
    """One epoch's posterior before and after recalibration."""

    epoch: int
    prior: Belief
    posterior: Belief
    recalibrated: Belief


@dataclass(frozen=True)
class BeliefStream:
    """The calibrated belief stream Algorithm 1 returns."""

    updates: tuple[PosteriorUpdate, ...]
    converged_at: int | None

    def beliefs(self) -> tuple[Belief, ...]:
        return tuple(update.recalibrated for update in self.updates)

    def final(self) -> Belief:
        return self.updates[-1].recalibrated

    def uncertainty_trace(self) -> Array:
        return np.asarray(
            [belief.total_uncertainty() for belief in self.beliefs()], dtype=np.float64
        )

    def label(self) -> dict[str, object]:
        return {
            "epochs": len(self.updates),
            "converged_at": self.converged_at,
            "final": self.final().label(),
            "recalibrations": sum(
                1
                for update in self.updates
                if not np.allclose(
                    update.posterior.standard_deviation,
                    update.recalibrated.standard_deviation,
                )
            ),
        }


class VariationalEstimator(nn.Module):
    """The encoder half of the state-space model: ``q_psi(s_t | b_{t-1}, z_t)``."""

    def __init__(
        self, embedding_dim: int, latent_dim: int, hidden_dim: int, axes: int = AXIS_COUNT
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.axes = axes
        self.embedding_dim = embedding_dim
        self.recurrent = nn.GRUCell(input_size=embedding_dim + 2 * axes, hidden_size=hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.to_mean = nn.Linear(latent_dim, axes)
        self.to_log_variance = nn.Linear(latent_dim, axes)

    def forward(
        self, embedding: Tensor, previous: Belief, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Posterior mean and log-variance for one epoch, with the next memory.

        The returned hidden state is the recurrent memory the next epoch starts
        from; the caller threads it so the sequence is processed causally.
        """
        joined: Tensor = torch.cat(
            [embedding, torch.as_tensor(previous.to_tensor(), dtype=embedding.dtype)]
        ).unsqueeze(0)
        memory = self.initial_state(embedding) if hidden is None else hidden.unsqueeze(0)
        updated: Tensor = self.recurrent(joined, memory)
        latent: Tensor = torch.tanh(self.to_latent(updated))
        mean: Tensor = self.to_mean(latent)
        log_variance: Tensor = torch.clamp(self.to_log_variance(latent), min=-12.0, max=6.0)
        return mean.squeeze(0), log_variance.squeeze(0), updated.squeeze(0)

    def initial_state(self, embedding: Tensor) -> Tensor:
        """A zero recurrent memory of the module's width."""
        return torch.zeros(
            1, self.recurrent.hidden_size, dtype=embedding.dtype, device=embedding.device
        )

    def posterior(self, embedding: Tensor, previous: Belief) -> Belief:
        mean, log_variance, _ = self.forward(embedding, previous)
        standard_deviation = torch.exp(0.5 * log_variance).detach().cpu().numpy()
        return Belief(
            mean=mean.detach().cpu().numpy().astype(np.float64),
            standard_deviation=np.asarray(standard_deviation, dtype=np.float64),
        )

    def step(
        self, embedding: Tensor, previous: Belief, hidden: Tensor | None = None
    ) -> tuple[Belief, Tensor]:
        """Posterior plus the recurrent state, for the streaming loop of Algorithm 1."""
        mean, log_variance, memory = self.forward(embedding, previous, hidden)
        standard_deviation = torch.exp(0.5 * log_variance)
        belief = Belief(
            mean=mean.detach().cpu().numpy().astype(np.float64),
            standard_deviation=standard_deviation.detach().cpu().numpy().astype(np.float64),
        )
        return belief, memory
