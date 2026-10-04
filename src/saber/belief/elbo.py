"""The variational lower bound and its auxiliary prediction.

Ref: Sec. 2.3 -- "An estimator undergoes training through the optimization
process of the variational lower bound, which employs the use of an auxiliary
prediction unit to introduce the morphology embeddings for the upcoming epoch.
This way, the impression can be prevented from converting into a static
compilation of features."

The bound has three terms: reconstruction of the current embedding, the
Kullback-Leibler divergence from the atlas prior, and the auxiliary prediction
of the *next* epoch's embedding. The third is what stops the latent code from
becoming a static summary of appearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from saber.belief.priors import AtlasPrior


@dataclass(frozen=True)
class VariationalLoss:
    """The three terms and their weighted sum."""

    reconstruction: float
    divergence: float
    auxiliary: float
    total: float

    def label(self) -> dict[str, float]:
        return {
            "reconstruction": round(self.reconstruction, 9),
            "divergence": round(self.divergence, 9),
            "auxiliary": round(self.auxiliary, 9),
            "total": round(self.total, 9),
        }


class BeliefDecoder(nn.Module):
    """Decodes a belief back to the embedding space and predicts the next embedding.

    The latent code is the sampled four-axis microenvironment state, not the
    estimator's internal width: the article's latent variable is the state the
    planner reasons over, so the bound reconstructs the embedding from the axes.
    """

    def __init__(self, embedding_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.reconstruct = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        return self.reconstruct(latent), self.forward_model(latent)


def reparameterise(
    mean: Tensor, log_variance: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """The standard Gaussian reparameterisation, with an explicit generator seed."""
    standard_deviation = torch.exp(0.5 * log_variance)
    noise = torch.randn(
        standard_deviation.shape, generator=generator, dtype=standard_deviation.dtype
    )
    return mean + standard_deviation * noise


def kl_to_prior(mean: Tensor, log_variance: Tensor, prior: AtlasPrior) -> Tensor:
    """KL divergence of a diagonal Gaussian from the atlas prior.

    The prior is the same object the recalibration step re-ties to, so the
    training objective and the drift guard cannot disagree about what the atlas
    says.
    """
    prior_mean = torch.as_tensor(prior.means, dtype=mean.dtype)
    prior_variance = torch.as_tensor(prior.variances(), dtype=mean.dtype)
    divergence = 0.5 * (
        torch.exp(log_variance) / prior_variance
        + (mean - prior_mean) ** 2 / prior_variance
        - 1.0
        + torch.log(prior_variance)
        - log_variance
    )
    return torch.sum(divergence)


def next_embedding_loss(predicted: Tensor, observed: Tensor) -> Tensor:
    """Mean squared error of the auxiliary next-epoch embedding prediction."""
    return torch.mean((predicted - observed) ** 2)


def elbo_terms(
    decoder: BeliefDecoder,
    embedding: Tensor,
    next_embedding: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    prior: AtlasPrior,
    kl_weight: float,
    auxiliary_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, VariationalLoss]:
    """The negative variational bound plus its decomposed terms.

    ``total`` is the quantity to minimise: reconstruction error, the weighted KL
    and the weighted auxiliary prediction, which is the bound the article
    describes with the opposite sign.
    """
    latent = reparameterise(mean, log_variance, generator)
    reconstruction, prediction = decoder(latent)
    reconstruction_loss = torch.mean((reconstruction - embedding) ** 2)
    divergence = kl_to_prior(mean, log_variance, prior)
    auxiliary = next_embedding_loss(prediction, next_embedding)
    total = reconstruction_loss + kl_weight * divergence + auxiliary_weight * auxiliary
    return total, VariationalLoss(
        reconstruction=float(reconstruction_loss.detach()),
        divergence=float(divergence.detach()),
        auxiliary=float(auxiliary.detach()),
        total=float(total.detach()),
    )
