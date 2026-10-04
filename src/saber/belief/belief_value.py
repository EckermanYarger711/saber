"""The belief-value function.

Ref: Sec. 2.5, Eq. (7) -- the planner maximises
``G(a | b) + gamma V_omega(b') - lambda C(a)`` with "learned belief-value function
``V_omega``". Sec. 2.7 gives where its ground truth comes from: the twin supplies
"the ground-truth policy value" that the logged-data evaluation is checked
against. Table 2's Tier-1 row removes the function and keeps only the myopic
currency, which is why it lives in its own module.

The training target is the discounted continuation information the twin
realises from the belief, so the function is fitted to the twin rather than to a
hand-built potential.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from saber.belief.estimator import Belief

Array = NDArray[np.float64]


class BeliefValue(nn.Module):
    """A two-layer map from a belief to its expected discounted information."""

    def __init__(self, dimension: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dimension = dimension
        self.network = nn.Sequential(
            nn.Linear(2 * dimension, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, belief: Tensor) -> Tensor:
        value: Tensor = self.network(belief)
        return value.squeeze(-1)

    def of(self, belief: Belief) -> float:
        """The scalar value of one belief, without gradients."""
        with torch.no_grad():
            tensor = torch.as_tensor(belief.to_tensor(), dtype=torch.float32).unsqueeze(0)
            return float(self.forward(tensor)[0])


@dataclass(frozen=True)
class BeliefValueFit:
    """The fitted model plus the evidence of the fit."""

    model: BeliefValue
    epochs: int
    final_loss: float
    initial_loss: float

    def improved(self) -> bool:
        return self.final_loss < self.initial_loss

    def label(self) -> dict[str, float | int | bool]:
        return {
            "epochs": self.epochs,
            "initial_loss": round(self.initial_loss, 9),
            "final_loss": round(self.final_loss, 9),
            "improved": self.improved(),
        }


def belief_tensor(beliefs: tuple[Belief, ...]) -> Tensor:
    return torch.stack(
        [torch.as_tensor(belief.to_tensor(), dtype=torch.float32) for belief in beliefs], dim=0
    )


def fit_belief_value(
    beliefs: tuple[Belief, ...],
    targets: Array,
    epochs: int = 200,
    learning_rate: float = 5e-3,
    seed: int = 0,
) -> BeliefValueFit:
    """Fit ``V_omega`` by least squares against the twin's realised continuations."""
    if not beliefs:
        raise ValueError("the fit needs at least one belief")
    torch.manual_seed(seed)
    dimension = beliefs[0].dimension
    model = BeliefValue(dimension)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs = belief_tensor(beliefs)
    desired = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    with torch.no_grad():
        initial = float(torch.mean((model(inputs) - desired) ** 2))
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = torch.mean((model(inputs) - desired) ** 2)
        torch.autograd.backward(loss)
        optimiser.step()
    with torch.no_grad():
        final = float(torch.mean((model(inputs) - desired) ** 2))
    return BeliefValueFit(model=model, epochs=epochs, final_loss=final, initial_loss=initial)
