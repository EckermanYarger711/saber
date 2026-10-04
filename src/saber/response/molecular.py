"""The molecular-graph half of the drug representation.

Ref: Sec. 2.6 -- the representation ``u`` is built from "a molecular graph and a
hashed fingerprint", so the head consumes both: a message-passing encoding of the
graph and the folded bit vector of :mod:`saber.response.fingerprint`.

Message passing with a fixed number of rounds is the standard local aggregation;
the article prints no width or round count, so both are engineering defaults from
the response block of the configuration.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from saber.response.fingerprint import ATOM_CODES, Molecule

Array = NDArray[np.float64]

ATOM_SYMBOLS: tuple[str, ...] = tuple(ATOM_CODES)
BOND_ORDERS: tuple[int, ...] = (1, 2, 3)


def atom_features(molecule: Molecule) -> Tensor:
    """One-hot atom symbol plus a degree slot, per atom."""
    rows = np.zeros((len(molecule.atoms), len(ATOM_SYMBOLS) + 1), dtype=np.float32)
    for index, symbol in enumerate(molecule.atoms):
        if symbol in ATOM_CODES:
            rows[index, ATOM_SYMBOLS.index(symbol)] = 1.0
    degrees = np.zeros(len(molecule.atoms), dtype=np.float32)
    for left, right, _ in molecule.bonds:
        degrees[left] += 1.0
        degrees[right] += 1.0
    rows[:, -1] = degrees / max(1.0, float(degrees.max()))
    return torch.as_tensor(rows)


def bond_index(molecule: Molecule) -> tuple[Tensor, Tensor, Tensor]:
    """Edge lists and edge features of the undirected bond set."""
    sources: list[int] = []
    targets: list[int] = []
    orders: list[float] = []
    for left, right, order in molecule.bonds:
        sources.extend([left, right])
        targets.extend([right, left])
        orders.extend([float(order), float(order)])
    edge_index = torch.as_tensor([sources, targets], dtype=torch.long)
    rows = [
        [1.0 if int(order) == candidate else 0.0 for candidate in BOND_ORDERS] for order in orders
    ]
    features = torch.as_tensor(rows, dtype=torch.float32)
    return edge_index, features, torch.as_tensor(orders, dtype=torch.float32)


class GraphEncoder(nn.Module):
    """A fixed-round message-passing encoder over a molecule's graph."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        rounds: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.node_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.message = nn.ModuleList(
            [nn.Linear(2 * hidden_dim + len(BOND_ORDERS), hidden_dim) for _ in range(rounds)]
        )
        self.update = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(rounds)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, molecule: Molecule) -> Tensor:
        features = atom_features(molecule)
        edge_index, edge_features, _ = bond_index(molecule)
        hidden: Tensor = torch.tanh(self.node_projection(features))
        for round_index in range(self.rounds):
            if edge_index.numel() == 0:
                break
            sources = hidden[edge_index[0]]
            targets = hidden[edge_index[1]]
            messages = torch.tanh(
                self.message[round_index](torch.cat([sources, targets, edge_features], dim=-1))
            )
            aggregated = torch.zeros_like(hidden)
            aggregated.index_add_(0, edge_index[1], messages)
            hidden = torch.tanh(self.update[round_index](torch.cat([hidden, aggregated], dim=-1)))
        pooled: Tensor = torch.mean(hidden, dim=0)
        encoded: Tensor = self.readout(pooled)
        return encoded
