"""The decision currency.

Ref: Eq. (4) -- ``rho(a | b) = G(a | b) / C(a)``, called "the currency of the
problem"; Sec. 2.5 -- "the constrained optimization is done using receding-horizon
planning through Lagrangian relaxation"; Table 2's Tier-2 row ``-rho currency
(gain only)`` removes this ratio and keeps the raw gain.

The ratio is the only place the specimen's integrity enters a decision, which is
the article's central claim; the ``-damage constraint`` row of Table 2 is the
same code path with the ratio replaced by the gain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from saber.spec.actions import Action
from saber.spec.damage import DamageFunctional, DamageTable

Array = NDArray[np.float64]


def currency(gain: float, action: Action, damage: DamageFunctional) -> float:
    """``rho`` for one action; an action that costs nothing carries infinite currency."""
    cost = damage.of(action)
    if cost <= 0.0:
        return float("inf") if gain > 0.0 else 0.0
    return gain / cost


def currency_vector(gains: Array, costs: Array) -> Array:
    """``rho`` for a whole candidate set, elementwise."""
    values = np.asarray(gains, dtype=np.float64)
    price = np.asarray(costs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(price > 0.0, values / price, np.inf)
    return np.asarray(ratio, dtype=np.float64)


def rank_by_currency(gains: Array, table: DamageTable, indices: Array) -> NDArray[np.int64]:
    """Candidate indices ordered by currency, best first."""
    ratios = currency_vector(np.asarray(gains)[indices], table.costs[indices])
    order = np.argsort(-ratios, kind="stable")
    return np.asarray(indices[order], dtype=np.int64)


def currency_per_unit_damage(total_gain: float, total_damage: float) -> float:
    """The session-level read-out of Table 1 Panel B: information per unit damage."""
    if total_damage <= 0.0:
        return 0.0
    return total_gain / total_damage
