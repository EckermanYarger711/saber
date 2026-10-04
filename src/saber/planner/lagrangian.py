"""The dual variable of the damage constraint.

Ref: Eq. (7) --

    ``lambda_{t+1} = [lambda_t + eta (sum_{t'} C(a_{t'}) - D)]^+``

and Algorithm 2 line 10, which carries the same update. Sec. 2.5 records the
requirement the update serves: "``lambda_t`` has to be updated so that the
realized cumulative damages come close to the budget but do not exceed it."

The projection onto the non-negative orthant is what makes an overspend raise
the price of damage and an underspend lower it, and it is what keeps the
multiplier from turning negative when the budget is not binding -- without the
projection, an unbinding budget would reward damage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualState:
    """The multiplier and the running spend it is updated from."""

    multiplier: float
    spent: float

    @classmethod
    def initial(cls, multiplier: float = 0.0) -> DualState:
        return cls(multiplier=multiplier, spent=0.0)

    def projected_gap(self, budget: float) -> float:
        """``sum_{t'} C(a_{t'}) - D``: how far the spend is from the budget."""
        return self.spent - budget

    def label(self) -> dict[str, float]:
        return {"multiplier": round(self.multiplier, 9), "spent": round(self.spent, 9)}


def dual_update(state: DualState, budget: float, step: float) -> DualState:
    """One projected subgradient step of the multiplier."""
    gap = state.projected_gap(budget)
    return DualState(multiplier=max(0.0, state.multiplier + step * gap), spent=state.spent)


def charge(state: DualState, damage: float) -> DualState:
    """Add one action's damage to the running spend."""
    return DualState(multiplier=state.multiplier, spent=state.spent + damage)


def lagrangian_score(
    gain: float, damage: float, multiplier: float, continuation: float = 0.0
) -> float:
    """``G(a | b) + gamma V_omega(b') - lambda C(a)``, the objective of Eq. (7).

    ``continuation`` is the already-discounted belief-value term; passing zero
    gives the myopic variant that Table 2's first Tier-1 row isolates.
    """
    return gain + continuation - multiplier * damage
