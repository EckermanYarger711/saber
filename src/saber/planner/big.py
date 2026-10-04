"""Algorithm 2: budgeted information-gain acquisition.

Ref: Algorithm 2 (BIG) and Sec. 2.5, Eq. (7).

The listing is transcribed in order: the remaining budget defines the feasible
set, each feasible action's gain and currency are estimated, the Lagrangian
objective is maximised, the action is executed and the belief updated, the
remaining budget and the multiplier are carried forward, and the loop stops on
information saturation or on an exhausted horizon. Algorithm 2 returns the action
sequence and the realised damage ``D - D_rem``; the trace keeps both, so the
iso-resource comparison can read the spend off directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from saber.belief.estimator import Belief
from saber.planner.info_gain import EncodingTable
from saber.planner.lagrangian import DualState, dual_update
from saber.planner.policies import (
    Candidates,
    DecisionContext,
    big,
)
from saber.runtime.config import PlannerSpec
from saber.spec.actions import Action
from saber.spec.damage import DamageTable

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]

ValueFunction = Callable[[Belief], float]


@dataclass(frozen=True)
class ActionSet:
    """The frozen action set with its damage and design tables.

    ``well_slices`` is what makes one decision a per-well choice: a session
    monitors ``wells_per_cycle`` wells over the session's epochs, so decision
    ``k`` acts on well ``k % wells_per_cycle`` and searches only that well's
    actions. Searching the whole plate at every decision would let one decision
    spend the budget of another well's clock, which the article's allocation
    across "wells, fields, channels and epochs" does not permit.
    """

    damage: DamageTable
    encodings: EncodingTable
    travel: Array
    reference_mask: Array
    well_slices: tuple[IndexArray, ...]
    actions: tuple[Action, ...]

    @property
    def size(self) -> int:
        return int(self.damage.costs.size)

    @property
    def wells(self) -> int:
        return len(self.well_slices)

    def label(self) -> dict[str, object]:
        return {
            "actions": self.size,
            "wells": self.wells,
            "damage": self.damage.label(),
            "encodings": self.encodings.label(),
            "reference_slots": int(np.count_nonzero(self.reference_mask)),
        }


@dataclass(frozen=True)
class BigStep:
    """One executed decision of Algorithm 2."""

    step: int
    action_index: int
    gain: float
    damage: float
    multiplier: float
    remaining: float
    feasible: int

    def label(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "action_index": self.action_index,
            "gain": round(self.gain, 9),
            "damage": round(self.damage, 9),
            "multiplier": round(self.multiplier, 9),
            "remaining": round(self.remaining, 9),
        }


@dataclass(frozen=True)
class BigTrace:
    """The whole session Algorithm 2 returns."""

    steps: tuple[BigStep, ...]
    damage: float
    information: float
    stopped: str
    budget: float

    def exposures(self) -> int:
        return len(self.steps)

    def currency(self) -> float:
        return self.information / self.damage if self.damage > 0.0 else 0.0

    def label(self) -> dict[str, float | int | str]:
        return {
            "steps": len(self.steps),
            "damage": round(self.damage, 9),
            "information": round(self.information, 9),
            "currency": round(self.currency(), 9),
            "budget": round(self.budget, 9),
            "stopped": self.stopped,
        }


def candidates_for(
    action_set: ActionSet,
    belief: Belief,
    remaining: float,
    prune_top_k: int,
    well: int,
) -> Candidates:
    """The feasible, pruned candidate set of Algorithm 2 lines 3-7 for one well."""
    in_well = action_set.well_slices[well % action_set.wells]
    feasible = np.asarray(
        in_well[action_set.damage.costs[in_well] <= remaining + 1e-12], dtype=np.int64
    )
    if feasible.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return Candidates(
            indices=np.zeros(0, dtype=np.int64),
            gains=empty,
            costs=empty,
            travel=empty,
            noise=empty,
            designs=np.zeros((0, belief.dimension), dtype=np.float64),
            reference_slots=np.zeros(0, dtype=np.int64),
        )
    gains = action_set.encodings.gains(belief)[feasible]
    if prune_top_k < feasible.size:
        keep = np.argsort(-gains, kind="stable")[:prune_top_k]
        keep.sort()
        feasible = np.asarray(feasible[keep], dtype=np.int64)
        gains = np.asarray(gains[keep], dtype=np.float64)
    slots = np.flatnonzero(action_set.reference_mask[feasible])
    return Candidates(
        indices=feasible,
        gains=gains,
        costs=action_set.damage.costs[feasible],
        travel=action_set.travel[feasible],
        noise=action_set.encodings.noises[feasible],
        designs=action_set.encodings.designs[feasible],
        reference_slots=np.asarray(slots, dtype=np.int64),
    )


@dataclass
class BigPlanner:
    """The planner of Eq. (7) with the dual update of Algorithm 2 line 10."""

    spec: PlannerSpec
    action_set: ActionSet
    budget: float
    value_function: ValueFunction | None = None
    dual: DualState = field(default_factory=DualState.initial)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @classmethod
    def build(
        cls,
        spec: PlannerSpec,
        action_set: ActionSet,
        budget: float,
        value_function: ValueFunction | None = None,
        seed: int = 0,
    ) -> BigPlanner:
        return cls(
            spec=spec,
            action_set=action_set,
            budget=budget,
            value_function=value_function,
            dual=DualState.initial(spec.lambda_init),
            rng=np.random.default_rng(seed),
        )

    def select(
        self, belief: Belief, remaining: float, epoch: int, well: int
    ) -> tuple[int, float, int]:
        """Maximise the Lagrangian objective over the feasible, pruned set.

        Returns the chosen global action index, the score it won with, and the
        feasible-set size, which the trace records so a reader can see when the
        budget, rather than the objective, decided the session.
        """
        candidates = candidates_for(self.action_set, belief, remaining, self.spec.prune_top_k, well)
        if candidates.size == 0:
            return -1, 0.0, 0
        values = None
        if self.value_function is not None:
            values = np.asarray(
                [
                    self.spec.discount
                    * float(
                        self.value_function(self.action_set.encodings.posterior(belief, int(index)))
                    )
                    for index in candidates.indices
                ],
                dtype=np.float64,
            )
        context = DecisionContext(
            belief=belief,
            candidates=candidates,
            spec=self.spec,
            epoch=epoch,
            multiplier=self.dual.multiplier,
            remaining=remaining,
            rng=self.rng,
            position_values=values,
        )
        decision = big(context)
        return decision.index, decision.score, candidates.size

    def run(self, belief: Belief, observe: Callable[[Belief, int, Action], Belief]) -> BigTrace:
        """Algorithm 2 over an observation callback.

        ``observe(current, index, action)`` returns the belief after executing the
        chosen action; the callback is what lets the same planner run against the
        twin and against a logged stream without the planner knowing which it is
        on, and it is the same callback type the baseline policies receive.
        """
        steps: list[BigStep] = []
        current = belief
        remaining = self.budget
        information = 0.0
        stopped = "horizon"
        for decision in range(self.spec.horizon):
            if remaining <= 0.0:
                stopped = "budget"
                break
            well = decision % self.spec.wells_per_cycle
            epoch = decision // self.spec.wells_per_cycle
            index, _score, feasible = self.select(current, remaining, epoch, well)
            if index < 0:
                stopped = "infeasible"
                break
            gain = self.action_set.encodings.gain(index, current)
            if gain < self.spec.saturation_threshold:
                stopped = "saturation"
                break
            cost = self.action_set.damage.cost(index)
            information += gain
            remaining = max(0.0, remaining - cost)
            self.dual = dual_update(charge(self.dual, cost), self.budget, self.spec.dual_step)
            current = observe(current, index, self.action_set.actions[index])
            steps.append(
                BigStep(
                    step=decision,
                    action_index=index,
                    gain=gain,
                    damage=cost,
                    multiplier=self.dual.multiplier,
                    remaining=remaining,
                    feasible=feasible,
                )
            )
        return BigTrace(
            steps=tuple(steps),
            damage=self.budget - remaining,
            information=information,
            stopped=stopped,
            budget=self.budget,
        )


def charge(state: DualState, damage: float) -> DualState:
    """Charge one action's damage before the multiplier's own subgradient step."""
    return DualState(multiplier=state.multiplier, spent=state.spent + damage)


def plan_session(
    spec: PlannerSpec,
    action_set: ActionSet,
    belief: Belief,
    budget: float,
    observe: Callable[[Belief, int, Action], Belief],
    value_function: ValueFunction | None = None,
    seed: int = 0,
) -> BigTrace:
    """Convenience constructor for a single-policy session."""
    planner = BigPlanner.build(spec, action_set, budget, value_function, seed)
    return planner.run(belief, observe)


def reference_schedule(
    action_set: ActionSet, horizon: int, wells_per_cycle: int
) -> tuple[IndexArray, ...]:
    """The reference protocol's action indices for each decision of a session.

    Ref: Sec. 3.1 -- the protocol images every well on three channels per epoch, so
    the decision that acts on well ``w`` at the session's ``e``-th epoch delivers
    the reference frames of ``(w, e)``. This is the sequence the ``fixed cadence``
    row of Table 1 follows and the sequence ``D`` is defined from.
    """
    schedule: list[IndexArray] = []
    for decision in range(horizon):
        well = decision % max(1, wells_per_cycle)
        in_well = action_set.well_slices[well % action_set.wells]
        slots = np.asarray(
            [index for index in in_well if bool(action_set.reference_mask[index])],
            dtype=np.int64,
        )
        schedule.append(slots)
    return tuple(schedule)
