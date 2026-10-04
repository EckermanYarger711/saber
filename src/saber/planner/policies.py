"""The compared acquisition policies.

Ref: Table 1's row set and the accompanying Methods. Two of the rows are the
article's own constructions; the release builds all of them on the same feasible
set, the same gain table and the same damage table, so a row differs from another
row only through its selection rule.

``trajectory_pomdp`` is our own construction with the constraint on the robot's
travel cost rather than on the specimen, which is the contrast the article draws:
the ablation's point is that "it doesn't matter that some constraint is used but
what the parameter is".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.belief.estimator import Belief
from saber.planner.currency import currency_vector
from saber.runtime.config import PlannerSpec

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]

POLICY_NAMES: tuple[str, ...] = (
    "fixed_cadence",
    "uniform_random",
    "protocol_heuristic",
    "expected_improvement",
    "bayes_adaptive",
    "lagrangian_policy",
    "trajectory_pomdp",
    "bald",
    "entropy_greedy",
    "big",
)

TRAVEL_PENALTY = 0.15


@dataclass(frozen=True)
class Candidates:
    """One decision's feasible set, with everything a selection rule may read."""

    indices: IndexArray
    gains: Array
    costs: Array
    travel: Array
    noise: Array
    designs: Array
    reference_slots: IndexArray

    @property
    def size(self) -> int:
        return int(self.indices.size)

    def label(self) -> dict[str, int]:
        return {"feasible": self.size, "reference_slots": int(self.reference_slots.size)}


@dataclass(frozen=True)
class PolicyDecision:
    """Which action a policy selected and what it was worth."""

    position: int
    index: int
    gain: float
    damage: float
    score: float

    def label(self) -> dict[str, float | int]:
        return {
            "action_index": self.index,
            "gain": round(self.gain, 9),
            "damage": round(self.damage, 9),
            "score": round(self.score, 9),
        }


@dataclass(frozen=True)
class DecisionContext:
    """Everything a policy may read when it selects the next action."""

    belief: Belief
    candidates: Candidates
    spec: PlannerSpec
    epoch: int
    multiplier: float
    remaining: float
    rng: np.random.Generator
    position_values: Array | None = None


Policy = Callable[[DecisionContext], PolicyDecision]


def _pick(context: DecisionContext, position: int, score: float) -> PolicyDecision:
    candidates = context.candidates
    return PolicyDecision(
        position=position,
        index=int(candidates.indices[position]),
        gain=float(candidates.gains[position]),
        damage=float(candidates.costs[position]),
        score=float(score),
    )


def _argmax(values: Array) -> int:
    return int(np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")[0])


def fixed_cadence(context: DecisionContext) -> PolicyDecision:
    """The reference protocol: one frame per well per epoch, cheapest channel first.

    Ref: Table 1 -- "Fixed cadence (every 6 h, 3 channels)". The selection is
    deterministic in the epoch and in the feasible set, which is what makes it a
    cadence rather than a policy.
    """
    slots = context.candidates.reference_slots
    if slots.size == 0:
        return _pick(context, 0, score=float(context.candidates.gains[0]))
    position = int(slots[context.epoch % slots.size])
    return _pick(context, position, score=float(context.candidates.gains[position]))


def uniform_random(context: DecisionContext) -> PolicyDecision:
    position = int(context.rng.integers(0, context.candidates.size))
    return _pick(context, position, score=float(context.candidates.gains[position]))


def protocol_heuristic(context: DecisionContext) -> PolicyDecision:
    """A hand-built rule: prefer a short-exposure label-free frame.

    The rule uses the domain knowledge that a mid-exposure label-free frame is
    informative without costing what a fluorescence frame costs, but it never
    estimates an information gain.
    """
    scores = -np.abs(context.candidates.noise - 0.08) - 0.02 * context.candidates.costs
    position = _argmax(scores)
    return _pick(context, position, score=float(scores[position]))


def expected_improvement(context: DecisionContext) -> PolicyDecision:
    """Information gain over the square root of damage.

    An improvement-shaped rule: it trades information against cost but has no
    multiplier and no belief value, so it cannot price a marginal unit of damage.
    """
    scores = context.candidates.gains / np.sqrt(np.clip(context.candidates.costs, 1e-9, None))
    position = _argmax(scores)
    return _pick(context, position, score=float(scores[position]))


def bayes_adaptive(context: DecisionContext) -> PolicyDecision:
    """A single cadence chosen for the population, re-phased once per epoch.

    The rule advances the reference slot by the belief's total uncertainty, which
    is the least adaptive rule that still reads the belief.
    """
    slots = context.candidates.reference_slots
    offset = int(round(float(np.sum(context.belief.standard_deviation)) * 10.0))
    if slots.size == 0:
        return _pick(context, 0, score=float(context.candidates.gains[0]))
    position = int(slots[(context.epoch + offset) % slots.size])
    return _pick(context, position, score=float(context.candidates.gains[position]))


def lagrangian_policy(context: DecisionContext) -> PolicyDecision:
    """The Lagrangian objective with a frozen multiplier.

    This is the baseline the article places below the specimen-constrained
    planner: the same objective as Eq. (7) with the dual update removed, which is
    the difference between a constraint being present and the specimen being the
    thing that is priced.
    """
    scores = context.candidates.gains - context.multiplier * context.candidates.costs
    position = _argmax(scores)
    return _pick(context, position, score=float(scores[position]))


def trajectory_pomdp(context: DecisionContext) -> PolicyDecision:
    """The constraint placed on the robot's travel instead of on the specimen.

    Ref: Table 1's dagger footnote -- "Our own construction: the constraint acts
    on the robot's trajectory cost, not on the specimen (see Methods)."
    """
    scores = context.candidates.gains - TRAVEL_PENALTY * context.candidates.travel
    position = _argmax(scores)
    return _pick(context, position, score=float(scores[position]))


def bald(context: DecisionContext) -> PolicyDecision:
    """Uncertainty sampling on a sampled sub-composition of the axes.

    BALD scores the mutual information between the observation and the latent
    state; the sampled variant here restricts that information to a random half of
    the axes, which is the sampling behaviour the family is known for and is why
    it sits below the entropy greedy in the article's Panel A.
    """
    axis_mask = np.asarray(context.rng.random(context.belief.dimension) < 0.5, dtype=np.float64)
    if not axis_mask.any():
        axis_mask[0] = 1.0
    quadratic = np.sum(
        context.candidates.designs**2 * (context.belief.variance() * axis_mask)[None, :], axis=1
    )
    scores = 0.5 * np.log1p(quadratic / context.candidates.noise)
    position = _argmax(scores)
    return _pick(context, position, score=float(scores[position]))


def entropy_greedy(context: DecisionContext) -> PolicyDecision:
    """Pure information greedy with no damage term: the unconstrained policy."""
    position = _argmax(context.candidates.gains)
    return _pick(context, position, score=float(context.candidates.gains[position]))


def big(context: DecisionContext) -> PolicyDecision:
    """The release's planner: the currency, the belief value and the priced damage.

    Ref: Eq. (7) -- ``G(a | b) + gamma V_omega(b') - lambda C(a)``, read together
    with Sec. 2.5's statement that "the constraint is involved in the
    decision-making process via ``rho``". Under ``currency: rho`` the information
    term is the currency rescaled by the candidate set's mean damage, so it stays
    in damage units and can be traded against the other two terms; under
    ``currency: gain`` the raw gain is used instead, which is Table 2's Tier-2
    ``-rho currency`` row.
    """
    continuation = (
        0.0
        if context.position_values is None
        else float(np.asarray(context.position_values).reshape(-1)[0])
    )
    costs = context.candidates.costs
    if context.spec.currency == "rho":
        information = currency_vector(context.candidates.gains, costs) * float(np.mean(costs))
    else:
        information = context.candidates.gains
    scores = information + continuation - context.multiplier * costs
    position = _argmax(scores)
    return _pick(context, position, score=float(scores[position]))


BUILDERS: dict[str, Policy] = {
    "fixed_cadence": fixed_cadence,
    "uniform_random": uniform_random,
    "protocol_heuristic": protocol_heuristic,
    "expected_improvement": expected_improvement,
    "bayes_adaptive": bayes_adaptive,
    "lagrangian_policy": lagrangian_policy,
    "trajectory_pomdp": trajectory_pomdp,
    "bald": bald,
    "entropy_greedy": entropy_greedy,
    "big": big,
}


def build_policy(name: str) -> Policy:
    if name not in BUILDERS:
        raise KeyError(f"{name} is not one of the compared acquisition policies")
    return BUILDERS[name]
