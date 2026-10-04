"""Session-level evaluation of the acquisition policies.

Ref: Sec. 3.1 (the iso-resource comparison), Sec. 3.2 (the unbinding-budget
panel), Sec. 3.3 (the matched-damage panel), Sec. 3.5 (the regime
stratification).

One session monitors ``wells_per_cycle`` wells over the clock's epochs and has
``horizon`` decisions, one per (well, epoch) pair. Every policy runs against the
same action set, the same gain table, the same damage table and the same belief
updates, which is what makes the rows of Table 1 comparable: the damage budget is
the only thing held fixed and the selection rule is the only thing that varies.

``fixed_cadence`` is a protocol rather than an action selector: the reference
protocol delivers the whole frame set of a well and an epoch, so its branch of the
session loop executes every frame of the decision's schedule entry, while every
other policy spends one action per decision. That asymmetry is the article's own
contrast between a fixed schedule and a planner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.belief.estimator import Belief
from saber.planner.big import ActionSet, BigPlanner, candidates_for, reference_schedule
from saber.planner.info_gain import observation_noise as photometric_noise
from saber.planner.lagrangian import DualState
from saber.planner.policies import DecisionContext, Policy, build_policy
from saber.runtime.config import DishSpec, PlannerSpec
from saber.spec.actions import Action

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]

Observation = Callable[[Belief, int, Action], Belief]


@dataclass(frozen=True)
class SessionOutcome:
    """One policy's realised session."""

    policy: str
    information: float
    damage: float
    budget: float
    action_indices: IndexArray
    gains: Array
    belief_trace: tuple[Belief, ...]
    stopped: str

    def exposures(self) -> int:
        return int(self.action_indices.size)

    def currency(self) -> float:
        return self.information / self.damage if self.damage > 0.0 else 0.0

    def budget_usage_percent(self) -> float:
        return 100.0 * self.damage / self.budget if self.budget > 0.0 else 0.0

    def final_belief(self) -> Belief:
        return self.belief_trace[-1]

    def label(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy,
            "information": round(self.information, 9),
            "damage": round(self.damage, 9),
            "currency": round(self.currency(), 9),
            "exposures": self.exposures(),
            "budget_usage_percent": round(self.budget_usage_percent(), 4),
            "stopped": self.stopped,
        }


def run_policy_session(
    policy_name: str,
    spec: PlannerSpec,
    action_set: ActionSet,
    actions: tuple[Action, ...],
    belief: Belief,
    budget: float,
    observe: Observation,
    *,
    seed: int = 0,
    value_function: Callable[[Belief], float] | None = None,
    update_dual: bool = False,
) -> SessionOutcome:
    """Run one policy over a session.

    ``update_dual`` separates the release's planner from the frozen-multiplier
    baseline; both are otherwise the same objective, which is the comparison the
    method section sets up.
    """
    if policy_name == "big":
        return _run_big(spec, action_set, actions, belief, budget, observe, value_function, seed)
    generator = np.random.default_rng(seed)
    schedule = reference_schedule(action_set, spec.horizon, spec.wells_per_cycle)
    policy: Policy = build_policy(policy_name)
    current = belief
    remaining = budget
    information = 0.0
    chosen: list[int] = []
    gains: list[float] = []
    trace: list[Belief] = [current]
    dual = DualState.initial(spec.lambda_init)
    stopped = "horizon"
    for index_decision in range(spec.horizon):
        well = index_decision % max(1, spec.wells_per_cycle)
        candidates = candidates_for(action_set, current, remaining, spec.prune_top_k, well)
        if candidates.size == 0:
            stopped = "infeasible"
            break
        if policy_name == "fixed_cadence":
            frames = [
                int(index)
                for index in schedule[index_decision]
                if action_set.damage.cost(int(index)) <= remaining + 1e-12
            ]
            if not frames:
                stopped = "infeasible"
                break
            for frame in frames:
                gains.append(action_set.encodings.gain(frame, current))
                information += gains[-1]
                chosen.append(frame)
                remaining = max(0.0, remaining - action_set.damage.cost(frame))
                current = observe(current, frame, actions[frame])
                trace.append(current)
            dual = DualState(
                multiplier=dual.multiplier,
                spent=dual.spent + sum(action_set.damage.cost(frame) for frame in frames),
            )
            continue
        values = None
        if value_function is not None:
            values = np.asarray(
                [
                    spec.discount
                    * float(value_function(action_set.encodings.posterior(current, int(index))))
                    for index in candidates.indices
                ],
                dtype=np.float64,
            )
        context = DecisionContext(
            belief=current,
            candidates=candidates,
            spec=spec,
            epoch=index_decision // max(1, spec.wells_per_cycle),
            multiplier=dual.multiplier,
            remaining=remaining,
            rng=generator,
            position_values=values,
        )
        selection = policy(context)
        gain = action_set.encodings.gain(selection.index, current)
        if gain < spec.saturation_threshold:
            stopped = "saturation"
            break
        cost = action_set.damage.cost(selection.index)
        information += gain
        remaining = max(0.0, remaining - cost)
        chosen.append(selection.index)
        gains.append(gain)
        spent = dual.spent + cost
        multiplier = (
            max(0.0, dual.multiplier + spec.dual_step * (spent - budget))
            if update_dual
            else dual.multiplier
        )
        dual = DualState(multiplier=multiplier, spent=spent)
        current = observe(current, selection.index, actions[selection.index])
        trace.append(current)
    return SessionOutcome(
        policy=policy_name,
        information=information,
        damage=budget - remaining,
        budget=budget,
        action_indices=np.asarray(chosen, dtype=np.int64),
        gains=np.asarray(gains, dtype=np.float64),
        belief_trace=tuple(trace),
        stopped=stopped,
    )


def _run_big(
    spec: PlannerSpec,
    action_set: ActionSet,
    actions: tuple[Action, ...],
    belief: Belief,
    budget: float,
    observe: Observation,
    value_function: Callable[[Belief], float] | None,
    seed: int,
) -> SessionOutcome:
    """The Algorithm 2 path, wrapped into the same outcome record."""
    planner = BigPlanner.build(spec, action_set, budget, value_function, seed)
    beliefs: list[Belief] = [belief]

    def step_observer(current: Belief, index: int, action: Action) -> Belief:
        updated = observe(current, index, action)
        beliefs.append(updated)
        return updated

    trace = planner.run(belief, step_observer)
    return SessionOutcome(
        policy="big",
        information=trace.information,
        damage=trace.damage,
        budget=budget,
        action_indices=np.asarray([step.action_index for step in trace.steps], dtype=np.int64),
        gains=np.asarray([step.gain for step in trace.steps], dtype=np.float64),
        belief_trace=tuple(beliefs),
        stopped=trace.stopped,
    )


PROCESS_NOISE = 0.02


def logged_observer(
    recorded_axes: Array,
    spec: DishSpec,
    seed: int,
) -> Observation:
    """An observation callback driven by a recorded axis trajectory.

    Ref: the abstract's "retrospective secondary analysis of previously logged
    trajectories". The callback advances one epoch per call and updates the belief
    with a Kalman step whose process-noise floor keeps the filter able to track a
    moving state: without the floor the belief variance collapses after the first
    observation and the filter stops following the record. The observation's own
    noise shrinks with the exposure the action commands, so a longer or brighter
    frame leaves a better estimate, which is what makes two policies' state
    estimates differ at all.
    """
    generator = np.random.default_rng(seed)
    epochs = len(recorded_axes)
    counter = {"epoch": 0}

    def observe(belief: Belief, index: int, action: Action) -> Belief:
        epoch = counter["epoch"] % epochs if epochs else 0
        counter["epoch"] += 1
        truth = np.asarray(recorded_axes[epoch], dtype=np.float64) if epochs else belief.mean
        noise = photometric_noise(action, spec)
        draw = truth + generator.normal(scale=noise, size=truth.shape)
        measurement_variance = noise**2
        prior_variance = belief.variance() + PROCESS_NOISE
        gain = prior_variance / (prior_variance + measurement_variance)
        mean = belief.mean + gain * (draw - belief.mean)
        variance = (1.0 - gain) * prior_variance
        return Belief(
            mean=np.asarray(mean, dtype=np.float64),
            standard_deviation=np.asarray(
                np.sqrt(np.clip(variance, 1e-12, None)), dtype=np.float64
            ),
        )

    return observe


def action_sequence_indices(outcomes: Sequence[SessionOutcome]) -> IndexArray:
    """Every action index a set of sessions executed, in order."""
    if not outcomes:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(
        np.concatenate([outcome.action_indices for outcome in outcomes]), dtype=np.int64
    )
