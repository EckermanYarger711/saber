"""The coupled twin: growth under the oxygen field it creates.

Ref: Sec. 2.7 (the two roles of the twin), Sec. 2.3 (the four axes the twin's
state is read onto), Sec. 4.1 (the twin's growth fidelity is evaluated, not
assumed).

Each epoch advances the oxygen field over the sub-steps of the clock, then grows
the population logistically with the non-hypoxic area as the availability, then
re-realises the population with agents and feeds the occupied area back as the
sink. The feedback is what makes a dense well differ from a sparse one, which is
the mechanism Sec. 3.5 reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.runtime.config import TwinSpec
from saber.twin.clock import SUBSTEPS_PER_EPOCH, EpochClock, SessionSchedule
from saber.twin.growth import (
    Agent,
    GrowthRecord,
    biomass_of,
    colony_radius,
    logistic_step,
    occupancy_field,
    seed_agents,
    step_agents,
)
from saber.twin.oxygen import (
    OxygenField,
    diffuse,
    hypoxic_fraction,
    stable_substep_hours,
    uniform_field,
)

GROWTH_SUBSTEP_FRACTION = 0.25

Array = NDArray[np.float64]

MEDIUM_TENSION = 1.0
AXIS_NAMES = ("oxygen", "stiffness", "coculture", "heterogeneity")


@dataclass(frozen=True)
class TwinState:
    """The twin at one epoch: field, population and the derived axes."""

    epoch: int
    time_hours: float
    oxygen: OxygenField
    agents: tuple[Agent, ...]
    biomass: float
    stiffness: float
    coculture: float

    def axes(self) -> tuple[float, float, float, float]:
        """The four microenvironment coordinates of Sec. 2.3.

        Oxygen and stiffness are read directly off the twin's own field and
        matrix state; the co-culture coordinate is the fraction of the
        population that is stromal, and the inter-organ heterogeneity index is
        the spread of instance size within the specimen.
        """
        oxygen_regime = float(self.oxygen.mean())
        radii = np.asarray([agent.radius for agent in self.agents], dtype=np.float64)
        spread = float(np.std(radii) / np.mean(radii)) if radii.size and np.mean(radii) > 0 else 0.0
        return (oxygen_regime, self.stiffness, self.coculture, spread)

    def label(self) -> dict[str, float | int | list[float]]:
        axes = self.axes()
        return {
            "epoch": self.epoch,
            "time_hours": round(self.time_hours, 6),
            "biomass": round(self.biomass, 9),
            "agents": len(self.agents),
            "oxygen_mean": round(float(self.oxygen.mean()), 9),
            "colony_radius": round(colony_radius(self.agents), 9),
            "axes": [round(value, 9) for value in axes],
        }


@dataclass(frozen=True)
class TwinTrajectory:
    """A whole session: the states, the growth record and the schedule."""

    states: tuple[TwinState, ...]
    growth: GrowthRecord
    schedule: SessionSchedule
    substep_hours: float

    def final(self) -> TwinState:
        return self.states[-1]

    def axes_by_epoch(self) -> Array:
        rows = [state.axes() for state in self.states]
        return np.asarray(rows, dtype=np.float64)

    def label(self) -> dict[str, object]:
        return {
            "schedule": self.schedule.label(),
            "final": self.final().label(),
        }


def _grid_spacing(size: int) -> float:
    return 1.0 / float(size - 1)


def run_twin(spec: TwinSpec, schedule: SessionSchedule, seed: int) -> TwinTrajectory:
    """Integrate the coupled system over the schedule's epochs."""
    rng = np.random.default_rng(seed)
    clock: EpochClock = schedule.clock
    spacing = _grid_spacing(spec.grid)
    oxygen = uniform_field(spec.grid, spacing, MEDIUM_TENSION)
    agents = seed_agents(spec.seed_radius * spacing * 8.0, 4, rng, spec.seed_radius * spacing * 2.0)
    biomass = biomass_of(agents)
    stiffness = 0.5
    coculture = 0.25
    dt = min(clock.substep_hours(), stable_substep_hours(spec.oxygen_diffusivity, spacing))

    states: list[TwinState] = []
    biomass_series: list[float] = [biomass]
    counts: list[int] = [len(agents)]
    mean_radius: list[float] = [biomass / max(len(agents), 1)]
    outer: list[float] = [colony_radius(agents)]
    hypoxic_series: list[float] = [hypoxic_fraction(oxygen, spec.hypoxic_threshold)]

    for epoch in range(clock.epochs):
        for _ in range(SUBSTEPS_PER_EPOCH):
            occupancy = occupancy_field(agents, spec.grid, spacing)
            oxygen = diffuse(
                oxygen, spec.oxygen_diffusivity, spec.oxygen_consumption, occupancy, dt
            )
        hypoxic = hypoxic_fraction(oxygen, spec.hypoxic_threshold)
        availability = 1.0 - hypoxic
        biomass = _advance_growth(biomass, spec, availability, clock.epoch_hours)
        agents = step_agents(
            agents,
            biomass,
            spacing,
            rng,
            detachment_rate=spec.detachment_rate,
            hypoxic=hypoxic,
        )
        stiffness = stiffness + spec.stiffness_relaxation * (availability - stiffness)
        coculture = coculture + 0.02 * (availability - coculture)
        states.append(
            TwinState(
                epoch=epoch,
                time_hours=(epoch + 1) * clock.epoch_hours,
                oxygen=oxygen,
                agents=agents,
                biomass=biomass,
                stiffness=stiffness,
                coculture=coculture,
            )
        )
        biomass_series.append(biomass)
        counts.append(len(agents))
        mean_radius.append(biomass / max(len(agents), 1))
        outer.append(colony_radius(agents))
        hypoxic_series.append(hypoxic)

    record = GrowthRecord(
        times_hours=clock.times_hours(),
        biomass=np.asarray(biomass_series[1:], dtype=np.float64),
        agent_counts=np.asarray(counts[1:], dtype=np.int64),
        mean_radius=np.asarray(mean_radius[1:], dtype=np.float64),
        colony_radius=np.asarray(outer[1:], dtype=np.float64),
        hypoxic_fraction=np.asarray(hypoxic_series[1:], dtype=np.float64),
    )
    return TwinTrajectory(states=tuple(states), growth=record, schedule=schedule, substep_hours=dt)


def _advance_growth(biomass: float, spec: TwinSpec, availability: float, hours: float) -> float:
    """Advance the growth law in sub-steps small enough that it cannot overshoot.

    The explicit Euler step of a logistic law overshoots its capacity once
    ``rate * availability * h`` approaches one, which a six-hour epoch with a
    per-hour rate near a half does. Sub-stepping to a fixed fraction keeps the
    biomass monotone and inside the capacity, which the fidelity checks assert.
    """
    step_hours = GROWTH_SUBSTEP_FRACTION / max(spec.growth_rate, 1e-9)
    steps = max(1, int(np.ceil(hours / step_hours)))
    dt = hours / float(steps)
    value = biomass
    for _ in range(steps):
        value = logistic_step(value, spec.growth_rate, spec.carrying_capacity, availability, dt)
    return value


def open_loop_growth(spec: TwinSpec, times: Array, availability: float = 1.0) -> Array:
    """The same growth law with the oxygen feedback held fixed.

    Used by the fidelity checks, which compare the coupled loop against the
    logistic closed form on the constant-oxygen control where the closed form is
    exact.
    """
    from saber.twin.growth import logistic_reference

    return logistic_reference(
        spec.growth_rate * availability,
        spec.carrying_capacity,
        spec.seed_radius * 0.05,
        times,
    )
