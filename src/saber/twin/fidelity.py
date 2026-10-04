"""Twin fidelity: the invariants the coupled system has to satisfy.

Ref: Sec. 2.7 -- "its fidelity is evaluated based on data concerning its growth
patterns"; Sec. 4.1 -- "a digital twin whose accuracy we assess instead of making
assumptions about".

The quantities here are the ones that can be evaluated without the study's own
logged trajectories: the coupled loop against the same growth law run open-loop,
the reaction-diffusion residual and its closed-form equilibrium, and the
structural invariants (monotone biomass, non-negative oxygen, an agent
population whose realised area reproduces the biomass the growth law carries).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.twin.coupled import TwinTrajectory
from saber.twin.growth import biomass_of
from saber.twin.oxygen import OxygenField, laplacian

Array = NDArray[np.float64]


def relative_gap(reference: Array, observed: Array) -> float:
    """Mean absolute deviation, relative to the reference's mean magnitude."""
    baseline = float(np.mean(np.abs(reference)))
    if baseline <= 0.0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(reference)))) / baseline


def gap_percent(reference: Array, observed: Array) -> float:
    """The same quantity the article prints, as a percentage."""
    return 100.0 * relative_gap(reference, observed)


def oxygen_invariant_error(
    field: OxygenField, consumption: float, drainage: Array, diffusivity: float
) -> float:
    """Largest interior residual of ``D lap(c) - consumption * drainage``.

    Zero at a settled field and non-zero while it is still relaxing, which is why
    the check runs it on a field the solver has driven to equilibrium.
    """
    residual = diffusivity * laplacian(field.concentration, field.spacing) - (
        consumption * np.asarray(drainage, dtype=np.float64)
    )
    return float(np.max(np.abs(residual[1:-1, 1:-1])))


def capacity_overshoot(biomass: Array, capacity: float) -> float:
    """How far the realised biomass passes the carrying capacity, relative to it."""
    if capacity <= 0.0:
        return 0.0
    series = np.asarray(biomass, dtype=np.float64)
    return float(max(0.0, float(np.max(series)) - capacity) / capacity)


def agent_area_error(trajectory: TwinTrajectory) -> float:
    """Gap between the agent population's realised area and the carried biomass.

    This is the twin's coupling invariant: the growth law carries a biomass, and
    the agents are supposed to be exactly its spatial realisation.
    """
    worst = 0.0
    for state in trajectory.states:
        realised = biomass_of(state.agents)
        if state.biomass <= 0.0:
            continue
        worst = max(worst, abs(realised - state.biomass) / state.biomass)
    return float(worst)


def monotone_series(values: Array, tolerance: float = 1e-9) -> bool:
    series = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.diff(series) >= -tolerance))


@dataclass(frozen=True)
class FidelityReport:
    """The twin's own accuracy read-out on the release's substrate."""

    coupled_final_biomass: float
    open_loop_final_biomass: float
    growth_gap_percent: float
    equilibrium_error: float
    final_oxygen_residual: float
    agent_area_error: float
    capacity_overshoot: float
    biomass_monotone: bool
    oxygen_non_negative: bool

    def label(self) -> dict[str, float | bool]:
        return {
            "coupled_final_biomass": round(self.coupled_final_biomass, 9),
            "open_loop_final_biomass": round(self.open_loop_final_biomass, 9),
            "growth_gap_percent": round(self.growth_gap_percent, 6),
            "equilibrium_error": round(self.equilibrium_error, 9),
            "final_oxygen_residual": round(self.final_oxygen_residual, 9),
            "agent_area_error": round(self.agent_area_error, 12),
            "capacity_overshoot": round(self.capacity_overshoot, 9),
            "biomass_monotone": self.biomass_monotone,
            "oxygen_non_negative": self.oxygen_non_negative,
        }


def assess(
    trajectory: TwinTrajectory,
    open_loop: Array,
    consumption: float,
    diffusivity: float,
    drainage: Array,
    settling_error: float,
    capacity: float,
) -> FidelityReport:
    """Bundle the twin's invariants into one report."""
    growth = trajectory.growth
    final_state = trajectory.final()
    return FidelityReport(
        coupled_final_biomass=float(growth.biomass[-1]),
        open_loop_final_biomass=float(open_loop[-1]),
        growth_gap_percent=gap_percent(open_loop, growth.biomass),
        equilibrium_error=settling_error,
        final_oxygen_residual=oxygen_invariant_error(
            final_state.oxygen, consumption, drainage, diffusivity
        ),
        agent_area_error=agent_area_error(trajectory),
        capacity_overshoot=capacity_overshoot(growth.biomass, capacity),
        biomass_monotone=monotone_series(growth.biomass, tolerance=1e-9),
        oxygen_non_negative=bool(np.min(final_state.oxygen.concentration) >= 0.0),
    )
