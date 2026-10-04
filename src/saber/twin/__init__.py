"""The digital twin: organoid growth coupled to oxygen and nutrient delivery.

Ref: Sec. 2.7 -- the twin is an agent-oriented model of organoid growth linked to
a reaction-diffusion approach to the supply of oxygen and nutrients, calibrated
against public organoid growth and size.

The twin has two roles in the article: the training environment for the planner
and actuation policy, and the source of the ground-truth policy value that the
logged-data evaluation is checked against. Both need the same object, so the
coupling lives in one place rather than in two.
"""

from saber.twin.clock import EpochClock, SessionSchedule
from saber.twin.coupled import TwinState, TwinTrajectory, run_twin
from saber.twin.fidelity import (
    FidelityReport,
    agent_area_error,
    capacity_overshoot,
    gap_percent,
    relative_gap,
)
from saber.twin.growth import (
    Agent,
    GrowthRecord,
    colony_radius,
    logistic_reference,
    step_agents,
)
from saber.twin.oxygen import (
    OxygenField,
    analytic_dirichlet_steady_state,
    diffuse,
    equilibrium_error,
    hypoxic_fraction,
    relax_to_equilibrium,
    uniform_field,
)

__all__ = [
    "Agent",
    "EpochClock",
    "FidelityReport",
    "GrowthRecord",
    "OxygenField",
    "SessionSchedule",
    "TwinState",
    "TwinTrajectory",
    "agent_area_error",
    "analytic_dirichlet_steady_state",
    "capacity_overshoot",
    "colony_radius",
    "diffuse",
    "equilibrium_error",
    "gap_percent",
    "hypoxic_fraction",
    "logistic_reference",
    "relative_gap",
    "relax_to_equilibrium",
    "run_twin",
    "step_agents",
    "uniform_field",
]
