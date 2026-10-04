"""The twin: the oxygen solver, the growth law, the coupling and the fidelity report."""

from __future__ import annotations

import numpy as np

from saber.cli.context import ReleaseContext
from saber.twin.clock import SessionSchedule
from saber.twin.coupled import run_twin
from saber.twin.fidelity import agent_area_error, capacity_overshoot, gap_percent
from saber.twin.growth import biomass_of, logistic_reference
from saber.twin.oxygen import (
    boundary_value,
    diffuse,
    equilibrium_error,
    hypoxic_fraction,
    laplacian,
    relax_to_equilibrium,
    stable_substep_hours,
    uniform_field,
)

SETTLING_STEPS = 40000


def test_schedule_agrees_with_the_dish_cadence(context: ReleaseContext) -> None:
    schedule = SessionSchedule.build(
        epochs=context.twin_spec.epochs,
        epoch_hours=context.twin_spec.epoch_hours,
        reference_epoch_hours=context.dish.reference_epoch_hours,
    )
    assert schedule.cadence_agrees()


def test_laplacian_of_a_linear_field_is_zero() -> None:
    axis = np.arange(8, dtype=np.float64)
    plane = np.add.outer(axis, axis)
    assert float(np.max(np.abs(laplacian(plane, 1.0)[1:-1, 1:-1]))) <= 1e-12


def test_solver_equilibrium_matches_the_closed_form(context: ReleaseContext) -> None:
    spec = context.twin_spec
    spacing = 1.0 / float(spec.grid - 1)
    settled = relax_to_equilibrium(
        uniform_field(spec.grid, spacing, 1.0),
        spec.oxygen_diffusivity,
        spec.oxygen_consumption,
        np.ones((spec.grid, spec.grid), dtype=np.float64),
        SETTLING_STEPS,
    )
    assert equilibrium_error(settled, spec.oxygen_consumption, spec.oxygen_diffusivity) <= 5e-4


def test_substep_is_inside_the_stability_bound(context: ReleaseContext) -> None:
    spacing = 1.0 / float(context.twin_spec.grid - 1)
    bound = spacing * spacing / (4.0 * context.twin_spec.oxygen_diffusivity)
    assert stable_substep_hours(context.twin_spec.oxygen_diffusivity, spacing) <= bound


def test_diffusion_keeps_the_boundary_held(context: ReleaseContext) -> None:
    spec = context.twin_spec
    spacing = 1.0 / float(spec.grid - 1)
    field = uniform_field(spec.grid, spacing, 1.0)
    stepped = diffuse(
        field,
        spec.oxygen_diffusivity,
        spec.oxygen_consumption,
        np.ones((spec.grid, spec.grid)),
        1e-4,
    )
    assert boundary_value(stepped) == 1.0
    assert hypoxic_fraction(stepped, spec.hypoxic_threshold) == 0.0


def test_logistic_reference_matches_its_closed_form() -> None:
    times = np.asarray([0.0, 6.0, 12.0, 24.0])
    values = logistic_reference(0.22, 1.0, 0.05, times)
    assert values[0] == 0.05
    assert np.all(np.diff(values) > 0.0)
    assert float(values[-1]) < 1.0


def test_twin_biomass_is_monotone_and_within_capacity(context: ReleaseContext) -> None:
    trajectory = run_twin(context.twin_spec, context.schedule, 20260929)
    series = trajectory.growth.biomass
    assert np.all(np.diff(series) >= -1e-12)
    assert capacity_overshoot(series, context.twin_spec.carrying_capacity) <= 1e-9


def test_twin_agents_realise_the_carried_biomass(context: ReleaseContext) -> None:
    trajectory = run_twin(context.twin_spec, context.schedule, 20260929)
    assert agent_area_error(trajectory) <= 1e-9
    assert biomass_of(trajectory.final().agents) > 0.0


def test_twin_axes_are_the_four_coordinates(context: ReleaseContext) -> None:
    trajectory = run_twin(context.twin_spec, context.schedule, 20260929)
    assert trajectory.axes_by_epoch().shape == (context.twin_spec.epochs, 4)


def test_gap_percent_is_zero_for_identical_series() -> None:
    series = np.linspace(0.1, 0.9, 8)
    assert gap_percent(series, series) == 0.0
