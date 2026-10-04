"""The action set, the damage functional, the budget and the session's arithmetic."""

from __future__ import annotations

import numpy as np

from saber.cli.context import ReleaseContext
from saber.planner.big import reference_schedule
from saber.spec.actions import (
    Action,
    enumerate_actions,
    reference_channel_count,
    reference_template_mask,
    travel_offsets,
    well_slices,
)
from saber.spec.budget import fixed_cadence_damage, session_reference_budget
from saber.spec.damage import DamageFunctional, photon_dose, stage_time


def test_action_set_is_the_product_of_its_factors(context: ReleaseContext) -> None:
    space = enumerate_actions(context.dish)
    factors = space.partition()
    product = 1
    for value in factors.values():
        product *= value
    assert space.size == product


def test_well_slices_partition_the_action_set(context: ReleaseContext) -> None:
    space = enumerate_actions(context.dish)
    slices = well_slices(space)
    joined = np.concatenate(slices)
    assert len(slices) == context.dish.wells
    assert np.array_equal(np.sort(joined), np.arange(space.size))
    for well, entry in enumerate(slices):
        assert {space.actions[index].well for index in entry} == {well}


def test_reference_template_is_one_field_per_well(context: ReleaseContext) -> None:
    space = enumerate_actions(context.dish)
    mask = reference_template_mask(space)
    expected = context.dish.wells * reference_channel_count(context.dish)
    assert int(np.count_nonzero(mask)) == expected
    assert all(space.actions[index].field == 0 for index in np.flatnonzero(mask))


def test_damage_is_a_function_of_the_action_alone(context: ReleaseContext) -> None:
    functional = DamageFunctional.build(context.dish)
    action = functional.actions[123]
    assert functional.of(action) == functional.of(action)
    assert functional.of(action) >= 0.0


def test_photon_dose_scales_with_exposure(context: ReleaseContext) -> None:
    spec = context.dish
    dim = Action(0, 0, 0, spec.exposure_levels_ms[0], 0.0, 0, 0.0)
    bright = Action(0, 0, 0, spec.exposure_levels_ms[-1], 0.0, 0, 0.0)
    assert photon_dose(bright, spec) > photon_dose(dim, spec)


def test_stage_time_is_measured_from_home(context: ReleaseContext) -> None:
    spec = context.dish
    home = Action(0, 0, 0, spec.exposure_levels_ms[0], 0.0, 0, 0.0)
    far = Action(3, 4, 0, spec.exposure_levels_ms[0], 0.0, 0, 0.0)
    assert stage_time(home, spec) == 0.0
    assert stage_time(far, spec) > stage_time(home, spec)


def test_travel_offsets_match_the_action_fields(context: ReleaseContext) -> None:
    space = enumerate_actions(context.dish)
    offsets = travel_offsets(space)
    for index in (0, 7, space.size - 1):
        action = space.actions[index]
        assert offsets[index] == float(action.well * context.dish.fields_of_view + action.field)


def test_budget_equals_the_reference_sequence_damage(context: ReleaseContext) -> None:
    functional = context.functional
    schedule = reference_schedule(
        context.action_set, context.planner.horizon, context.planner.wells_per_cycle
    )
    computed = session_reference_budget(functional, schedule)
    assert computed.total == context.budget.total


def test_session_horizon_covers_two_wells_per_epoch(context: ReleaseContext) -> None:
    assert context.planner.horizon % context.planner.wells_per_cycle == 0
    epochs = context.planner.horizon // context.planner.wells_per_cycle
    assert epochs == context.twin_spec.epochs


def test_fixed_cadence_damage_scales_with_epochs(context: ReleaseContext) -> None:
    one = fixed_cadence_damage(context.dish, 1, context.functional)
    two = fixed_cadence_damage(context.dish, 2, context.functional)
    assert abs(two - 2.0 * one) <= 1e-9


def test_uniform_weights_change_the_costs(context: ReleaseContext) -> None:
    uniform = context.functional.uniform_weights()
    sample = context.functional.actions[::4096]
    assert any(abs(uniform.of(action) - context.functional.of(action)) > 1e-12 for action in sample)


def test_action_ordering_is_deterministic(context: ReleaseContext) -> None:
    first = enumerate_actions(context.dish).actions
    second = enumerate_actions(context.dish).actions
    assert first == second
