"""The action set of the platform.

Ref: Eq. (1) --
``A = A_field x A_ch x A_exp x A_depth x A_fluid x A_dose``.

The abstract widens the allocation axis to "wells, fields, channels and
epochs", so the well index is carried as an outer factor of the field axis
here; the paper's ``A_field`` is read as the pair (well, field), which is the
only reading under which a single budget is shared across a plate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from saber.runtime.config import DishSpec

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

CHANNEL_EXCITATION = {
    "brightfield": 0.35,
    "phase_contrast": 0.55,
    "fluorescence": 1.0,
}


@dataclass(frozen=True, order=True)
class Action:
    """One instrument command.

    ``well`` and ``field`` index the stage's field axis; ``exposure_ms``,
    ``depth_um``, ``fluidics`` and ``dose_M`` are the remaining four factors of
    Eq. (1) that the instrument can vary per decision.
    """

    well: int
    field: int
    channel: int
    exposure_ms: float
    depth_um: float
    fluidics: int
    dose_M: float

    def key(self) -> tuple[int, int, int, float, float, int, float]:
        return (
            self.well,
            self.field,
            self.channel,
            self.exposure_ms,
            self.depth_um,
            self.fluidics,
            self.dose_M,
        )


@dataclass(frozen=True)
class ActionSpace:
    """An enumerated action set built from a dish configuration."""

    spec: DishSpec
    fields: tuple[tuple[int, int], ...]
    actions: tuple[Action, ...]

    @property
    def size(self) -> int:
        return len(self.actions)

    def partition(self) -> dict[str, int]:
        """Factor cardinalities, kept for the action-set size arithmetic."""
        return {
            "fields": len(self.fields),
            "channels": len(self.spec.channels),
            "exposures": len(self.spec.exposure_levels_ms),
            "depths": len(self.spec.axial_depths_um),
            "fluidics": len(self.spec.fluidics_options),
            "doses": len(self.spec.dose_levels_M),
        }

    def by_well(self) -> dict[int, tuple[Action, ...]]:
        buckets: dict[int, list[Action]] = {}
        for action in self.actions:
            buckets.setdefault(action.well, []).append(action)
        return {well: tuple(items) for well, items in buckets.items()}

    def field_actions(self, well: int, field: int) -> tuple[Action, ...]:
        return tuple(
            action for action in self.actions if action.well == well and action.field == field
        )

    def exposure_curve(self, well: int, field: int, channel: int) -> tuple[Action, ...]:
        """The exposure ladder of a fixed field and channel: the axis of Theorem 1."""
        return tuple(
            action
            for action in self.actions
            if action.well == well
            and action.field == field
            and action.channel == channel
            and action.depth_um == 0.0
            and action.fluidics == 0
        )


def enumerate_actions(spec: DishSpec) -> ActionSpace:
    """Enumerate ``A``.

    The enumeration order is fixed by the factor order of the configuration, so
    two runs over the same configuration produce the same action indices, which
    the planner's pruning relies on.
    """
    fields = tuple(
        (well, field) for well, field in product(range(spec.wells), range(spec.fields_of_view))
    )
    fluidics = (0, 1) if "exchange" in spec.fluidics_options else (0,)
    actions = tuple(
        Action(
            well=well,
            field=field,
            channel=channel,
            exposure_ms=exposure,
            depth_um=depth,
            fluidics=fluid,
            dose_M=dose,
        )
        for well, field in fields
        for channel in range(len(spec.channels))
        for exposure in spec.exposure_levels_ms
        for depth in spec.axial_depths_um
        for fluid in fluidics
        for dose in spec.dose_levels_M
    )
    return ActionSpace(spec=spec, fields=fields, actions=actions)


def well_of(space: ActionSpace) -> NDArray[np.int64]:
    """The well every action belongs to, in the action set's own order."""
    return np.asarray([action.well for action in space.actions], dtype=np.int64)


def well_slices(space: ActionSpace) -> tuple[NDArray[np.int64], ...]:
    """The action indices of each well, so a decision can be restricted to one well."""
    wells = well_of(space)
    return tuple(
        np.asarray(np.flatnonzero(wells == well), dtype=np.int64)
        for well in range(space.spec.wells)
    )


def travel_offsets(space: ActionSpace) -> Array:
    """Stage travel distance of every action from the home position, in fields."""
    return np.asarray(
        [float(action.well * space.spec.fields_of_view + action.field) for action in space.actions],
        dtype=np.float64,
    )


def reference_channel_count(spec: DishSpec) -> int:
    """How many channels the reference protocol images."""
    return int(min(spec.reference_channels, len(spec.channels)))


def reference_template_mask(space: ActionSpace, exposure_ms: float | None = None) -> Array:
    """Which actions are a reference-protocol frame.

    Ref: Sec. 3.1 -- the article's cadence photographs "the wells ... with the same
    channel and the same exposure for a defined number of times", delivering three
    channels in six hours per well. The template is therefore one field per well,
    on the first ``reference_channels`` channels, at a single exposure, in focus,
    without an exchange or a dose. The ``fixed cadence`` policy and the budget's
    arithmetic both read this mask, so the two cannot disagree about the protocol.
    """
    spec = space.spec
    level = (
        spec.exposure_levels_ms[len(spec.exposure_levels_ms) // 2]
        if exposure_ms is None
        else exposure_ms
    )
    flags = [
        action.field == 0
        and action.channel < reference_channel_count(spec)
        and abs(action.exposure_ms - level) <= 1e-9
        and abs(action.depth_um) <= 1e-9
        and action.fluidics == 0
        and abs(action.dose_M) <= 1e-300
        for action in space.actions
    ]
    return np.asarray(flags, dtype=bool)
