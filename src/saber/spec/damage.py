"""The damage functional.

Ref: Eq. (2) and Eq. (6) (the article prints the same functional twice) --

    C(a) = w_e d(a) + w_d z(a) + w_m 1[a in A_fluid] + w_s tau(a)

with ``d(a)`` the delivered photon dose, ``z(a)`` the out-of-focus axial depth,
``tau(a)`` the stage transition time, and the four weights non-negative and
"instrumental in carrying the required units for expressing the above four
parameters as a single normalised dosage" (Sec. 2.4). Normalisation rescales
each term by its own maximum over the action set, so the shipped weights are
dimensionless trade-offs between the four damage channels rather than unit
conversions.

A plate's action set is enumerated once and the four terms are kept as columns,
so a planner's feasible set is a comparison against a precomputed cost vector
rather than one Python call per action per step.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.runtime.config import DishSpec
from saber.spec.actions import CHANNEL_EXCITATION, Action, enumerate_actions

Array = NDArray[np.float64]


def photon_dose(action: Action, spec: DishSpec) -> float:
    """``d(a)``: the supplied photon dose, in seconds of excitation.

    The excitation coefficient of a channel multiplies the commanded exposure;
    fluorescence is the term the article singles out as the phototoxic one
    (Introduction; Sec. 2.2), and it is normalised to unit excitation.
    """
    channel = spec.channels[action.channel]
    return action.exposure_ms * CHANNEL_EXCITATION[channel] / 1000.0


def defocus_depth(action: Action) -> float:
    """``z(a)``: the out-of-focus axial depth of the commanded plane."""
    return abs(action.depth_um)


def stage_time(action: Action, spec: DishSpec) -> float:
    """``tau(a)``: stage travel time to the commanded field.

    Measured from the home position, so the functional stays a function of the
    action alone as Eq. (2) prints it; a session-relative reading would make
    ``C`` depend on the order of the actions and Eq. (2) is not written that way.
    """
    offset = action.well * spec.fields_of_view + action.field
    return spec.stage_seconds_per_field * float(offset)


@dataclass(frozen=True)
class Normalisation:
    """Per-term maxima over an action set, used to form the normalised dosage."""

    dose: float
    depth: float
    stage: float

    @classmethod
    def of(cls, actions: Iterable[Action], spec: DishSpec) -> Normalisation:
        doses = [photon_dose(action, spec) for action in actions]
        depths = [defocus_depth(action) for action in actions]
        stages = [stage_time(action, spec) for action in actions]
        return cls(
            dose=max(doses) if doses else 1.0,
            depth=max(depths) if depths else 1.0,
            stage=max(stages) if stages else 1.0,
        )

    def scale(self) -> tuple[float, float, float]:
        return (self.dose or 1.0, self.depth or 1.0, self.stage or 1.0)


@dataclass(frozen=True)
class DamageFunctional:
    """``C`` for one dish configuration, with the action set frozen at construction."""

    spec: DishSpec
    actions: tuple[Action, ...]
    normalisation: Normalisation

    @classmethod
    def build(cls, spec: DishSpec) -> DamageFunctional:
        space = enumerate_actions(spec)
        return cls(
            spec=spec,
            actions=space.actions,
            normalisation=Normalisation.of(space.actions, spec),
        )

    def terms(self, action: Action) -> tuple[float, float, float, float]:
        """The four normalised terms of Eq. (2), before the weights."""
        dose_scale, depth_scale, stage_scale = self.normalisation.scale()
        return (
            photon_dose(action, self.spec) / dose_scale,
            defocus_depth(action) / depth_scale,
            float(action.fluidics),
            stage_time(action, self.spec) / stage_scale,
        )

    def of(self, action: Action) -> float:
        weights = self.spec.weights
        dose, depth, fluid, stage = self.terms(action)
        return (
            weights.exposure * dose
            + weights.defocus * depth
            + weights.fluidics * fluid
            + weights.stage * stage
        )

    def of_sequence(self, actions: Iterable[Action]) -> float:
        return float(sum(self.of(action) for action in actions))

    def table(self) -> DamageTable:
        """The whole action set's costs as one vector."""
        rows = np.asarray([self.terms(action) for action in self.actions], dtype=np.float64)
        return DamageTable(rows=rows, weights=self.spec.weights.as_tuple())

    def uniform_weights(self) -> DamageFunctional:
        """The same functional with all four weights equal.

        Used by the ``uniform weights`` ablation of Table 2, where the article
        reports that the damage weighting itself carries information.
        """
        from saber.runtime.config import DamageWeights

        weight = sum(self.spec.weights.as_tuple()) / 4.0
        uniform = DamageWeights(exposure=weight, defocus=weight, fluidics=weight, stage=weight)
        spec = DishSpec(
            wells=self.spec.wells,
            fields_of_view=self.spec.fields_of_view,
            channels=self.spec.channels,
            exposure_levels_ms=self.spec.exposure_levels_ms,
            axial_depths_um=self.spec.axial_depths_um,
            fluidics_options=self.spec.fluidics_options,
            dose_levels_M=self.spec.dose_levels_M,
            reference_channels=self.spec.reference_channels,
            reference_epoch_hours=self.spec.reference_epoch_hours,
            stage_seconds_per_field=self.spec.stage_seconds_per_field,
            weights=uniform,
        )
        return DamageFunctional(spec=spec, actions=self.actions, normalisation=self.normalisation)

    def currency(self, gain: float, action: Action) -> float:
        """``rho(a | b) = G(a | b) / C(a)``, Eq. (4)."""
        cost = self.of(action)
        if cost <= 0.0:
            return float("inf") if gain > 0.0 else 0.0
        return gain / cost


@dataclass(frozen=True)
class DamageTable:
    """Per-action damage for a frozen action set, as four columns and their weights."""

    rows: Array
    weights: tuple[float, float, float, float]

    @property
    def costs(self) -> Array:
        weight_vector = np.asarray(self.weights, dtype=np.float64)
        return np.asarray(self.rows @ weight_vector, dtype=np.float64)

    def cost(self, index: int) -> float:
        return float(self.costs[index])

    def feasible(self, remaining: float) -> Array:
        return np.asarray(self.costs <= remaining + 1e-12, dtype=bool)

    def feasible_indices(self, remaining: float) -> Array:
        return np.asarray(np.flatnonzero(self.feasible(remaining)), dtype=np.int64)

    def label(self) -> dict[str, float]:
        costs = self.costs
        return {
            "actions": int(costs.size),
            "minimum": round(float(costs.min()), 9),
            "maximum": round(float(costs.max()), 9),
            "total": round(float(costs.sum()), 9),
        }


def damage_of(action: Action, spec: DishSpec) -> float:
    """Convenience constructor for a single action outside a frozen action set."""
    functional = DamageFunctional.build(spec)
    return functional.of(action)
