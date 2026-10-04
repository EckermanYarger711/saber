"""The damage budget ``D`` and the iso-resource arithmetic.

Ref: Sec. 2.4 -- "The D budget then becomes equal to the total impact of a
benchmark protocol during the same wall-clock time frame; therefore, every
comparison conducted in Section 3 can be classified as an iso-resource
comparison instead of iso-step comparison." Sec. 3.1 fixes the benchmark
protocol: the reference delivers three channels in six hours.

The consequence that the release leans on is arithmetic: with ``D`` defined as
the fixed-cadence protocol's own damage, the fixed-cadence policy in Table 1
consumes exactly 100% of the budget, which is what Panel B prints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.runtime.config import DishSpec
from saber.spec.actions import Action
from saber.spec.damage import DamageFunctional

REFERENCE_PROTOCOL = "fixed cadence (every 6 h, 3 channels)"


def reference_frames(functional: DamageFunctional) -> tuple[Action, ...]:
    """The reference protocol's actions for a single epoch.

    The selection is read off the same template mask the ``fixed cadence`` policy
    uses (:func:`saber.spec.actions.reference_template_mask`), so the budget the
    article defines as "the total damage that a reference protocol would incur"
    and the damage the fixed-cadence row of Table 1 actually spends cannot drift
    apart.
    """
    from saber.spec.actions import ActionSpace, reference_template_mask

    space = ActionSpace(
        spec=functional.spec,
        fields=tuple(
            (well, field)
            for well in range(functional.spec.wells)
            for field in range(functional.spec.fields_of_view)
        ),
        actions=functional.actions,
    )
    mask = reference_template_mask(space)
    return tuple(action for action, selected in zip(functional.actions, mask) if bool(selected))


@dataclass(frozen=True)
class SessionBudget:
    """A plate's damage budget and the protocol that defines it."""

    total: float
    epochs: int
    per_epoch: float
    protocol: str

    def remaining(self, spent: float) -> float:
        return max(0.0, self.total - spent)

    def usage_percent(self, spent: float) -> float:
        if self.total <= 0.0:
            return 0.0
        return 100.0 * spent / self.total

    def scale(self, factor: float) -> SessionBudget:
        """A budget widened by ``factor``, used for the unbinding-budget panel."""
        return SessionBudget(
            total=self.total * factor,
            epochs=self.epochs,
            per_epoch=self.per_epoch * factor,
            protocol=f"{self.protocol} x{factor:g}",
        )

    def label(self) -> dict[str, float | int | str]:
        return {
            "protocol": self.protocol,
            "epochs": self.epochs,
            "total": round(self.total, 9),
            "per_epoch": round(self.per_epoch, 9),
        }


def reference_budget(
    spec: DishSpec, epochs: int, functional: DamageFunctional | None = None
) -> SessionBudget:
    """``D`` for one plate under the article's own definition."""
    damage = functional or DamageFunctional.build(spec)
    per_epoch = damage.of_sequence(reference_frames(damage))
    return SessionBudget(
        total=per_epoch * float(epochs),
        epochs=epochs,
        per_epoch=per_epoch,
        protocol=REFERENCE_PROTOCOL,
    )


def fixed_cadence_damage(
    spec: DishSpec, epochs: int, functional: DamageFunctional | None = None
) -> float:
    """Total damage the reference protocol incurs over a session of ``epochs``.

    The budget and this quantity are computed from the same reference frames, so
    the fixed-cadence policy in Table 1 spends the budget exactly.
    """
    damage = functional or DamageFunctional.build(spec)
    return damage.of_sequence(reference_frames(damage)) * float(epochs)


def session_reference_budget(
    functional: DamageFunctional, schedule: Sequence[Sequence[int] | NDArray[np.int64]]
) -> SessionBudget:
    """``D`` for an executed session: the reference protocol's own sequence damage.

    Ref: Sec. 2.4 -- ``D`` equals "the total impact of a benchmark protocol during
    the same wall-clock time frame". The sequence is evaluated action by action, so
    the budget and the ``fixed cadence`` row of Table 1 cannot disagree: the cadence
    is that sequence.
    """
    per_decision = [
        float(sum(functional.of(functional.actions[int(index)]) for index in decision))
        for decision in schedule
    ]
    total = float(sum(per_decision))
    epochs = len(schedule)
    return SessionBudget(
        total=total,
        epochs=epochs,
        per_epoch=total / float(epochs) if epochs else 0.0,
        protocol=REFERENCE_PROTOCOL,
    )


def iso_resource_ratio(spent: float, reference: float) -> float:
    """The iso-resource factor of two policies, ``spent / reference``."""
    if reference <= 0.0:
        return 0.0
    return spent / reference


def exposure_count(actions: tuple[Action, ...]) -> int:
    """Frames that actually deliver photons; fluid exchange and dose do not."""
    return sum(1 for action in actions if action.exposure_ms > 0.0)


def saved_exposure_percent(
    policy_actions: tuple[Action, ...], reference: tuple[Action, ...]
) -> float:
    """Exposures saved against the reference protocol at matched accuracy.

    Ref: Table 1 caption -- "'Saved' is the reduction in exposures relative to
    the fixed-cadence protocol at equal state-estimation accuracy."
    """
    baseline = exposure_count(reference)
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - exposure_count(policy_actions)) / baseline
