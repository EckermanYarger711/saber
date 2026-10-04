"""Oxygen and nutrient delivery by reaction-diffusion.

Ref: Sec. 2.7 -- the growth model is "linked to a reaction-diffusion approach to
supply of oxygen and nutrients". Sec. 4.1 quantifies how far the twin's own
growth sits from the logged growth, so the field solver is written to have a
closed-form equilibrium that the checks compare against rather than a tolerance
chosen by inspection.

The discretisation is an explicit five-point Laplacian with a zero-order sink
whose strength is the occupancy field, and Dirichlet boundaries held at the
medium's oxygen tension. :func:`stable_substep_hours` reports the explicit
scheme's stability limit so a configuration cannot silently run past it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class OxygenField:
    """A square, unit-side oxygen concentration field."""

    concentration: Array
    spacing: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.concentration.shape[0]), int(self.concentration.shape[1]))

    def mean(self) -> float:
        return float(np.mean(self.concentration))

    def minimum(self) -> float:
        return float(np.min(self.concentration))

    def with_concentration(self, concentration: Array) -> OxygenField:
        return OxygenField(concentration=concentration, spacing=self.spacing)

    def hypoxic_mask(self, threshold: float) -> Array:
        return np.asarray(self.concentration < threshold, dtype=bool)


def uniform_field(size: int, spacing: float, tension: float) -> OxygenField:
    """A field held uniformly at the medium's tension."""
    return OxygenField(
        concentration=np.full((size, size), tension, dtype=np.float64), spacing=spacing
    )


def boundary_value(field: OxygenField) -> float:
    """The medium tension the Dirichlet boundary holds."""
    return float(0.5 * (field.concentration[0, 0] + field.concentration[0, -1]))


def laplacian(field: Array, spacing: float) -> Array:
    """Five-point Laplacian; the boundary rows are handled by the caller."""
    value = np.zeros_like(field)
    value[1:-1, 1:-1] = (
        field[2:, 1:-1]
        + field[:-2, 1:-1]
        + field[1:-1, 2:]
        + field[1:-1, :-2]
        - 4.0 * field[1:-1, 1:-1]
    ) / (spacing * spacing)
    return np.asarray(value, dtype=np.float64)


def stable_substep_hours(diffusivity: float, spacing: float, safety: float = 0.4) -> float:
    """The explicit scheme's stability limit, ``h <= safety * dx^2 / (4 D)``."""
    if diffusivity <= 0.0:
        return float("inf")
    return safety * spacing * spacing / (4.0 * diffusivity)


def diffuse(
    field: OxygenField, diffusivity: float, consumption: float, drainage: Array, dt: float
) -> OxygenField:
    """One explicit step of ``dc/dt = D lap(c) - consumption * drainage``.

    ``drainage`` is the per-pixel sink weight; the growth layer supplies it from
    the occupied area, which is how the two halves of the twin are coupled.
    """
    boundary = boundary_value(field)
    value = field.concentration + dt * (
        diffusivity * laplacian(field.concentration, field.spacing) - consumption * drainage
    )
    value[0, :] = boundary
    value[-1, :] = boundary
    value[:, 0] = boundary
    value[:, -1] = boundary
    return field.with_concentration(np.asarray(np.clip(value, 0.0, None), dtype=np.float64))


def analytic_dirichlet_steady_state(
    size: int,
    spacing: float,
    boundary: float,
    consumption: float,
    diffusivity: float,
    terms: int = 41,
) -> Array:
    """Exact 2-D equilibrium of ``D lap(c) = U`` with ``c = boundary`` on the rim.

    Separating ``c = boundary - (U / D) g`` leaves ``lap(g) = -1`` with a zero
    rim, whose odd-mode sine series is

        ``g(x, y) = sum_{m,n odd} 16 sin(m pi x) sin(n pi y) / (pi^4 m n (m^2 + n^2))``.

    The series is the independent reference the solver is measured against: the
    one-dimensional quadratic is *not* a solution here, because it does not
    vanish along the two edges it is constant on.
    """
    axis = np.arange(size, dtype=np.float64) * spacing
    field = np.zeros((size, size), dtype=np.float64)
    modes = np.arange(1, terms + 1, 2, dtype=np.float64)
    for m in modes:
        sin_m = np.sin(m * np.pi * axis)
        for n in modes:
            coefficient = 16.0 / (np.pi**4 * m * n * (m * m + n * n))
            field += coefficient * np.outer(sin_m, np.sin(n * np.pi * axis))
    return np.asarray(boundary - (consumption / diffusivity) * field, dtype=np.float64)


def relax_to_equilibrium(
    field: OxygenField,
    diffusivity: float,
    consumption: float,
    drainage: Array,
    steps: int,
) -> OxygenField:
    """Advance the explicit scheme until the boundary-value problem settles."""
    dt = stable_substep_hours(diffusivity, field.spacing)
    relaxed = field
    for _ in range(steps):
        relaxed = diffuse(relaxed, diffusivity, consumption, drainage, dt)
    return relaxed


def equilibrium_error(field: OxygenField, consumption: float, diffusivity: float) -> float:
    """Largest deviation of the solver's field from the closed-form equilibrium."""
    reference = analytic_dirichlet_steady_state(
        field.shape[0], field.spacing, boundary_value(field), consumption, diffusivity
    )
    return float(np.max(np.abs(field.concentration - reference)))


def hypoxic_fraction(field: OxygenField, threshold: float) -> float:
    return float(np.mean(field.hypoxic_mask(threshold)))
