"""Agent-oriented organoid growth.

Ref: Sec. 2.7 -- an agent-oriented growth model calibrated against public organoid
growth and size.

The population is grown logistically with the local oxygen availability as the
limiting factor, and the agents are the spatial realisation of that population:
their radii are solved so that the summed agent area reproduces the population
biomass exactly. Keeping the two views exactly consistent is what lets the
verification layer compare the agent loop against the logistic closed form
instead of against a tolerance chosen by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Agent:
    """One organoid instance: a disc of radius ``radius`` at ``position``."""

    position: tuple[float, float]
    radius: float

    def area(self) -> float:
        return float(np.pi * self.radius * self.radius)


@dataclass(frozen=True)
class GrowthRecord:
    """The twin's growth trajectory over a session."""

    times_hours: Array
    biomass: Array
    agent_counts: Array
    mean_radius: Array
    colony_radius: Array
    hypoxic_fraction: Array

    def final_biomass(self) -> float:
        return float(self.biomass[-1])

    def label(self) -> dict[str, float]:
        return {
            "final_biomass": round(self.final_biomass(), 9),
            "final_agents": int(self.agent_counts[-1]),
            "final_colony_radius": round(float(self.colony_radius[-1]), 9),
            "final_hypoxic_fraction": round(float(self.hypoxic_fraction[-1]), 9),
        }


def biomass_of(agents: tuple[Agent, ...]) -> float:
    return float(sum(agent.area() for agent in agents))


def colony_radius(agents: tuple[Agent, ...]) -> float:
    """Outer radius of the colony, disc edges included."""
    if not agents:
        return 0.0
    centroid = np.mean([agent.position for agent in agents], axis=0)
    return max(
        float(np.linalg.norm(np.asarray(agent.position) - centroid)) + agent.radius
        for agent in agents
    )


def seed_agents(
    radius: float, count: int, rng: np.random.Generator, spread: float
) -> tuple[Agent, ...]:
    """A small, centrally placed founding population."""
    offsets = rng.normal(scale=spread, size=(count, 2))
    return tuple(
        Agent(position=(float(offset[0]), float(offset[1])), radius=radius) for offset in offsets
    )


def logistic_reference(rate: float, capacity: float, initial: float, times: Array) -> Array:
    """Exact logistic trajectory, ``B(t) = K / (1 + (K/B0 - 1) e^{-r t})``."""
    ratio = capacity / initial - 1.0
    values = capacity / (1.0 + ratio * np.exp(-rate * np.asarray(times, dtype=np.float64)))
    return np.asarray(values, dtype=np.float64)


def occupancy_field(agents: tuple[Agent, ...], size: int, spacing: float) -> Array:
    """Per-pixel sink weight: 1 inside an agent's disc, smeared over its edge.

    Fractional coverage is used rather than a binary mask so that the oxygen
    solver's drainage does not jump when an agent's radius crosses a pixel
    boundary, which would put a step into the reaction-diffusion term.
    """
    axis = np.arange(size, dtype=np.float64) * spacing
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    sink = np.zeros((size, size), dtype=np.float64)
    for agent in agents:
        distance = np.sqrt((grid_x - agent.position[0]) ** 2 + (grid_y - agent.position[1]) ** 2)
        coverage = np.clip((agent.radius + 0.5 * spacing - distance) / spacing, 0.0, 1.0)
        sink = np.maximum(sink, coverage)
    return np.asarray(np.clip(sink, 0.0, 1.0), dtype=np.float64)


def _rescale(agents: tuple[Agent, ...], target: float) -> tuple[Agent, ...]:
    """Scale every radius so the summed area equals ``target``."""
    current = biomass_of(agents)
    if current <= 0.0 or target <= 0.0:
        return agents
    factor = float(np.sqrt(target / current))
    return tuple(Agent(position=agent.position, radius=agent.radius * factor) for agent in agents)


def step_agents(
    agents: tuple[Agent, ...],
    target_biomass: float,
    spacing: float,
    rng: np.random.Generator,
    detachment_rate: float = 0.0,
    hypoxic: float = 0.0,
    spawn_scale: float = 1.0,
) -> tuple[Agent, ...]:
    """Realise a new target biomass with the agent population.

    Agents detach at the rate the hypoxic fraction drives, and new agents appear
    on the colony's rim once the biomass per agent grows past what a single
    instance can carry, which is the crowding the article attributes to dense
    wells.
    """
    surviving = list(agents)
    if hypoxic > 0.0 and detachment_rate > 0.0 and surviving:
        count = int(round(hypoxic * detachment_rate * len(surviving)))
        for _ in range(min(count, len(surviving))):
            if not surviving:
                break
            victims = np.argsort([agent.radius for agent in surviving])
            surviving.pop(int(victims[0]))
    if not surviving:
        surviving = list(seed_agents(0.6 * spacing, 1, rng, 0.5 * spacing))
    current = biomass_of(tuple(surviving))
    mean_area = max(current / len(surviving), 1e-9)
    target_count = max(1, int(round(target_biomass / (mean_area * spawn_scale))))
    if target_count > len(surviving):
        centroid = np.mean([agent.position for agent in surviving], axis=0)
        for _ in range(target_count - len(surviving)):
            anchor = surviving[int(rng.integers(0, len(surviving)))]
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            direction = np.asarray([np.cos(angle), np.sin(angle)])
            base = np.asarray(anchor.position)
            rim = base + direction * (anchor.radius + spacing)
            drift = 0.2 * (rim - centroid)
            surviving.append(
                Agent(
                    position=(float(rim[0] + drift[0]), float(rim[1] + drift[1])),
                    radius=anchor.radius,
                )
            )
    elif target_count < len(surviving):
        order = np.argsort([-agent.radius for agent in surviving])
        keep = sorted(int(index) for index in order[:target_count])
        surviving = [surviving[index] for index in keep]
    return _rescale(tuple(surviving), target_biomass)


def logistic_step(
    biomass: float, rate: float, capacity: float, availability: float, dt: float
) -> float:
    """One explicit step of ``dB/dt = r * availability * B * (1 - B/K)``."""
    growth = rate * availability * biomass * (1.0 - biomass / capacity)
    return float(max(0.0, biomass + dt * growth))
