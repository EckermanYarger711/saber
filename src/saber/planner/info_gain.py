"""Expected information gain of an action under a belief.

Ref: Sec. 2.5 -- "In ``G(a | b) = E_o[H[b] - H[b' | a, o]]``, the decision
currency is known as ``rho(a | b) = G(a | b)/C(a)``"; Sec. 1.1.2 -- the planner
"maximising ``I(s_t; o_t | b_t, a_t)``" spends damage on the latent state rather
than on pixel-space entropy.

An action is mapped to an observation design over the four axes and to a noise
variance that grows with the delivered dose, which is the mechanism the
Introduction names: "the observation noise ... increases according to the dose
administered". One action delivers one scalar observation of a linear combination
of the latent axes, so the expected gain is the exact rank-one expression

    ``G(a | b) = 0.5 * log(1 + d(a)^T Sigma d(a) / nu(a))``.

The rank-one reading is what makes Theorem 1's premise hold: the accumulated
precision is the prior precision plus the exposure count times ``d d^T / nu``, and
the log-determinant of an additively updated precision is concave in that count.
The verification layer recomputes the same quantity from the matrix-valued Kalman
form in :mod:`saber.theory.information`, so the sweep is measured against an
independent implementation rather than against itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.belief.axes import AXIS_COUNT
from saber.belief.estimator import Belief
from saber.runtime.config import DishSpec
from saber.spec.actions import CHANNEL_EXCITATION, Action

Array = NDArray[np.float64]

DESIGN_SCALE = 1.25
BASELINE_NOISE = 0.10
SHORT_EXPOSURE_PENALTY = 3.0
DEPTH_NOISE_COEFFICIENT = 0.001

CHANNEL_DESIGN: dict[str, tuple[float, float, float, float]] = {
    "brightfield": (0.35, 0.10, 0.10, 0.20),
    "phase_contrast": (0.25, 0.45, 0.20, 0.25),
    "fluorescence": (0.15, 0.30, 0.60, 0.35),
}


def observation_noise(action: Action, spec: DishSpec) -> float:
    """``nu(a)``: the observation noise of the frame the action commands.

    Ref: the Introduction's mechanism -- "the observation noise ... increases
    according to the dose administered" -- read together with Sec. 2.5's statement
    that the noise rises with the exposure. A short or defocused frame is a noisier
    estimate of the microenvironment than a long, focused one, which is the trade
    the damage functional exists to price.

    The same function is the estimator's measurement noise in
    :func:`saber.planner.rollout.logged_observer`, so the planner's model of its own
    observation cannot drift from the filter it is planning against.
    """
    level = max(spec.exposure_levels_ms)
    fraction = min(1.0, action.exposure_ms / level) if level > 0.0 else 0.0
    return float(
        BASELINE_NOISE * (1.0 + SHORT_EXPOSURE_PENALTY * (1.0 - fraction))
        + DEPTH_NOISE_COEFFICIENT * abs(action.depth_um)
    )


def design_vector(action: Action, spec: DishSpec) -> Array:
    """The action's design over the four axes.

    A bright-field frame carries the oxygen and heterogeneity information an
    unstained culture can show; phase contrast adds the matrix boundary; a
    fluorescence frame reports the co-culture composition directly. The
    amplitudes are fixed by the modality and the exposure and depth scale them,
    which is why the same channel at a longer exposure is a different design
    rather than a repetition of the same one.
    """
    excitation = CHANNEL_EXCITATION[spec.channels[action.channel]]
    exposure = min(1.0, action.exposure_ms / max(spec.exposure_levels_ms))
    focus = 1.0 / (1.0 + abs(action.depth_um) / 10.0)
    base = CHANNEL_DESIGN.get(spec.channels[action.channel], (0.25,) * AXIS_COUNT)
    return np.asarray(
        np.asarray(base, dtype=np.float64) * excitation * exposure * focus * DESIGN_SCALE,
        dtype=np.float64,
    )


@dataclass(frozen=True)
class EncodingTable:
    """Designs and noise variances for a frozen action set, as two arrays."""

    designs: Array
    noises: Array

    @property
    def size(self) -> int:
        return int(self.designs.shape[0])

    def quadratic(self, belief: Belief, mask: Array | None = None) -> Array:
        """``d(a)^T Sigma d(a)`` for every action, optionally over a subset of axes."""
        weights = belief.variance() if mask is None else belief.variance() * mask
        total = np.asarray(self.designs**2 * weights[None, :], dtype=np.float64)
        return np.asarray(np.sum(total, axis=1), dtype=np.float64)

    def gains(self, belief: Belief, mask: Array | None = None) -> Array:
        """``G(a | b)`` for every action."""
        ratio = self.quadratic(belief, mask) / self.noises
        return np.asarray(0.5 * np.log1p(ratio), dtype=np.float64)

    def gain(self, index: int, belief: Belief) -> float:
        return float(self.gains(belief)[index])

    def posterior(self, belief: Belief, index: int) -> Belief:
        """The belief an action's observation would leave, in expectation.

        The mean is unchanged because the observation is centred on the belief's
        own mean; only the variance contracts, by the same factor the gain was
        computed from.
        """
        variance = belief.variance()
        quadratic = float(self.quadratic(belief)[index])
        contracted = variance - (variance * self.designs[index]) ** 2 / (
            self.noises[index] + quadratic
        )
        return Belief(
            mean=belief.mean.copy(),
            standard_deviation=np.asarray(
                np.sqrt(np.clip(contracted, 0.0, None)), dtype=np.float64
            ),
        )

    def pruned(self, belief: Belief, top_k: int) -> Array:
        """The action-set pruning the article's planner operates under.

        Ref: Sec. 2.5 -- "In accordance with the planning horizon and action-set
        pruning parameters, one decision can be made within the control window."
        The prune is by raw information gain, before the belief-value and damage
        terms, so the search width is bounded without biasing what follows.
        """
        gains = self.gains(belief)
        if top_k >= gains.size:
            return np.asarray(np.arange(gains.size), dtype=np.int64)
        order = np.argsort(-gains, kind="stable")
        return np.asarray(np.sort(order[:top_k]), dtype=np.int64)

    def label(self) -> dict[str, float]:
        return {
            "actions": self.size,
            "noise_min": round(float(self.noises.min()), 9),
            "noise_max": round(float(self.noises.max()), 9),
        }


def encode_actions(actions: tuple[Action, ...], spec: DishSpec) -> EncodingTable:
    """Designs and noise variances of a frozen action set."""
    designs = np.asarray([design_vector(action, spec) for action in actions], dtype=np.float64)
    noises = np.asarray([observation_noise(action, spec) for action in actions], dtype=np.float64)
    return EncodingTable(designs=designs, noises=noises)
