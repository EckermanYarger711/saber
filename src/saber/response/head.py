"""The response head.

Ref: Sec. 2.6 -- the head predicts the expected dose-response summary from the
belief and the drug representation; the unconditioned model "follows the same
structure and the same training dataset ... except for the fact that instead of
replacing the believed coordinates, the believed coordinates are omitted
completely". Sec. 2.6 also propagates the estimator's uncertainty: "we propagate
the uncertainty ``sigma_t`` into the predictions, using it as an input parameter
and providing results on the per-corpus basis rather than the pooled approach".

The summary's support is the dose range the corpus registry prints for the
response benchmark: a Hill fit over ``1e-10`` to ``1e-4`` molar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from saber.belief.estimator import Belief
from saber.response.fingerprint import Molecule
from saber.response.molecular import ATOM_SYMBOLS, GraphEncoder

Array = NDArray[np.float64]

DOSE_RANGE_M: tuple[float, float] = (1.0e-10, 1.0e-4)


def hill_curve(doses: Array, top: float, bottom: float, log_ic50: float, slope: float) -> Array:
    """Four-parameter Hill curve over log10 dose.

    ``response = bottom + (top - bottom) / (1 + 10 ** (slope * (log10 d - log IC50)))``.
    """
    log_dose = np.log10(np.clip(np.asarray(doses, dtype=np.float64), 1e-300, None))
    return np.asarray(
        bottom + (top - bottom) / (1.0 + 10.0 ** (slope * (log_dose - log_ic50))),
        dtype=np.float64,
    )


def hill_fit(doses: Array, response: Array) -> tuple[float, float, float, float]:
    """Least-squares Hill parameters, used to build the corpus's summary targets.

    The article's registry records that the benchmark's own fits are filtered at
    ``R2 >= 0.3``; the same statistic is returned by :func:`hill_r_squared` so a
    curve can be screened on the release side too.
    """
    from scipy.optimize import curve_fit

    log_dose = np.log10(np.clip(np.asarray(doses, dtype=np.float64), 1e-300, None))
    observed = np.asarray(response, dtype=np.float64)
    guess = (
        float(np.max(observed)),
        float(np.min(observed)),
        float(np.median(log_dose)),
        1.0,
    )

    def model(x: Array, top: float, bottom: float, log_ic50: float, slope: float) -> Array:
        return bottom + (top - bottom) / (1.0 + 10.0 ** (slope * (x - log_ic50)))

    fitted, _ = curve_fit(model, log_dose, observed, p0=guess, maxfev=20000)
    return (float(fitted[0]), float(fitted[1]), float(fitted[2]), float(fitted[3]))


def hill_r_squared(
    doses: Array, response: Array, parameters: tuple[float, float, float, float]
) -> float:
    from saber.evaluation.metrics import r_squared

    predicted = hill_curve(doses, *parameters)
    return r_squared(predicted, np.asarray(response, dtype=np.float64))


def summarise_curve(parameters: tuple[float, float, float, float], points: int) -> Array:
    """The curve's value at ``points`` log-spaced doses over the printed range.

    The summary is a curve sample rather than the parameters, because the
    article's head outputs a dose-response summary and the transfer statistic is
    computed on it.
    """
    log_low = np.log10(DOSE_RANGE_M[0])
    log_high = np.log10(DOSE_RANGE_M[1])
    doses = 10.0 ** np.linspace(log_low, log_high, points)
    return hill_curve(doses, *parameters)


def dose_response_summary(doses: Array, response: Array, points: int = 5) -> Array:
    """Fit a corpus curve and return its summary."""
    return summarise_curve(hill_fit(doses, response), points)


@dataclass(frozen=True)
class ResponsePrediction:
    """One predicted summary with the uncertainty it was propagated from."""

    summary: Array
    uncertainty: Array
    conditioned: bool

    def label(self) -> dict[str, object]:
        return {
            "conditioned": self.conditioned,
            "points": int(self.summary.size),
            "summary": [round(float(value), 9) for value in self.summary],
            "uncertainty": [round(float(value), 9) for value in self.uncertainty],
        }


class ResponseHead(nn.Module):
    """``f_phi``: the conditioned head and, by construction, its unconditioned control.

    The two variants share every weight and every input except the belief
    coordinates themselves, so the ablation's difference is the conditioning and
    nothing else.
    """

    def __init__(
        self,
        belief_dim: int,
        graph_dim: int,
        fingerprint_bits: int,
        hidden_dim: int,
        points: int,
        conditioned: bool,
        message_passing_rounds: int = 3,
    ) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.points = points
        self.belief_dim = belief_dim
        self.encoder = GraphEncoder(
            node_feature_dim=len(ATOM_SYMBOLS) + 1,
            hidden_dim=hidden_dim,
            rounds=message_passing_rounds,
            output_dim=graph_dim,
        )
        conditioned_width = 2 * belief_dim if conditioned else 0
        self.trunk = nn.Sequential(
            nn.Linear(graph_dim + fingerprint_bits + conditioned_width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, points),
        )

    def forward(self, belief: Belief, fingerprint: Array, molecule: Molecule) -> ResponsePrediction:
        predicted = self.predict_tensor(belief, fingerprint, molecule)
        summary = predicted.detach().cpu().numpy().astype(np.float64)
        return ResponsePrediction(
            summary=summary,
            uncertainty=np.asarray(belief.standard_deviation, dtype=np.float64),
            conditioned=self.conditioned,
        )

    def predict_tensor(self, belief: Belief, fingerprint: Array, molecule: Molecule) -> Tensor:
        """The head's raw output, which keeps the autograd graph for fitting."""
        graph: Tensor = self.encoder(molecule)
        pieces: list[Tensor] = [
            graph,
            torch.as_tensor(np.asarray(fingerprint, dtype=np.float32)),
        ]
        if self.conditioned:
            pieces.append(torch.as_tensor(belief.to_tensor(), dtype=torch.float32))
        joined: Tensor = torch.cat(pieces, dim=-1)
        predicted: Tensor = self.trunk(joined)
        return predicted


def predict_summaries(
    head: ResponseHead,
    beliefs: tuple[Belief, ...],
    fingerprints: Array,
    molecules: tuple[Molecule, ...],
) -> Array:
    """One predicted summary per (belief, molecule) pair."""
    rows = []
    for belief, fingerprint, molecule in zip(beliefs, fingerprints, molecules):
        rows.append(head(belief, fingerprint, molecule).summary)
    return np.asarray(np.stack(rows, axis=0), dtype=np.float64)
