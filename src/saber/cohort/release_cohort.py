"""The release cohort and its on-disk form.

The article's evaluation is a retrospective analysis of previously logged
trajectories that are not deposited, and its supplementary registry names the
public archives those trajectories would be assembled from. The release cannot
re-derive the study's own trajectories, so it runs on a schema-compatible cohort
cohort built from its own configuration: the corpus sizes, epoch count and dose
support are the article's printed values, and every axis value, observation and
dose-response summary is the release's own.

Nothing read from this module is a manuscript result. The cohort is the release's
substrate; the checks that compare the manuscript's own arithmetic are separate
and live in :mod:`saber.verification.manuscript`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saber.cohort.registry import CorpusCatalogue
from saber.cohort.schema import CohortRecord, SpecimenRecord
from saber.response.head import DOSE_RANGE_M, hill_curve
from saber.runtime.atomic import write_json, write_text
from saber.runtime.config import PanelSpec, ResponseSpec, TwinSpec
from saber.twin.clock import SessionSchedule
from saber.twin.coupled import run_twin

Array = NDArray[np.float64]

STRATUM_RATES: tuple[tuple[str, float, tuple[float, float, float, float]], ...] = (
    ("normoxic-mono-culture", 0.08, (0.88, 0.45, 0.15, 0.12)),
    ("coculture", 0.17, (0.80, 0.52, 0.42, 0.20)),
    ("immune-compartment", 0.24, (0.72, 0.58, 0.70, 0.28)),
    ("matrix-stiff", 0.33, (0.66, 0.78, 0.38, 0.34)),
    ("hypoxic", 0.41, (0.24, 0.62, 0.30, 0.46)),
)

AXIS_DRIFT = 0.035
OBSERVATION_NOISE = 0.045
COCULTURE_WEIGHT = 0.9
OXYGEN_WEIGHT = 0.7


@dataclass(frozen=True)
class ReleaseCohortSpec:
    """The parameters of the cohort: the article's printed counts and the session."""

    panel: PanelSpec
    response: ResponseSpec
    epochs: int
    epoch_hours: float
    release_seed: int

    @classmethod
    def from_configs(
        cls,
        panel: PanelSpec,
        response: ResponseSpec,
        twin: TwinSpec,
        seed: int,
    ) -> ReleaseCohortSpec:
        return cls(
            panel=panel,
            response=response,
            epochs=twin.epochs,
            epoch_hours=twin.epoch_hours,
            release_seed=seed,
        )

    def label(self) -> dict[str, object]:
        return {
            "corpora": dict(self.panel.corpora),
            "specimens": self.panel.specimens,
            "epochs": self.epochs,
            "epoch_hours": self.epoch_hours,
            "modality": self.panel.modality,
            "seed": self.release_seed,
        }


def _stratum_for(
    corpus_rank: int, index: int
) -> tuple[str, float, tuple[float, float, float, float]]:
    """Assign the five strata round-robin, rotated per corpus.

    Rotating by the corpus keeps every stratum represented in a corpus of any
    size -- including the bounded slices a command runs at -- so a stratum-level
    read-out is never an artefact of how many specimens were instantiated.
    """
    return STRATUM_RATES[(index + corpus_rank) % len(STRATUM_RATES)]


def _trajectory(
    rng: np.random.Generator,
    epochs: int,
    stratum: str,
    rate: float,
    centre: tuple[float, float, float, float],
) -> Array:
    """An axis trajectory whose regime changes at the stratum's transition rate.

    The regime changes are what the article's frontier result is measured against:
    a specimen in the hypoxic stratum crosses a regime boundary roughly every two
    epochs while a normoxic mono-culture specimen barely moves.
    """
    axes = np.zeros((epochs, 4), dtype=np.float64)
    current = np.asarray(centre, dtype=np.float64)
    drift = np.zeros(4, dtype=np.float64)
    for epoch in range(epochs):
        if rng.random() < rate:
            drift = rng.normal(scale=AXIS_DRIFT * 4.0, size=4)
        current = np.clip(current + drift * 0.5 + rng.normal(scale=AXIS_DRIFT, size=4), 0.02, 0.98)
        axes[epoch] = current
    if stratum == "hypoxic":
        axes[:, 0] = np.clip(axes[:, 0] - 0.25, 0.02, 0.98)
    return axes


def assemble_release_cohort(spec: ReleaseCohortSpec, catalogue: CorpusCatalogue) -> CohortRecord:
    """Assemble the cohort: one record per printed specimen."""
    rng = np.random.default_rng(spec.release_seed)
    specimens: list[SpecimenRecord] = []
    for corpus_rank, (corpus, count) in enumerate(sorted(catalogue.counts.items())):
        for index in range(count):
            stratum, rate, centre = _stratum_for(corpus_rank, index)
            axes = _trajectory(rng, spec.epochs, stratum, rate, centre)
            observations = np.clip(
                axes + rng.normal(scale=OBSERVATION_NOISE, size=axes.shape), 0.0, 1.0
            )
            top = float(rng.uniform(0.85, 1.05))
            bottom = float(rng.uniform(0.05, 0.25))
            log_ic50 = -6.0 + 2.0 * (
                COCULTURE_WEIGHT * (axes[-1, 2] - 0.4) + OXYGEN_WEIGHT * (0.5 - axes[-1, 0])
            )
            slope = float(rng.uniform(0.8, 1.6))
            doses = np.logspace(
                np.log10(DOSE_RANGE_M[0]),
                np.log10(DOSE_RANGE_M[1]),
                spec.response.dose_summary_points,
            )
            summary = hill_curve(doses, top, bottom, log_ic50, slope)
            specimens.append(
                SpecimenRecord(
                    identity=f"{corpus}-{index + 1:04d}",
                    corpus=corpus,
                    stratum=stratum,
                    transition_rate=rate,
                    axes=axes,
                    observations=observations,
                    dose_response=summary,
                )
            )
    return CohortRecord(
        specimens=tuple(specimens),
        epoch_hours=spec.epoch_hours,
        dose_points=spec.response.dose_summary_points,
    )


def write_cohort(cohort: CohortRecord, root: object) -> tuple[str, str]:
    """Persist the cohort as a compressed array archive plus a JSON index.

    The archive holds the numeric arrays and the index holds the string fields and
    the shapes, so a reader can load either without parsing the other.
    """
    from pathlib import Path

    destination = Path(str(root))
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Array] = {}
    for specimen in cohort.specimens:
        arrays[f"axes/{specimen.identity}"] = specimen.axes
        arrays[f"observations/{specimen.identity}"] = specimen.observations
        arrays[f"dose_response/{specimen.identity}"] = specimen.dose_response
    archive = destination / "release_cohort.npz"
    writer: Any = np.savez_compressed
    writer(archive, **arrays)
    index = {
        "epoch_hours": cohort.epoch_hours,
        "dose_points": cohort.dose_points,
        "specimens": [specimen.label() for specimen in cohort.specimens],
    }
    index_path = destination / "cohort_index.json"
    write_json(index_path, index)
    write_text(
        destination / "corpus_counts.txt",
        "\n".join(f"{corpus} {len(cohort.by_corpus(corpus))}" for corpus in cohort.corpora())
        + "\n",
    )
    return (archive.name, index_path.name)


def read_cohort(root: object) -> CohortRecord:
    """Read back a persisted cohort."""
    from pathlib import Path

    destination = Path(str(root))
    index = json.loads((destination / "cohort_index.json").read_text(encoding="utf-8"))
    archive = np.load(destination / "release_cohort.npz")
    specimens = tuple(
        SpecimenRecord(
            identity=entry["identity"],
            corpus=entry["corpus"],
            stratum=entry["stratum"],
            transition_rate=float(entry["transition_rate"]),
            axes=np.asarray(archive[f"axes/{entry['identity']}"], dtype=np.float64),
            observations=np.asarray(archive[f"observations/{entry['identity']}"], dtype=np.float64),
            dose_response=np.asarray(
                archive[f"dose_response/{entry['identity']}"], dtype=np.float64
            ),
        )
        for entry in index["specimens"]
    )
    return CohortRecord(
        specimens=specimens,
        epoch_hours=float(index["epoch_hours"]),
        dose_points=int(index["dose_points"]),
    )


def release_cohort_twin_axes(
    spec: ReleaseCohortSpec, schedule: SessionSchedule, twin_spec: TwinSpec, seed: int
) -> Array:
    """The twin's own axis trajectory, used as the ground truth the twin supplies.

    Ref: Sec. 2.7 -- the twin is "the source of the ground truth policy value".
    """
    trajectory = run_twin(twin_spec, schedule, seed)
    return trajectory.axes_by_epoch()
