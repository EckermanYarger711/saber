"""Atlas priors for the four axes.

Ref: Sec. 2.3 -- "It has been demonstrated that ... ; they are all related to a
transcriptional program that was discovered in public, single-cell atlases made
from liver and liver tumour tissues, so therefore, the position of the sample on
an axis has a biological significance in addition to merely numerical one."
Algorithm 1's recalibration step re-ties a drifting axis to its prior ``p_k``.

The atlases supply the axis *semantics*, not numeric coordinates, and the
article prints no prior values. The shipped priors are therefore declared
engineering defaults on the unit interval the release's axes live on, and the
recalibration threshold is what decides when they are re-applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saber.belief.axes import AXIS_COUNT, AXIS_NAMES

Array = NDArray[np.float64]

ATLAS_SOURCES: tuple[tuple[str, str], ...] = (
    ("GSE115469", "human liver immune atlas, liver-resident immune and parenchymal profiles"),
    ("GSE146409", "human liver tumour-microenvironment atlas, tumour and adjacent tissue"),
    ("10.1038/s41586-019-1373-2", "human liver cell atlas, expected cell-type composition"),
)


@dataclass(frozen=True)
class AtlasPrior:
    """Per-axis prior means and standard deviations plus the axis order."""

    names: tuple[str, ...]
    means: Array
    standard_deviations: Array

    @property
    def dimension(self) -> int:
        return len(self.names)

    def variances(self) -> Array:
        return np.asarray(self.standard_deviations**2, dtype=np.float64)

    def standardise(self, values: Array) -> Array:
        return np.asarray((values - self.means) / self.standard_deviations, dtype=np.float64)

    def label(self) -> dict[str, object]:
        return {
            "axes": list(self.names),
            "means": [round(float(value), 9) for value in self.means],
            "standard_deviations": [round(float(value), 9) for value in self.standard_deviations],
            "sources": [accession for accession, _ in ATLAS_SOURCES],
        }


def default_atlas_prior() -> AtlasPrior:
    """The release's prior on the four axes.

    Oxygen is placed high because the medium is held at full tension at the rim;
    stiffness is centred because the matrix is the culture's own variable;
    co-culture is placed at a quarter because the article's mono-culture stratum
    is the reference; heterogeneity is placed low because a specimen is a single
    organoid line.
    """
    return AtlasPrior(
        names=AXIS_NAMES,
        means=np.asarray([0.85, 0.50, 0.25, 0.20], dtype=np.float64),
        standard_deviations=np.asarray([0.12, 0.15, 0.18, 0.10], dtype=np.float64),
    )


def prior_from_means(means: Array, scale: float = 0.15) -> AtlasPrior:
    """A prior of the right shape re-centred on supplied means."""
    values = np.asarray(means, dtype=np.float64)
    if values.size != AXIS_COUNT:
        raise ValueError("the prior needs one mean per axis")
    return AtlasPrior(
        names=AXIS_NAMES,
        means=values,
        standard_deviations=np.full(AXIS_COUNT, scale, dtype=np.float64),
    )
