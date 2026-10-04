"""The four microenvironment axes.

Ref: Sec. 2.3 -- "``s_t``'s latent state is built on a four-dimensional
coordinate system, with each of its microenvironment axis representing one
coordinate; this system includes an oxygen regime coordinate highlighting
hypoxic and normoxic conditions, a matrix stiffness regime coordinate, a
co-culturing composition coordinate ... and an inter-organ heterogeneity index
encapsulating the findings of these parameters together on a specimen level."

The article also records that each axis is tied to a transcriptional program
found in public liver and liver-tumour atlases, so a coordinate has biological
rather than only numerical meaning. The axis semantics are what the priors in
:mod:`saber.belief.priors` encode.
"""

from __future__ import annotations

AXIS_NAMES: tuple[str, ...] = ("oxygen", "stiffness", "coculture", "heterogeneity")
AXIS_COUNT = len(AXIS_NAMES)


def axis_index(name: str) -> int:
    if name not in AXIS_NAMES:
        raise KeyError(f"{name} is not one of the four microenvironment axes")
    return AXIS_NAMES.index(name)


def describe() -> dict[str, str]:
    return {
        "oxygen": "oxygen regime: hypoxic at the low end, normoxic at the high end",
        "stiffness": "matrix stiffness regime of the surrounding gel",
        "coculture": "co-culture composition between mono-culture and stromal co-culture",
        "heterogeneity": "inter-organ heterogeneity index over the other three axes",
    }


def regime_label(axes: tuple[float, ...]) -> str:
    """The stratum name the article's regime stratification uses.

    Ref: Sec. 3.5 and the Discussion, which report the planner's advantage over
    the fixed cadence separately for the hypoxic, matrix-stiff, co-culture,
    immune-compartment and normoxic mono-culture strata.
    """
    oxygen, stiffness, coculture, _ = axes
    if oxygen < 0.35:
        return "hypoxic"
    if stiffness > 0.66:
        return "matrix-stiff"
    if coculture > 0.6:
        return "immune-compartment"
    if coculture > 0.35:
        return "coculture"
    return "normoxic-mono-culture"
