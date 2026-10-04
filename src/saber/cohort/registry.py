"""The catalogued corpus and the resource registry.

Ref: Sec. 3.1 -- the test set is "4180 tracked organoid samples collected from five
publicly available organoid imaging datasets. Corpora A, B, C, D, and E includes
1246, 873, 964, 621 and 476 specimens respectively", plus "five public response
corpora which were published through a common benchmark"; Supplementary Table A1
-- the registry of every resource with "its version, accession or URL, licence and
access date".

The registry is transcribed here because it is the release's provenance record:
every external resource the article names appears once, with the licence it was
catalogued under and the role it plays. A resource whose licence could not be
established is recorded as catalogued-without-licence rather than dropped, so the
transcription cannot be read as a claim that it was used.
"""

from __future__ import annotations

from dataclasses import dataclass

from saber.runtime.config import PanelSpec


@dataclass(frozen=True)
class Resource:
    """One entry of the registry."""

    name: str
    version: str
    licence: str
    accessed: str
    role: str

    def label(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "accessed": self.accessed,
            "role": self.role,
        }


RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="Image Data Resource",
        version="per-study idrNNNN",
        licence="CC BY 4.0",
        accessed="2026-09-27",
        role="perception pre-training and cross-dataset transfer",
    ),
    Resource(
        name="BioImage Archive",
        version="live archive",
        licence="mostly CC0 / CC BY",
        accessed="2026-09-27",
        role="supplementary perception evaluation",
    ),
    Resource(
        name="Brain-organoid imaging set",
        version="10.1038/s41597-024-03330-z",
        licence="CC BY 4.0",
        accessed="2026-09-27",
        role="tracking benchmark, 1400 frames of 64 trackable organoids from four clones",
    ),
    Resource(
        name="Cell Tracking Challenge",
        version="continuously updated",
        licence="CC BY 4.0",
        accessed="2026-09-27",
        role="joint segmentation and tracking ceiling reference",
    ),
    Resource(
        name="LIVECell",
        version="10.1038/s41592-021-01249-6",
        licence="CC BY 4.0",
        accessed="2026-09-27",
        role="label-free segmentation ceiling reference",
    ),
    Resource(
        name="CPJUMP1 (Cell Painting)",
        version="JUMP public release",
        licence="CC0 / CC BY 4.0",
        accessed="2026-09-27",
        role="morphology-encoder pre-training",
    ),
    Resource(
        name="IMPROVE response benchmark",
        version="benchmark-data-pilot1",
        licence="per-source licences",
        accessed="2026-09-27",
        role="primary response corpora, Hill fits at R2 >= 0.3 over 1e-10 to 1e-4 M",
    ),
    Resource(
        name="Human liver immune atlas (MacParland)",
        version="GSE115469",
        licence="GEO public access",
        accessed="2026-09-27",
        role="microenvironment axis semantics",
    ),
    Resource(
        name="Human liver tumour-microenvironment atlas",
        version="GSE146409",
        licence="GEO public access",
        accessed="2026-09-27",
        role="co-culture composition prior",
    ),
    Resource(
        name="Human liver cell atlas",
        version="10.1038/s41586-019-1373-2",
        licence="public",
        accessed="2026-09-27",
        role="expected cell-type composition prior",
    ),
    Resource(
        name="Public robot trajectories",
        version="LeRobot format",
        licence="CC BY 4.0 / Apache-2.0",
        accessed="2026-09-27",
        role="actuation pre-training and the logged-trajectory evaluation corpus",
    ),
    Resource(
        name="ARCHS4",
        version="live public",
        licence="public",
        accessed="2026-09-27",
        role="auxiliary uniformly reprocessed bulk transcriptomes",
    ),
    Resource(
        name="PRISM",
        version="10.1038/s43018-019-0018-6",
        licence="public supplementary",
        accessed="2026-09-27",
        role="auxiliary viability profiling corpus",
    ),
    Resource(
        name="L1000 / CMap",
        version="CLUE portal",
        licence="public portal terms",
        accessed="2026-09-27",
        role="auxiliary perturbation transcriptomics",
    ),
    Resource(
        name="DepMap",
        version="10.1016/j.cell.2017.06.010",
        licence="CC BY 4.0",
        accessed="2026-09-27",
        role="auxiliary multi-omics features",
    ),
    Resource(
        name="TissueNet",
        version="10.1038/s41587-021-01094-0",
        licence="CC BY 4.0",
        accessed="2026-09-27",
        role="auxiliary tissue segmentation pre-training",
    ),
)

CORPUS_ROLES: dict[str, str] = {
    "A": "cell tracking and label-free segmentation benchmark, brightest specimen counts",
    "B": "cell tracking benchmark, phase-contrast arm",
    "C": "perturbation-morphology corpus, the densest response source",
    "D": "perturbation-morphology corpus, second response source",
    "E": "label-free corpus, the smallest source",
}


@dataclass(frozen=True)
class CorpusCatalogue:
    """The five imaging corpora and their printed specimen counts."""

    counts: dict[str, int]
    modality: str

    @classmethod
    def from_panel(cls, panel: PanelSpec) -> CorpusCatalogue:
        return cls(counts=dict(panel.corpora), modality=panel.modality)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def roles(self) -> dict[str, str]:
        return {name: CORPUS_ROLES[name] for name in self.counts}

    def label(self) -> dict[str, object]:
        return {
            "counts": dict(self.counts),
            "total": self.total,
            "modality": self.modality,
            "roles": self.roles(),
        }


def registry_rows() -> list[dict[str, str]]:
    return [resource.label() for resource in RESOURCES]


def resources_without_licence_position() -> tuple[str, ...]:
    """Registry entries whose catalogue position is a licence-unknown holder.

    The table's own caption records the policy: "A resource whose licence could
    not be established at access time was excluded from the primary layer rather
    than used silently." The registry therefore has to be able to say which of
    its entries are not in the primary layer.
    """
    return tuple(
        resource.name
        for resource in RESOURCES
        if resource.licence == "public" or resource.licence == "GEO public access"
    )
