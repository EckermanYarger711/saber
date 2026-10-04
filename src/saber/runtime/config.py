"""Typed configuration surface.

Every leaf of every shipped YAML is read through this module; the integrity
check ``configuration_leaves_are_consumed`` walks the same files and fails if a
declared key reaches no consumer, so a config file cannot drift away from the
code that is supposed to obey it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_TITLE = (
    "A Closed-Loop Bio-Robotic Platform for Monitoring Liver Cancer Organoids "
    "and Tumor-Microenvironment-Aware Drug Screening"
)
RELEASE_SLUG = "saber"


def repo_root(anchor: Path | None = None) -> Path:
    """Locate the release root from any file inside it.

    The anchor defaults to this module, which sits three levels below the root.
    """
    if anchor is None:
        return Path(__file__).resolve().parents[3]
    resolved = anchor.resolve()
    if resolved.is_dir():
        return resolved
    return resolved.parent


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not hold a mapping")
    return payload


@dataclass(frozen=True)
class PanelSpec:
    """The evaluation substrate: five corpora and their specimen splits.

    Ref: Sec. 3.1 (test set of 4180 tracked organoid samples across corpora
    A-E); Supplementary Table A1 (IMPROVE row prints the 0.8/0.1/0.1 splits).
    """

    corpora: dict[str, int]
    modality: str
    frames_per_specimen: int
    frame_size: int
    split_fractions: tuple[float, float, float]
    specimen_group_split: bool

    @classmethod
    def load(cls, path: Path) -> PanelSpec:
        block = _load(path)["panel"]
        fractions = block["split_fractions"]
        return cls(
            corpora={str(key): int(value) for key, value in block["corpora"].items()},
            modality=str(block["modality"]),
            frames_per_specimen=int(block["frames_per_specimen"]),
            frame_size=int(block["frame_size"]),
            split_fractions=(float(fractions[0]), float(fractions[1]), float(fractions[2])),
            specimen_group_split=bool(block["specimen_group_split"]),
        )

    @property
    def specimens(self) -> int:
        return int(sum(self.corpora.values()))

    def names(self) -> tuple[str, ...]:
        return tuple(self.corpora)


@dataclass(frozen=True)
class DamageWeights:
    """The four non-negative weights of Eq. (2).

    The article states that the weights are calibrated from instrument register
    logs but does not print them; the shipped values are engineering defaults.
    """

    exposure: float
    defocus: float
    fluidics: float
    stage: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.exposure, self.defocus, self.fluidics, self.stage)


@dataclass(frozen=True)
class DishSpec:
    """The action set of Eq. (1) and the reference protocol that fixes D.

    Ref: Eq. (1); Sec. 2.4 (``D`` equals the impact of a benchmark protocol over
    the same wall-clock window); Sec. 3.1 (three channels in six hours).
    """

    wells: int
    fields_of_view: int
    channels: tuple[str, ...]
    exposure_levels_ms: tuple[float, ...]
    axial_depths_um: tuple[float, ...]
    fluidics_options: tuple[str, ...]
    dose_levels_M: tuple[float, ...]
    reference_channels: int
    reference_epoch_hours: float
    stage_seconds_per_field: float
    weights: DamageWeights

    @classmethod
    def load(cls, path: Path) -> DishSpec:
        block = _load(path)["dish"]
        weights = block["damage_weights"]
        reference = block["reference_protocol"]
        return cls(
            wells=int(block["wells"]),
            fields_of_view=int(block["fields_of_view"]),
            channels=tuple(str(name) for name in block["channels"]),
            exposure_levels_ms=tuple(float(value) for value in block["exposure_levels_ms"]),
            axial_depths_um=tuple(float(value) for value in block["axial_depths_um"]),
            fluidics_options=tuple(str(name) for name in block["fluidics_options"]),
            dose_levels_M=tuple(float(value) for value in block["dose_levels_M"]),
            reference_channels=int(reference["channels"]),
            reference_epoch_hours=float(reference["epoch_hours"]),
            stage_seconds_per_field=float(block["stage_seconds_per_field"]),
            weights=DamageWeights(
                exposure=float(weights["exposure"]),
                defocus=float(weights["defocus"]),
                fluidics=float(weights["fluidics"]),
                stage=float(weights["stage"]),
            ),
        )


@dataclass(frozen=True)
class PlannerSpec:
    """The receding-horizon planner of Eq. (7) and Algorithm 2.

    ``variant`` selects which of the compared policies in Table 1 is built;
    the remaining fields are shared by every variant so the comparison is
    iso-resource rather than iso-step-settings.
    """

    variant: str
    discount: float
    horizon: int
    wells_per_cycle: int
    dual_step: float
    lambda_init: float
    saturation_threshold: float
    prune_top_k: int
    control_window_s: float
    budget_scale: float
    belief_value: bool
    uncertainty_propagation: bool
    twin_based_training: bool
    currency: str

    @classmethod
    def load(cls, path: Path) -> PlannerSpec:
        block = _load(path)["planner"]
        return cls(
            variant=str(block["variant"]),
            discount=float(block["discount"]),
            horizon=int(block["horizon"]),
            wells_per_cycle=int(block["wells_per_cycle"]),
            dual_step=float(block["dual_step"]),
            lambda_init=float(block["lambda_init"]),
            saturation_threshold=float(block["saturation_threshold"]),
            prune_top_k=int(block["prune_top_k"]),
            control_window_s=float(block["control_window_s"]),
            budget_scale=float(block["budget_scale"]),
            belief_value=bool(block["belief_value"]),
            uncertainty_propagation=bool(block["uncertainty_propagation"]),
            twin_based_training=bool(block["twin_based_training"]),
            currency=str(block["currency"]),
        )


@dataclass(frozen=True)
class PerceptionSpec:
    """The perception front end of Sec. 2.2, including its ablations.

    ``use_adapters`` and ``swap_morphology`` are the two Tier-3 switches of
    Table 2: "frozen backbones, no adapters" and "morphology backbone B (swap)".
    ``channels`` is the inference modality set, so the label-free-only row and
    the label-free-plus-fluorescence row are two configurations rather than two
    code paths.
    """

    vision_width: int
    vision_depth: int
    morphology_width: int
    morphology_depth: int
    texture_scales: int
    adapter_rank: int
    adapter_alpha: float
    adapter_dropout: float
    embedding_dim: int
    segmenter_hidden: int
    tracker_gate: float
    channels: tuple[str, ...]
    auxiliary_channels: tuple[str, ...]
    use_adapters: bool
    swap_morphology: bool

    @classmethod
    def load(cls, path: Path) -> PerceptionSpec:
        block = _load(path)["perception"]
        return cls(
            vision_width=int(block["vision_width"]),
            vision_depth=int(block["vision_depth"]),
            morphology_width=int(block["morphology_width"]),
            morphology_depth=int(block["morphology_depth"]),
            texture_scales=int(block["texture_scales"]),
            adapter_rank=int(block["adapter_rank"]),
            adapter_alpha=float(block["adapter_alpha"]),
            adapter_dropout=float(block["adapter_dropout"]),
            embedding_dim=int(block["embedding_dim"]),
            segmenter_hidden=int(block["segmenter_hidden"]),
            tracker_gate=float(block["tracker_gate"]),
            channels=tuple(str(name) for name in block["channels"]),
            auxiliary_channels=tuple(str(name) for name in block["auxiliary_channels"]),
            use_adapters=bool(block["use_adapters"]),
            swap_morphology=bool(block["swap_morphology"]),
        )


@dataclass(frozen=True)
class BeliefSpec:
    """The four-axis variational estimator of Sec. 2.3 and Algorithm 1."""

    axes: tuple[str, ...]
    recalibration_threshold: float
    convergence_epsilon: float
    convergence_patience: int
    latent_dim: int
    hidden_dim: int
    kl_weight: float
    auxiliary_prediction: bool

    @classmethod
    def load(cls, path: Path) -> BeliefSpec:
        block = _load(path)["belief"]
        return cls(
            axes=tuple(str(name) for name in block["axes"]),
            recalibration_threshold=float(block["recalibration_threshold"]),
            convergence_epsilon=float(block["convergence_epsilon"]),
            convergence_patience=int(block["convergence_patience"]),
            latent_dim=int(block["latent_dim"]),
            hidden_dim=int(block["hidden_dim"]),
            kl_weight=float(block["kl_weight"]),
            auxiliary_prediction=bool(block["auxiliary_prediction"]),
        )


@dataclass(frozen=True)
class ResponseSpec:
    """The microenvironment-conditioned response head of Sec. 2.6 and 3.6."""

    conditioned: bool
    fingerprint_bits: int
    fingerprint_radius: int
    message_passing_rounds: int
    node_feature_dim: int
    hidden_dim: int
    dose_summary_points: int
    effective_sample_floor: float
    transfer_metric: str

    @classmethod
    def load(cls, path: Path) -> ResponseSpec:
        block = _load(path)["response"]
        return cls(
            conditioned=bool(block["conditioned"]),
            fingerprint_bits=int(block["fingerprint_bits"]),
            fingerprint_radius=int(block["fingerprint_radius"]),
            message_passing_rounds=int(block["message_passing_rounds"]),
            node_feature_dim=int(block["node_feature_dim"]),
            hidden_dim=int(block["hidden_dim"]),
            dose_summary_points=int(block["dose_summary_points"]),
            effective_sample_floor=float(block["effective_sample_floor"]),
            transfer_metric=str(block["transfer_metric"]),
        )


@dataclass(frozen=True)
class TwinSpec:
    """The organoid-growth / oxygen-delivery twin of Sec. 2.7.

    The twin supplies the ground-truth policy value and the training
    environment for the planner; its own growth fidelity is what the release
    evaluates against, rather than assumes.
    """

    grid: int
    epochs: int
    epoch_hours: float
    oxygen_diffusivity: float
    oxygen_consumption: float
    growth_rate: float
    carrying_capacity: float
    detachment_rate: float
    stiffness_relaxation: float
    seed_radius: float
    hypoxic_threshold: float

    @classmethod
    def load(cls, path: Path) -> TwinSpec:
        block = _load(path)["twin"]
        return cls(
            grid=int(block["grid"]),
            epochs=int(block["epochs"]),
            epoch_hours=float(block["epoch_hours"]),
            oxygen_diffusivity=float(block["oxygen_diffusivity"]),
            oxygen_consumption=float(block["oxygen_consumption"]),
            growth_rate=float(block["growth_rate"]),
            carrying_capacity=float(block["carrying_capacity"]),
            detachment_rate=float(block["detachment_rate"]),
            stiffness_relaxation=float(block["stiffness_relaxation"]),
            seed_radius=float(block["seed_radius"]),
            hypoxic_threshold=float(block["hypoxic_threshold"]),
        )


@dataclass(frozen=True)
class RunSpec:
    """One named run: the observation budget setting plus its blocks."""

    name: str
    kind: str
    title: str
    seed: int
    seed_count: int
    panel_path: Path
    dish_path: Path
    perception_path: Path
    planner_path: Path
    belief_path: Path
    response_path: Path
    twin_path: Path
    root: Path = field(repr=False, default=Path("."))

    def panel(self) -> PanelSpec:
        return PanelSpec.load(self.root / self.panel_path)

    def dish(self) -> DishSpec:
        return DishSpec.load(self.root / self.dish_path)

    def perception(self) -> PerceptionSpec:
        return PerceptionSpec.load(self.root / self.perception_path)

    def planner(self) -> PlannerSpec:
        return PlannerSpec.load(self.root / self.planner_path)

    def belief(self) -> BeliefSpec:
        return BeliefSpec.load(self.root / self.belief_path)

    def response(self) -> ResponseSpec:
        return ResponseSpec.load(self.root / self.response_path)

    def twin(self) -> TwinSpec:
        return TwinSpec.load(self.root / self.twin_path)

    def block_paths(self) -> tuple[Path, ...]:
        return (
            self.panel_path,
            self.dish_path,
            self.perception_path,
            self.planner_path,
            self.belief_path,
            self.response_path,
            self.twin_path,
        )

    def label(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "title": self.title, "seed": self.seed}


def load_run_spec(path: Path) -> RunSpec:
    """Read a run file and resolve its block paths against the release root."""
    resolved = path.resolve()
    root = resolved
    for _ in range(4):
        if (root / "configs").is_dir() and (root / "pyproject.toml").is_file():
            break
        root = root.parent
    payload = _load(resolved)
    run = payload["run"]
    blocks = payload["blocks"]
    relative = {key: Path(str(value)) for key, value in blocks.items()}
    return RunSpec(
        name=str(run["name"]),
        kind=str(run["kind"]),
        title=str(run["title"]),
        seed=int(run["seed"]),
        seed_count=int(run["seed_count"]),
        panel_path=relative["panel"],
        dish_path=relative["dish"],
        perception_path=relative["perception"],
        planner_path=relative["planner"],
        belief_path=relative["belief"],
        response_path=relative["response"],
        twin_path=relative["twin"],
        root=root,
    )
