"""The shared release context every command builds on.

One run file selects one configuration of every block, and this module resolves
that selection once: the action set of Eq. (1), the damage functional, the
reference budget ``D``, the release cohort, the twin's session and the perception,
belief and response specifications. Commands then differ only in which part of the
context they exercise.

``scale`` bounds how many specimens of each corpus a command instantiates. The
article's evaluation covers all 4180; a command that reports a session-scale
number records the scale it actually ran at rather than implying the full one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from saber.belief.estimator import VariationalEstimator
from saber.belief.priors import AtlasPrior, default_atlas_prior
from saber.cohort.registry import CorpusCatalogue
from saber.cohort.release_cohort import ReleaseCohortSpec, assemble_release_cohort
from saber.cohort.schema import CohortRecord, SpecimenRecord
from saber.perception.head import PerceptionFrontend
from saber.planner.big import ActionSet, reference_schedule
from saber.planner.info_gain import encode_actions
from saber.response.head import ResponseHead
from saber.runtime.config import (
    BeliefSpec,
    DishSpec,
    PanelSpec,
    PerceptionSpec,
    PlannerSpec,
    ResponseSpec,
    RunSpec,
    TwinSpec,
    load_run_spec,
    repo_root,
)
from saber.runtime.seeding import set_seed
from saber.spec.actions import (
    ActionSpace,
    enumerate_actions,
    reference_template_mask,
    travel_offsets,
    well_slices,
)
from saber.spec.budget import SessionBudget, session_reference_budget
from saber.spec.damage import DamageFunctional
from saber.twin.clock import EpochClock, SessionSchedule

LOGGER = logging.getLogger("saber.context")

DEFAULT_RUN = "configs/run/primary.yaml"
SCALE_ALL = 0


@dataclass(frozen=True)
class ReleaseContext:
    """Everything a command needs, resolved from one run selection."""

    root: Path
    run: RunSpec
    panel: PanelSpec
    dish: DishSpec
    perception_spec: PerceptionSpec
    planner: PlannerSpec
    belief_spec: BeliefSpec
    response_spec: ResponseSpec
    twin_spec: TwinSpec
    catalogue: CorpusCatalogue
    cohort: CohortRecord
    space: ActionSpace
    functional: DamageFunctional
    action_set: ActionSet
    budget: SessionBudget
    schedule: SessionSchedule
    prior: AtlasPrior
    scale: int

    def clock(self) -> EpochClock:
        return self.schedule.clock

    def build_frontend(self) -> PerceptionFrontend:
        """Build the front end with its initialisation pinned to the run's seed.

        Every network in this release is an untrained engineering default: the
        article deposits no checkpoints, so the adapters' initialisation is a
        choice rather than a loaded state. Pinning it to the run's seed is what
        makes an instance count or an embedding norm a property of the release
        rather than of the process that happened to construct the module.
        """
        from saber.perception.adapters import LoRAConfig
        from saber.perception.backbones import BackboneConfig

        set_seed(self.run.seed)
        config = self.perception_spec
        return PerceptionFrontend(
            in_channels=len(config.channels),
            backbone=BackboneConfig(
                vision_width=config.vision_width,
                vision_depth=config.vision_depth,
                morphology_width=config.morphology_width,
                morphology_depth=config.morphology_depth,
                texture_scales=config.texture_scales,
            ),
            adapter=LoRAConfig(
                rank=max(1, config.adapter_rank),
                alpha=config.adapter_alpha,
                dropout=config.adapter_dropout,
            ),
            embedding_dim=config.embedding_dim,
            hidden=config.segmenter_hidden,
            use_adapters=config.use_adapters,
        )

    def build_estimator(self) -> VariationalEstimator:
        set_seed(self.run.seed)
        return VariationalEstimator(
            embedding_dim=self.perception_spec.embedding_dim,
            latent_dim=self.belief_spec.latent_dim,
            hidden_dim=self.belief_spec.hidden_dim,
            axes=len(self.belief_spec.axes),
        )

    def build_response_head(self) -> ResponseHead:
        set_seed(self.run.seed)
        return ResponseHead(
            belief_dim=len(self.belief_spec.axes),
            graph_dim=self.response_spec.hidden_dim // 2,
            fingerprint_bits=self.response_spec.fingerprint_bits,
            hidden_dim=self.response_spec.hidden_dim,
            points=self.response_spec.dose_summary_points,
            conditioned=self.response_spec.conditioned,
            message_passing_rounds=self.response_spec.message_passing_rounds,
        )

    def specimens(self) -> tuple[SpecimenRecord, ...]:
        return self.cohort.specimens

    def label(self) -> dict[str, object]:
        return {
            "run": self.run.label(),
            "panel": {"specimens": self.panel.specimens, "corpora": dict(self.panel.corpora)},
            "dish": {"wells": self.dish.wells, "actions": self.space.size},
            "budget": self.budget.label(),
            "session": self.schedule.label(),
            "scale": self.scale,
            "cohort": self.cohort.summary(),
        }


def slice_cohort(cohort: CohortRecord, scale: int) -> CohortRecord:
    """Bound a cohort to ``scale`` specimens per corpus.

    A zero scale means the whole cohort. The slice is taken in the cohort's own
    order so two commands at the same scale see the same specimens.
    """
    if scale <= 0:
        return cohort
    kept: list[SpecimenRecord] = []
    seen: dict[str, int] = {}
    for specimen in cohort.specimens:
        count = seen.get(specimen.corpus, 0)
        if count >= scale:
            continue
        seen[specimen.corpus] = count + 1
        kept.append(specimen)
    return CohortRecord(
        specimens=tuple(kept),
        epoch_hours=cohort.epoch_hours,
        dose_points=cohort.dose_points,
    )


def build_context(
    root: Path | None = None, run: str = DEFAULT_RUN, scale: int = SCALE_ALL
) -> ReleaseContext:
    """Resolve one run selection into a context."""
    release_root = repo_root(root) if root is not None else repo_root()
    run_path = Path(run)
    if not run_path.is_absolute():
        run_path = release_root / run_path
    spec = load_run_spec(run_path)
    panel = spec.panel()
    dish = spec.dish()
    catalogue = CorpusCatalogue.from_panel(panel)
    twin_spec = spec.twin()
    schedule = SessionSchedule.build(
        epochs=twin_spec.epochs,
        epoch_hours=twin_spec.epoch_hours,
        reference_epoch_hours=dish.reference_epoch_hours,
    )
    functional = DamageFunctional.build(dish)
    space = enumerate_actions(dish)
    action_set = ActionSet(
        damage=functional.table(),
        encodings=encode_actions(space.actions, dish),
        travel=travel_offsets(space),
        reference_mask=reference_template_mask(space),
        well_slices=well_slices(space),
        actions=space.actions,
    )
    planner_spec = spec.planner()
    reference_plan = reference_schedule(
        action_set, planner_spec.horizon, planner_spec.wells_per_cycle
    )
    budget = session_reference_budget(functional, reference_plan).scale(planner_spec.budget_scale)
    cohort = assemble_release_cohort(
        ReleaseCohortSpec.from_configs(panel, spec.response(), twin_spec, spec.seed), catalogue
    )
    context = ReleaseContext(
        root=release_root,
        run=spec,
        panel=panel,
        dish=dish,
        perception_spec=spec.perception(),
        planner=spec.planner(),
        belief_spec=spec.belief(),
        response_spec=spec.response(),
        twin_spec=twin_spec,
        catalogue=catalogue,
        cohort=slice_cohort(cohort, scale),
        space=space,
        functional=functional,
        action_set=action_set,
        budget=budget,
        schedule=schedule,
        prior=default_atlas_prior(),
        scale=scale,
    )
    LOGGER.info(
        "context: run=%s actions=%d budget=%.6g specimens=%d",
        spec.name,
        space.size,
        budget.total,
        context.cohort.size(),
    )
    return context
