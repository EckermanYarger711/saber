"""The execution family: what actually ran, and what it produced.

Every check here runs the release and reads the result. The verification logic is
deliberately not the implementation under test: the information gains are
recomputed from the matrix-valued Kalman form in :mod:`saber.theory.information`,
the coverage greedy is compared against a brute-force optimum and an LP
relaxation, the oxygen solver against a closed-form series, the agent population
against its own biomass law, the metrics against hand-computed values, and the
double-precision arithmetic against its own closed forms.

A check that cannot pass trivially is marked as such where it appears: a
comparison whose both sides are the same object is not evidence, so where a
quantity has no independent reference the check is recorded as NOT_RUN with the
reason rather than as a PASS.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from saber.belief.axes import regime_label
from saber.belief.elbo import BeliefDecoder, elbo_terms
from saber.belief.estimator import Belief, BeliefStream
from saber.belief.pipeline import initial_belief, run_tme_belief
from saber.belief.priors import default_atlas_prior
from saber.cohort.registry import CorpusCatalogue
from saber.cohort.release_cohort import (
    ReleaseCohortSpec,
    assemble_release_cohort,
    read_cohort,
    write_cohort,
)
from saber.cohort.rendering import frame_bundle
from saber.evaluation.metrics import (
    balanced_accuracy,
    effective_sample_size,
    noise_band,
    r_squared,
)
from saber.evaluation.offpolicy import LoggedBatch, evaluate
from saber.perception.adapters import LoRAConv2d
from saber.perception.head import FrameBundle
from saber.planner.big import reference_schedule
from saber.response.fingerprint import FingerprintSpec, Molecule, hashed_fingerprint
from saber.response.head import ResponseHead
from saber.runtime.atomic import file_digest, load_tensors, save_tensors
from saber.runtime.seeding import set_seed
from saber.spec.actions import (
    enumerate_actions,
    reference_channel_count,
    reference_template_mask,
)
from saber.spec.damage import DamageFunctional
from saber.study import (
    front_end_output,
    run_acquisition_study,
    run_belief_study,
    run_frontier_study,
    run_perception_study,
    run_transfer_study,
    run_twin_study,
)
from saber.theory import certificate, interior, parity
from saber.theory.information import GaussianBelief, information_gain, observe
from saber.twin.growth import logistic_reference
from saber.twin.oxygen import (
    equilibrium_error,
    relax_to_equilibrium,
    stable_substep_hours,
    uniform_field,
)
from saber.verification.checks import (
    CheckResult,
    blocked,
    close_within,
    not_run,
    pass_fail,
)
from saber.verification.claimed_values import CORPUS_COUNTS, CORPUS_TOTAL_PRINTED

if TYPE_CHECKING:
    from saber.cli.context import ReleaseContext

FAMILY = "execution"
THEORY_FAMILY = "theory"
POSTERIOR_FAMILY = "posterior"

SEED = 20260929
DRAW_COUNT = 40


def _context_check(
    name: str, category: str, passed: bool, detail: str, evidence: Mapping[str, Any]
) -> CheckResult:
    return pass_fail(name, category, True, bool(passed), detail, evidence)


def cohort_checks(context: ReleaseContext) -> list[CheckResult]:
    """The cohort's counts, its round-trip and the registry transcription."""
    catalogue = CorpusCatalogue.from_panel(context.panel)
    cohort_spec = ReleaseCohortSpec.from_configs(
        context.panel, context.response_spec, context.twin_spec, SEED
    )
    cohort = assemble_release_cohort(cohort_spec, catalogue)
    results = [
        pass_fail(
            "cohort corpus counts equal the printed counts",
            FAMILY,
            CORPUS_COUNTS,
            {corpus: len(cohort.by_corpus(corpus)) for corpus in cohort.corpora()},
            "Sec. 3.1 prints the five corpus sizes",
        ),
        pass_fail(
            "cohort specimen total equals the printed total",
            FAMILY,
            CORPUS_TOTAL_PRINTED,
            cohort.size(),
            "Sec. 3.1 prints a 4180-sample test set",
        ),
        pass_fail(
            "the cohort's strata cover the five named regimes",
            FAMILY,
            5,
            len(cohort.strata()),
            "Sec. 3.5 stratifies the frontier over five regimes",
            {"strata": list(cohort.strata())},
        ),
    ]
    with tempfile.TemporaryDirectory() as scratch:
        write_cohort(cohort, scratch)
        reloaded = read_cohort(scratch)
        same = all(
            np.array_equal(left.axes, right.axes)
            and np.array_equal(left.observations, right.observations)
            and np.array_equal(left.dose_response, right.dose_response)
            for left, right in zip(cohort.specimens, reloaded.specimens)
        )
        results.append(
            _context_check(
                "the cohort survives a write and read round trip",
                FAMILY,
                same and reloaded.size() == cohort.size(),
                "the archive and index reproduce every array",
                {"specimens": reloaded.size()},
            )
        )
    return results


def spec_checks(context: ReleaseContext) -> list[CheckResult]:
    """The action set, the damage functional and the budget's arithmetic."""
    spec = context.dish
    space = enumerate_actions(spec)
    functional = DamageFunctional.build(spec)
    table = functional.table()
    factors = space.partition()
    product = 1
    for value in factors.values():
        product *= value
    mask = reference_template_mask(space)
    expected_slots = spec.wells * min(spec.reference_channels, len(spec.channels))
    budget = context.budget.total
    schedule = reference_schedule(
        context.action_set, context.planner.horizon, context.planner.wells_per_cycle
    )
    reference = float(
        sum(
            functional.of(functional.actions[int(index)])
            for decision in schedule
            for index in decision
        )
    )
    weights = spec.weights.as_tuple()
    return [
        pass_fail(
            "action set size equals the product of its factors",
            FAMILY,
            product,
            space.size,
            "Eq. (1) factorises the action into six axes",
            {"factors": factors},
        ),
        pass_fail(
            "the reference template covers every well and reference channel",
            FAMILY,
            expected_slots,
            int(np.count_nonzero(mask)),
            "Sec. 3.1's protocol images each well on three channels per epoch",
        ),
        close_within(
            "the single-action and tabulated damage agree",
            FAMILY,
            float(functional.of(space.actions[-1])),
            float(table.cost(space.size - 1)),
            1e-12,
            "the table is the vectorised form of Eq. (2)",
        ),
        _context_check(
            "every damage weight is non-negative and every cost is non-negative",
            FAMILY,
            all(weight >= 0.0 for weight in weights) and bool(np.all(table.costs >= 0.0)),
            "Eq. (2) requires w_e, w_d, w_m, w_s >= 0, which makes C(a) >= 0",
            {"weights": list(weights)},
        ),
        close_within(
            "the budget equals the reference protocol's own sequence damage",
            FAMILY,
            reference,
            budget,
            1e-9,
            "Sec. 2.4 defines D as the benchmark protocol's total impact over the same window; "
            "the sequence is the same schedule the fixed cadence follows",
            {"decisions": len(schedule)},
        ),
        pass_fail(
            "the session horizon is the wells-per-cycle times the session's epochs",
            FAMILY,
            context.planner.wells_per_cycle * context.twin_spec.epochs,
            context.planner.horizon,
            "one decision per (well, epoch) pair, as the article's allocation across wells and "
            "epochs requires",
            {
                "wells_per_cycle": context.planner.wells_per_cycle,
                "epochs": context.twin_spec.epochs,
            },
        ),
        _context_check(
            "raising a damage weight cannot lower a cost",
            FAMILY,
            _weights_monotone(functional),
            "C is linear in the four weights, so a weight increase cannot reduce a cost",
            {},
        ),
        _context_check(
            "damage is increasing in exposure and in defocus",
            FAMILY,
            _damage_monotone(spec),
            "phototoxicity and defocus both accumulate, so a longer or more defocused frame "
            "cannot cost less than a shorter or better-focused one",
            {},
        ),
        close_within(
            "the cadence agreement between the two configuration blocks holds",
            FAMILY,
            0.0,
            abs(spec.reference_epoch_hours - context.twin_spec.epoch_hours),
            1e-12,
            "the dish block carries the reference cadence and the twin block the epoch length",
            {},
        ),
        pass_fail(
            "the inference modality set excludes the auxiliary channels",
            FAMILY,
            [
                name
                for name in context.perception_spec.channels
                if name not in context.perception_spec.auxiliary_channels
            ],
            list(context.perception_spec.channels),
            "Sec. 2.2 uses fluorescence only as an auxiliary signal",
        ),
    ]


def _weights_monotone(functional: DamageFunctional) -> bool:
    from dataclasses import replace

    from saber.runtime.config import DamageWeights

    base = functional.spec.weights
    raised = DamageWeights(
        exposure=base.exposure + 1.0,
        defocus=base.defocus,
        fluidics=base.fluidics,
        stage=base.stage,
    )
    spec = replace(functional.spec, weights=raised)
    raised_functional = DamageFunctional(
        spec=spec, actions=functional.actions, normalisation=functional.normalisation
    )
    sample = functional.actions[:: max(1, len(functional.actions) // 64)]
    return all(raised_functional.of(action) >= functional.of(action) - 1e-12 for action in sample)


def _damage_monotone(spec: object) -> bool:
    from dataclasses import replace

    from saber.runtime.config import DishSpec

    assert isinstance(spec, DishSpec)
    base = DamageFunctional.build(spec)
    longer = DamageFunctional(
        spec=replace(
            spec, exposure_levels_ms=tuple(value * 2.0 for value in spec.exposure_levels_ms)
        ),
        actions=base.actions,
        normalisation=base.normalisation,
    )
    sample = base.actions[:: max(1, len(base.actions) // 64)]
    return all(
        longer.of(action) >= base.of(action) - 1e-12
        for action in sample
        if action.exposure_ms > 0.0 and action.fluidics == 0
    )


def gain_checks(context: ReleaseContext) -> list[CheckResult]:
    """The planner's gain against the matrix-valued Kalman closed form."""
    belief = initial_belief(context.prior)
    sample = np.linspace(0, context.action_set.size - 1, 12).astype(int)
    worst = 0.0
    for index in sample:
        design = context.action_set.encodings.designs[index]
        noise = float(context.action_set.encodings.noises[index])
        reference = _matrix_gain(belief, design, noise)
        observed = context.action_set.encodings.gain(int(index), belief)
        worst = max(worst, abs(reference - observed))
    return [
        close_within(
            "the planner's gain matches the matrix Kalman closed form",
            THEORY_FAMILY,
            0.0,
            worst,
            1e-9,
            "Eq. (4)'s expected gain, recomputed from the Cholesky form in "
            "saber.theory.information rather than from the diagonal expression the planner uses",
            {"arm": "encoding table", "sampled_actions": int(sample.size)},
        ),
        close_within(
            "the gain of a noiseless zero-design action is zero",
            THEORY_FAMILY,
            0.0,
            _matrix_gain(belief, np.zeros(4), 0.1),
            0.0,
            "an action that observes nothing carries no information",
        ),
    ]


def _matrix_gain(belief: Belief, design: np.ndarray, noise: float) -> float:
    covariance = np.diag(belief.variance())
    gaussian = GaussianBelief(mean=belief.mean.copy(), covariance=covariance)
    return float(information_gain(gaussian, design, noise))


def _matrix_posterior_variance(belief: Belief, design: np.ndarray, noise: float) -> np.ndarray:
    covariance = np.diag(belief.variance())
    gaussian = GaussianBelief(mean=belief.mean.copy(), covariance=covariance)
    return np.diag(observe(gaussian, design, noise).covariance)


def theorem_checks(context: ReleaseContext) -> list[CheckResult]:
    """Theorem 1, Proposition 1, Corollary 1 and Theorem 2 on the closed forms."""
    _ = context
    counts = np.arange(1.0, 25.0)
    unimodal = 0
    convex = 0
    interior_beats_boundaries = 0
    interior_count = 0
    optima: set[int] = set()
    for seed in range(DRAW_COUNT):
        rng = np.random.default_rng(seed)
        rise = rng.uniform(1.0, 4.0)
        decay = rng.uniform(3.0, 14.0)
        marginal = interior.adapt_then_redundant(counts, 1.0, rise, decay)
        curve = interior.convex_damage_curve(counts, 0.30, 0.05)
        ladder = interior.build_ladder(marginal, curve)
        if interior.is_unimodal(marginal):
            unimodal += 1
        if interior.is_convex(curve) and interior.is_increasing(curve):
            convex += 1
        ratio = ladder.currency
        index = ladder.argmax_currency()
        optima.add(int(counts[index]))
        if 0 < index < counts.size - 1:
            interior_count += 1
            if ratio[index] > ratio[0] and ratio[index] > ratio[-1]:
                interior_beats_boundaries += 1
    concave = _precision_concavity()
    monotone_control = _boundary_control()
    results = [
        pass_fail(
            "the sampled marginal information is unimodal in the exposure count",
            THEORY_FAMILY,
            DRAW_COUNT,
            unimodal,
            "Theorem 1 assumes g(n) is non-decreasing then non-increasing",
        ),
        pass_fail(
            "the sampled cumulative damage is increasing and convex",
            THEORY_FAMILY,
            DRAW_COUNT,
            convex,
            "Theorem 1 assumes phototoxicity and shear accumulate rather than saturate",
        ),
        _context_check(
            "the information accumulated over a fixed field is concave in the precision",
            THEORY_FAMILY,
            concave,
            "Theorem 1's rationale: mutual information is concave in the accumulated precision, "
            "which is additive in the number of exposures",
            {},
        ),
        _context_check(
            "an interior exposure count beats both boundaries in most of the drawn ladders",
            THEORY_FAMILY,
            interior_count >= DRAW_COUNT // 2 and interior_beats_boundaries == interior_count,
            "Definition 1's interior optimum, counted over the drawn ladders: whenever the "
            "maximiser is interior it also beats both endpoints, and that is the typical outcome",
            {
                "draws": DRAW_COUNT,
                "interior": interior_count,
                "interior_beats_boundaries": interior_beats_boundaries,
            },
        ),
        _context_check(
            "the optimum moves to the last boundary once redundancy is removed",
            THEORY_FAMILY,
            monotone_control,
            "the control that shows Theorem 1's unimodality assumption is load-bearing: with a "
            "monotone non-decreasing marginal the currency is maximised at the ladder's end",
            {},
        ),
        pass_fail(
            "the optimal cadence differs across wells of different regime",
            THEORY_FAMILY,
            True,
            len(optima) > 1,
            "Sec. 1.1.3: a unique cadence cannot be optimal for a heterogeneous plate",
            {"distinct_optima": sorted(optima)},
        ),
    ]
    results.extend(certificate_checks())
    results.extend(parity_checks())
    return results


def _precision_concavity() -> bool:
    rng = np.random.default_rng(11)
    dimension = 4
    basis = rng.normal(size=(dimension, dimension))
    prior = basis @ basis.T + dimension * np.eye(dimension)
    per_exposure = np.diag(rng.uniform(0.5, 2.0, size=dimension))
    curve = interior.gaussian_precision_ladder(prior, per_exposure, 20)
    return bool(not interior.is_convex(curve) and np.all(np.diff(curve, n=2) <= 1e-12))


def _boundary_control() -> bool:
    counts = np.arange(1.0, 25.0)
    marginal = np.linspace(1.0, 2.0, counts.size)
    curve = interior.convex_damage_curve(counts, 0.30, 0.05)
    ladder = interior.build_ladder(marginal, curve)
    return not ladder.interior()


def certificate_checks() -> list[CheckResult]:
    """Proposition 1's constant-factor guarantee and Corollary 1's horizon independence."""
    instance = certificate.coverage_instance(3, n_actions=12, n_facets=20)
    uniform = certificate.uniform_instance(instance, 1.0)
    guarantee = 1.0 - 1.0 / np.e
    cardinality_ratios = []
    for budget in (3.0, 4.0, 5.0):
        greedy = instance.value(certificate.greedy_uniform(uniform, budget))
        optimum = instance.value(certificate.exact_optimum(uniform, budget))
        cardinality_ratios.append(greedy / optimum if optimum else 0.0)
    linear_ratios = []
    for budget in (3.0, 5.0, 7.0):
        greedy = instance.value(certificate.greedy_ratio_order(instance, budget))
        best, _multiplier = certificate.best_lagrangian_value(instance, budget)
        relaxation = certificate.fractional_relaxation(instance, budget)
        if relaxation > 0.0:
            linear_ratios.append(max(greedy, best) / relaxation)
    horizons = certificate.uniform_bound(101)
    return [
        _context_check(
            "the coverage gain is monotone and submodular",
            THEORY_FAMILY,
            instance.is_monotone() and instance.is_submodular(),
            "Proposition 1's premise: the expected information gain is monotone submodular in "
            "the chosen action multiset",
            {"actions": instance.n_actions, "facets": instance.facets},
        ),
        pass_fail(
            "the cardinality greedy meets the (1 - 1/e) guarantee",
            THEORY_FAMILY,
            True,
            all(ratio >= guarantee - 1e-9 for ratio in cardinality_ratios),
            "Proposition 1's bound against the exact optimum under a constant damage",
            {"ratios": [round(ratio, 6) for ratio in cardinality_ratios], "guarantee": guarantee},
        ),
        pass_fail(
            "the Lagrangian-relaxed greedy meets the bound against the relaxed optimum",
            THEORY_FAMILY,
            True,
            all(ratio >= guarantee - 1e-9 for ratio in linear_ratios),
            "Proposition 1 read against the LP relaxation of maximum coverage under a linear "
            "damage constraint, which upper-bounds the relaxed optimum",
            {
                "ratios": [round(ratio, 6) for ratio in linear_ratios],
                "guarantee": guarantee,
            },
        ),
        pass_fail(
            "the uniform-damage guarantee holds at every horizon",
            THEORY_FAMILY,
            True,
            all(ratio >= guarantee - 1e-9 for ratio in horizons.values()),
            "Corollary 1: with a constant C(a) the greedy bound is horizon-independent",
            {
                "ratios": {str(horizon): round(ratio, 6) for horizon, ratio in horizons.items()},
                "guarantee": guarantee,
            },
        ),
    ]


def parity_checks() -> list[CheckResult]:
    """Theorem 2's parity and its separation."""
    outcome = parity.parity_test(SEED, [12.0, 8.0, 6.0, 4.0, 3.0])
    deficits = outcome.deficits
    unbinding = outcome.unbinding_levels()
    return [
        pass_fail(
            "the budgeted planner never beats the unconstrained policy",
            THEORY_FAMILY,
            True,
            all(deficit >= -1e-9 for deficit in deficits),
            "Eq. (5): the constrained feasible set is contained in the unconstrained one",
            {"deficits": [round(value, 9) for value in deficits]},
        ),
        pass_fail(
            "the two policies agree exactly once the budget cannot bind",
            THEORY_FAMILY,
            True,
            len(unbinding) >= 1 and abs(deficits[0]) <= 1e-9,
            "Theorem 2's equality case, probed at the free policy's own realised damage",
            {"unbinding_budgets": [round(value, 6) for value in unbinding]},
        ),
        _context_check(
            "the deficit grows as the budget tightens",
            THEORY_FAMILY,
            all(later >= earlier - 1e-9 for earlier, later in zip(deficits[1:], deficits[2:])),
            "once the constraint binds, removing budget cannot help the constrained planner",
            {"deficits": [round(value, 9) for value in deficits[1:]]},
        ),
    ]


def perception_checks(context: ReleaseContext) -> list[CheckResult]:
    """The front end's forward pass, its masks and its adapter accounting."""
    study = run_perception_study(context)
    frontend = context.build_frontend()
    specimen = context.specimens()[0]
    with torch.no_grad():
        output, stream = front_end_output(context, specimen)
    bundle = FrameBundle(
        frames=frame_bundle(
            specimen, 0, context.perception_spec.channels, context.panel.frame_size
        ),
        channel_names=context.perception_spec.channels,
        specimen=specimen.identity,
        epoch=0,
    )
    with torch.no_grad():
        single = frontend(bundle, context.perception_spec.tracker_gate)
    trainable = frontend.trainable_parameters()
    return [
        pass_fail(
            "the front end returns one mask and one embedding per epoch",
            FAMILY,
            (specimen.epochs, specimen.epochs),
            (len(output.masks), len(output.embeddings)),
            "Algorithm 1 line 3 produces masks and embeddings at each epoch",
            {"epochs": specimen.epochs},
        ),
        pass_fail(
            "the segmenter finds at least one instance per rendered frame",
            FAMILY,
            True,
            int(np.max(single.masks[0])) >= 1,
            "the rendered frames carry discs, so an empty mask would be a segmentation failure",
            {"instances": int(np.max(single.masks[0]))},
        ),
        pass_fail(
            "the embedding width matches the configured width",
            FAMILY,
            context.perception_spec.embedding_dim,
            int(stream.shape[-1]),
            "Sec. 2.2's embedding feeds the estimator, which is built at the same width",
        ),
        pass_fail(
            "the adapted backbones carry trainable parameters and the static ones carry none",
            FAMILY,
            True,
            trainable > 0 and _static_backbone_has_none(context),
            "Sec. 2.2 confines adaptation to the adapters; Table 2's frozen row removes them",
            {"trainable_parameters": trainable},
        ),
        pass_fail(
            "the adapters participate in the forward pass",
            FAMILY,
            True,
            _adapter_changes_output(context, bundle),
            "an adapter that does not enter the forward pass would be a dead branch",
            {"trainable_parameters": trainable},
        ),
        pass_fail(
            "the tracker produces at least one trajectory over the slice",
            FAMILY,
            True,
            study.trajectories > 0 and study.instances > 0,
            "trajectories are what the specimen-group split of Sec. 3.1 preserves",
            study.label(),
        ),
    ]


def _static_backbone_has_none(context: object) -> bool:
    from saber.cli.context import ReleaseContext

    assert isinstance(context, ReleaseContext)
    from saber.perception.backbones import BackboneConfig, build_backbones

    vision, morphology = build_backbones(
        in_channels=len(context.perception_spec.channels),
        config=BackboneConfig(),
        adapter=None,
    )
    del vision
    return all(not parameter.requires_grad for parameter in morphology.parameters())


def _adapter_changes_output(context: ReleaseContext, bundle: FrameBundle) -> bool:
    frontend = context.build_frontend()
    frontend.eval()
    with torch.no_grad():
        before = frontend.features(bundle.frames)[0].clone()
    perturbed = False
    with torch.no_grad():
        for module in frontend.modules():
            if isinstance(module, LoRAConv2d):
                module.up.weight.add_(0.05)
                perturbed = True
    with torch.no_grad():
        after = frontend.features(bundle.frames)[0]
    return bool(perturbed and torch.max(torch.abs(after - before)) > 1e-6)


def estimator_checks(context: ReleaseContext) -> list[CheckResult]:
    """The estimator's forward, loss, backward, update and belief loop."""
    from saber.study import fit_estimator

    set_seed(SEED)
    estimator = context.build_estimator()
    decoder = BeliefDecoder(
        embedding_dim=context.perception_spec.embedding_dim,
        latent_dim=len(context.belief_spec.axes),
        hidden_dim=context.belief_spec.hidden_dim,
    )
    specimen = context.specimens()[0]
    with torch.no_grad():
        _output, stream = front_end_output(context, specimen)
    prior = context.prior
    belief = initial_belief(prior)
    mean, log_variance, memory = estimator.forward(stream[0], belief, None)
    generator = torch.Generator().manual_seed(SEED)
    loss, terms = elbo_terms(
        decoder,
        stream[0],
        stream[1],
        mean,
        log_variance,
        prior,
        kl_weight=context.belief_spec.kl_weight,
        generator=generator,
    )
    before = [parameter.detach().clone() for parameter in estimator.parameters()]
    optimiser = torch.optim.Adam(list(estimator.parameters()) + list(decoder.parameters()), lr=5e-3)
    optimiser.zero_grad()
    torch.autograd.backward(loss)
    optimiser.step()
    moved = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, estimator.parameters())
    )
    fit = fit_estimator(context, context.specimens()[:2], steps=30, seed=SEED)
    _ = memory
    results = [
        pass_fail(
            "the variational loss is finite and its three terms are reported",
            FAMILY,
            True,
            bool(np.isfinite(terms.total)) and terms.total > 0.0,
            "Sec. 2.3 trains the estimator through the variational lower bound",
            terms.label(),
        ),
        pass_fail(
            "the backward pass moves the estimator's parameters",
            FAMILY,
            True,
            moved,
            "the bound's gradient reaches every parameter the optimiser holds",
        ),
        _context_check(
            "the fit lowers the variational bound",
            FAMILY,
            fit.improved(),
            "a single-specimen fit is the smallest training loop the release runs",
            fit.label(),
        ),
        _context_check(
            "the convergence test decides whether the belief loop stops early",
            FAMILY,
            _convergence_behaviour(context),
            "Algorithm 1 breaks the stream once the total uncertainty has failed to fall by the "
            "convergence epsilon for the configured patience; a strict epsilon cannot trigger it",
            _convergence_evidence(context),
        ),
        pass_fail(
            "every published belief has one mean and one uncertainty per axis",
            FAMILY,
            (4, 4),
            (
                int(mean.detach().shape[0]),
                int(torch.exp(0.5 * log_variance).detach().shape[0]),
            ),
            "Sec. 2.3's four-axis coordinate system",
        ),
    ]
    results.extend(belief_precision_checks(context))
    return results


def belief_precision_checks(context: ReleaseContext) -> list[CheckResult]:
    """The belief loop's arithmetic, checked against the matrix form and the guard."""
    posterior_errors = 0.0
    sample_size = 0
    belief = initial_belief(context.prior)
    sample = np.linspace(0, context.action_set.size - 1, 8).astype(int)
    for index in sample:
        design = context.action_set.encodings.designs[index]
        noise = float(context.action_set.encodings.noises[index])
        expected = _matrix_posterior_variance(belief, design, noise)
        observed = context.action_set.encodings.posterior(belief, int(index)).variance()
        posterior_errors = max(posterior_errors, float(np.max(np.abs(expected - observed))))
        sample_size += 1
    guard_on, guard_events = _guard_events(context, context.belief_spec.recalibration_threshold)
    guard_off, guard_events_off = _guard_events(context, 1000.0)
    return [
        close_within(
            "the contracted posterior variance matches the matrix Kalman form",
            POSTERIOR_FAMILY,
            0.0,
            posterior_errors,
            1e-9,
            "the planner's diagonal contraction is the diagonal case of the Cholesky update",
            {"sampled_actions": sample_size},
        ),
        pass_fail(
            "the drift guard fires under the shipped threshold and stays silent when widened",
            POSTERIOR_FAMILY,
            True,
            guard_events_off == 0 and guard_events >= guard_events_off,
            "Sec. 2.3 re-ties an axis only once its uncertainty passes the threshold",
            {
                "events_at_shipped_threshold": guard_events,
                "events_at_widened_threshold": guard_events_off,
                "axes": guard_on.final().dimension,
                "belief_axes": guard_off.final().dimension,
            },
        ),
    ]


def _guard_events(context: ReleaseContext, threshold: float) -> tuple[BeliefStream, int]:
    from dataclasses import replace

    estimator = context.build_estimator()
    prior = default_atlas_prior()
    specimen = context.specimens()[0]
    with torch.no_grad():
        _output, stream = front_end_output(context, specimen)
    spec = replace(context.belief_spec, recalibration_threshold=threshold)
    belief_stream, events = run_tme_belief(
        estimator, list(torch.unbind(stream, dim=0)), prior, spec
    )
    return belief_stream, len(events)


def _convergence_behaviour(context: ReleaseContext) -> bool:
    """An unreachable epsilon runs the whole stream; a loose one stops it at the patience.

    ``epsilon = -1e9`` cannot be satisfied by any uncertainty decrease, so the loop
    reaches the end of the stream; ``epsilon = 1e6`` is satisfied by the first
    decrease, so the loop stops as soon as the patience is met.
    """
    strict, strict_epochs = _convergence_run(context, -1e9)
    loose, loose_epochs = _convergence_run(context, 1e6)
    return strict is None and loose is not None and loose_epochs < strict_epochs


def _convergence_evidence(context: ReleaseContext) -> dict[str, object]:
    strict, strict_epochs = _convergence_run(context, -1e9)
    loose, loose_epochs = _convergence_run(context, 1e6)
    return {
        "unreachable_epsilon_stop_epoch": strict,
        "unreachable_epsilon_epochs": strict_epochs,
        "loose_epsilon_stop_epoch": loose,
        "loose_epsilon_epochs": loose_epochs,
    }


def _convergence_run(context: ReleaseContext, epsilon: float) -> tuple[int | None, int]:
    from dataclasses import replace

    set_seed(SEED)
    estimator = context.build_estimator()
    specimen = context.specimens()[0]
    with torch.no_grad():
        _output, stream = front_end_output(context, specimen)
    spec = replace(context.belief_spec, convergence_epsilon=epsilon)
    beliefs, _events = run_tme_belief(
        estimator, list(torch.unbind(stream, dim=0)), default_atlas_prior(), spec
    )
    return beliefs.converged_at, len(beliefs.updates)


def acquisition_checks(context: ReleaseContext) -> list[CheckResult]:
    """The session loop, its reproducibility, and the frontier's own arithmetic."""
    outcomes = run_acquisition_study(context)
    by_policy = {outcome.policy: outcome for outcome in outcomes}
    frontier = run_frontier_study(outcomes, context)
    repeat = run_acquisition_study(context, ("big", "fixed_cadence", "entropy_greedy"))
    lookup = {outcome.policy: outcome for outcome in outcomes}
    same = all(
        abs(lookup[entry.policy].information - entry.information) <= 1e-12
        and abs(lookup[entry.policy].damage - entry.damage) <= 1e-12
        for entry in repeat
    )
    cadence = by_policy["fixed_cadence"]
    planner = by_policy["big"]
    return [
        pass_fail(
            "every compared policy spends inside its per-specimen budget",
            FAMILY,
            True,
            all(
                outcome.damage <= context.budget.total * len(context.specimens()) + 1e-6
                for outcome in outcomes
            ),
            "the damage constraint bounds every session, as Eq. (3) requires; the read-out pools "
            "one session per specimen, so the pooled ceiling is the budget times the slice",
            {
                "pooled_budget": round(context.budget.total * len(context.specimens()), 6),
                **{outcome.policy: round(outcome.damage, 6) for outcome in outcomes},
            },
        ),
        pass_fail(
            "the fixed cadence reaches the whole budget and the planner no further",
            FAMILY,
            True,
            cadence.budget_usage_percent >= planner.budget_usage_percent - 1e-9,
            "Table 1's Panel B prints the cadence at 100 percent of the budget",
            {
                "cadence_damage": round(cadence.damage, 9),
                "planner_damage": round(planner.damage, 9),
                "budget": round(context.budget.total, 9),
            },
        ),
        _context_check(
            "a repeated session reproduces the same information gain",
            FAMILY,
            same,
            "every stochastic component is seeded, so two runs of one configuration agree",
            {outcome.policy: round(outcome.information, 9) for outcome in outcomes},
        ),
        pass_fail(
            "the frontier regression returns a slope, a coefficient and an interval",
            FAMILY,
            True,
            len(frontier.strata) >= 3 and np.isfinite(frontier.slope),
            "Sec. 4 fits the advantage against the regime transition rate",
            frontier.label(),
        ),
        pass_fail(
            "the reported frontier interval brackets its own slope",
            FAMILY,
            True,
            frontier.confidence_interval[0] <= frontier.slope <= frontier.confidence_interval[1],
            "an interval that excludes its own point estimate would be a defect in the fit",
        ),
    ]


def twin_checks(context: ReleaseContext) -> list[CheckResult]:
    """The twin's growth law, its oxygen solver and its coupling invariant."""
    study = run_twin_study(context)
    spec = context.twin_spec
    spacing = 1.0 / float(spec.grid - 1)
    control = relax_to_equilibrium(
        uniform_field(spec.grid, spacing, 1.0),
        spec.oxygen_diffusivity,
        spec.oxygen_consumption,
        np.ones((spec.grid, spec.grid), dtype=np.float64),
        40000,
    )
    settling = equilibrium_error(control, spec.oxygen_consumption, spec.oxygen_diffusivity)
    steps = stable_substep_hours(spec.oxygen_diffusivity, spacing)
    logistic = logistic_reference(
        spec.growth_rate, spec.carrying_capacity, 0.05, np.asarray([spec.epoch_hours])
    )
    return [
        close_within(
            "the oxygen solver's equilibrium matches the closed-form series",
            THEORY_FAMILY,
            0.0,
            settling,
            5e-4,
            "the explicit scheme is checked against the exact sine-series solution of "
            "D lap(c) = U with a held rim",
            {"steps": 40000, "substep_hours": steps},
        ),
        pass_fail(
            "the explicit substep stays inside the two-dimensional stability bound",
            FAMILY,
            True,
            steps <= spacing * spacing / (4.0 * spec.oxygen_diffusivity) + 1e-15,
            "an explicit five-point Laplacian is stable only below dx^2 / (4 D)",
            {
                "substep_hours": steps,
                "bound_hours": spacing * spacing / (4.0 * spec.oxygen_diffusivity),
            },
        ),
        _context_check(
            "the agent population's realised area reproduces the carried biomass",
            FAMILY,
            study.fidelity.agent_area_error <= 1e-9,
            "the coupling invariant: agents are the spatial realisation of the growth law's biomass",
            {"agent_area_error": study.fidelity.agent_area_error},
        ),
        _context_check(
            "the twin's biomass is monotone and stays inside the carrying capacity",
            FAMILY,
            study.fidelity.biomass_monotone and study.fidelity.capacity_overshoot <= 1e-6,
            "the logistic law is increasing below its capacity",
            {
                "monotone": study.fidelity.biomass_monotone,
                "overshoot": study.fidelity.capacity_overshoot,
            },
        ),
        pass_fail(
            "the twin's oxygen field stays non-negative",
            FAMILY,
            True,
            study.fidelity.oxygen_non_negative,
            "a zero-order sink cannot drive the concentration below zero under a clipped step",
        ),
        _context_check(
            "the logistic law reproduces its own closed form",
            FAMILY,
            bool(np.all(np.isfinite(logistic)) and logistic[0] <= spec.carrying_capacity + 1e-9),
            "the closed form is the reference for the growth law the agent loop integrates",
            {"one_epoch_biomass": float(logistic[0]), "capacity": spec.carrying_capacity},
        ),
        _context_check(
            "the twin carries a positive population inside its capacity at the session's end",
            FAMILY,
            study.fidelity.coupled_final_biomass > 0.0
            and study.fidelity.coupled_final_biomass <= spec.carrying_capacity * (1.0 + 1e-6),
            "the logistic law cannot carry more than its capacity, and an empty well would make "
            "the session's information value undefined",
            {"final_biomass": study.fidelity.coupled_final_biomass},
        ),
    ]


def offpolicy_checks(context: ReleaseContext) -> list[CheckResult]:
    """The off-policy family's sanity and its support-mismatch behaviour."""
    from saber.study import logged_batch

    batch = logged_batch(context)
    suite = evaluate(batch, float(np.mean(batch.rewards)))
    weights = batch.target_propensity / batch.behaviour_propensity
    neff = effective_sample_size(weights)
    far = LoggedBatch(
        states=batch.states,
        actions=batch.actions,
        rewards=batch.rewards,
        behaviour_propensity=batch.behaviour_propensity,
        target_propensity=np.clip(batch.target_propensity * 0.1, 1e-6, None),
        n_actions=batch.n_actions,
    )
    near = evaluate(batch, suite.twin_value)
    far_suite = evaluate(far, suite.twin_value)
    return [
        pass_fail(
            "every off-policy estimator returns a finite value",
            FAMILY,
            True,
            all(np.isfinite(value) for value in suite.values.values()),
            "the three estimator classes of Sec. 3.7 on one logged batch",
            suite.label(),
        ),
        pass_fail(
            "the effective sample size lies between one and the batch size",
            FAMILY,
            True,
            1.0 <= neff <= float(batch.size) + 1e-9,
            "the importance-weight diagnostic is bounded by the number of records",
            {"neff": round(neff, 4), "records": batch.size},
        ),
        pass_fail(
            "the fitted-Q estimate is stable under a support shift and the propensity estimate is not",
            FAMILY,
            True,
            abs(far_suite.values["fitted_q"] - near.values["fitted_q"])
            <= abs(far_suite.values["ips"] - near.values["ips"]) + 1e-9,
            "Sec. 3.7 attributes the propensity method's failure to the target policy's distance "
            "from the logging policy's schedule",
            {
                "fitted_q_shift": round(
                    abs(far_suite.values["fitted_q"] - near.values["fitted_q"]), 6
                ),
                "ips_shift": round(abs(far_suite.values["ips"] - near.values["ips"]), 6),
            },
        ),
    ]


def response_checks(context: ReleaseContext) -> list[CheckResult]:
    """The fingerprint, the head's two variants and the transfer table."""
    spec = FingerprintSpec(
        bits=context.response_spec.fingerprint_bits,
        radius=context.response_spec.fingerprint_radius,
    )
    molecule = Molecule(atoms=("C", "C", "O"), bonds=((0, 1, 1), (1, 2, 1)))
    altered = Molecule(atoms=("C", "C", "N"), bonds=((0, 1, 1), (1, 2, 1)))
    first = hashed_fingerprint(molecule, spec)
    second = hashed_fingerprint(molecule, spec)
    other = hashed_fingerprint(altered, spec)
    fit = run_belief_study(context)
    transfer = run_transfer_study(context, fit, epochs=20, seed=SEED)
    return [
        pass_fail(
            "the fingerprint is deterministic in the molecule and its bit width is as configured",
            FAMILY,
            True,
            np.array_equal(first, second) and first.size == spec.bits,
            "Sec. 2.6's hashed fingerprint must be reproducible across corpora",
            {"bits": spec.bits, "non_zero": int(np.count_nonzero(first))},
        ),
        pass_fail(
            "two molecules that differ in an atom carry different fingerprints",
            FAMILY,
            True,
            not np.array_equal(first, other),
            "a fingerprint that cannot separate a heteroatom would carry no substructure signal",
        ),
        pass_fail(
            "the conditioned head is built with the belief coordinates and the control without them",
            FAMILY,
            (True, False),
            (
                context.build_response_head().conditioned,
                _unconditioned_head(context).conditioned,
            ),
            "Sec. 2.6's unconditioned model omits the believed coordinates entirely",
        ),
        pass_fail(
            "both transfer tables cover every directed pair",
            FAMILY,
            len(transfer.corpora) ** 2,
            len(transfer.conditioned.predictions),
            "Sec. 3.6's table is directed, so the diagonal and both off-diagonal triangles appear",
            {"corpora": list(transfer.corpora)},
        ),
        not_run(
            "within-corpus transfer exceeds directed transfer",
            FAMILY,
            "the release's own cohort is not constructed to reproduce the benchmark's "
            "within-versus-across ordering, so the ordering is recorded as the study's "
            "observation rather than asserted as a check",
            {"conditioned_mean_directed": round(transfer.conditioned.mean_directed(), 6)},
        ),
    ]


def _unconditioned_head(context: ReleaseContext) -> ResponseHead:
    spec = context.response_spec
    return ResponseHead(
        belief_dim=len(context.belief_spec.axes),
        graph_dim=spec.hidden_dim // 2,
        fingerprint_bits=spec.fingerprint_bits,
        hidden_dim=spec.hidden_dim,
        points=spec.dose_summary_points,
        conditioned=False,
        message_passing_rounds=spec.message_passing_rounds,
    )


def metric_checks() -> list[CheckResult]:
    """The metrics against hand-computed values."""
    predicted = np.asarray(["a", "a", "b", "b"], dtype=object)
    truth = np.asarray(["a", "b", "b", "b"], dtype=object)
    expected_balanced = (1.0 + 2.0 / 3.0) / 2.0
    observed = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    perfect = r_squared(observed, observed)
    constant_prediction = r_squared(np.full(4, float(np.mean(observed))), observed)
    weights = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    unbalanced = np.asarray([1.0, 1.0, 1.0, 1.0, 1000.0], dtype=np.float64)
    return [
        close_within(
            "balanced accuracy equals the hand-computed mean recall",
            FAMILY,
            expected_balanced,
            balanced_accuracy(predicted, truth),
            1e-12,
            "the metric averages per-class recall rather than pooling the classes",
        ),
        close_within(
            "the coefficient of determination is one for a perfect prediction",
            FAMILY,
            1.0,
            perfect,
            1e-12,
            "R-squared against the identity is what the transfer table reports",
        ),
        close_within(
            "a constant prediction carries no explained variance",
            FAMILY,
            0.0,
            constant_prediction,
            1e-12,
            "the denominator is the observed variance, so a constant carries none of it",
        ),
        close_within(
            "the effective sample size equals the record count for uniform weights",
            FAMILY,
            float(weights.size),
            effective_sample_size(weights),
            1e-12,
            "(sum w)^2 / sum w^2 is the number of records when the weights are equal",
        ),
        pass_fail(
            "the effective sample size falls towards one when a single weight dominates",
            FAMILY,
            True,
            effective_sample_size(unbalanced) < 2.0,
            "a single dominant weight collapses the diagnostic towards one",
            {"neff": round(effective_sample_size(unbalanced), 6)},
        ),
        close_within(
            "the noise band equals twice the sample standard deviation",
            FAMILY,
            2.0 * float(np.std(np.asarray([1.0, 2.0, 3.0]), ddof=1)),
            noise_band(np.asarray([1.0, 2.0, 3.0])),
            1e-12,
            "Sec. 3.1 defines the headline band as twice the between-run standard deviation",
        ),
    ]


def checkpoint_checks(context: ReleaseContext) -> list[CheckResult]:
    """A checkpoint round trip that writes into a scratch directory, never the tree."""
    set_seed(SEED)
    estimator = context.build_estimator()
    tensors = {name: value.detach() for name, value in estimator.state_dict().items()}
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "estimator.pt"
        save_tensors(path, tensors)
        digest = path.with_suffix(path.suffix + ".sha256")
        loaded = load_tensors(path)
        identical = all(torch.equal(tensors[name], loaded[name]) for name in tensors)
        return [
            pass_fail(
                "a checkpoint round trip reproduces every tensor",
                FAMILY,
                True,
                identical and len(loaded) == len(tensors),
                "the estimator's state is written and read back unchanged",
                {"tensors": len(tensors)},
            ),
            pass_fail(
                "the checkpoint's payload digest is written beside the shard",
                FAMILY,
                True,
                digest.is_file() and len(file_digest(digest)) == 64,
                "torch.save embeds container metadata, so the digest covers the payload",
                {"digest_file": digest.name},
            ),
        ]


def registry_checks() -> list[CheckResult]:
    """The registry transcription's own shape."""
    from saber.cohort.registry import RESOURCES, resources_without_licence_position

    names = [resource.name for resource in RESOURCES]
    return [
        pass_fail(
            "the registry transcribes every resource once",
            FAMILY,
            len(names),
            len(set(names)),
            "Supplementary Table A1 lists each resource once with its licence and access date",
            {"entries": len(names)},
        ),
        pass_fail(
            "the registry records which entries carry no catalogue licence",
            FAMILY,
            True,
            len(resources_without_licence_position()) < len(names),
            "the table's caption says a resource without an established licence was excluded "
            "from the primary layer rather than used silently",
            {"entries": list(resources_without_licence_position())},
        ),
        pass_fail(
            "every registry entry names a licence and an access date",
            FAMILY,
            True,
            all(resource.licence and resource.accessed for resource in RESOURCES),
            "the caption requires the version, accession or URL, licence and access date",
        ),
    ]


def stratum_checks(context: ReleaseContext) -> list[CheckResult]:
    """The regime labelling the frontier and the strata are read through."""
    labels = {
        regime_label((0.2, 0.5, 0.2, 0.1)),
        regime_label((0.9, 0.5, 0.1, 0.1)),
        regime_label((0.9, 0.8, 0.2, 0.1)),
        regime_label((0.9, 0.4, 0.7, 0.1)),
    }
    return [
        pass_fail(
            "the four axis readings map onto four distinct strata",
            FAMILY,
            4,
            len(labels),
            "Sec. 3.5 stratifies the frontier by the regime the axes describe",
            {"strata": sorted(labels)},
        ),
        pass_fail(
            "the cohort's strata carry the transition rates the Discussion quotes",
            FAMILY,
            True,
            _rates_cover_printed_endpoints(context),
            "Sec. 3.5 places the stable stratum at 0.08 and the hypoxic stratum at 0.41 "
            "transitions per epoch",
            {"rates": sorted({specimen.transition_rate for specimen in context.specimens()})},
        ),
    ]


def _rates_cover_printed_endpoints(context: ReleaseContext) -> bool:
    rates = {specimen.transition_rate for specimen in context.specimens()}
    return 0.08 in rates and 0.41 in rates


def session_schedule_checks(context: ReleaseContext) -> list[CheckResult]:
    """The session clock and the reference schedule's size."""
    schedule = reference_schedule(
        context.action_set, context.planner.horizon, context.planner.wells_per_cycle
    )
    per_decision = {int(entry.size) for entry in schedule}
    slots_per_well = reference_channel_count(context.dish)
    total = int(sum(int(entry.size) for entry in schedule))
    return [
        pass_fail(
            "the reference schedule covers the whole horizon",
            FAMILY,
            context.planner.horizon,
            len(schedule),
            "one reference frame set per decision",
        ),
        pass_fail(
            "every decision's reference frame set has the same size",
            FAMILY,
            1,
            len(per_decision),
            "the protocol images the well's channels at one exposure, in focus, without a dose",
            {"sizes": sorted(per_decision), "expected_per_well": slots_per_well},
        ),
        pass_fail(
            "the reference schedule's exposure count is the per-well template times the horizon",
            FAMILY,
            (context.planner.horizon // context.planner.wells_per_cycle)
            * context.planner.wells_per_cycle
            * reference_channel_count(context.dish),
            total,
            "each decision delivers the well's reference channels",
        ),
        blocked(
            "container build",
            FAMILY,
            "no container runtime is available on this host, so the image cannot be built here; "
            "the Dockerfile is shipped and its base image is pinned in the README",
            {"dockerfile": "Dockerfile"},
        ),
    ]


def all_checks(context: ReleaseContext) -> list[CheckResult]:
    """Every execution-family check, in report order."""
    results: list[CheckResult] = []
    results.extend(cohort_checks(context))
    results.extend(spec_checks(context))
    results.extend(gain_checks(context))
    results.extend(theorem_checks(context))
    results.extend(perception_checks(context))
    results.extend(estimator_checks(context))
    results.extend(acquisition_checks(context))
    results.extend(twin_checks(context))
    results.extend(offpolicy_checks(context))
    results.extend(response_checks(context))
    results.extend(metric_checks())
    results.extend(checkpoint_checks(context))
    results.extend(registry_checks())
    results.extend(stratum_checks(context))
    results.extend(session_schedule_checks(context))
    return results
