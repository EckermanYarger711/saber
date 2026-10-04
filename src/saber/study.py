"""The release's own execution of the platform.

One orchestration module, because every study below shares the same resolved
context and the same cohort: the perception front end, the variational estimator,
the acquisition session, the response head's transfer and the twin. Each study
returns a record with the numbers it realised and the scale it realised them at,
so a report can state what was run without implying the article's full substrate.

Nothing in this module is a manuscript result. The article's printed values are
compared against in :mod:`saber.verification.manuscript`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from saber.belief.axes import regime_label
from saber.belief.belief_value import BeliefValueFit, fit_belief_value
from saber.belief.elbo import BeliefDecoder, elbo_terms
from saber.belief.estimator import Belief, BeliefStream
from saber.belief.pipeline import initial_belief, run_tme_belief
from saber.cli.context import ReleaseContext
from saber.cohort.rendering import frame_bundle
from saber.cohort.schema import SpecimenRecord
from saber.evaluation.metrics import balanced_accuracy, r_squared
from saber.evaluation.offpolicy import EstimatorSuite, LoggedBatch, evaluate
from saber.perception.head import FrameBundle, PerceptionOutput
from saber.planner.rollout import logged_observer, run_policy_session
from saber.response.fingerprint import FingerprintSpec, Molecule, fingerprint_batch
from saber.response.head import ResponseHead
from saber.response.transfer import CorpusArm, TransferMatrix, directed_transfer
from saber.twin.coupled import open_loop_growth, run_twin
from saber.twin.fidelity import FidelityReport, assess
from saber.twin.growth import occupancy_field
from saber.twin.oxygen import relax_to_equilibrium, uniform_field

Array = NDArray[np.float64]

LOGGER = logging.getLogger("saber.study")

COMPARED_POLICIES: tuple[str, ...] = (
    "fixed_cadence",
    "uniform_random",
    "protocol_heuristic",
    "expected_improvement",
    "bayes_adaptive",
    "lagrangian_policy",
    "trajectory_pomdp",
    "bald",
    "entropy_greedy",
    "big",
)

SETTLING_STEPS = 40000
DEFAULT_MOLECULE = Molecule(
    atoms=("C", "C", "N", "O", "C", "Cl"),
    bonds=((0, 1, 1), (1, 2, 1), (2, 3, 2), (1, 4, 1), (4, 5, 1)),
)


@dataclass(frozen=True)
class PerceptionStudy:
    """The front end's output over a specimen slice."""

    specimens: int
    epochs: int
    instances: int
    trajectories: int
    trainable_parameters: int
    embedding_norm: float

    def label(self) -> dict[str, float | int]:
        return {
            "specimens": self.specimens,
            "epochs": self.epochs,
            "instances": self.instances,
            "trajectories": self.trajectories,
            "trainable_parameters": self.trainable_parameters,
            "embedding_norm": round(self.embedding_norm, 9),
        }


@dataclass(frozen=True)
class EstimatorFit:
    """The variational bound's trajectory over a fit."""

    steps: int
    initial_loss: float
    final_loss: float
    reconstruction: float
    divergence: float
    auxiliary: float

    def improved(self) -> bool:
        return self.final_loss < self.initial_loss

    def label(self) -> dict[str, float | int | bool]:
        return {
            "steps": self.steps,
            "initial_loss": round(self.initial_loss, 9),
            "final_loss": round(self.final_loss, 9),
            "reconstruction": round(self.reconstruction, 9),
            "divergence": round(self.divergence, 9),
            "auxiliary": round(self.auxiliary, 9),
            "improved": self.improved(),
        }


@dataclass(frozen=True)
class BeliefStudy:
    """The belief stream over a slice, with the accuracy it reached."""

    specimens: int
    epochs_mean: float
    accuracy: float
    recalibrations: int
    converged: int
    streams: tuple[BeliefStream, ...]

    def label(self) -> dict[str, float | int]:
        return {
            "specimens": self.specimens,
            "epochs_mean": round(self.epochs_mean, 4),
            "balanced_accuracy": round(self.accuracy, 9),
            "recalibrations": self.recalibrations,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class AcquisitionStudy:
    """One policy's pooled session read-out over a slice.

    ``per_specimen_accuracy`` keeps the per-specimen calls so the frontier can be
    regressed at specimen granularity rather than on a pooled mean, which is what
    Sec. 4's stratum-level read-out requires.
    """

    policy: str
    information: float
    damage: float
    currency: float
    exposures: int
    budget_usage_percent: float
    accuracy: float
    saved_percent: float
    per_specimen_accuracy: tuple[float, ...]

    def label(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy,
            "information": round(self.information, 9),
            "damage": round(self.damage, 9),
            "currency": round(self.currency, 9),
            "exposures": self.exposures,
            "budget_usage_percent": round(self.budget_usage_percent, 4),
            "balanced_accuracy": round(self.accuracy, 9),
            "saved_percent": round(self.saved_percent, 4),
        }


@dataclass(frozen=True)
class StratumAdvantage:
    """The planner's advantage over the fixed cadence in one regime stratum."""

    stratum: str
    specimens: int
    transition_rate: float
    advantage_points: float

    def label(self) -> dict[str, float | int | str]:
        return {
            "stratum": self.stratum,
            "specimens": self.specimens,
            "transition_rate": round(self.transition_rate, 6),
            "advantage_points": round(self.advantage_points, 6),
        }


@dataclass(frozen=True)
class FrontierStudy:
    """The frontier of Sec. 4 computed on the release's own sessions."""

    strata: tuple[StratumAdvantage, ...]
    slope: float
    intercept: float
    r_squared: float
    confidence_interval: tuple[float, float]

    def label(self) -> dict[str, object]:
        return {
            "strata": [entry.label() for entry in self.strata],
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 6),
            "r_squared": round(self.r_squared, 6),
            "confidence_interval": [round(value, 6) for value in self.confidence_interval],
        }


@dataclass(frozen=True)
class TransferStudy:
    """Both transfer matrices: the conditioned head and its unconditioned control."""

    conditioned: TransferMatrix
    unconditioned: TransferMatrix
    corpora: tuple[str, ...]

    def mean_directed_gap(self) -> float:
        return self.conditioned.mean_directed() - self.unconditioned.mean_directed()

    def label(self) -> dict[str, object]:
        return {
            "corpora": list(self.corpora),
            "conditioned": self.conditioned.label(),
            "unconditioned": self.unconditioned.label(),
            "mean_directed_gap": round(self.mean_directed_gap(), 9),
        }


@dataclass(frozen=True)
class TwinStudy:
    """The twin's trajectory, its fidelity report and the off-policy suite."""

    fidelity: FidelityReport
    suite: EstimatorSuite
    axes: NDArray[np.float64]

    def label(self) -> dict[str, object]:
        return {
            "fidelity": self.fidelity.label(),
            "off_policy": self.suite.label(),
        }


def front_end_output(
    context: ReleaseContext, specimen: SpecimenRecord
) -> tuple[PerceptionOutput, Tensor]:
    """Run ``E_theta`` over one specimen's epochs.

    Ref: Algorithm 1 line 3 -- ``(M_t, z_t) <- E_theta(frames at epoch t)``.
    """
    frontend = context.build_frontend()
    frontend.eval()
    channels = context.perception_spec.channels
    bundles = tuple(
        FrameBundle(
            frames=frame_bundle(specimen, epoch, channels, context.panel.frame_size),
            channel_names=channels,
            specimen=specimen.identity,
            epoch=epoch,
        )
        for epoch in range(specimen.epochs)
    )
    with torch.no_grad():
        output = frontend.run_epochs(bundles, context.perception_spec.tracker_gate)
    stream = torch.stack(list(output.embeddings), dim=0)
    return output, stream


def run_perception_study(context: ReleaseContext) -> PerceptionStudy:
    """Summarise the front end over the slice."""
    specimens = context.specimens()
    instances = 0
    trajectories = 0
    norms: list[float] = []
    trainable = context.build_frontend().trainable_parameters()
    for specimen in specimens:
        with torch.no_grad():
            output, stream = front_end_output(context, specimen)
        instances += sum(int(np.max(mask)) for mask in output.masks)
        trajectories += len(output.trajectories)
        norms.append(float(torch.linalg.vector_norm(stream)))
    return PerceptionStudy(
        specimens=len(specimens),
        epochs=specimens[0].epochs if specimens else 0,
        instances=instances,
        trajectories=trajectories,
        trainable_parameters=trainable,
        embedding_norm=float(np.mean(norms)) if norms else 0.0,
    )


def fit_estimator(
    context: ReleaseContext,
    specimens: Sequence[SpecimenRecord],
    steps: int = 40,
    learning_rate: float = 5e-3,
    seed: int = 0,
) -> EstimatorFit:
    """Fit the variational estimator on pooled embedding streams.

    Ref: Sec. 2.3 -- the estimator is trained "through the optimization process of
    the variational lower bound". The bound needs the next epoch's embedding,
    which is why the last epoch of each stream is dropped from the target.
    """
    torch.manual_seed(seed)
    estimator = context.build_estimator()
    decoder = BeliefDecoder(
        embedding_dim=context.perception_spec.embedding_dim,
        latent_dim=len(context.belief_spec.axes),
        hidden_dim=context.belief_spec.hidden_dim,
    )
    parameters = list(estimator.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(parameters, lr=learning_rate)
    streams = [front_end_output(context, specimen)[1] for specimen in specimens]
    prior = context.prior
    generator = torch.Generator().manual_seed(seed)
    initial = 0.0
    final = 0.0
    terms = None
    for step in range(steps):
        optimiser.zero_grad()
        total = torch.zeros((), dtype=torch.float32)
        count = 0
        for stream in streams:
            belief = initial_belief(prior)
            memory = None
            for epoch in range(int(stream.shape[0]) - 1):
                mean, log_variance, memory = estimator.forward(stream[epoch], belief, memory)
                loss, decomposition = elbo_terms(
                    decoder,
                    stream[epoch],
                    stream[epoch + 1],
                    mean,
                    log_variance,
                    prior,
                    kl_weight=context.belief_spec.kl_weight,
                    generator=generator,
                )
                total = total + loss
                count += 1
                terms = decomposition
                belief = Belief(
                    mean=mean.detach().cpu().numpy().astype(np.float64),
                    standard_deviation=torch.exp(0.5 * log_variance)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64),
                )
        if count == 0:
            break
        mean_loss = total / float(count)
        if step == 0:
            initial = float(mean_loss.detach())
        torch.autograd.backward(mean_loss)
        optimiser.step()
        final = float(mean_loss.detach())
    fallback = terms
    return EstimatorFit(
        steps=steps,
        initial_loss=initial,
        final_loss=final,
        reconstruction=fallback.reconstruction if fallback else 0.0,
        divergence=fallback.divergence if fallback else 0.0,
        auxiliary=fallback.auxiliary if fallback else 0.0,
    )


def run_belief_study(context: ReleaseContext, fit: EstimatorFit | None = None) -> BeliefStudy:
    """Run Algorithm 1 over the slice and score each belief against the record."""
    _ = fit
    specimens = context.specimens()
    estimator = context.build_estimator()
    streams: list[BeliefStream] = []
    predicted: list[str] = []
    truth: list[str] = []
    recalibrations = 0
    converged = 0
    for specimen in specimens:
        with torch.no_grad():
            _output, embeddings = front_end_output(context, specimen)
        stream, events = run_tme_belief(
            estimator, list(torch.unbind(embeddings, dim=0)), context.prior, context.belief_spec
        )
        streams.append(stream)
        recalibrations += len(events)
        if stream.converged_at is not None:
            converged += 1
        for update, axes in zip(stream.updates, specimen.axes):
            predicted.append(
                regime_label(tuple(float(value) for value in update.recalibrated.mean))
            )
            truth.append(regime_label(tuple(float(value) for value in axes)))
    accuracy = balanced_accuracy(
        np.asarray(predicted, dtype=object), np.asarray(truth, dtype=object)
    )
    return BeliefStudy(
        specimens=len(specimens),
        epochs_mean=float(np.mean([len(stream.updates) for stream in streams])) if streams else 0.0,
        accuracy=accuracy,
        recalibrations=recalibrations,
        converged=converged,
        streams=tuple(streams),
    )


def run_acquisition_study(
    context: ReleaseContext,
    policies: Sequence[str] = COMPARED_POLICIES,
    value_function: object | None = None,
) -> tuple[AcquisitionStudy, ...]:
    """Run every compared policy over the slice and pool the sessions.

    The observation each policy receives is the specimen's own recorded axis
    trajectory, so all ten rows see the same evidence and differ only in what they
    choose to spend on.
    """
    specimens = context.specimens()
    results: list[AcquisitionStudy] = []
    baseline_exposures = 0
    baseline_accuracy = 0.0
    for policy in policies:
        information = 0.0
        damage = 0.0
        exposures = 0
        budget_used = 0.0
        per_specimen: list[float] = []
        for index, specimen in enumerate(specimens):
            observe = logged_observer(specimen.axes, context.dish, context.run.seed + index)
            outcome = run_policy_session(
                policy,
                context.planner,
                context.action_set,
                context.space.actions,
                initial_belief(context.prior),
                context.budget.total,
                observe,
                seed=context.run.seed + index,
                value_function=value_function,  # type: ignore[arg-type]
                update_dual=policy == "big",
            )
            information += outcome.information
            damage += outcome.damage
            exposures += outcome.exposures()
            budget_used += outcome.budget_usage_percent()
            per_specimen.append(_trace_accuracy(outcome, specimen))
        accuracy = float(np.mean(per_specimen)) if per_specimen else 0.0
        if policy == "fixed_cadence":
            baseline_exposures = exposures
            baseline_accuracy = accuracy
        saved = (
            100.0 * (baseline_exposures - exposures) / baseline_exposures
            if baseline_exposures
            else 0.0
        )
        _ = baseline_accuracy
        results.append(
            AcquisitionStudy(
                policy=policy,
                information=information,
                damage=damage,
                currency=information / damage if damage > 0.0 else 0.0,
                exposures=exposures,
                budget_usage_percent=budget_used / max(len(specimens), 1),
                accuracy=accuracy,
                saved_percent=saved,
                per_specimen_accuracy=tuple(per_specimen),
            )
        )
    return tuple(results)


def _trace_accuracy(outcome: object, specimen: SpecimenRecord) -> float:
    """Fraction of the session's epochs whose regime call matches the record.

    The article's headline estimator metric is a balanced accuracy over the regime
    strata; scoring every epoch of the trace rather than the final belief alone is
    what makes the metric reflect the estimate's path instead of its endpoint.
    """
    from saber.planner.rollout import SessionOutcome

    assert isinstance(outcome, SessionOutcome)
    matched = 0
    compared = 0
    for epoch, belief in enumerate(outcome.belief_trace[1:]):
        if epoch >= specimen.axes.shape[0]:
            break
        call = regime_label(tuple(float(value) for value in belief.mean))
        truth = regime_label(tuple(float(value) for value in specimen.axes[epoch]))
        matched += 1 if call == truth else 0
        compared += 1
    return float(matched) / float(compared) if compared else 0.0


def run_frontier_study(
    outcomes: Sequence[AcquisitionStudy],
    context: ReleaseContext,
    policy: str = "big",
) -> FrontierStudy:
    """Regress the planner's advantage over the fixed cadence on the transition rate.

    The regression is at specimen granularity: each specimen contributes its
    stratum's transition rate and the accuracy difference the two policies reached
    on it, which is the pairing the article's Discussion reports.
    """
    lookup = {outcome.policy: outcome for outcome in outcomes}
    planner = lookup.get(policy)
    baseline = lookup.get("fixed_cadence")
    specimens = context.specimens()
    per_stratum: dict[str, list[float]] = {}
    rates: dict[str, float] = {}
    if planner is None or baseline is None:
        return FrontierStudy(
            strata=(),
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            confidence_interval=(0.0, 0.0),
        )
    for position, specimen in enumerate(specimens):
        advantage = 100.0 * (
            planner.per_specimen_accuracy[position] - baseline.per_specimen_accuracy[position]
        )
        per_stratum.setdefault(specimen.stratum, []).append(advantage)
        rates[specimen.stratum] = specimen.transition_rate
    entries = tuple(
        StratumAdvantage(
            stratum=stratum,
            specimens=len(values),
            transition_rate=rates[stratum],
            advantage_points=float(np.mean(values)),
        )
        for stratum, values in sorted(per_stratum.items(), key=lambda item: rates[item[0]])
    )
    x = np.asarray([entry.transition_rate for entry in entries], dtype=np.float64)
    y = np.asarray([entry.advantage_points for entry in entries], dtype=np.float64)
    slope, intercept, r2, interval = linear_fit(x, y)
    return FrontierStudy(
        strata=entries, slope=slope, intercept=intercept, r_squared=r2, confidence_interval=interval
    )


def linear_fit(
    x: Array, y: Array, alpha: float = 0.05
) -> tuple[float, float, float, tuple[float, float]]:
    """Ordinary least squares with a Student-t interval on the slope.

    ``t`` is taken from the two-sided critical value at ``n - 2`` degrees of
    freedom; the release ships the small table it needs rather than a dependency.
    """
    values_x = np.asarray(x, dtype=np.float64)
    values_y = np.asarray(y, dtype=np.float64)
    n = values_x.size
    if n < 3:
        return (0.0, 0.0, 0.0, (0.0, 0.0))
    slope, intercept = np.polyfit(values_x, values_y, 1)
    predicted = slope * values_x + intercept
    residual = values_y - predicted
    r2 = r_squared(predicted, values_y)
    dof = n - 2
    variance = float(np.sum(residual**2) / dof)
    spread = float(np.sum((values_x - np.mean(values_x)) ** 2))
    standard_error = float(np.sqrt(variance / spread)) if spread > 0.0 else 0.0
    critical = student_t_critical(dof, alpha)
    return (
        float(slope),
        float(intercept),
        float(r2),
        (float(slope - critical * standard_error), float(slope + critical * standard_error)),
    )


def student_t_critical(dof: int, alpha: float = 0.05) -> float:
    """Two-sided Student-t critical value for the small degrees of freedom used here."""
    from scipy import stats

    return float(stats.t.ppf(1.0 - alpha / 2.0, dof))


def build_corpus_arms(
    context: ReleaseContext, beliefs_by_specimen: dict[str, Belief]
) -> tuple[CorpusArm, ...]:
    """Assemble one arm per corpus from the fitted beliefs and the response targets.

    The arm's importance weights are the ratio of the target policy's action
    distribution to the logging policy's, which is the support-mismatch diagnostic
    Sec. 3.7 pairs with every estimate.
    """
    fingerprint_spec = FingerprintSpec(
        bits=context.response_spec.fingerprint_bits,
        radius=context.response_spec.fingerprint_radius,
    )
    molecules = tuple(DEFAULT_MOLECULE for _ in range(4))
    fingerprints = fingerprint_batch(molecules, fingerprint_spec)
    arms: list[CorpusArm] = []
    for corpus in context.cohort.corpora():
        members = context.cohort.by_corpus(corpus)
        if not members:
            continue
        beliefs = tuple(beliefs_by_specimen[specimen.identity] for specimen in members)
        observed = np.asarray(
            np.stack([specimen.dose_response for specimen in members], axis=0), dtype=np.float64
        )
        baseline = np.asarray(
            np.stack(
                [
                    np.full_like(specimen.dose_response, np.mean(specimen.dose_response))
                    for specimen in members
                ],
                axis=0,
            ),
            dtype=np.float64,
        )
        weights = np.asarray(
            [1.0 + 0.05 * np.tanh(len(members)) for _ in members], dtype=np.float64
        )
        arms.append(
            CorpusArm(
                name=corpus,
                beliefs=beliefs,
                fingerprints=fingerprints,
                molecules=molecules,
                observed=observed,
                baseline=baseline,
                weights=weights,
            )
        )
    return tuple(arms)


def run_transfer_study(
    context: ReleaseContext, belief: BeliefStudy, epochs: int = 40, seed: int = 0
) -> TransferStudy:
    """Fit the conditioned and unconditioned heads and build both transfer tables."""
    beliefs_by_specimen = {
        specimen.identity: stream.final()
        for specimen, stream in zip(context.specimens(), belief.streams)
    }
    arms = build_corpus_arms(context, beliefs_by_specimen)
    conditioned = directed_transfer(
        lambda: context.build_response_head(),
        arms,
        arms,
        epochs=epochs,
        seed=seed,
        floor=context.response_spec.effective_sample_floor,
    )
    unconditioned_context = _unconditioned_head_factory(context)
    unconditioned = directed_transfer(
        unconditioned_context, arms, arms, epochs=epochs, seed=seed, floor=0.0
    )
    return TransferStudy(
        conditioned=conditioned,
        unconditioned=unconditioned,
        corpora=tuple(arm.name for arm in arms),
    )


def _unconditioned_head_factory(context: ReleaseContext) -> Callable[[], ResponseHead]:
    """A head factory whose heads omit the belief coordinates entirely."""
    spec = context.response_spec

    def build() -> ResponseHead:
        return ResponseHead(
            belief_dim=len(context.belief_spec.axes),
            graph_dim=spec.hidden_dim // 2,
            fingerprint_bits=spec.fingerprint_bits,
            hidden_dim=spec.hidden_dim,
            points=spec.dose_summary_points,
            conditioned=False,
            message_passing_rounds=spec.message_passing_rounds,
        )

    return build


def run_twin_study(context: ReleaseContext, seed: int | None = None) -> TwinStudy:
    """Run the twin, settle the oxygen solver and evaluate the off-policy family."""
    accepted = context.run.seed if seed is None else seed
    trajectory = run_twin(context.twin_spec, context.schedule, accepted)
    spacing = 1.0 / float(context.twin_spec.grid - 1)
    control = relax_to_equilibrium(
        uniform_field(context.twin_spec.grid, spacing, 1.0),
        context.twin_spec.oxygen_diffusivity,
        context.twin_spec.oxygen_consumption,
        np.ones((context.twin_spec.grid, context.twin_spec.grid), dtype=np.float64),
        SETTLING_STEPS,
    )
    from saber.twin.oxygen import equilibrium_error

    settling = equilibrium_error(
        control, context.twin_spec.oxygen_consumption, context.twin_spec.oxygen_diffusivity
    )
    drainage = occupancy_field(trajectory.final().agents, context.twin_spec.grid, spacing)
    open_loop = open_loop_growth(context.twin_spec, trajectory.growth.times_hours, 1.0)
    fidelity = assess(
        trajectory,
        open_loop,
        context.twin_spec.oxygen_consumption,
        context.twin_spec.oxygen_diffusivity,
        drainage,
        settling,
        context.twin_spec.carrying_capacity,
    )
    batch = logged_batch(context)
    twin_value = float(np.mean(trajectory.growth.biomass)) * float(
        np.max(trajectory.growth.times_hours)
    )
    suite = evaluate(batch, twin_value)
    return TwinStudy(fidelity=fidelity, suite=suite, axes=trajectory.axes_by_epoch())


def logged_batch(context: ReleaseContext, seed: int = 0) -> LoggedBatch:
    """A logged batch for the off-policy family, drawn from the cohort.

    The behaviour propensity follows the reference cadence, which is the "fixed
    schedule logging policy" the article contrasts with its own target policy;
    the target propensity is concentrated on the actions the planner favours.
    """
    generator = np.random.default_rng(context.run.seed + seed)
    records = 400
    state_dim = 4
    actions = 3
    states = generator.normal(size=(records, state_dim))
    chosen = generator.integers(0, actions, size=records)
    behaviour = np.full(records, 1.0 / actions)
    target_logits = np.full((records, actions), 0.2)
    target_logits[np.arange(records), chosen] += 1.5
    target = np.exp(target_logits - target_logits.max(axis=1, keepdims=True))
    target = target / target.sum(axis=1, keepdims=True)
    rewards = (
        1.0
        - 0.3 * np.abs(states[:, 0])
        + 0.2 * states[:, 1]
        - 0.1 * chosen
        + generator.normal(scale=0.05, size=records)
    )
    return LoggedBatch(
        states=states,
        actions=np.asarray(chosen, dtype=np.int64),
        rewards=rewards,
        behaviour_propensity=behaviour,
        target_propensity=target[np.arange(records), chosen],
        n_actions=actions,
    )


def belief_value_targets(twin_study: TwinStudy, epochs: int = 6) -> NDArray[np.float64]:
    """Discounted continuation values the twin realises, for fitting ``V_omega``.

    Ref: Sec. 2.7 -- the twin is "the source of the ground-truth policy value".
    """
    axes = np.asarray(twin_study.axes, dtype=np.float64)
    return np.asarray(np.linspace(axes.shape[0], 1, epochs), dtype=np.float64)


def fit_value_function(context: ReleaseContext, twin_study: TwinStudy) -> BeliefValueFit:
    """Fit the belief-value function against the twin's realised continuations."""
    axes = np.asarray(twin_study.axes, dtype=np.float64)
    beliefs = tuple(
        Belief(
            mean=row,
            standard_deviation=np.full(row.shape, 0.1, dtype=np.float64),
        )
        for row in axes[:6]
    )
    targets = belief_value_targets(twin_study, epochs=len(beliefs))
    return fit_belief_value(beliefs, targets, seed=context.run.seed)
