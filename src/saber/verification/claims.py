"""The paper-claim to code mapping.

Each entry names the article's statement, the location it is printed at, the code
that carries it, and the checks that decide whether the carrying code behaves.
The anchors are ``path::symbol`` pairs and are resolved against the tree, so a
rename breaks the map rather than silently pointing at nothing.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from saber.verification.checks import CheckResult

CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "C1",
        "statement": "The action set is the product of the field, channel, exposure, axial-depth, fluidics and dose axes.",
        "paper_location": "Eq. (1), Sec. 1.1.1",
        "code_anchors": [
            "src/saber/spec/actions.py::Action",
            "src/saber/spec/actions.py::ActionSpace",
            "src/saber/spec/actions.py::enumerate_actions",
        ],
        "checks": [
            "action set size equals the product of its factors",
            "the session horizon is the wells-per-cycle times the session's epochs",
        ],
    },
    {
        "id": "C2",
        "statement": "Damage is the weighted sum of the photon dose, the out-of-focus depth, the fluidic exchange indicator and the stage time.",
        "paper_location": "Eq. (2) and Eq. (6), Sec. 1.1.1 and Sec. 2.4",
        "code_anchors": [
            "src/saber/spec/damage.py::DamageFunctional",
            "src/saber/spec/damage.py::photon_dose",
            "src/saber/spec/damage.py::stage_time",
        ],
        "checks": [
            "every damage weight is non-negative and every cost is non-negative",
            "raising a damage weight cannot lower a cost",
            "damage is increasing in exposure and in defocus",
            "the single-action and tabulated damage agree",
        ],
    },
    {
        "id": "C3",
        "statement": "The budget equals the total damage a reference protocol incurs over the same wall-clock window, which makes every comparison iso-resource.",
        "paper_location": "Sec. 2.4 and Sec. 3.1",
        "code_anchors": [
            "src/saber/spec/budget.py::session_reference_budget",
            "src/saber/spec/budget.py::reference_frames",
            "src/saber/planner/big.py::reference_schedule",
        ],
        "checks": [
            "the budget equals the reference protocol's own sequence damage",
            "the reference schedule covers the whole horizon",
            "every decision's reference frame set has the same size",
            "the fixed cadence reaches the whole budget and the planner no further",
        ],
    },
    {
        "id": "C4",
        "statement": "The monitoring objective maximises expected discounted information about the latent state subject to the cumulative damage constraint.",
        "paper_location": "Eq. (3), Sec. 1.1.2",
        "code_anchors": [
            "src/saber/planner/currency.py::currency",
            "src/saber/planner/info_gain.py::EncodingTable",
            "src/saber/spec/budget.py::SessionBudget",
        ],
        "checks": [
            "the planner's gain matches the matrix Kalman closed form",
            "every compared policy spends inside its per-specimen budget",
        ],
    },
    {
        "id": "C5",
        "statement": "The decision currency is the expected information gain per unit of specimen damage.",
        "paper_location": "Eq. (4) and Sec. 2.5",
        "code_anchors": [
            "src/saber/planner/currency.py::currency_vector",
            "src/saber/planner/currency.py::currency_per_unit_damage",
        ],
        "checks": [
            "every compared policy spends inside its per-specimen budget",
            "a repeated session reproduces the same information gain",
        ],
    },
    {
        "id": "C6",
        "statement": "The constrained problem is solved by receding-horizon planning with a Lagrangian relaxation whose multiplier is updated from the running damage.",
        "paper_location": "Eq. (7) and Sec. 2.5",
        "code_anchors": [
            "src/saber/planner/lagrangian.py::dual_update",
            "src/saber/planner/lagrangian.py::lagrangian_score",
            "src/saber/planner/big.py::BigPlanner",
        ],
        "checks": [
            "every compared policy spends inside its per-specimen budget",
            "the budgeted planner never beats the unconstrained policy",
        ],
    },
    {
        "id": "C7",
        "statement": "On a fixed field the optimal exposure count is interior, because marginal information is unimodal and cumulative damage is increasing and convex.",
        "paper_location": "Definition 1 and Theorem 1, Sec. 1.1.3",
        "code_anchors": [
            "src/saber/theory/interior.py::ExposureLadder",
            "src/saber/theory/interior.py::interior_optimum",
            "src/saber/theory/interior.py::is_unimodal",
        ],
        "checks": [
            "the sampled marginal information is unimodal in the exposure count",
            "the sampled cumulative damage is increasing and convex",
            "the information accumulated over a fixed field is concave in the precision",
            "an interior exposure count beats both boundaries in most of the drawn ladders",
            "the optimum moves to the last boundary once redundancy is removed",
            "the optimal cadence differs across wells of different regime",
        ],
    },
    {
        "id": "C8",
        "statement": "Greedy selection under Lagrangian relaxation attains a constant-factor guarantee against the optimum of the relaxed submodular problem.",
        "paper_location": "Proposition 1, Sec. 1.1.3",
        "code_anchors": [
            "src/saber/theory/certificate.py::greedy_ratio_order",
            "src/saber/theory/certificate.py::fractional_relaxation",
            "src/saber/theory/certificate.py::best_lagrangian_value",
        ],
        "checks": [
            "the coverage gain is monotone and submodular",
            "the Lagrangian-relaxed greedy meets the bound against the relaxed optimum",
            "the cardinality greedy meets the (1 - 1/e) guarantee",
        ],
    },
    {
        "id": "C9",
        "statement": "With a uniform per-action damage the greedy bound holds with a constant factor independent of the horizon.",
        "paper_location": "Corollary 1, Sec. 1.1.3",
        "code_anchors": [
            "src/saber/theory/certificate.py::uniform_instance",
            "src/saber/theory/certificate.py::uniform_bound",
        ],
        "checks": ["the uniform-damage guarantee holds at every horizon"],
    },
    {
        "id": "C10",
        "statement": "A constrained planner cannot beat an unconstrained information-greedy policy, and the two agree once the budget stops binding.",
        "paper_location": "Theorem 2 and Eq. (5), Sec. 1.1.3",
        "code_anchors": [
            "src/saber/theory/parity.py::parity_test",
            "src/saber/theory/parity.py::rollout_episode",
            "src/saber/theory/parity.py::unconstrained_value",
        ],
        "checks": [
            "the budgeted planner never beats the unconstrained policy",
            "the two policies agree exactly once the budget cannot bind",
            "the deficit grows as the budget tightens",
        ],
    },
    {
        "id": "C11",
        "statement": "The perception front end adapts two pretrained backbones through low-rank adapters and emits instance masks, trajectories and one morphology embedding per instance.",
        "paper_location": "Sec. 2.2",
        "code_anchors": [
            "src/saber/perception/adapters.py::LoRAConv2d",
            "src/saber/perception/head.py::PerceptionFrontend",
            "src/saber/perception/segmenter.py::connected_instances",
            "src/saber/perception/tracker.py::associate",
        ],
        "checks": [
            "the front end returns one mask and one embedding per epoch",
            "the segmenter finds at least one instance per rendered frame",
            "the embedding width matches the configured width",
            "the adapted backbones carry trainable parameters and the static ones carry none",
            "the adapters participate in the forward pass",
            "the tracker produces at least one trajectory over the slice",
        ],
    },
    {
        "id": "C12",
        "statement": "A repeating variational state-space model maintains a calibrated four-axis belief, re-ties a drifting axis to its atlas prior, and stops once the total uncertainty stops falling.",
        "paper_location": "Sec. 2.3 and Algorithm 1",
        "code_anchors": [
            "src/saber/belief/estimator.py::VariationalEstimator",
            "src/saber/belief/pipeline.py::run_tme_belief",
            "src/saber/belief/recalibrate.py::recalibrate",
            "src/saber/belief/elbo.py::elbo_terms",
        ],
        "checks": [
            "the variational loss is finite and its three terms are reported",
            "the backward pass moves the estimator's parameters",
            "the fit lowers the variational bound",
            "the convergence test decides whether the belief loop stops early",
            "the drift guard fires under the shipped threshold and stays silent when widened",
            "every published belief has one mean and one uncertainty per axis",
        ],
    },
    {
        "id": "C13",
        "statement": "The budgeted information-gain planner searches the feasible, pruned action set, executes the Lagrangian maximiser, and stops on information saturation.",
        "paper_location": "Algorithm 2, Sec. 2.5",
        "code_anchors": [
            "src/saber/planner/big.py::candidates_for",
            "src/saber/planner/big.py::BigPlanner",
            "src/saber/planner/info_gain.py::EncodingTable",
        ],
        "checks": [
            "every compared policy spends inside its per-specimen budget",
            "the session horizon is the wells-per-cycle times the session's epochs",
            "the planner's gain matches the matrix Kalman closed form",
        ],
    },
    {
        "id": "C14",
        "statement": "The response head predicts a dose-response summary from the belief and a drug representation built from a molecular graph and a hashed fingerprint, with the believed coordinates omitted in the unconditioned control.",
        "paper_location": "Sec. 2.6 and Sec. 3.6",
        "code_anchors": [
            "src/saber/response/head.py::ResponseHead",
            "src/saber/response/fingerprint.py::hashed_fingerprint",
            "src/saber/response/molecular.py::GraphEncoder",
            "src/saber/response/transfer.py::directed_transfer",
        ],
        "checks": [
            "the fingerprint is deterministic in the molecule and its bit width is as configured",
            "two molecules that differ in an atom carry different fingerprints",
            "the conditioned head is built with the belief coordinates and the control without them",
            "both transfer tables cover every directed pair",
        ],
    },
    {
        "id": "C15",
        "statement": "The digital twin couples an agent-oriented growth model to a reaction-diffusion oxygen supply, and supplies the ground-truth policy value.",
        "paper_location": "Sec. 2.7 and Sec. 4.1",
        "code_anchors": [
            "src/saber/twin/coupled.py::run_twin",
            "src/saber/twin/oxygen.py::diffuse",
            "src/saber/twin/growth.py::logistic_step",
            "src/saber/twin/fidelity.py::assess",
        ],
        "checks": [
            "the oxygen solver's equilibrium matches the closed-form series",
            "the explicit substep stays inside the two-dimensional stability bound",
            "the agent population's realised area reproduces the carried biomass",
            "the twin's biomass is monotone and stays inside the carrying capacity",
            "the twin's oxygen field stays non-negative",
            "the logistic law reproduces its own closed form",
        ],
    },
    {
        "id": "C16",
        "statement": "The closed-loop policy value is estimated retrospectively with a family of off-policy estimators whose conformity with the twin's value is itself measured.",
        "paper_location": "Sec. 3.7",
        "code_anchors": [
            "src/saber/evaluation/offpolicy.py::ips",
            "src/saber/evaluation/offpolicy.py::doubly_robust",
            "src/saber/evaluation/offpolicy.py::fitted_q",
            "src/saber/evaluation/offpolicy.py::evaluate",
        ],
        "checks": [
            "every off-policy estimator returns a finite value",
            "the effective sample size lies between one and the batch size",
            "the fitted-Q estimate is stable under a support shift and the propensity estimate is not",
        ],
    },
    {
        "id": "C17",
        "statement": "The measured advantage of budgeted acquisition grows with the rate at which a well's microenvironment changes regime.",
        "paper_location": "Sec. 3.5, the Discussion and Sec. 4.1",
        "code_anchors": [
            "src/saber/study.py::run_frontier_study",
            "src/saber/study.py::linear_fit",
            "src/saber/belief/axes.py::regime_label",
        ],
        "checks": [
            "the frontier regression returns a slope, a coefficient and an interval",
            "the reported frontier interval brackets its own slope",
            "the four axis readings map onto four distinct strata",
            "the cohort's strata carry the transition rates the Discussion quotes",
        ],
    },
)

DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "D1",
        "paper_location": "Data Availability",
        "departs": "The article's evaluation is a retrospective secondary analysis of logged "
        "trajectories that are not deposited, and its benchmark environment, planner "
        "implementation and twin are promised only on request. The release therefore runs on "
        "a release cohort it builds from its own configuration.",
        "reason": "Without the study's trajectories no table value can be recomputed from this "
        "release. Every cohort-level quantity is reported as the release's own on its cohort, "
        "and the article's printed arithmetic is checked separately from its own tables.",
    },
    {
        "id": "D2",
        "paper_location": "Sec. 2.8 and the Appendix",
        "departs": "Sec. 2.8 locates the corpus splits, the metric definitions, the estimator "
        "suite and the statistical procedure in the Appendix, and the registry's caption counts "
        "ten supplementary tables; the PDF carries only Table A1.",
        "reason": "The missing material carries the hyper-parameters the article says every "
        "reported number traces to. Where a value is needed and not printed, the release uses a "
        "labelled engineering default exposed in the configuration, and records the dependency.",
    },
    {
        "id": "D3",
        "paper_location": "Sec. 2.4, Eq. (2)",
        "departs": "The damage weights are 'calibrated weights from the register logs of an "
        "instrument' and are not printed; the release ships exposure 1.0, defocus 0.5, fluidics "
        "2.0 and stage 0.25.",
        "reason": "The weights are the trade-offs the currency is built from, so they must exist. "
        "They live in the dish block, are varied by the uniform-weights ablation, and are marked "
        "here rather than presented as the article's.",
    },
    {
        "id": "D4",
        "paper_location": "Sec. 2.2",
        "departs": "The two backbones' widths and depths, the adapter rank and the embedding "
        "width are not printed. The release uses 24/2 and 32/2 with rank 8 and a 32-wide "
        "embedding.",
        "reason": "Sec. 2.2 names the architectural family and confines adaptation to the "
        "adapters; the widths are engineering defaults in the perception block, and the two "
        "Tier-3 rows vary the backbone and the adapters so the ablation is still the article's.",
    },
    {
        "id": "D5",
        "paper_location": "Algorithm 2, Sec. 2.5",
        "departs": "The planning horizon, the dual step, the saturation threshold, the pruning "
        "width and the wells per session cycle are not printed. The release uses a 48-decision "
        "session over two wells at the reference cadence, a dual step of 0.05 and a pruning "
        "width of 96.",
        "reason": "The session length has to be finite and is an execution scale, not a result: "
        "the horizon is the article's own wells-times-epochs structure evaluated over two wells, "
        "and every report records the scale it ran at.",
    },
    {
        "id": "D6",
        "paper_location": "Sec. 2.3",
        "departs": "The atlas priors supply the four axes' semantics; the article prints no "
        "prior means or standard deviations.",
        "reason": "The recalibration step re-ties a drifting axis to its prior and the "
        "variational bound diverges from it, so a prior must exist. The shipped prior is on the "
        "unit interval the axes live on and is declared in the belief block.",
    },
    {
        "id": "D7",
        "paper_location": "Sec. 3.6",
        "departs": "The article states that an effective-sample-size lower limit was fixed "
        "beforehand and prints only the observed range 412 to 8940. The release carries a floor "
        "of 400 in the response block.",
        "reason": "The floor is a reporting gate rather than a result; it is recorded as an "
        "engineering default because the printed limit is what the article declines to give.",
    },
    {
        "id": "D8",
        "paper_location": "Sec. 2.7 and Sec. 4.1",
        "departs": "The twin's growth fidelity band is printed as two percentages, 9.4 for twin "
        "growth and 24.8 for recorded growth, without the series they were computed from.",
        "reason": "The band cannot be recomputed without the recorded growth series, so the "
        "release evaluates the twin's own invariants and records the printed band as not run "
        "rather than as a reproduced number.",
    },
)


def _defined_symbols(path: Path) -> set[str]:
    """Top-level names a module defines, read from its own syntax tree."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def resolve_anchors(root: Path, anchors: Sequence[str]) -> list[dict[str, object]]:
    """Resolve each ``path::symbol`` anchor against the tree."""
    resolved: list[dict[str, object]] = []
    for anchor in anchors:
        path_text, _, symbol = anchor.partition("::")
        path = root / path_text
        if not path.is_file():
            resolved.append({"anchor": anchor, "resolved": False, "reason": "file missing"})
            continue
        present = symbol in _defined_symbols(path)
        resolved.append(
            {
                "anchor": anchor,
                "resolved": bool(present),
                "reason": "symbol present" if present else "symbol not defined at module level",
            }
        )
    return resolved


def build_claim_map(root: Path, results: Sequence[CheckResult]) -> dict[str, Any]:
    """Bind the mapped checks' statuses to each claim and resolve its anchors."""
    statuses = {result.name: result.status for result in results}
    entries: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for claim in CLAIMS:
        check_statuses: dict[str, str] = {}
        for name in claim["checks"]:
            if name not in statuses:
                unmatched.append(name)
                continue
            check_statuses[name] = statuses[name]
        anchors = resolve_anchors(root, claim["code_anchors"])
        if any(not entry["resolved"] for entry in anchors) or any(
            status == "FAIL" for status in check_statuses.values()
        ):
            verification = "FAIL"
        elif any(status in {"NOT_RUN", "BLOCKED"} for status in check_statuses.values()):
            verification = "PARTIALLY_VERIFIED"
        else:
            verification = "PASS"
        entries.append(
            {
                "id": claim["id"],
                "statement": claim["statement"],
                "paper_location": claim["paper_location"],
                "code_anchors": list(claim["code_anchors"]),
                "anchors": anchors,
                "checks": list(claim["checks"]),
                "check_statuses": check_statuses,
                "verification": verification,
            }
        )
    return {
        "n_claims": len(entries),
        "n_deviations": len(DEVIATIONS),
        "claims": entries,
        "deviations": [dict(entry) for entry in DEVIATIONS],
        "unmatched_check_names": sorted(set(unmatched)),
    }
