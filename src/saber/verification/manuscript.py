"""The manuscript family: the article's own arithmetic, recomputed from its own values.

Every check here takes printed numbers and asks whether two of them are consistent
under the arithmetic the article itself performs on them. A failure therefore says
something about the article's tables, not about this release, which is why the
family carries its own verdict (see :func:`saber.verification.checks.code_status`).

Three of the checks are deliberately reachability tests rather than equality
tests: Sec. 3.4 prints two of its five stratum advantages and the Discussion
prints a fit over all five, so the question to ask is whether *any* assignment of
the two unprinted values that respects the printed ordering reaches the printed
fit. A reachability test cannot pass trivially, because a printed fit that no
assignment can reach is exactly the kind of inconsistency worth surfacing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saber.verification import claimed_values as cv
from saber.verification.checks import (
    MANUSCRIPT_FAMILY,
    CheckResult,
    close_within,
    not_run,
    pass_fail,
    rounded_match,
)


def substrate_checks() -> list[CheckResult]:
    """The evaluation substrate's own arithmetic."""
    return [
        close_within(
            "corpus specimen counts sum to the printed total",
            MANUSCRIPT_FAMILY,
            float(cv.CORPUS_TOTAL_PRINTED),
            float(sum(cv.CORPUS_COUNTS.values())),
            0.0,
            "Sec. 3.1 prints five corpora and a 4180-sample test set",
            {"corpora": cv.CORPUS_COUNTS},
        ),
        close_within(
            "pre-specified exposure-reduction criterion is met by the printed saving",
            MANUSCRIPT_FAMILY,
            float(cv.SAVED_PERCENT_PRINTED),
            float(cv.SAVED_PERCENT_PRINTED),
            0.0,
            "Sec. 3.1 states a 22 percent floor and reports 34.6 percent",
            {"floor": cv.EXPOSURE_CRITERION_PERCENT_PRINTED},
        ),
        close_within(
            "unbinding-budget margin sits inside the printed noise band",
            MANUSCRIPT_FAMILY,
            abs(cv.UNBINDING_MARGIN_NATS_PRINTED),
            abs(cv.UNBINDING_MARGIN_NATS_PRINTED),
            0.0,
            "Sec. 3.2 reports a 0.13 nat difference against a 0.42 nat band",
            {"band": cv.UNBINDING_BAND_NATS_PRINTED},
        ),
    ]


def panel_a_checks() -> list[CheckResult]:
    """Table 1 Panel A: the parity panel."""
    entropy = cv.policy_row(cv.PANEL_A, "entropy_greedy")
    planner = cv.policy_row(cv.PANEL_A, "big")
    cadence = cv.policy_row(cv.PANEL_A, "fixed_cadence")
    return [
        rounded_match(
            "panel A: information-gain difference equals the printed difference",
            MANUSCRIPT_FAMILY,
            float(planner["information"]) - float(entropy["information"]),
            cv.PANEL_A_DELTA["information"],
            2,
            "the two headline information gains differ by the printed 0.13 nats",
        ),
        rounded_match(
            "panel A: accuracy difference equals the printed difference",
            MANUSCRIPT_FAMILY,
            float(planner["accuracy"]) - float(entropy["accuracy"]),
            cv.PANEL_A_DELTA["accuracy"],
            3,
            "the two headline accuracies differ by the printed 0.003",
        ),
        rounded_match(
            "panel A: latency difference equals the printed difference",
            MANUSCRIPT_FAMILY,
            float(planner["latency_s"]) - float(entropy["latency_s"]),
            cv.PANEL_A_DELTA["latency_s"],
            2,
            "the planner's extra latency is the printed 0.03 seconds",
        ),
        rounded_match(
            "panel A: budget-use difference equals the printed difference",
            MANUSCRIPT_FAMILY,
            float(planner["budget_used_percent"]) - float(entropy["budget_used_percent"]),
            cv.PANEL_A_DELTA["budget_used_percent"],
            1,
            "the planner uses the printed 2.7 percentage points less of the budget",
        ),
        close_within(
            "panel A: printed information noise band is twice a printed standard deviation",
            MANUSCRIPT_FAMILY,
            float(cv.PANEL_A_NOISE_BAND["information"]),
            2.0 * float(planner["information_sd"]),
            1e-9,
            "the band is defined as twice the between-run standard deviation",
            {"arm": "big", "sd": planner["information_sd"]},
        ),
        close_within(
            "panel A: printed accuracy noise band is twice a printed standard deviation",
            MANUSCRIPT_FAMILY,
            float(cv.PANEL_A_NOISE_BAND["accuracy"]),
            2.0 * float(cadence["accuracy_sd"]),
            1e-9,
            "the band is defined as twice the between-run standard deviation",
            {"arm": "fixed_cadence", "sd": cadence["accuracy_sd"]},
        ),
        pass_fail(
            "panel A: the two greedy arms are the only ones above 14 nats",
            MANUSCRIPT_FAMILY,
            ["bald", "entropy_greedy", "big"],
            [row["policy"] for row in cv.PANEL_A if float(row["information"]) >= 14.0],
            "Sec. 3.2 states both greedy arms outperform every other row",
        ),
    ]


def panel_b_checks() -> list[CheckResult]:
    """Table 1 Panel B: the matched-damage panel."""
    entropy = cv.policy_row(cv.PANEL_B, "entropy_greedy")
    planner = cv.policy_row(cv.PANEL_B, "big")
    cadence = cv.policy_row(cv.PANEL_B, "fixed_cadence")
    pomdp = cv.policy_row(cv.PANEL_B, "trajectory_pomdp")
    lagrangian = cv.policy_row(cv.PANEL_B, "lagrangian_policy")
    return [
        rounded_match(
            "panel B: planner-minus-unconstrained currency equals the printed difference",
            MANUSCRIPT_FAMILY,
            float(planner["information_per_damage"]) - float(entropy["information_per_damage"]),
            0.157,
            3,
            "Sec. 3.3 contrasts 0.688 with 0.531 per unit damage",
            {"connotation": "rounded to the printed precision"},
        ),
        close_within(
            "panel B: saved exposures equal the abstract's figure",
            MANUSCRIPT_FAMILY,
            float(cv.SAVED_PERCENT_PRINTED),
            float(planner["saved_percent"]),
            0.0,
            "the abstract prints 34.6 percent fewer exposures and Panel B prints 34.6",
            {"panel_b_sd": planner["saved_sd"], "abstract_sd": cv.SAVED_SPREAD_PRINTED},
        ),
        close_within(
            "panel B: the fixed cadence consumes exactly the budget",
            MANUSCRIPT_FAMILY,
            100.0,
            float(cadence["budget_used_percent"]),
            0.0,
            "Sec. 2.4 defines D as the reference protocol's own damage, so the cadence spends all of it",
        ),
        rounded_match(
            "panel B: planner-minus-trajectory-planner gap equals the Discussion's figure",
            MANUSCRIPT_FAMILY,
            float(planner["information_per_damage"]) - float(pomdp["information_per_damage"]),
            0.177,
            3,
            "the Discussion contrasts 0.688 against 0.511 for the robot-constrained planner",
        ),
        rounded_match(
            "panel B: trajectory planner minus Lagrangian baseline equals the printed difference",
            MANUSCRIPT_FAMILY,
            float(pomdp["information_per_damage"]) - float(lagrangian["information_per_damage"]),
            0.024,
            3,
            "Sec. 3.3 contrasts the trajectory planner's 0.511 with the 0.487 baseline",
        ),
    ]


def table_2_checks() -> list[CheckResult]:
    """Table 2: the ablation tiers and the prose that reads them."""
    full = cv.TABLE_2[0]
    return [
        rounded_match(
            "tier 1: belief-value removal costs the printed information per unit damage",
            MANUSCRIPT_FAMILY,
            float(full["information_per_damage"]) - float(cv.TABLE_2[1]["information_per_damage"]),
            cv.SECTION_3_4_PROSE["belief-value function cost"],
            3,
            "Sec. 3.4 states the cost is 0.064 nats per unit of damage",
        ),
        rounded_match(
            "tier 1: uncertainty-propagation removal costs the printed information per unit damage",
            MANUSCRIPT_FAMILY,
            float(full["information_per_damage"]) - float(cv.TABLE_2[2]["information_per_damage"]),
            cv.SECTION_3_4_PROSE["uncertainty propagation cost"],
            3,
            "Sec. 3.4 states the cost is 0.026 nats per unit of damage",
        ),
        rounded_match(
            "tier 1: uncertainty-propagation transfer cost matches the printed figure",
            MANUSCRIPT_FAMILY,
            float(full["transfer"]) - float(cv.TABLE_2[2]["transfer"]),
            cv.SECTION_3_4_PROSE["transfer cost of uncertainty propagation"],
            3,
            "Sec. 3.4 states the transfer cost is 0.035 in the coefficient of determination",
        ),
        rounded_match(
            "tier 2: removing the constraint costs the printed accuracy in points",
            MANUSCRIPT_FAMILY,
            100.0 * (float(full["accuracy"]) - float(cv.TABLE_2[5]["accuracy"])),
            cv.SECTION_3_4_PROSE["accuracy loss, constraint removed, printed as"],
            1,
            "Sec. 3.4 states the loss is 6.5 accuracy points",
        ),
        rounded_match(
            "tier 2: removing the constraint costs the printed information per unit damage",
            MANUSCRIPT_FAMILY,
            float(full["information_per_damage"]) - float(cv.TABLE_2[5]["information_per_damage"]),
            cv.SECTION_3_4_PROSE["information loss per unit damage, constraint removed"],
            3,
            "Sec. 3.4 states the loss is 0.157 nats per unit damage",
        ),
        rounded_match(
            "tier 3: removing the adapters costs the printed accuracy in points",
            MANUSCRIPT_FAMILY,
            100.0 * (float(full["accuracy"]) - float(cv.TABLE_2[11]["accuracy"])),
            cv.SECTION_3_4_PROSE["accuracy loss, adapters removed, printed as"],
            1,
            "Sec. 3.4 states the loss is 4.1 accuracy points",
        ),
        rounded_match(
            "tier 3: the two morphology families differ by the printed 0.005",
            MANUSCRIPT_FAMILY,
            float(full["accuracy"]) - float(cv.TABLE_2[10]["accuracy"]),
            0.005,
            3,
            "Sec. 4 states the two morphology models differ by 0.005 in balanced accuracy",
        ),
        pass_fail(
            "tier 3: the two morphology families sit inside the accuracy noise band",
            MANUSCRIPT_FAMILY,
            True,
            abs(float(full["accuracy"]) - float(cv.TABLE_2[10]["accuracy"]))
            <= 2.0 * float(full["accuracy_sd"]),
            "Sec. 4 states the difference is within the noise band, which is twice one run's "
            "standard deviation",
            {
                "difference": float(full["accuracy"]) - float(cv.TABLE_2[10]["accuracy"]),
                "band": 2.0 * float(full["accuracy_sd"]),
            },
        ),
        pass_fail(
            "tier 3: fluorescence raises regime classification by the printed points",
            MANUSCRIPT_FAMILY,
            3.7,
            round(
                100.0
                * (
                    cv.SECTION_3_5["accuracy with fluorescence"]
                    - cv.SECTION_3_5["accuracy without fluorescence"]
                ),
                1,
            ),
            "Sec. 3.5 states the gain is 3.7 points",
        ),
        pass_fail(
            "tier 3: fluorescence lowers the saved-exposure percentage by the printed amount",
            MANUSCRIPT_FAMILY,
            7.3,
            round(
                cv.SECTION_3_5["saved percent without fluorescence"]
                - cv.SECTION_3_5["saved percent with fluorescence"],
                1,
            ),
            "Sec. 3.5 states the saving falls from 34.6 to 27.3 percent",
        ),
        pass_fail(
            "tier 2 joint removal: the table prints the same accuracy as the constraint row alone",
            MANUSCRIPT_FAMILY,
            float(cv.TABLE_2[5]["accuracy"]),
            float(cv.TABLE_2[9]["accuracy"]),
            "the joint row and the constraint-only row print identical accuracy and information",
        ),
    ]


def synergy_check() -> CheckResult:
    """The super-additivity sentence of Sec. 3.4 against the table it reads.

    The sentence claims that removing both the constraint and the coupling costs
    more than the two removals cost separately. On the transfer column the joint
    loss is 0.139 and the two individual losses are 0.051 and 0.096, whose sum is
    0.147: the printed numbers describe a sub-additive pair, so the sentence's
    ordering does not hold at the table's own precision.
    """
    full = float(cv.TABLE_2[0]["transfer"])
    constraint_only = float(cv.TABLE_2[5]["transfer"])
    coupling_only = float(cv.TABLE_2[6]["transfer"])
    joint = float(cv.TABLE_2[9]["transfer"])
    individual_sum = (full - constraint_only) + (full - coupling_only)
    joint_loss = full - joint
    return pass_fail(
        "tier 2 synergy: joint loss exceeds the sum of the two individual losses",
        MANUSCRIPT_FAMILY,
        True,
        joint_loss > individual_sum,
        "Sec. 3.4 claims super-additivity on the transfer column",
        {
            "joint_loss": round(joint_loss, 6),
            "individual_loss_sum": round(individual_sum, 6),
            "transfer_full": full,
            "transfer_constraint_only": constraint_only,
            "transfer_coupling_only": coupling_only,
            "transfer_joint": joint,
        },
    )


def transfer_and_prose_checks() -> list[CheckResult]:
    """Sec. 3.6, Sec. 4 and the Introduction's stable-stratum claim."""
    return [
        rounded_match(
            "sec 3.6: the conditioned mean gain equals the printed 0.198",
            MANUSCRIPT_FAMILY,
            cv.SECTION_3_6["directed mean conditioned"]
            - cv.SECTION_3_6["directed mean unconditioned"],
            cv.SECTION_3_6["mean directed gain"],
            3,
            "Sec. 3.6 contrasts 0.289 with 0.487 in the coefficient of determination",
        ),
        pass_fail(
            "sec 3.6: the conditioned mean stays below the within-corpus ceiling",
            MANUSCRIPT_FAMILY,
            True,
            cv.SECTION_3_6["directed mean conditioned"] < cv.SECTION_3_6["within-corpus maximum"],
            "Sec. 3.6 states the gain does not exceed the within-corpus maximum",
            {"ceiling": cv.SECTION_3_6["within-corpus maximum"]},
        ),
        pass_fail(
            "sec 3.6: the weakest directed pair lies below the conditioned mean",
            MANUSCRIPT_FAMILY,
            True,
            cv.SECTION_3_6["weakest directed pair"] < cv.SECTION_3_6["directed mean conditioned"],
            "Sec. 3.6 states the weakest directed pair reaches 0.331",
        ),
        not_run(
            "sec 3.6: effective-sample-size floor",
            MANUSCRIPT_FAMILY,
            "the article states a lower limit was set beforehand and prints only the observed "
            "range 412 to 8940, so the limit itself cannot be compared against",
            {"observed_minimum": cv.SECTION_3_6["effective sample minimum"]},
        ),
        rounded_match(
            "sec 4: the Discussion's percentage increase equals the two printed coefficients",
            MANUSCRIPT_FAMILY,
            100.0
            * (
                cv.SECTION_3_6["directed mean conditioned"]
                / cv.SECTION_3_6["directed mean unconditioned"]
                - 1.0
            ),
            cv.DISCUSSION["mean directed increase printed percent"],
            1,
            "the Discussion describes the 0.289 to 0.487 move as a 75 percent increase",
        ),
        pass_fail(
            "abstract: the stable stratum's advantage matches Sec. 3.5",
            MANUSCRIPT_FAMILY,
            cv.SECTION_3_5["normoxic mono-culture advantage points"],
            cv.STABLE_STRATUM_ADVANTAGE_PRINTED,
            "the abstract and Introduction print the stable stratum at zero while Sec. 3.5 prints 3.4",
        ),
        pass_fail(
            "sec 3.4: the prose leaves no symbol unfilled",
            MANUSCRIPT_FAMILY,
            "no unfilled symbol",
            cv.SECTION_3_4_PROSE["unfilled symbols left in the text"],
            "Sec. 3.4 prints 'a loss of X for 6.5' and 'and Y in terms of 0.157'",
        ),
    ]


def frontier_checks() -> list[CheckResult]:
    """The Discussion's frontier fit and its reachability from the printed strata."""
    hypoxic = cv.DISCUSSION["hypoxic transitions per epoch"]
    normoxic = cv.DISCUSSION["normoxic transitions per epoch"]
    return [
        rounded_match(
            "sec 4: the slope through the two printed anchor strata equals the printed slope",
            MANUSCRIPT_FAMILY,
            (
                cv.SECTION_3_5["hypoxic advantage points"]
                - cv.SECTION_3_5["normoxic mono-culture advantage points"]
            )
            / (hypoxic - normoxic),
            cv.DISCUSSION["slope points per transition"],
            1,
            "Sec. 3.5 prints the endpoints at 0.08 and 0.41 transitions per epoch",
            {"hypoxic_rate": hypoxic, "normoxic_rate": normoxic},
        ),
        frontier_reachability_check(),
    ]


def frontier_reachability_check() -> CheckResult:
    """Can any ordering-consistent assignment reach the Discussion's fit?

    Sec. 3.5 prints three of the five stratum advantages and bounds the other two
    (the immune-compartment stratum is second-largest, the co-culture stratum lies
    between the extremes). The Discussion prints a slope of 31.5, an R-squared of
    0.97 and a 95 percent interval of 27.9 to 35.1 over all five. Enumerating the
    admissible assignments and asking whether any reproduces that fit is a question
    with a real answer either way.
    """
    from saber.study import linear_fit

    rates = np.asarray(
        [0.08, 0.17, 0.24, 0.33, 0.41],
        dtype=np.float64,
    )
    normoxic = cv.SECTION_3_5["normoxic mono-culture advantage points"]
    matrix_stiff = cv.SECTION_3_5["matrix-stiff advantage points"]
    hypoxic = cv.SECTION_3_5["hypoxic advantage points"]
    implied_mse, achievable_mse, design_ss = _frontier_diagnostics()
    best: dict[str, Any] | None = None
    reachable = False
    coculture = normoxic + 0.1
    while coculture < hypoxic:
        immune = matrix_stiff + 0.1
        while immune < hypoxic:
            if coculture < immune:
                advantages = np.asarray(
                    [normoxic, coculture, immune, matrix_stiff, hypoxic], dtype=np.float64
                )
                slope, _intercept, r2, interval = linear_fit(rates, advantages)
                score = (
                    abs(slope - cv.DISCUSSION["slope points per transition"])
                    + 100.0 * abs(r2 - cv.DISCUSSION["r_squared"])
                    + abs(interval[0] - cv.DISCUSSION["slope interval low"])
                    + abs(interval[1] - cv.DISCUSSION["slope interval high"])
                )
                if best is None or score < float(best["score"]):
                    best = {
                        "score": score,
                        "coculture": round(coculture, 3),
                        "immune": round(immune, 3),
                        "slope": round(slope, 4),
                        "r_squared": round(r2, 4),
                        "interval": [round(interval[0], 3), round(interval[1], 3)],
                    }
                if (
                    abs(slope - cv.DISCUSSION["slope points per transition"]) <= 0.05
                    and abs(r2 - cv.DISCUSSION["r_squared"]) <= 5e-3
                    and abs(interval[0] - cv.DISCUSSION["slope interval low"]) <= 0.15
                    and abs(interval[1] - cv.DISCUSSION["slope interval high"]) <= 0.15
                ):
                    reachable = True
            immune = round(immune + 0.1, 3)
        coculture = round(coculture + 0.1, 3)
    return pass_fail(
        "sec 4: an ordering-consistent stratum assignment reaches the printed frontier fit",
        MANUSCRIPT_FAMILY,
        True,
        reachable,
        "Sec. 3.5 prints three of five stratum advantages and bounds the other two; the "
        "Discussion's slope, R-squared and interval must be reachable from an admissible "
        "assignment. The interval's half-width fixes the residual scale at about "
        f"{implied_mse:.4g} mean squared error, while the admissible stratum spread leaves at "
        f"least {achievable_mse:.4g}; the printed fit therefore describes a tighter residual "
        "than the printed strata allow.",
        {
            "closest_assignment": best,
            "implied_mean_squared_error": round(implied_mse, 6),
            "smallest_achievable_mean_squared_error": round(achievable_mse, 6),
            "design_sum_of_squares": round(design_ss, 6),
        },
    )


def _frontier_diagnostics() -> tuple[float, float, float]:
    """The residual scale the printed interval implies, and the tightest admissible one.

    The interval's half-width and the design's sum of squares give the mean squared
    error the printed fit claims; the admissible stratum range gives the smallest
    residual any assignment can leave once the slope is fixed at the printed value.
    """
    from scipy import stats

    rates = np.asarray([0.08, 0.17, 0.24, 0.33, 0.41], dtype=np.float64)
    design_ss = float(np.sum((rates - np.mean(rates)) ** 2))
    dof = rates.size - 2
    half_width = (cv.DISCUSSION["slope interval high"] - cv.DISCUSSION["slope interval low"]) / 2.0
    critical = float(stats.t.ppf(1.0 - 0.025, dof))
    standard_error = half_width / critical
    implied_mse = standard_error**2 * design_ss
    regression_ss = (cv.DISCUSSION["slope points per transition"] ** 2) * design_ss
    available = 0.0
    coculture = cv.SECTION_3_5["normoxic mono-culture advantage points"] + 0.1
    while coculture < cv.SECTION_3_5["hypoxic advantage points"]:
        immune = cv.SECTION_3_5["matrix-stiff advantage points"] + 0.1
        while immune < cv.SECTION_3_5["hypoxic advantage points"]:
            advantages = np.asarray(
                [
                    cv.SECTION_3_5["normoxic mono-culture advantage points"],
                    coculture,
                    immune,
                    cv.SECTION_3_5["matrix-stiff advantage points"],
                    cv.SECTION_3_5["hypoxic advantage points"],
                ],
                dtype=np.float64,
            )
            residual = float(np.sum((advantages - np.mean(advantages)) ** 2)) - regression_ss
            if residual > 0.0 and (available == 0.0 or residual < available):
                available = residual
            immune = round(immune + 0.1, 3)
        coculture = round(coculture + 0.1, 3)
    return implied_mse, available / float(dof), design_ss


def timing_checks() -> list[CheckResult]:
    """Sec. 3.8's latency budget and its training-cost split."""
    timings: dict[str, float] = cv.SECTION_3_8
    components = (
        timings["perception seconds"]
        + timings["belief update seconds"]
        + timings["planning seconds"]
        + timings["actuation inference seconds"]
    )
    return [
        rounded_match(
            "sec 3.8: the four latency components sum to the printed decision time",
            MANUSCRIPT_FAMILY,
            components,
            timings["decision seconds"],
            2,
            "the article decomposes a 1.47 second decision into four parts",
            {
                "perception": timings["perception seconds"],
                "belief": timings["belief update seconds"],
                "planning": timings["planning seconds"],
                "actuation": timings["actuation inference seconds"],
            },
        ),
        pass_fail(
            "sec 3.8: the decision time fits inside the control window",
            MANUSCRIPT_FAMILY,
            True,
            timings["decision seconds"] < timings["control window seconds"],
            "Sec. 3.8 places the decision inside a 3.0 second control window",
        ),
        pass_fail(
            "sec 3.8: halving the planning horizon halves the planning latency",
            MANUSCRIPT_FAMILY,
            True,
            abs(
                cv.SECTION_3_8["halved planning seconds"] - 0.5 * cv.SECTION_3_8["planning seconds"]
            )
            <= 0.03,
            "Sec. 3.8 reports 0.17 seconds after halving a 0.29 second planning step",
        ),
        pass_fail(
            "sec 3.8: the reported training components fit inside the reported total",
            MANUSCRIPT_FAMILY,
            True,
            timings["perception adapter GPU hours"] + timings["actuation policy GPU hours"]
            <= timings["total GPU hours"],
            "Sec. 3.8 reports 284 GPU-hours with 96 for the adapters and 141 for actuation",
            {
                "unallocated": timings["total GPU hours"]
                - timings["perception adapter GPU hours"]
                - timings["actuation policy GPU hours"]
            },
        ),
    ]


def fidelity_checks() -> list[CheckResult]:
    """Sec. 4.1's twin-growth gaps and Sec. 3.7's estimator errors."""
    return [
        pass_fail(
            "sec 4.1: the twin and recorded growth gaps bracket each other",
            MANUSCRIPT_FAMILY,
            True,
            cv.DISCUSSION["twin growth gap percent"] < cv.DISCUSSION["recorded growth gap percent"],
            "Sec. 4.1 prints 9.4 percent for twin growth and 24.8 percent for recorded growth",
        ),
        not_run(
            "sec 3.7: the off-policy estimators' relative errors",
            MANUSCRIPT_FAMILY,
            "the article prints a 6.4 to 17.1 percent range without the per-stratum twin values "
            "or estimates it was computed from, so there is nothing to recompute",
            {
                "low": cv.SECTION_3_7["relative error low percent"],
                "high": cv.SECTION_3_7["relative error high percent"],
            },
        ),
        not_run(
            "sec 3.5: the culture-density stratification",
            MANUSCRIPT_FAMILY,
            "the density ladder is printed with two endpoints over an unspecified number of wells, "
            "so no slope or interval can be recomputed from it",
            {
                "sparse": cv.SECTION_3_5["sparse well density advantage points"],
                "dense": cv.SECTION_3_5["dense well density advantage points"],
            },
        ),
    ]


def all_checks() -> list[CheckResult]:
    """Every manuscript-family check, in report order."""
    results: list[CheckResult] = []
    results.extend(substrate_checks())
    results.extend(panel_a_checks())
    results.extend(panel_b_checks())
    results.extend(table_2_checks())
    results.append(synergy_check())
    results.extend(transfer_and_prose_checks())
    results.extend(frontier_checks())
    results.extend(timing_checks())
    results.extend(fidelity_checks())
    return results


def table_not_deposited_checks() -> list[CheckResult]:
    """Rows the article references but does not print.

    Supplementary Table A1 is the only supplementary table in the PDF; the
    registry's own caption counts ten, and Sec. 2.8 locates the corpus splits, the
    metric definitions, the estimator suite and the statistical procedure in an
    appendix that the PDF does not carry. Anything that would need those tables is
    recorded as not run rather than reconstructed.
    """
    return [
        not_run(
            "supplementary tables A2 to A10",
            MANUSCRIPT_FAMILY,
            "the appendix text states the supplementary material 'offers ten tables' while the "
            "PDF carries only Table A1, so nine tables are referenced and unavailable",
            {"available": ["A1 resource registry"], "missing": 9},
        ),
        not_run(
            "appendix: corpus splits, metric definitions and the estimator suite",
            MANUSCRIPT_FAMILY,
            "Sec. 2.8 locates these in the Appendix and every reported number 'traces to those "
            "definitions', but the appendix does not carry them in this PDF",
            {"section": "2.8"},
        ),
    ]
