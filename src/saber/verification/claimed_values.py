"""The manuscript's printed values, with their anchors.

Every number here was transcribed from the article; nothing is recomputed. The
anchors name the table, panel or section the value sits in, so a failing check can
be traced back to the page it came from rather than to a summary.

Two conventions matter for reading the tables below. First, every value keeps the
printed precision: where the article prints three decimals, the transcription
carries three decimals, because the checks compare at the precision the table
prints. Second, a value that the article prints inside prose (Sec. 3.4, Sec. 3.5,
Sec. 3.8, the Discussion) is kept beside its sentence's own wording, since those
sentences are where the release's manuscript findings live.
"""

from __future__ import annotations

from typing import Any

# Sec. 3.1: the evaluation substrate.
CORPUS_COUNTS: dict[str, int] = {"A": 1246, "B": 873, "C": 964, "D": 621, "E": 476}
CORPUS_TOTAL_PRINTED = 4180
RUNS_PRINTED = 5
SAVED_PERCENT_PRINTED = 34.6
SAVED_SPREAD_PRINTED = 1.8
EXPOSURE_CRITERION_PERCENT_PRINTED = 22.0
UNBINDING_MARGIN_NATS_PRINTED = 0.13
UNBINDING_BAND_NATS_PRINTED = 0.42

# Table 1, Panel A: budget does not bind.
PANEL_A: tuple[dict[str, Any], ...] = (
    {
        "policy": "fixed_cadence",
        "information": 11.40,
        "information_sd": 0.28,
        "accuracy": 0.712,
        "accuracy_sd": 0.011,
        "latency_s": 0.31,
        "budget_used_percent": 18.4,
    },
    {
        "policy": "uniform_random",
        "information": 12.05,
        "information_sd": 0.31,
        "accuracy": 0.734,
        "accuracy_sd": 0.013,
        "latency_s": 0.29,
        "budget_used_percent": 21.7,
    },
    {
        "policy": "protocol_heuristic",
        "information": 12.42,
        "information_sd": 0.25,
        "accuracy": 0.756,
        "accuracy_sd": 0.012,
        "latency_s": 0.34,
        "budget_used_percent": 26.1,
    },
    {
        "policy": "expected_improvement",
        "information": 12.88,
        "information_sd": 0.26,
        "accuracy": 0.791,
        "accuracy_sd": 0.011,
        "latency_s": 1.24,
        "budget_used_percent": 41.3,
    },
    {
        "policy": "bayes_adaptive",
        "information": 13.02,
        "information_sd": 0.23,
        "accuracy": 0.774,
        "accuracy_sd": 0.011,
        "latency_s": 0.88,
        "budget_used_percent": 33.5,
    },
    {
        "policy": "lagrangian_policy",
        "information": 13.31,
        "information_sd": 0.29,
        "accuracy": 0.798,
        "accuracy_sd": 0.012,
        "latency_s": 1.52,
        "budget_used_percent": 47.9,
    },
    {
        "policy": "trajectory_pomdp",
        "information": 13.74,
        "information_sd": 0.24,
        "accuracy": 0.806,
        "accuracy_sd": 0.010,
        "latency_s": 1.86,
        "budget_used_percent": 52.4,
    },
    {
        "policy": "bald",
        "information": 14.61,
        "information_sd": 0.22,
        "accuracy": 0.833,
        "accuracy_sd": 0.009,
        "latency_s": 1.41,
        "budget_used_percent": 71.2,
    },
    {
        "policy": "entropy_greedy",
        "information": 14.95,
        "information_sd": 0.19,
        "accuracy": 0.842,
        "accuracy_sd": 0.008,
        "latency_s": 1.44,
        "budget_used_percent": 96.8,
    },
    {
        "policy": "big",
        "information": 14.82,
        "information_sd": 0.21,
        "accuracy": 0.839,
        "accuracy_sd": 0.009,
        "latency_s": 1.47,
        "budget_used_percent": 94.1,
    },
)

PANEL_A_DELTA: dict[str, float] = {
    "information": -0.13,
    "accuracy": -0.003,
    "latency_s": 0.03,
    "budget_used_percent": -2.7,
}

PANEL_A_NOISE_BAND: dict[str, float] = {"information": 0.42, "accuracy": 0.022}

# Table 1, Panel B: matched damage budget.
PANEL_B: tuple[dict[str, Any], ...] = (
    {
        "policy": "fixed_cadence",
        "information_per_damage": 0.412,
        "sd": 0.015,
        "accuracy": 0.712,
        "accuracy_sd": 0.011,
        "saved_percent": 0.0,
        "saved_sd": 0.0,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "uniform_random",
        "information_per_damage": 0.438,
        "sd": 0.017,
        "accuracy": 0.734,
        "accuracy_sd": 0.013,
        "saved_percent": 3.1,
        "saved_sd": 0.9,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "protocol_heuristic",
        "information_per_damage": 0.442,
        "sd": 0.018,
        "accuracy": 0.736,
        "accuracy_sd": 0.012,
        "saved_percent": 3.6,
        "saved_sd": 0.8,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "expected_improvement",
        "information_per_damage": 0.449,
        "sd": 0.019,
        "accuracy": 0.741,
        "accuracy_sd": 0.011,
        "saved_percent": 4.2,
        "saved_sd": 1.1,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "bayes_adaptive",
        "information_per_damage": 0.478,
        "sd": 0.017,
        "accuracy": 0.752,
        "accuracy_sd": 0.011,
        "saved_percent": 6.9,
        "saved_sd": 1.2,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "lagrangian_policy",
        "information_per_damage": 0.487,
        "sd": 0.020,
        "accuracy": 0.749,
        "accuracy_sd": 0.012,
        "saved_percent": 7.6,
        "saved_sd": 1.3,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "trajectory_pomdp",
        "information_per_damage": 0.511,
        "sd": 0.017,
        "accuracy": 0.758,
        "accuracy_sd": 0.010,
        "saved_percent": 9.8,
        "saved_sd": 1.1,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "bald",
        "information_per_damage": 0.522,
        "sd": 0.018,
        "accuracy": 0.762,
        "accuracy_sd": 0.009,
        "saved_percent": 11.1,
        "saved_sd": 1.2,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "entropy_greedy",
        "information_per_damage": 0.531,
        "sd": 0.016,
        "accuracy": 0.769,
        "accuracy_sd": 0.010,
        "saved_percent": 12.4,
        "saved_sd": 1.3,
        "budget_used_percent": 100.0,
    },
    {
        "policy": "big",
        "information_per_damage": 0.688,
        "sd": 0.021,
        "accuracy": 0.834,
        "accuracy_sd": 0.009,
        "saved_percent": 34.6,
        "saved_sd": 1.8,
        "budget_used_percent": 99.4,
    },
)

# Table 2: the three ablation tiers.
TABLE_2: tuple[dict[str, Any], ...] = (
    {
        "tier": "-",
        "variant": "Full SABER",
        "accuracy": 0.834,
        "accuracy_sd": 0.009,
        "information_per_damage": 0.688,
        "info_sd": 0.021,
        "transfer": 0.487,
        "transfer_sd": 0.014,
        "saved_percent": 34.6,
        "saved_sd": 1.8,
    },
    {
        "tier": "1",
        "variant": "-belief-value function (myopic rho)",
        "accuracy": 0.808,
        "accuracy_sd": 0.011,
        "information_per_damage": 0.624,
        "info_sd": 0.023,
        "transfer": 0.487,
        "transfer_sd": 0.014,
        "saved_percent": 27.1,
        "saved_sd": 1.6,
    },
    {
        "tier": "1",
        "variant": "-uncertainty propagation into TC-Rx",
        "accuracy": 0.821,
        "accuracy_sd": 0.010,
        "information_per_damage": 0.661,
        "info_sd": 0.020,
        "transfer": 0.452,
        "transfer_sd": 0.015,
        "saved_percent": 32.4,
        "saved_sd": 1.7,
    },
    {
        "tier": "1",
        "variant": "-twin-based policy training",
        "accuracy": 0.796,
        "accuracy_sd": 0.012,
        "information_per_damage": 0.639,
        "info_sd": 0.022,
        "transfer": 0.471,
        "transfer_sd": 0.014,
        "saved_percent": 22.8,
        "saved_sd": 2.1,
    },
    {
        "tier": "1",
        "variant": "-recalibration of drifting axes",
        "accuracy": 0.827,
        "accuracy_sd": 0.010,
        "information_per_damage": 0.672,
        "info_sd": 0.019,
        "transfer": 0.479,
        "transfer_sd": 0.013,
        "saved_percent": 33.2,
        "saved_sd": 1.6,
    },
    {
        "tier": "2",
        "variant": "-damage constraint (unconstrained objective)",
        "accuracy": 0.769,
        "accuracy_sd": 0.010,
        "information_per_damage": 0.531,
        "info_sd": 0.016,
        "transfer": 0.436,
        "transfer_sd": 0.015,
        "saved_percent": 12.4,
        "saved_sd": 1.3,
    },
    {
        "tier": "2",
        "variant": "-microenvironment coupling (unconditioned head)",
        "accuracy": 0.834,
        "accuracy_sd": 0.009,
        "information_per_damage": 0.688,
        "info_sd": 0.021,
        "transfer": 0.391,
        "transfer_sd": 0.013,
        "saved_percent": 34.6,
        "saved_sd": 1.8,
    },
    {
        "tier": "2",
        "variant": "-damage weighting (uniform weights)",
        "accuracy": 0.816,
        "accuracy_sd": 0.011,
        "information_per_damage": 0.641,
        "info_sd": 0.022,
        "transfer": 0.487,
        "transfer_sd": 0.014,
        "saved_percent": 26.4,
        "saved_sd": 1.9,
    },
    {
        "tier": "2",
        "variant": "-rho currency (gain only)",
        "accuracy": 0.786,
        "accuracy_sd": 0.011,
        "information_per_damage": 0.548,
        "info_sd": 0.019,
        "transfer": 0.443,
        "transfer_sd": 0.014,
        "saved_percent": 15.7,
        "saved_sd": 1.5,
    },
    {
        "tier": "2",
        "variant": "joint -constraint -coupling",
        "accuracy": 0.769,
        "accuracy_sd": 0.010,
        "information_per_damage": 0.531,
        "info_sd": 0.016,
        "transfer": 0.348,
        "transfer_sd": 0.015,
        "saved_percent": 12.4,
        "saved_sd": 1.3,
    },
    {
        "tier": "3",
        "variant": "morphology backbone B (swap)",
        "accuracy": 0.829,
        "accuracy_sd": 0.010,
        "information_per_damage": 0.681,
        "info_sd": 0.021,
        "transfer": 0.479,
        "transfer_sd": 0.015,
        "saved_percent": 34.0,
        "saved_sd": 1.8,
    },
    {
        "tier": "3",
        "variant": "frozen backbones, no adapters",
        "accuracy": 0.793,
        "accuracy_sd": 0.012,
        "information_per_damage": 0.651,
        "info_sd": 0.023,
        "transfer": 0.458,
        "transfer_sd": 0.016,
        "saved_percent": 30.6,
        "saved_sd": 2.0,
    },
    {
        "tier": "3",
        "variant": "label-free only (inference configuration)",
        "accuracy": 0.834,
        "accuracy_sd": 0.009,
        "information_per_damage": 0.688,
        "info_sd": 0.021,
        "transfer": 0.487,
        "transfer_sd": 0.014,
        "saved_percent": 34.6,
        "saved_sd": 1.8,
    },
    {
        "tier": "3",
        "variant": "label-free + fluorescence at inference",
        "accuracy": 0.871,
        "accuracy_sd": 0.008,
        "information_per_damage": 0.664,
        "info_sd": 0.022,
        "transfer": 0.512,
        "transfer_sd": 0.013,
        "saved_percent": 27.3,
        "saved_sd": 2.2,
    },
)

# Sec. 3.4, the prose that reads the ablation table.
SECTION_3_4_PROSE: dict[str, Any] = {
    "accuracy loss, constraint removed, printed as": 6.5,
    "information loss per unit damage, constraint removed": 0.157,
    "belief-value function cost": 0.064,
    "uncertainty propagation cost": 0.026,
    "transfer cost of uncertainty propagation": 0.035,
    "accuracy loss, adapters removed, printed as": 4.1,
    "unfilled symbols left in the text": "X and Y",
}

# Sec. 3.5: the regime stratification and the modality cost.
SECTION_3_5: dict[str, float] = {
    "hypoxic advantage points": 13.8,
    "matrix-stiff advantage points": 12.6,
    "normoxic mono-culture advantage points": 3.4,
    "sparse well density advantage points": 7.1,
    "dense well density advantage points": 14.9,
    "accuracy with fluorescence": 0.871,
    "accuracy without fluorescence": 0.834,
    "fluorescence accuracy gain points": 3.7,
    "saved percent without fluorescence": 34.6,
    "saved percent with fluorescence": 27.3,
}

# Sec. 3.6: directed cross-corpus transfer.
SECTION_3_6: dict[str, float] = {
    "within-corpus minimum": 0.704,
    "within-corpus maximum": 0.842,
    "directed mean unconditioned": 0.289,
    "directed mean conditioned": 0.487,
    "mean directed gain": 0.198,
    "weakest directed pair": 0.331,
    "effective sample minimum": 412.0,
    "effective sample maximum": 8940.0,
}

# Sec. 3.7: the closed-loop evaluation.
SECTION_3_7: dict[str, float] = {
    "twin value low": 0.752,
    "twin value low sd": 0.014,
    "twin value high": 0.868,
    "twin value high sd": 0.011,
    "relative error low percent": 6.4,
    "relative error high percent": 17.1,
}

# Sec. 3.8: computational cost.
SECTION_3_8: dict[str, float] = {
    "decision seconds": 1.47,
    "decision seconds sd": 0.08,
    "perception seconds": 0.62,
    "belief update seconds": 0.41,
    "planning seconds": 0.29,
    "actuation inference seconds": 0.15,
    "control window seconds": 3.0,
    "halved planning seconds": 0.17,
    "halved planning cost nats": 0.019,
    "total GPU hours": 284.0,
    "perception adapter GPU hours": 96.0,
    "actuation policy GPU hours": 141.0,
}

# The Discussion's frontier fit and the limitations' fidelity gaps.
DISCUSSION: dict[str, float] = {
    "slope points per transition": 31.5,
    "slope interval low": 27.9,
    "slope interval high": 35.1,
    "r_squared": 0.97,
    "hypoxic transitions per epoch": 0.41,
    "normoxic transitions per epoch": 0.08,
    "twin growth gap percent": 9.4,
    "recorded growth gap percent": 24.8,
    "mean directed increase printed percent": 75.0,
}

# The abstract and the Introduction print the stable stratum's advantage as zero.
STABLE_STRATUM_ADVANTAGE_PRINTED: float = 0.0


def policy_row(table: tuple[dict[str, Any], ...], policy: str) -> dict[str, Any]:
    for row in table:
        if row["policy"] == policy:
            return row
    raise KeyError(f"{policy} is not a row of the table")
