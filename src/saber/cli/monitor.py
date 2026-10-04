"""Run the acquisition comparison and report it.

Ref: Table 1 (both panels), Table 2 (the ablation tiers) and Sec. 3.5 (the
regime stratification). One command per run selection, so an ablation is a
different ``--run`` rather than a different code path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from saber.cli.context import DEFAULT_RUN, SCALE_ALL, build_context
from saber.evaluation.summary import render_key_values, render_table, write_report
from saber.runtime.seeding import set_seed
from saber.study import COMPARED_POLICIES, run_acquisition_study, run_frontier_study

LOGGER = logging.getLogger("saber.monitor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run one acquisition configuration")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--scale", type=int, default=SCALE_ALL)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument(
        "--policies",
        default="",
        help="comma-separated subset of the compared policies; empty runs all of them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    context = build_context(args.root, args.run, args.scale)
    set_seed(context.run.seed)
    selected = (
        tuple(name.strip() for name in args.policies.split(",") if name.strip())
        if args.policies
        else COMPARED_POLICIES
    )
    outcomes = run_acquisition_study(context, selected)
    frontier = run_frontier_study(outcomes, context)
    destination = args.out if args.out.is_absolute() else context.root / args.out
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / f"acquisition_{context.run.name}.txt"
    write_report(
        report,
        render_table(
            [outcome.label() for outcome in outcomes],
            [
                "policy",
                "information",
                "damage",
                "currency",
                "exposures",
                "budget_usage_percent",
                "balanced_accuracy",
                "saved_percent",
            ],
            title=f"acquisition comparison, run {context.run.name}",
        )
        + "\n"
        + render_table(
            [entry.label() for entry in frontier.strata],
            ["stratum", "specimens", "transition_rate", "advantage_points"],
            title="regime stratification",
        )
        + "\n"
        + render_key_values(
            {
                "run": context.run.name,
                "specimens": context.cohort.size(),
                "scale (specimens per corpus)": context.scale,
                "budget": context.budget.total,
                "frontier slope": frontier.slope,
                "frontier r_squared": frontier.r_squared,
                "frontier interval": list(frontier.confidence_interval),
            }
        ),
    )
    LOGGER.info("wrote %s", report.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
