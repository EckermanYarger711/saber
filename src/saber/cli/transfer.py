"""Fit the response head and report the directed cross-corpus transfer.

Ref: Sec. 2.6 (the response head and its unconditioned control), Sec. 3.6 (the
directed transfer table with its retention and effective-sample-size columns).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from saber.cli.context import DEFAULT_RUN, SCALE_ALL, build_context
from saber.evaluation.summary import render_key_values, render_table, write_report
from saber.runtime.seeding import set_seed
from saber.study import fit_estimator, run_belief_study, run_transfer_study

LOGGER = logging.getLogger("saber.transfer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fit the response head and transfer it")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--scale", type=int, default=SCALE_ALL)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("artefacts"))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    context = build_context(args.root, args.run, args.scale)
    set_seed(context.run.seed)
    fit = fit_estimator(context, context.specimens(), steps=12, seed=context.run.seed)
    belief = run_belief_study(context, fit)
    study = run_transfer_study(context, belief, epochs=args.epochs, seed=context.run.seed)
    destination = args.out if args.out.is_absolute() else context.root / args.out
    destination.mkdir(parents=True, exist_ok=True)
    from saber.response.transfer import retention_table

    report = destination / f"transfer_{context.run.name}.txt"
    body = render_table(
        retention_table(study.conditioned),
        ["source", "target", "r_squared", "retention", "neff"],
        title="microenvironment-conditioned head",
    )
    body += "\n" + render_table(
        retention_table(study.unconditioned),
        ["source", "target", "r_squared", "retention", "neff"],
        title="unconditioned head",
    )
    body += "\n" + render_key_values(
        {
            "run": context.run.name,
            "corpora": list(study.corpora),
            "conditioned mean directed": study.conditioned.mean_directed(),
            "unconditioned mean directed": study.unconditioned.mean_directed(),
            "gap": study.mean_directed_gap(),
            "within-corpus range": list(study.conditioned.within_corpus_range()),
        }
    )
    write_report(report, body)
    LOGGER.info("wrote %s", report.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
