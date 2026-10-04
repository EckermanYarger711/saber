"""Run the twin and report its fidelity and the off-policy evaluation.

Ref: Sec. 2.7 (the twin's two roles), Sec. 3.7 (the closed-loop policy value
estimated retrospectively), Sec. 4.1 (the twin's growth fidelity is evaluated
rather than assumed).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from saber.cli.context import DEFAULT_RUN, SCALE_ALL, build_context
from saber.evaluation.summary import render_key_values, write_report
from saber.runtime.seeding import set_seed
from saber.study import fit_value_function, run_twin_study

LOGGER = logging.getLogger("saber.twin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the twin and its evaluation")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--scale", type=int, default=SCALE_ALL)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    context = build_context(args.root, args.run, args.scale)
    set_seed(context.run.seed)
    study = run_twin_study(context)
    value = fit_value_function(context, study)
    destination = args.out if args.out.is_absolute() else context.root / args.out
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / f"twin_{context.run.name}.txt"
    write_report(
        report,
        render_key_values(
            {
                "run": context.run.name,
                "session": context.schedule.label(),
                **{f"fidelity {key}": value_ for key, value_ in study.fidelity.label().items()},
                **{f"off-policy {key}": value_ for key, value_ in study.suite.label().items()},
                **{f"value fit {key}": value_ for key, value_ in value.label().items()},
            }
        ),
    )
    LOGGER.info("wrote %s", report.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
