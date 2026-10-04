"""Fit the belief stack and report the state-estimation read-out.

Ref: Sec. 2.2 (the front end), Sec. 2.3 and Algorithm 1 (the belief loop),
Sec. 3.1 (the balanced-accuracy criterion the estimator is judged on).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from saber.cli.context import DEFAULT_RUN, SCALE_ALL, build_context
from saber.evaluation.summary import render_key_values, write_report
from saber.runtime.atomic import save_tensors
from saber.runtime.seeding import set_seed
from saber.study import fit_estimator, run_belief_study, run_perception_study

LOGGER = logging.getLogger("saber.estimate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fit and run the belief stack")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--scale", type=int, default=SCALE_ALL)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("artefacts"))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    context = build_context(args.root, args.run, args.scale)
    set_seed(context.run.seed)
    perception = run_perception_study(context)
    fit = fit_estimator(context, context.specimens(), steps=args.steps, seed=context.run.seed)
    belief = run_belief_study(context, fit)
    destination = args.out if args.out.is_absolute() else context.root / args.out
    destination.mkdir(parents=True, exist_ok=True)
    estimator = context.build_estimator()
    save_tensors(
        destination / f"estimator_{context.run.name}.pt",
        {name: value.detach() for name, value in estimator.state_dict().items()},
    )
    report = destination / f"belief_{context.run.name}.txt"
    write_report(
        report,
        render_key_values(
            {
                "run": context.run.name,
                "specimens": context.cohort.size(),
                **{f"perception {key}": value for key, value in perception.label().items()},
                **{f"fit {key}": value for key, value in fit.label().items()},
                **{f"belief {key}": value for key, value in belief.label().items()},
            }
        ),
    )
    LOGGER.info("wrote %s", report.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
