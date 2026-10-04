"""Build the release's release cohort and write it to a directory.

Ref: Sec. 3.1 -- the evaluation substrate is five corpora at printed specimen
counts; Supplementary Table A1 -- the resource registry that says what those
corpora are. The command writes the cohort the rest of the release reads, plus
the registry transcription, so a reader can see both the substrate and its
provenance in one place.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from saber.cli.context import DEFAULT_RUN, SCALE_ALL, build_context
from saber.cohort.registry import registry_rows, resources_without_licence_position
from saber.evaluation.summary import render_key_values, render_table, write_report
from saber.runtime.atomic import write_json
from saber.runtime.seeding import set_seed

LOGGER = logging.getLogger("saber.build_cohort")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the release cohort")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=Path("cohort"))
    parser.add_argument(
        "--scale",
        type=int,
        default=SCALE_ALL,
        help="specimens per corpus; 0 keeps every printed specimen",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    context = build_context(args.root, args.run, args.scale)
    set_seed(context.run.seed)
    destination = args.out
    if not destination.is_absolute():
        destination = context.root / destination
    from saber.cohort.release_cohort import write_cohort

    _ = write_cohort(context.cohort, destination)
    write_json(destination / "resource_registry.json", {"resources": registry_rows()})
    write_report(
        destination / "resource_registry.txt",
        render_table(
            registry_rows(),
            ["name", "version", "licence", "accessed", "role"],
            title="resource registry",
        )
        + "\n"
        + render_key_values(
            {
                "registry entries": len(registry_rows()),
                "entries without a catalogue licence position": len(
                    resources_without_licence_position()
                ),
                "corpora": len(context.cohort.corpora()),
                "specimens written": context.cohort.size(),
            }
        ),
    )
    LOGGER.info("wrote the cohort to %s", destination.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
