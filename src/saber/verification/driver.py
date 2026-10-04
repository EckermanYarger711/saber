"""The verification driver: the two families, the gates, and the shipped artefacts.

Order matters and is fixed here. The pre-pass refreshes the inputs a check
consumes -- the dataset link list, the release cohort -- and then the checks run.
The gates come after the checks so a tool failure is recorded beside the science.
The artefacts are written report first, then the plain-text summary, then the
manifest last, because the manifest digests the tree that contains the other two
and anything written after it would leave a stale entry.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saber.cli.context import ReleaseContext, build_context
from saber.evaluation.summary import render_key_values, render_table, write_report
from saber.runtime.atomic import write_json
from saber.runtime.config import RELEASE_SLUG, repo_root
from saber.verification import execution as execution_family, manuscript as manuscript_family
from saber.verification.checks import (
    CheckResult,
    by_category,
    code_status,
    failures,
    overall_status,
    summarise,
)
from saber.verification.claims import build_claim_map
from saber.verification.hygiene import (
    config_hygiene_checks,
    dataset_link_checks,
    phrase_checks,
    privacy_checks,
    report_checks,
    source_length,
    structure_checks,
)
from saber.verification.manifest import MANIFEST_NAME, freshness_check, write_integrity_manifest

LOGGER = logging.getLogger("saber.verify")

CLAIM_FILE = "claim_to_code.json"
REPORT_FILE = "verification_report.json"
SUMMARY_FILE = "verification_summary.txt"
DEFAULT_RUN = "configs/run/primary.yaml"
VERIFICATION_SCALE = 1

GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff", (sys.executable, "-m", "ruff", "check", ".")),
    ("black", (sys.executable, "-m", "black", "--check", ".")),
    ("isort", (sys.executable, "-m", "isort", "--check-only", ".")),
    ("mypy", (sys.executable, "-m", "mypy", "--strict", "src/saber")),
    ("pytest", (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider")),
)

GATE_ENVIRONMENT = {
    "COLUMNS": "120",
    "LINES": "40",
    "NO_COLOR": "1",
    "PY_COLORS": "0",
    "PYTHONHASHSEED": "0",
}


@dataclass(frozen=True)
class DriverOptions:
    root: Path
    run: str
    scale: int
    probe_links: bool
    skip_gates: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the release's verification driver")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--scale", type=int, default=VERIFICATION_SCALE)
    parser.add_argument(
        "--probe-live",
        action="store_true",
        help="fetch the dataset links; the default pass opens no socket",
    )
    parser.add_argument("--skip-gates", action="store_true")
    return parser


def run_gates(root: Path) -> list[CheckResult]:
    """Each tool's exit code decides its gate; no output text is recorded."""
    results: list[CheckResult] = []
    environment = {**os.environ, **GATE_ENVIRONMENT}
    for label, command in GATES:
        try:
            completed = subprocess.run(
                list(command),
                cwd=root,
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            captured = completed.returncode
        except OSError as error:
            results.append(
                CheckResult(
                    name=f"gate {label}",
                    category="gate",
                    status="BLOCKED",
                    expected="the tool to run",
                    observed=type(error).__name__,
                    detail="the tool is not available in this environment",
                    evidence={"label": label},
                )
            )
            continue
        results.append(
            CheckResult(
                name=f"gate {label}",
                category="gate",
                status="PASS" if captured == 0 else "FAIL",
                expected=0,
                observed=captured,
                detail="the tool's exit code decides the gate",
                evidence={"label": label},
            )
        )
    return results


def gather(options: DriverOptions) -> tuple[list[CheckResult], dict[str, Any], ReleaseContext]:
    root = options.root
    context = build_context(root, options.run, options.scale)
    results: list[CheckResult] = []
    results.extend(manuscript_family.all_checks())
    results.extend(manuscript_family.table_not_deposited_checks())
    results.extend(execution_family.all_checks(context))
    results.extend(phrase_checks(root))
    results.extend(privacy_checks(root))
    results.extend(structure_checks(root))
    results.extend(report_checks(root))
    results.extend(config_hygiene_checks(root))
    results.extend(dataset_link_checks(root, probe_live=options.probe_links))
    results.append(
        CheckResult(
            name="manifest freshness",
            category="integrity",
            status="PASS" if freshness_check(root)["changed_count"] == 0 else "FAIL",
            expected="the shipped manifest describes the live tree",
            observed=freshness_check(root)["changed_count"],
            detail="compared in both directions; the digests themselves live in the manifest",
            evidence=freshness_check(root),
        )
    )
    if options.skip_gates:
        results.append(
            CheckResult(
                name="gate family",
                category="gate",
                status="NOT_RUN",
                expected="the tool gates to run",
                observed=None,
                detail="the gates were skipped for this invocation and are not shipped evidence",
                evidence={"label": "all"},
            )
        )
    else:
        results.extend(run_gates(root))
    claim_map = build_claim_map(root, results)
    results.extend(
        CheckResult(
            name=f"claim {entry['id']} maps to executed code",
            category="claim",
            status="PASS" if entry["verification"] in {"PASS", "PARTIALLY_VERIFIED"} else "FAIL",
            expected="PASS",
            observed=entry["verification"],
            detail=entry["statement"][:160],
            evidence={
                "anchors": entry["code_anchors"],
                "unresolved": [
                    anchor["anchor"] for anchor in entry["anchors"] if not anchor["resolved"]
                ],
            },
        )
        for entry in claim_map["claims"]
    )
    claim_map = build_claim_map(root, results)
    return results, claim_map, context


def render_summary(
    results: Sequence[CheckResult],
    claim_map: dict[str, Any],
    context: ReleaseContext,
) -> str:
    counts = summarise(results)
    discrepancies = [result for result in failures(results) if result.category == "manuscript"]
    failing = [
        {
            "check": result.name,
            "family": result.category,
            "expected": result.expected,
            "observed": result.observed,
        }
        for result in failures(results)
    ]
    return "\n".join(
        [
            "release root           : " + RELEASE_SLUG,
            f"run                    : {context.run.name} ({context.run.kind})",
            f"scale (per corpus)     : {context.scale}",
            "",
            "checks by family",
            render_table(by_category(results), ["family", "PASS", "FAIL", "NOT_RUN", "BLOCKED"]),
            render_key_values(
                {
                    "overall status": overall_status(results),
                    "code status (manuscript family excluded)": code_status(results),
                    "checks run": counts["total"],
                    "manuscript discrepancies": len(discrepancies),
                    "claims mapped": claim_map["n_claims"],
                    "deviations registered": claim_map["n_deviations"],
                    "source lines": context_payload_lines(context),
                }
            ),
            (
                render_table(failing, ["check", "family", "expected", "observed"], title="failures")
                if failing
                else "no failures"
            ),
            "",
        ]
    )


def context_payload_lines(context: ReleaseContext) -> int:
    from saber.verification.hygiene import source_length

    return int(source_length(context.root)["total"])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    root = repo_root(args.root) if args.root is not None else repo_root()
    options = DriverOptions(
        root=root,
        run=args.run,
        scale=args.scale,
        probe_links=args.probe_live,
        skip_gates=args.skip_gates,
    )
    results, claim_map, context = gather(options)
    counts = summarise(results)
    report = {
        "release_root": RELEASE_SLUG,
        "run": context.run.label(),
        "cohort": context.cohort.summary(),
        "budget": context.budget.label(),
        "session": context.schedule.label(),
        "action_set": context.action_set.label(),
        "scale": context.scale,
        "source_lines": source_length(root),
        "summary": {
            **counts,
            "overall_status": overall_status(results),
            "code_status": code_status(results),
            "manuscript_discrepancies": sum(
                1 for result in failures(results) if result.category == "manuscript"
            ),
        },
        "interpretation": (
            "A failing check in the manuscript family is a discrepancy inside the article's own "
            "printed tables, not a defect in this release. Every execution-family quantity is the "
            "release's own on its release cohort, at the scale recorded above; the article's "
            "evaluation is a retrospective secondary analysis of trajectories it does not deposit."
        ),
        "manuscript_discrepancies": [
            result.as_record() for result in failures(results) if result.category == "manuscript"
        ],
        "checks": [result.as_record() for result in results],
        "integrity_manifest": {"name": MANIFEST_NAME, "written_last": True},
    }
    claim_payload = {
        "release_root": RELEASE_SLUG,
        "run": context.run.label(),
        "n_claims": claim_map["n_claims"],
        "n_deviations": claim_map["n_deviations"],
        "claims": claim_map["claims"],
        "deviations": claim_map["deviations"],
        "unmatched_check_names": claim_map["unmatched_check_names"],
    }
    summary_text = render_summary(results, claim_map, context)
    write_json(root / CLAIM_FILE, claim_payload)
    write_json(root / REPORT_FILE, report)
    write_report(root / SUMMARY_FILE, summary_text)
    manifest_path = write_integrity_manifest(root)
    verification = freshness_check(root)
    LOGGER.info(
        "wrote %s, %s, %s and %s (manifest clean: %s)",
        CLAIM_FILE,
        REPORT_FILE,
        SUMMARY_FILE,
        manifest_path.name,
        verification["changed_count"] == 0,
    )
    if verification["changed_count"] != 0:
        LOGGER.error("the manifest does not describe the live tree: %s", verification)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
