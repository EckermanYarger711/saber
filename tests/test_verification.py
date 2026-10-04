"""The verification layer's own artefacts and the manuscript family's outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from saber.verification import manuscript
from saber.verification.checks import code_status, overall_status, summarise
from saber.verification.claims import DEVIATIONS, resolve_anchors
from saber.verification.manifest import MANIFEST_NAME, verify_manifest
from saber.verification.manuscript import all_checks, table_not_deposited_checks


def test_manuscript_family_reports_both_verdicts() -> None:
    results = all_checks()
    counts = summarise(results)
    assert counts["total"] == len(results)
    assert counts["PASS"] > 0
    assert overall_status(results) in {"PASS", "FAIL", "PARTIALLY_VERIFIED"}


def test_manuscript_discrepancies_are_reported_not_softened() -> None:
    results = all_checks()
    failing = [result.name for result in results if result.status == "FAIL"]
    assert failing
    assert code_status(results) in {"PASS", "PARTIALLY_VERIFIED", "FAIL"}


def test_table_not_deposited_checks_are_not_run() -> None:
    for result in table_not_deposited_checks():
        assert result.status == "NOT_RUN"
        assert result.detail


def test_frontier_findings_carry_their_arithmetic() -> None:
    check = next(
        result
        for result in manuscript.all_checks()
        if result.name.startswith("sec 4: an ordering-consistent")
    )
    assert "implied_mean_squared_error" in check.evidence
    assert "smallest_achievable_mean_squared_error" in check.evidence


def test_synergy_finding_names_the_columns() -> None:
    check = next(
        result for result in manuscript.all_checks() if result.name.startswith("tier 2 synergy")
    )
    assert check.evidence["joint_loss"] < check.evidence["individual_loss_sum"]


def test_every_claim_anchor_resolves(root: Path) -> None:
    for deviation in DEVIATIONS:
        assert deviation["paper_location"]
        assert deviation["reason"]
    from saber.verification.claims import CLAIMS

    unresolved = [
        entry
        for claim in CLAIMS
        for entry in resolve_anchors(root, claim["code_anchors"])
        if not entry["resolved"]
    ]
    assert unresolved == []


def test_shipped_artefacts_are_present_and_consistent(root: Path) -> None:
    claim_map = json.loads((root / "claim_to_code.json").read_text(encoding="utf-8"))
    report = json.loads((root / "verification_report.json").read_text(encoding="utf-8"))
    assert claim_map["n_claims"] == len(claim_map["claims"])
    assert claim_map["unmatched_check_names"] == []
    assert report["summary"]["total"] == len(report["checks"])
    assert report["integrity_manifest"]["written_last"]


def test_shipped_report_is_the_complete_one(root: Path) -> None:
    report = json.loads((root / "verification_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["total"] > 100


def test_manifest_describes_the_live_tree(root: Path) -> None:
    assert (root / MANIFEST_NAME).is_file()
    verification = verify_manifest(root)
    assert verification["intact"], verification


def test_manifest_skips_its_own_digest(root: Path) -> None:
    payload = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    listed = {entry["path"] for entry in payload["files"]}
    assert MANIFEST_NAME not in listed
    assert "verification_report.json" in listed
    assert "verification_summary.txt" in listed
