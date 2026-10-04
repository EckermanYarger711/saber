"""Result records for the verification families.

A check's status is one of PASS, FAIL, NOT_RUN or BLOCKED, and it is decided from
what actually happened rather than from a narrative. Two families are reported
separately: the ``manuscript`` family recomputes the article's own arithmetic from
its own printed values, and everything else is the release's own evidence. A
failing manuscript check is a discrepancy inside the article, not a defect in the
release, so the two families carry separate statuses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

STATUSES: tuple[str, ...] = ("PASS", "FAIL", "NOT_RUN", "BLOCKED")
MANUSCRIPT_FAMILY = "manuscript"


def jsonable(value: Any) -> Any:
    """Convert a check's values into what a JSON document can carry.

    A numpy scalar, a numpy array or a Path reaches a check's evidence often
    enough that the report writer would otherwise fail at serialisation time,
    which is the worst place to discover it: after the science has run.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    return str(value)


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome and the evidence it was decided from."""

    name: str
    category: str
    status: str
    expected: object
    observed: object
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.category,
            "status": self.status,
            "expected": jsonable(self.expected),
            "observed": jsonable(self.observed),
            "detail": self.detail,
            "evidence": jsonable(self.evidence),
        }


def summarise(results: Sequence[CheckResult]) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    counts["total"] = len(results)
    return counts


def failures(results: Iterable[CheckResult]) -> list[CheckResult]:
    return [result for result in results if result.status == "FAIL"]


def by_category(results: Sequence[CheckResult]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = buckets.setdefault(result.category, dict.fromkeys(STATUSES, 0))
        bucket[result.status] = bucket.get(result.status, 0) + 1
    return [{"family": name, **values} for name, values in sorted(buckets.items())]


def overall_status(results: Sequence[CheckResult]) -> str:
    """The strictest status over every family."""
    statuses = {result.status for result in results}
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "PARTIALLY_VERIFIED"
    if "NOT_RUN" in statuses:
        return "PARTIALLY_VERIFIED"
    return "PASS"


def code_status(results: Sequence[CheckResult]) -> str:
    """The same over the release's own families, with the manuscript family excluded.

    A discrepancy inside the article's printed tables says nothing about whether
    the release executes, so the two verdicts are reported beside each other
    rather than merged.
    """
    own = [result for result in results if result.category != MANUSCRIPT_FAMILY]
    return overall_status(own)


def pass_fail(
    name: str,
    category: str,
    expected: object,
    observed: object,
    detail: str,
    evidence: Mapping[str, Any] | None = None,
) -> CheckResult:
    """A check whose status is equality between the expected and observed values."""
    return CheckResult(
        name=name,
        category=category,
        status="PASS" if expected == observed else "FAIL",
        expected=expected,
        observed=observed,
        detail=detail,
        evidence=dict(evidence or {}),
    )


def close_within(
    name: str,
    category: str,
    expected: float,
    observed: float,
    tolerance: float,
    detail: str,
    evidence: Mapping[str, Any] | None = None,
) -> CheckResult:
    """A check on a real-valued comparison with a stated tolerance."""
    return CheckResult(
        name=name,
        category=category,
        status="PASS" if abs(expected - observed) <= tolerance else "FAIL",
        expected=expected,
        observed=observed,
        detail=detail,
        evidence={**dict(evidence or {}), "tolerance": tolerance},
    )


def rounded_match(
    name: str,
    category: str,
    computed: float,
    printed: float,
    digits: int,
    detail: str,
    evidence: dict[str, Any] | None = None,
) -> CheckResult:
    """A check that a printed effect equals a computed difference at its own precision.

    Comparing at the precision the table prints is what makes these checks stable:
    a value the article prints to three decimals is only claimed to three decimals.
    """
    rounded = round(computed, digits)
    return CheckResult(
        name=name,
        category=category,
        status="PASS" if abs(rounded - printed) <= 0.0 else "FAIL",
        expected=printed,
        observed=rounded,
        detail=detail,
        evidence={**dict(evidence or {}), "precise_difference": computed, "digits": digits},
    )


def not_run(
    name: str, category: str, reason: str, evidence: dict[str, Any] | None = None
) -> CheckResult:
    """A check that could not be decided, with the reason it could not."""
    return CheckResult(
        name=name,
        category=category,
        status="NOT_RUN",
        expected="a decision",
        observed=None,
        detail=reason,
        evidence=dict(evidence or {}),
    )


def blocked(
    name: str, category: str, reason: str, evidence: dict[str, Any] | None = None
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        status="BLOCKED",
        expected="a decision",
        observed=None,
        detail=reason,
        evidence=dict(evidence or {}),
    )
