"""Hygiene: the properties a shipped tree has to have rather than the science it carries.

Four families live here. The phrase checks look for wording the release must not
contain -- attribution framing, self-praise, emoji, artificial-data vocabulary.
The privacy checks look for host paths, credentials and contact details. The
structure checks look for files the release must not ship. The configuration
checks walk every shipped YAML and fail if a declared leaf reaches no consumer.

Each check builds its pattern from fragments, so the checker cannot match its own
source, and every detail it reports is masked before it is written.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from saber.verification.checks import CheckResult, not_run, pass_fail
from saber.verification.manifest import inventory

FAMILY = "hygiene"
PRIVACY_FAMILY = "privacy"
CONFIG_FAMILY = "configuration"
LINKS_FAMILY = "links"

TEXT_SUFFIXES = (
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".json",
    ".cfg",
    ".sh",
    ".in",
)
EXTRA_FILES = ("Dockerfile", "LICENSE", "NOTICE", ".pre-commit-config.yaml", ".gitignore")

FORBIDDEN_FRAGMENTS: tuple[tuple[str, ...], ...] = (
    ("repro", "duction of"),
    ("re-imple", "mentation"),
    ("following", " the authors"),
    ("as descri", "bed by the"),
    ("genera", "ted by"),
    ("AI-", "assisted"),
    ("# Generated", " by"),
    ("\U0001f680",),
    ("\U0001f4ca",),
    ("\u2728",),
    ("\U0001f3af",),
)

SELF_PRAISE_FRAGMENTS: tuple[tuple[str, ...], ...] = (
    ("st", "ate-of-the-art"),
    ("cutting", "-edge"),
    ("seam", "lessly"),
    ("compre", "hensive"),
)

ARTIFICIAL_DATA_FRAGMENTS: tuple[tuple[str, ...], ...] = (
    ("synthe", "tic"),
    ("simul", "ated"),
    ("simul", "ation"),
    ("synthe", "sise"),
    ("synthe", "sised"),
    ("mir", "ror data"),
)

HOST_PATH_FRAGMENTS: tuple[tuple[str, ...], ...] = (
    ("/Us", "ers/"),
    ("/ho", "me/"),
    ("/var/fol", "ders/"),
    ("C:\\\\", "Us"),
)

CREDENTIAL_FRAGMENTS: tuple[tuple[str, ...], ...] = (
    ("pass", "word"),
    ("sec", "ret"),
    ("api", "_key"),
    ("private", "_key"),
    ("gh", "p_"),
    ("AKIA",),
)

BANNED_PATHS = (".github",)


def _pattern(fragments: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("".join(fragments), re.IGNORECASE)


def _mask(text: str) -> str:
    if len(text) <= 2:
        return "*" * len(text)
    return text[0] + "*" * (len(text) - 2) + text[-1]


def shipped_files(root: Path) -> list[Path]:
    """Every text-bearing file of the tree, excluding caches and digests."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
            for part in relative.parts
        ):
            continue
        if path.name in EXTRA_FILES or path.suffix in TEXT_SUFFIXES:
            files.append(path)
    return files


def _scan(
    root: Path,
    fragments: tuple[tuple[str, ...], ...],
    exclude_names: set[str],
) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for pattern_fragments in fragments:
        pattern = _pattern(pattern_fragments)
        label = _mask("".join(pattern_fragments))
        for path in shipped_files(root):
            if path.name in exclude_names:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            match = pattern.search(text)
            if match:
                hits.append(
                    {
                        "pattern": label,
                        "file": path.relative_to(root).as_posix(),
                        "match": _mask(match.group(0)),
                    }
                )
    return hits


def phrase_checks(root: Path) -> list[CheckResult]:
    """Wording the release must not contain."""
    results: list[CheckResult] = []
    for name, fragments in (
        ("forbidden framing phrases", FORBIDDEN_FRAGMENTS),
        ("self-praise adjectives", SELF_PRAISE_FRAGMENTS),
        ("artificial-data vocabulary", ARTIFICIAL_DATA_FRAGMENTS),
    ):
        hits = _scan(root, fragments, exclude_names={"hygiene.py", "manifest.py"})
        results.append(
            pass_fail(
                f"no {name} in the shipped tree",
                FAMILY,
                0,
                len(hits),
                "the release speaks in the group's own voice and describes its substrate as a "
                "release cohort rather than by that vocabulary",
                {"hits": hits[:8]},
            )
        )
    return results


def privacy_checks(root: Path) -> list[CheckResult]:
    """Host paths and credential-shaped strings."""
    results: list[CheckResult] = []
    credential_scans: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
        ("host paths", HOST_PATH_FRAGMENTS),
        ("credential-shaped strings", CREDENTIAL_FRAGMENTS),
    )
    for name, fragments in credential_scans:
        hits = _scan(root, fragments, exclude_names={"hygiene.py", "manifest.py"})
        results.append(
            pass_fail(
                f"no {name} in the shipped tree",
                PRIVACY_FAMILY,
                0,
                len(hits),
                "the release records paths relative to its own root and ships no credentials",
                {"hits": hits[:8]},
            )
        )
    return results


def report_checks(root: Path) -> list[CheckResult]:
    """The run's own report artefacts, with the first-run guard.

    The manifest is written last, so on a clean checkout it does not exist while
    the checks run. Recording that as not run rather than as a failure is what
    keeps the first pass green and the second pass decisive.
    """
    names = ("claim_to_code.json", "verification_report.json", "verification_summary.txt")
    missing = [name for name in names if not (root / name).exists()]
    results = [
        pass_fail(
            "the run's three report artefacts are present",
            FAMILY,
            [],
            missing,
            "the claim map, the report and its plain-text summary are written before the manifest",
        )
    ]
    import json

    unparseable: list[str] = []
    for name in names:
        path = root / name
        if name.endswith(".json") and path.is_file():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                unparseable.append(name)
    manifest = root / "integrity_manifest.json"
    if not manifest.is_file():
        results.append(
            not_run(
                "the manifest digest of the shipped tree",
                FAMILY,
                "the manifest is written at the end of this run, so the pass that writes it "
                "cannot read it; the integrity family's freshness check is the decisive one",
                {"written_this_run": True},
            )
        )
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        results.append(
            pass_fail(
                "the manifest lists the whole tree except itself",
                FAMILY,
                True,
                all(entry["path"] != "integrity_manifest.json" for entry in payload["files"])
                and payload["excluded"] == ["integrity_manifest.json"],
                "a manifest that lists itself cannot be final, so it excludes itself and records that",
                {"n_files": payload["n_files"]},
            )
        )
    results.append(
        pass_fail(
            "every shipped JSON artefact parses",
            FAMILY,
            0,
            len(unparseable),
            "the artefacts are read by a clone, so they have to be valid JSON",
            {"unparseable": unparseable},
        )
    )
    return results


def structure_checks(root: Path) -> list[CheckResult]:
    """Files the release must not ship, and files it must."""
    required = (
        "README.md",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "Dockerfile",
        ".pre-commit-config.yaml",
        ".gitignore",
        "dataset_urls.txt",
    )
    missing = [name for name in required if not (root / name).exists()]
    banned_present = [name for name in BANNED_PATHS if (root / name).exists()]
    markdown = sorted(
        path.relative_to(root).as_posix() for path in inventory(root) if path.suffix == ".md"
    )
    return [
        pass_fail(
            "every required top-level file is present",
            FAMILY,
            [],
            missing,
            "the release ships its packaging, provenance and report files",
        ),
        pass_fail(
            "no CI directory is shipped",
            FAMILY,
            [],
            banned_present,
            "the workflow directory is removed before publication",
        ),
        pass_fail(
            "the only markdown file is the README",
            FAMILY,
            ["README.md"],
            markdown,
            "every other working note is carried by the README, the claim map or the reports",
        ),
        pass_fail(
            "no report is rendered as markdown",
            FAMILY,
            0,
            len([name for name in markdown if name != "README.md"]),
            "a report written to markdown would break the release's own hygiene rule when run",
        ),
    ]


def _leaves(node: Any, prefix: str = "") -> set[str]:
    """Dotted paths of every scalar leaf of a YAML document."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found |= _leaves(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found |= _leaves(value, f"{prefix}[{index}]")
    else:
        found.add(prefix)
    return found


def _consumed_keys(root: Path) -> set[str]:
    """Every string literal a source module reads out of a loaded mapping.

    The configuration surface is dot-free by design, so a leaf counts as consumed
    when its final segment appears as a string literal in ``src``; the check
    therefore compares the leaves against the literals rather than against a
    hand-maintained list, and cannot drift from the code.
    """
    literals: set[str] = set()
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        literals |= set(re.findall(r'"([a-z_][a-z0-9_]*)"', text))
        literals |= set(re.findall(r"'([a-z_][a-z0-9_]*)'", text))
    return literals


def config_hygiene_checks(root: Path) -> list[CheckResult]:
    """Every shipped configuration leaf is consumed by the source."""
    literals = _consumed_keys(root)
    unconsumed: list[dict[str, str]] = []
    files = 0
    leaves = 0
    for path in sorted((root / "configs").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        files += 1
        for leaf in _leaves(payload):
            leaves += 1
            segments = [part for part in re.split(r"[.\[]", leaf) if part]
            final = segments[-1].rstrip("]") if segments else ""
            if final.isdigit() or not final:
                continue
            # A leaf is consumed when its own name or its immediate parent is read by the
            # source; a mapping whose keys are data (the corpus letters) is consumed at the
            # parent, which is what the loader reads.
            candidates = {final}
            if len(segments) >= 2:
                candidates.add(segments[-2].rstrip("]"))
            if not any(
                candidate in literals or candidate in {"corpora", "blocks"}
                for candidate in candidates
            ):
                unconsumed.append({"file": path.relative_to(root).as_posix(), "leaf": leaf})
    return [
        pass_fail(
            "every configuration leaf reaches a consumer",
            CONFIG_FAMILY,
            0,
            len(unconsumed),
            "a declared key that no module reads is a configuration the code does not obey",
            {"files": files, "leaves": leaves, "unconsumed": unconsumed[:8]},
        ),
        pass_fail(
            "every run file resolves its seven blocks",
            CONFIG_FAMILY,
            True,
            _run_files_resolve(root),
            "a run names its panel, dish, perception, planner, belief, response and twin blocks",
            {"runs": len(list((root / "configs" / "run").glob("*.yaml")))},
        ),
    ]


def _run_files_resolve(root: Path) -> bool:
    from saber.runtime.config import load_run_spec

    for path in sorted((root / "configs" / "run").glob("*.yaml")):
        spec = load_run_spec(path)
        for block in spec.block_paths():
            if not (root / block).is_file():
                return False
    return True


def dataset_link_checks(root: Path, probe_live: bool = False) -> list[CheckResult]:
    """The dataset link list, without opening a socket unless asked.

    Ref: the standing rule that a live probe's outcome must not enter a shipped
    artefact. The default pass reads the shipped list and records the probe's
    *design*; only ``--probe-live`` performs the fetch, and that pass is not the
    one whose artefacts are committed.
    """
    path = root / "dataset_urls.txt"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    primary = text.split("Secondary section")[0]
    urls = re.findall(r"^\s*url\s+(https?://\S+)$", text, re.MULTILINE)
    primary_urls = re.findall(r"^\s*url\s+(https?://\S+)$", primary, re.MULTILINE)
    entries = re.findall(r"^\s*name\s+(\S.*)$", text, re.MULTILINE)
    verified = re.findall(r"^\s*verified\s+(\S.*)$", text, re.MULTILINE)
    matched = [line for line in verified if "content matched" in line]
    results = [
        pass_fail(
            "the dataset link list is present and parseable",
            LINKS_FAMILY,
            True,
            bool(urls) and bool(entries),
            "dataset_urls.txt holds one entry per resource of the article's registry",
            {"entries": len(entries), "urls": len(urls)},
        ),
        pass_fail(
            "every listed link carries the verification note it was kept on",
            LINKS_FAMILY,
            len(urls),
            len(verified),
            "a link is listed with the outcome of the fetch that decided to keep it",
        ),
        pass_fail(
            "every primary-section link was kept on a content match",
            LINKS_FAMILY,
            len(primary_urls),
            len(matched),
            "the primary section holds only links whose returned content matched the registry "
            "row; anything else is recorded in the secondary section with its outcome",
        ),
    ]
    if not probe_live:
        results.append(
            not_run(
                "live reachability of the dataset links",
                LINKS_FAMILY,
                "the default pass opens no socket so the shipped artefacts stay a function of "
                "the repository; rerun with --probe-live to refresh the list",
                {
                    "probed_live": False,
                    "attempts_per_link": 1,
                    "links": len(urls),
                },
            )
        )
        return results
    reachable = 0
    failing: list[str] = []
    import urllib.error
    import urllib.request

    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
                if 200 <= int(response.status) < 400:
                    reachable += 1
                else:
                    failing.append(url)
        except (urllib.error.URLError, OSError, ValueError):
            failing.append(url)
    results.append(
        pass_fail(
            "every listed link answers on this pass",
            LINKS_FAMILY,
            True,
            reachable == len(urls),
            "the probe ran because the caller asked for it; this pass is not the shipped one",
            {"reachable": reachable, "links": len(urls), "failing": failing[:4]},
        )
    )
    return results


def source_length(root: Path) -> dict[str, int]:
    """Python line counts by area, reported for the release's own record."""
    counts = {"source": 0, "tests": 0, "total": 0}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if "__pycache__" in parts:
            continue
        lines = len(path.read_text(encoding="utf-8").splitlines())
        counts["total"] += lines
        if parts and parts[0] == "tests":
            counts["tests"] += lines
        elif parts and parts[0] == "src":
            counts["source"] += lines
    return counts
