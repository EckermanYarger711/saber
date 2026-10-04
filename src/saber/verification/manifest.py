"""The integrity manifest: one digest per shipped file, and a digest of itself.

The manifest is always the last artefact written and never lists itself, so the
report and the manifest cannot both be final if the report embedded manifest
digests. ``verify_manifest`` compares the shipped manifest against the live tree
in both directions, which is what catches a file that was shipped but not listed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from saber.runtime.atomic import file_digest, read_json, write_json
from saber.runtime.config import RELEASE_SLUG

MANIFEST_NAME = "integrity_manifest.json"
SKIP_DIRECTORY_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "build",
    "dist",
    ".venv",
    "node_modules",
}
SKIP_TOP_LEVEL = {"runs", "artefacts", "cohort", "checkpoints", "data"}


def inventory(root: Path, exclude: Path | None = None) -> list[Path]:
    """Every shipped file, in a stable walk order.

    A skipped directory is skipped wherever it appears, but a data or artefact
    root is skipped only at the top level of the tree: a release has both a root
    ``data/`` and a package ``data/`` module, and skipping the second would drop
    a source file from the manifest while leaving the release looking complete.
    """
    excluded = exclude.resolve() if exclude is not None else None
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRECTORY_NAMES for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in SKIP_TOP_LEVEL:
            continue
        if excluded is not None and path.resolve() == excluded:
            continue
        files.append(path)
    return files


def manifest_digest(entries: list[dict[str, str]]) -> str:
    """Digest the inventory itself, so the manifest is checkable without re-hashing."""
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}:{entry['sha256']}\n".encode())
    return digest.hexdigest()


def build_manifest(root: Path, target: Path | None = None) -> dict[str, Any]:
    destination = (target or (root / MANIFEST_NAME)).resolve()
    entries = [
        {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
        for path in inventory(root, exclude=destination)
    ]
    return {
        "algorithm": "sha256",
        "root": RELEASE_SLUG,
        "excluded": [MANIFEST_NAME],
        "skipped_directories": sorted(SKIP_DIRECTORY_NAMES),
        "skipped_top_level": sorted(SKIP_TOP_LEVEL),
        "n_files": len(entries),
        "files": entries,
        "manifest_digest": manifest_digest(entries),
    }


def write_integrity_manifest(root: Path, target: Path | None = None) -> Path:
    destination = target or (root / MANIFEST_NAME)
    write_json(destination, build_manifest(root, target=destination))
    return destination


def verify_manifest(root: Path, target: Path | None = None) -> dict[str, Any]:
    """Compare the shipped manifest with the live tree, in both directions."""
    destination = target or (root / MANIFEST_NAME)
    if not destination.is_file():
        return {
            "n_listed": 0,
            "n_on_disk": len(inventory(root)),
            "stale": [],
            "unlisted": [],
            "digest_matches": False,
            "intact": False,
            "written_this_run": True,
        }
    payload = read_json(destination)
    listed = {entry["path"]: entry["sha256"] for entry in payload["files"]}
    on_disk = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in inventory(root, exclude=destination.resolve())
    }
    stale = sorted(name for name, digest in listed.items() if on_disk.get(name) != digest)
    unlisted = sorted(name for name in on_disk if name not in listed)
    recomputed = manifest_digest(payload["files"])
    return {
        "n_listed": len(listed),
        "n_on_disk": len(on_disk),
        "stale": stale,
        "unlisted": unlisted,
        "digest_matches": recomputed == payload["manifest_digest"],
        "intact": not stale and not unlisted and recomputed == payload["manifest_digest"],
    }


def freshness_check(root: Path) -> dict[str, Any]:
    """The manifest-freshness check's evidence, without recording either digest.

    Both digests are properties of the tree that contains the report being written,
    so recording them would make every run differ from the last and no fixed point
    would exist. The evidence describes the comparison and its outcome; the digest
    lives in the manifest, which is the artefact that carries it.
    """
    verification = verify_manifest(root)
    return {
        "files": verification["n_listed"],
        "on_disk": verification["n_on_disk"],
        "stale": verification["stale"],
        "unlisted": verification["unlisted"],
        "changed": len(verification["stale"]) + len(verification["unlisted"]),
        "changed_count": len(verification["stale"]) + len(verification["unlisted"]),
        "digest_matches": verification["digest_matches"],
        "first_run": not (root / MANIFEST_NAME).is_file(),
    }
