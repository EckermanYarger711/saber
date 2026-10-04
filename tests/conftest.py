"""Shared state for the suite.

The context is built once at the suite's own scale, so a test never instantiate the
article's full 4180-specimen cohort and the suite stays minutes rather than hours.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from saber.cli.context import ReleaseContext, build_context
from saber.runtime.config import repo_root

SUITE_SCALE = 1


@pytest.fixture(scope="session")
def root() -> Path:
    return repo_root(Path(__file__).resolve().parents[1] / "pyproject.toml")


@pytest.fixture(scope="session")
def context(root: Path) -> ReleaseContext:
    return build_context(root, "configs/run/primary.yaml", SUITE_SCALE)


@pytest.fixture(scope="session")
def rng() -> Iterator[object]:
    import numpy as np

    yield np.random.default_rng(20260929)


@pytest.fixture(scope="session")
def acquisition(context: ReleaseContext) -> tuple[object, ...]:
    """The acquisition comparison, run once for the suite."""
    from saber.study import run_acquisition_study

    return run_acquisition_study(context)


@pytest.fixture(scope="session")
def belief_study(context: ReleaseContext) -> object:
    from saber.study import run_belief_study

    return run_belief_study(context)


@pytest.fixture(scope="session")
def twin_study(context: ReleaseContext) -> object:
    from saber.study import run_twin_study

    return run_twin_study(context)
