"""A single seeding entry point.

Every stochastic component in the release is seeded through here, because the
shipped artefacts have to be a fixed point: a check that builds a module reports
a different number on every run otherwise.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the interpreter, NumPy and torch.

    The torch seed covers CPU and, when a device is present, CUDA; the
    deterministic algorithms switch is left off because no operation in this
    release needs it and enabling it costs occupancy on the reduction kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_rng(seed: int) -> np.random.Generator:
    """An independent generator, so a module can draw without touching global state."""
    return np.random.default_rng(seed)
