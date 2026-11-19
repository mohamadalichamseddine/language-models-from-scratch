"""Small helpers shared across the project."""

import random
from pathlib import Path

import numpy as np
import torch

from lm.data import ROOT


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly.

    Call this once at the start of every script. torch.manual_seed also
    seeds the GPU, if there is one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def repo_path(path):
    """Return a path relative to the repo root, with forward slashes.

    Saved records use it so they read the same on every machine. A path
    outside the repo is returned as an absolute path.
    """
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return str(path)
