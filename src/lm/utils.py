"""Small helpers shared across the project."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly.

    Call this once at the start of every script. torch.manual_seed also
    seeds the GPU, if there is one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
