import random

import numpy as np
import torch

from lm.utils import set_seed


def draw_random_numbers():
    """One number from each random generator that set_seed controls."""
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_same_seed_gives_same_numbers():
    set_seed(123)
    first = draw_random_numbers()
    set_seed(123)
    assert draw_random_numbers() == first


def test_different_seeds_give_different_numbers():
    set_seed(1)
    a = draw_random_numbers()
    set_seed(2)
    b = draw_random_numbers()
    assert a != b
