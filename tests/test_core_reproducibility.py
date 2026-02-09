"""Tests for edgevad.core.reproducibility."""

import logging

import numpy as np
import torch

from edgevad.core.reproducibility import set_seed


def test_set_seed_reproducible():
    logger = logging.getLogger("test_seed")
    set_seed(123, deterministic=False, logger=logger)
    a1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(123, deterministic=False, logger=logger)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)

    np.testing.assert_allclose(a1, a2, rtol=0.0, atol=0.0)
    assert torch.equal(t1, t2)


def test_set_seed_optional_logger():
    set_seed(42, deterministic=False)
    a = np.random.rand(3)
    set_seed(42, deterministic=False)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)
