"""Tests for edgevad.core.math_utils."""

import logging

import numpy as np
import torch

from edgevad.core.math_utils import l2_normalize, percentile, sanitize_array


def test_l2_normalize_unit_vectors():
    x = torch.randn(4, 8)
    y = l2_normalize(x)
    norms = y.norm(dim=1)
    np.testing.assert_allclose(norms.numpy(), np.ones(4), atol=1e-5)


def test_l2_normalize_fp16_zero_vector_finite():
    x = torch.zeros((2, 3), dtype=torch.float16)
    y = l2_normalize(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.equal(y, torch.zeros_like(y))


def test_l2_normalize_preserves_dtype():
    x = torch.randn(3, 4, dtype=torch.float16)
    y = l2_normalize(x)
    assert y.dtype == torch.float16


def test_percentile_basic():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5


def test_percentile_empty():
    assert percentile([], 50.0) is None


def test_sanitize_array_replaces_nonfinite_default():
    arr = np.array([0.5, np.nan, np.inf, -np.inf], dtype=np.float32)
    cleaned = sanitize_array(arr, "test", logging.getLogger("test"))
    assert cleaned.tolist() == [0.5, 0.0, 0.0, 0.0]
    assert np.isfinite(cleaned).all()


def test_sanitize_array_posinf_one():
    arr = np.array([0.5, np.inf], dtype=np.float32)
    cleaned = sanitize_array(arr, "test", logging.getLogger("test"), posinf=1.0)
    assert cleaned.tolist() == [0.5, 1.0]


def test_sanitize_array_all_finite_passthrough():
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cleaned = sanitize_array(arr, "test", logging.getLogger("test"))
    np.testing.assert_array_equal(arr, cleaned)
