"""Tests for edgevad.core.smoothing."""

import numpy as np

from edgevad.core.smoothing import gaussian_smooth, smooth_moving_average


def test_smooth_moving_average_length_preserved():
    x = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    y = smooth_moving_average(x, win=5)
    assert len(y) == len(x)
    assert np.isfinite(y).all()


def test_smooth_moving_average_win1_identity():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = smooth_moving_average(x, win=1)
    np.testing.assert_array_equal(x, y)


def test_gaussian_smooth_length_preserved():
    x = np.random.rand(50).astype(np.float32)
    y = gaussian_smooth(x, sigma=2.0)
    assert len(y) == len(x)
    assert np.isfinite(y).all()


def test_gaussian_smooth_sigma_zero_identity():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = gaussian_smooth(x, sigma=0.0)
    np.testing.assert_array_equal(x, y)


def test_gaussian_smooth_reduces_variance():
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    y = gaussian_smooth(x, sigma=2.0)
    assert np.std(y) < np.std(x)
