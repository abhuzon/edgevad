"""Temporal smoothing: moving average and Gaussian."""

from __future__ import annotations

import numpy as np


def smooth_moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="reflect")
    ys = np.convolve(xp, kernel, mode="valid")
    return ys.astype(np.float32)


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing via scipy. If sigma <= 0, returns copy."""
    if sigma <= 0:
        return x.copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(
        x.astype(np.float64), sigma=sigma, mode="reflect"
    ).astype(np.float32)
