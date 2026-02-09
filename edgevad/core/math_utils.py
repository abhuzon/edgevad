"""Math utilities: percentile, L2 normalization, array sanitization."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch


def percentile(x: List[float], p: float) -> Optional[float]:
    if not x:
        return None
    arr = np.asarray(x, dtype=np.float64)
    return float(np.percentile(arr, p))


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along dim=1 with a dtype-safe epsilon.

    Using float16 with a tiny epsilon underflows to zero (1e-12 -> 0) and
    produces NaNs when the norm is also zero. Compute the norm in float32,
    clamp to eps, then cast back to the original dtype to avoid NaN/Inf.
    """
    dtype = x.dtype
    norm = x.float().norm(dim=1, keepdim=True).clamp_min(eps)
    out = x.float() / norm
    return out.to(dtype)


def sanitize_array(
    arr,
    name: str,
    logger: logging.Logger,
    posinf: float = 0.0,
    neginf: float = 0.0,
    nan: float = 0.0,
) -> np.ndarray:
    """Replace non-finite values in array. Default posinf=0.0 (Avenue semantics)."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        bad = int((~finite).sum())
        logger.warning(f"{name}: replacing {bad} non-finite values")
        arr = np.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)
    return arr
