"""Tests for edgevad.core.metrics."""

import numpy as np
import pytest

from edgevad.core.metrics import compute_auc_ap, validate_metrics_schema


def test_compute_auc_ap_perfect():
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    auc, ap, reason = compute_auc_ap(y_true, y_score)
    assert auc == 1.0
    assert ap == 1.0
    assert reason is None


def test_compute_auc_ap_single_class():
    y_true = np.array([0, 0, 0], dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    auc, ap, reason = compute_auc_ap(y_true, y_score)
    assert auc is None
    assert ap is None
    assert reason == "only_one_class"


def test_compute_auc_ap_empty():
    y_true = np.array([], dtype=np.int32)
    y_score = np.array([], dtype=np.float32)
    auc, ap, reason = compute_auc_ap(y_true, y_score)
    assert auc is None
    assert reason == "no_gt"


def test_compute_auc_ap_returns_3_tuple():
    """Always returns 3-tuple (auc, ap, reason)."""
    y_true = np.array([0, 1], dtype=np.int32)
    y_score = np.array([0.3, 0.7], dtype=np.float32)
    result = compute_auc_ap(y_true, y_score)
    assert len(result) == 3


def test_validate_metrics_schema_valid():
    metrics = {
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "git_hash": None,
        "args": {},
        "env": {},
        "memory_bank": {},
        "overall": {"auc_raw": None, "auc_smooth": None, "ap_raw": None, "ap_smooth": None},
        "runtime": {"wall_seconds": 1.0, "proc_fps": 30.0},
        "per_video": [],
    }
    validate_metrics_schema(metrics)  # should not raise


def test_validate_metrics_schema_missing_overall():
    metrics = {
        "timestamp_utc": "2024-01-01T00:00:00Z",
        "git_hash": None,
        "args": {},
        "env": {},
        "memory_bank": {},
        "runtime": {"wall_seconds": 1.0, "proc_fps": 30.0},
        "per_video": [],
    }
    with pytest.raises(ValueError):
        validate_metrics_schema(metrics)
