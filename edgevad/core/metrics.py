"""AUC/AP metrics and schema validation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


def compute_auc_ap(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Compute ROC-AUC and Average Precision. Always returns 3-tuple (auc, ap, reason)."""
    if roc_auc_score is None or average_precision_score is None:
        return None, None, "sklearn_missing"
    y_true = y_true.astype(np.int32)
    if y_true.size == 0:
        return None, None, "no_gt"
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None, None, "only_one_class"
    try:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
        return auc, ap, None
    except Exception as e:
        return None, None, f"metric_error:{e}"


def validate_metrics_schema(metrics: Dict[str, Any]) -> None:
    required_top = {"timestamp_utc", "git_hash", "args", "env", "memory_bank", "overall", "runtime", "per_video"}
    missing_top = required_top.difference(metrics.keys())
    if missing_top:
        raise ValueError(f"metrics_full missing top-level keys: {sorted(missing_top)}")

    overall = metrics.get("overall", {})
    for key in ("auc_raw", "auc_smooth", "ap_raw", "ap_smooth"):
        if key not in overall:
            raise ValueError(f"metrics_full.overall missing key: {key}")

    runtime = metrics.get("runtime", {})
    for key in ("wall_seconds", "proc_fps"):
        if key not in runtime:
            raise ValueError(f"metrics_full.runtime missing key: {key}")
    if not isinstance(metrics.get("per_video"), list):
        raise ValueError("metrics_full.per_video must be a list")
