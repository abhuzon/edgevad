"""Dataset utilities for EdgeVAD."""

from .avenue_dataset import (
    AvenueVideoDataset,
    list_videos,
    load_avenue_gt_per_frame,
    resolve_gt_mat_path,
)

__all__ = [
    "AvenueVideoDataset",
    "list_videos",
    "load_avenue_gt_per_frame",
    "resolve_gt_mat_path",
]
