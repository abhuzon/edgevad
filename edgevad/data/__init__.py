"""Dataset utilities for EdgeVAD."""

from .avenue_dataset import (
    AvenueVideoDataset,
    list_videos,
    load_avenue_gt_per_frame,
    resolve_gt_mat_path,
)
from .shanghaitech_dataset import (
    list_videos as list_videos_shanghaitech,
    train_video_dir as shanghaitech_train_video_dir,
    test_video_dir as shanghaitech_test_video_dir,
    gt_dir as shanghaitech_gt_dir,
    load_shanghaitech_gt,
)

__all__ = [
    "AvenueVideoDataset",
    "list_videos",
    "load_avenue_gt_per_frame",
    "resolve_gt_mat_path",
    "list_videos_shanghaitech",
    "shanghaitech_train_video_dir",
    "shanghaitech_test_video_dir",
    "shanghaitech_gt_dir",
    "load_shanghaitech_gt",
]
