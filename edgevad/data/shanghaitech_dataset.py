from __future__ import annotations

"""
Minimal utilities for the ShanghaiTech anomaly dataset.

Provides:
- list_videos(): stable numeric-aware ordering of video files.
- path helpers for common train/test/gt layouts.
- load_shanghaitech_gt(): stub/utility to turn provided GT into per-frame 0/1 labels.

This is intentionally lightweight so it can be imported without the dataset present.
"""

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

__all__ = [
    "list_videos",
    "train_video_dir",
    "test_video_dir",
    "gt_dir",
    "load_shanghaitech_gt",
]


def _natural_key(path: Path) -> List[Any]:
    """Return a key for natural sorting (numeric-aware)."""
    parts: List[str] = re.split(r"(\d+)", path.stem)
    key: List[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        elif p:
            key.append(p.lower())
    return key


def list_videos(video_dir: str, ext: str = ".avi") -> List[str]:
    """
    List and numerically sort video files under `video_dir` matching `ext`.

    Returns absolute paths. Raises FileNotFoundError if none found.
    """
    video_root = Path(video_dir).expanduser().resolve()
    if not video_root.is_dir():
        raise NotADirectoryError(f"video_dir is not a directory: {video_root}")

    ext = ext.lower()
    videos = [p for p in video_root.iterdir() if p.is_file() and p.suffix.lower() == ext]
    videos = sorted(videos, key=_natural_key)
    if not videos:
        raise FileNotFoundError(f"No videos with extension '{ext}' found under: {video_root}")
    return [str(p) for p in videos]


def train_video_dir(root: str) -> str:
    """
    Convenience helper for a common ShanghaiTech layout:
      <root>/training/videos/*.avi
    """
    return str(Path(root).expanduser().resolve() / "training" / "videos")


def test_video_dir(root: str) -> str:
    """
    Convenience helper for a common ShanghaiTech layout:
      <root>/testing/videos/*.avi
    """
    return str(Path(root).expanduser().resolve() / "testing" / "videos")


def gt_dir(root: str) -> str:
    """
    Convenience helper for a common ShanghaiTech layout:
      <root>/testing/gt
    Adjust if your dataset layout differs.
    """
    return str(Path(root).expanduser().resolve() / "testing" / "gt")


def _to_binary_labels(arr: Iterable[Any]) -> np.ndarray:
    labels = np.asarray(list(arr)).astype(np.int64)
    return (labels > 0).astype(np.uint8).ravel()


def load_shanghaitech_gt(gt_input: Any, total_frames: Optional[int] = None) -> np.ndarray:
    """
    Best-effort GT loader that returns per-frame 0/1 labels.

    Accepts:
      - np.ndarray or list-like of frame labels (interpreted as >0 -> 1)
      - path to .npy containing a 1D array
      - path to .mat containing a key named 'gt' or 'frame_label' (if scipy is available)

    This is a stub: extend as needed to match your dataset variant.
    """
    if isinstance(gt_input, (list, tuple, np.ndarray)):
        labels = _to_binary_labels(gt_input)
    else:
        path = Path(str(gt_input)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"GT path does not exist: {path}")
        if path.suffix.lower() == ".npy":
            labels = _to_binary_labels(np.load(path, allow_pickle=True))
        elif path.suffix.lower() == ".mat":
            try:
                import scipy.io as sio  # optional dependency
            except Exception as e:  # pragma: no cover - optional path
                raise ImportError("scipy is required to load .mat GT files") from e
            mat = sio.loadmat(path)
            if "gt" in mat:
                labels = _to_binary_labels(mat["gt"])
            elif "frame_label" in mat:
                labels = _to_binary_labels(mat["frame_label"])
            else:
                raise KeyError(f"GT .mat missing 'gt' or 'frame_label' keys: {path}")
        else:
            raise NotImplementedError(
                f"Unsupported GT input: {path}. Provide a list/array or .npy/.mat file."
            )

    if total_frames is not None:
        if len(labels) < total_frames:
            labels = np.pad(labels, (0, total_frames - len(labels)), constant_values=0)
        elif len(labels) > total_frames:
            labels = labels[:total_frames]
    return labels.astype(np.uint8)
