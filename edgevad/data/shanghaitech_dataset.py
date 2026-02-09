from __future__ import annotations

"""
Minimal utilities for the ShanghaiTech anomaly dataset.

Provides:
- list_train_videos(): stable numeric-aware ordering of .avi files.
- list_test_clips(): list of test clip ids under testing/frames.
- list_clip_frames(): sorted list of frame image paths for a clip.
- GT helpers for test_frame_mask/test_pixel_mask layouts.

This is intentionally lightweight so it can be imported without the dataset present.
"""

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

__all__ = [
    "list_videos",
    "list_train_videos",
    "list_test_clips",
    "list_clip_frames",
    "resolve_shanghaitech_mask_path",
    "load_frame_mask",
    "load_pixel_mask",
    "get_gt_per_frame",
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
    List and numerically sort ShanghaiTech test inputs.

    Behavior:
      - If files matching `ext` exist directly under `video_dir`, return those.
      - Otherwise, return subdirectories that contain at least one image file.
    """
    video_root = Path(video_dir).expanduser().resolve()
    if not video_root.is_dir():
        raise NotADirectoryError(f"video_dir is not a directory: {video_root}")

    ext = ext.lower()
    videos = [p for p in video_root.iterdir() if p.is_file() and p.suffix.lower() == ext]
    videos = sorted(videos, key=_natural_key)
    if videos:
        return [str(p) for p in videos]

    image_exts = {".jpg", ".jpeg", ".png"}
    clip_dirs = []
    for p in video_root.iterdir():
        if not p.is_dir():
            continue
        if any(c.is_file() and c.suffix.lower() in image_exts for c in p.iterdir()):
            clip_dirs.append(p)
    clip_dirs = sorted(clip_dirs, key=_natural_key)
    if not clip_dirs:
        raise FileNotFoundError(
            f"No videos with extension '{ext}' or frame directories found under: {video_root}"
        )
    return [str(p) for p in clip_dirs]


def list_train_videos(train_videos_dir: str) -> List[str]:
    """
    List and numerically sort ShanghaiTech training videos.

    Expects:
      <root>/training/videos/*.avi
    """
    return list_videos(train_videos_dir, ext=".avi")


def list_test_clips(test_frames_dir: str) -> List[str]:
    """
    List ShanghaiTech test clip IDs under <root>/testing/frames/<clip_id>/.
    """
    root = Path(test_frames_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"test_frames_dir is not a directory: {root}")
    clips = [p for p in root.iterdir() if p.is_dir()]
    clips = sorted(clips, key=_natural_key)
    return [p.name for p in clips]


def list_clip_frames(test_frames_dir: str, clip_id: str) -> List[str]:
    """
    List and numerically sort frame image paths for a test clip.
    """
    clip_dir = Path(test_frames_dir).expanduser().resolve() / clip_id
    if not clip_dir.is_dir():
        raise NotADirectoryError(f"clip directory does not exist: {clip_dir}")
    frames = [
        p
        for p in clip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    frames = sorted(frames, key=_natural_key)
    if not frames:
        raise FileNotFoundError(f"No frame images found under: {clip_dir}")
    return [str(p) for p in frames]


def _clip_id_variants(clip_id: str) -> List[str]:
    match = re.match(r"^(\d+)[_ -](\d+)$", clip_id)
    if not match:
        return [clip_id]
    scene_str, index_str = match.groups()
    scene_int = int(scene_str)
    index_int = int(index_str)
    scene_variants = [scene_str, str(scene_int)]
    index_variants = [
        index_str,
        str(index_int),
        str(index_int).zfill(3),
        str(index_int).zfill(4),
    ]
    variants = []
    for s in scene_variants:
        for i in index_variants:
            variants.append(f"{s}_{i}")
    variants.insert(0, clip_id)
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def resolve_shanghaitech_mask_path(mask_dir: str, clip_id: str) -> Optional[Path]:
    """
    Resolve a mask path by trying common ShanghaiTech naming variants.

    Checks for: exact clip_id.npy, stripped leading zeros, and index widths 3/4.
    """
    if not mask_dir:
        return None
    root = Path(mask_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"mask_dir is not a directory: {root}")
    for variant in _clip_id_variants(clip_id):
        candidate = root / f"{variant}.npy"
        if candidate.is_file():
            return candidate
    return None


def load_frame_mask(frame_mask_dir: str, clip_id: str) -> np.ndarray:
    """
    Load per-frame labels for a clip as (T,) uint8 in {0,1}.
    """
    path = resolve_shanghaitech_mask_path(frame_mask_dir, clip_id)
    if path is None:
        raise FileNotFoundError(f"No frame mask found for clip '{clip_id}' under {frame_mask_dir}")
    arr = np.load(path, allow_pickle=True)
    labels = _to_binary_labels(np.asarray(arr).ravel())
    return labels.astype(np.uint8)


def _normalize_pixel_mask_shape(arr: np.ndarray) -> np.ndarray:
    """Normalize a 3D pixel mask to (T, H, W) ordering.

    Heuristic: ShanghaiTech stores masks as H,W,T (MATLAB convention) where T
    is typically much smaller than H and W.  We detect H,W,T only when dim-2 is
    **strictly** the smallest — this avoids mis-transposing shapes like (4,3,3)
    where dim-1 == dim-2 and the array is already T,H,W.
    """
    if arr.ndim != 3:
        raise ValueError(f"Pixel mask must be 3D, got shape {arr.shape}")
    shape = arr.shape
    # dim-0 is smallest or tied → already T,H,W
    if shape[0] <= shape[1] and shape[0] <= shape[2]:
        return arr
    # dim-2 is strictly smallest → H,W,T → transpose to T,H,W
    if shape[2] < shape[0] and shape[2] < shape[1]:
        return arr.transpose(2, 0, 1)
    # Ambiguous layout — keep as-is (assume T,H,W)
    return arr


def load_pixel_mask(pixel_mask_dir: str, clip_id: str) -> np.ndarray:
    """
    Load per-pixel masks as boolean array shaped (T, H, W).
    Accepts (T,H,W) or (H,W,T) by transposing.
    """
    path = resolve_shanghaitech_mask_path(pixel_mask_dir, clip_id)
    if path is None:
        raise FileNotFoundError(f"No pixel mask found for clip '{clip_id}' under {pixel_mask_dir}")
    arr = np.load(path, allow_pickle=True)
    arr = _normalize_pixel_mask_shape(np.asarray(arr))
    return (arr > 0)


def _resolve_gt_dirs(gt_root_or_testing_dir: str) -> tuple[Optional[str], Optional[str]]:
    root = Path(gt_root_or_testing_dir).expanduser().resolve()
    if not root.exists():
        return None, None
    testing_dir = root
    if (root / "testing").is_dir() and not (
        (root / "test_frame_mask").is_dir() or (root / "test_pixel_mask").is_dir()
    ):
        testing_dir = root / "testing"
    frame_mask_dir = testing_dir / "test_frame_mask"
    pixel_mask_dir = testing_dir / "test_pixel_mask"
    return (
        str(frame_mask_dir) if frame_mask_dir.is_dir() else None,
        str(pixel_mask_dir) if pixel_mask_dir.is_dir() else None,
    )


def get_gt_per_frame(gt_root_or_testing_dir: str, clip_id: str) -> Optional[np.ndarray]:
    """
    Resolve per-frame GT labels for a clip.

    Priority:
      1) testing/test_frame_mask/<clip_id>.npy
      2) testing/test_pixel_mask/<clip_id>.npy -> max over pixels
    Returns None if no GT is found.
    """
    frame_mask_dir, pixel_mask_dir = _resolve_gt_dirs(gt_root_or_testing_dir)
    if frame_mask_dir:
        path = resolve_shanghaitech_mask_path(frame_mask_dir, clip_id)
        if path is not None:
            return load_frame_mask(frame_mask_dir, clip_id)
    if pixel_mask_dir:
        path = resolve_shanghaitech_mask_path(pixel_mask_dir, clip_id)
        if path is not None:
            pixel_mask = load_pixel_mask(pixel_mask_dir, clip_id)
            labels = pixel_mask.reshape(pixel_mask.shape[0], -1).max(axis=1)
            return labels.astype(np.uint8)
    return None


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
