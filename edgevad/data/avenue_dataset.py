from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

__all__ = [
    "list_videos",
    "resolve_gt_mat_path",
    "load_avenue_gt_per_frame",
    "AvenueVideoDataset",
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


def resolve_gt_mat_path(gt_dir: str, video_path: str) -> str:
    """
    Resolve Avenue GT .mat path for a given video.

    Accepts both `01.avi -> 01_label.mat` and `01.avi -> 1_label.mat` patterns.
    Preference order:
      1) exact video stem match
      2) int(stem) without zero padding
      3) zero-padded (width >= 2)
    Raises FileNotFoundError with attempted candidates if nothing exists.
    """
    gt_root = Path(gt_dir).expanduser().resolve()
    if not gt_root.is_dir():
        raise NotADirectoryError(f"gt_dir is not a directory: {gt_root}")

    stem = Path(video_path).stem
    tried: List[Path] = []

    def _add_candidate(name: str) -> None:
        p = gt_root / f"{name}_label.mat"
        if p not in tried:
            tried.append(p)

    _add_candidate(stem)

    try:
        vid_int = int(stem)
    except Exception:
        vid_int = None

    if vid_int is not None:
        _add_candidate(str(vid_int))
        width = max(len(stem), 2)
        _add_candidate(f"{vid_int:0{width}d}")

    for candidate in tried:
        if candidate.is_file():
            return str(candidate)

    tried_str = ", ".join(str(p) for p in tried)
    raise FileNotFoundError(f"No GT .mat found for video '{video_path}'. Tried: {tried_str}")


def _labels_from_object_array(obj: np.ndarray) -> np.ndarray:
    cells = obj.ravel()
    labels = np.zeros(len(cells), dtype=np.uint8)
    for i, cell in enumerate(cells):
        m = np.asarray(cell)
        if m.dtype == object:
            m = np.asarray(m.tolist())
        labels[i] = 1 if (m > 0).any() else 0
    return labels


def _labels_from_numeric(vol: np.ndarray) -> np.ndarray:
    if vol.ndim == 3:
        # Prefer Avenue-style shapes: (H, W, T) then (T, H, W).
        candidate_axes = (2, 0, 1)
        best = None
        for priority, axis in enumerate(candidate_axes):
            other_axes = tuple(i for i in range(3) if i != axis)
            labels = (vol != 0).any(axis=other_axes)
            labels = np.asarray(labels, dtype=np.uint8).ravel()
            axis_len = vol.shape[axis]
            # Choose axis with largest length (plausible frame count); tie-break by Avenue-preferred order.
            if best is None or axis_len > best[0] or (axis_len == best[0] and priority < best[1]):
                best = (axis_len, priority, labels)
        if best is not None:
            return best[2]

    if vol.ndim == 2:
        h, w = vol.shape
        if h == 1 and w > 1:
            axis = 1
        elif w == 1 and h > 1:
            axis = 0
        else:
            axis = 0 if h >= w else 1  # larger dimension more likely to be time
        labels = (vol != 0).any(axis=1 - axis)
        return np.asarray(labels, dtype=np.uint8).ravel()

    flat = vol.ravel()
    return np.asarray(flat > 0, dtype=np.uint8)


def load_avenue_gt_per_frame(gt_mat_path: str) -> np.ndarray:
    """
    Load Avenue volLabel from MATLAB .mat into per-frame binary labels.

    Returns a 1D np.ndarray of ints in {0,1} with length=T frames.
    Raises KeyError if volLabel key is missing, ValueError if shape unsupported.
    """
    mat = sio.loadmat(gt_mat_path)
    if "volLabel" not in mat:
        raise KeyError(f"'volLabel' not found in GT mat: {gt_mat_path}")

    vol = np.asarray(mat["volLabel"])
    if vol.dtype == object:
        return _labels_from_object_array(vol)

    return _labels_from_numeric(vol)


def letterbox_image(
    image: np.ndarray,
    new_shape: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize with letterbox padding to square `new_shape` while preserving aspect ratio.

    Returns (padded_image, scale, (pad_x, pad_y)).
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image passed to letterbox_image")

    scale = min(float(new_shape) / h, float(new_shape) / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - new_unpad[0]
    pad_h = new_shape - new_unpad[1]

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (pad_left, pad_top)


def frame_to_tensor(frame: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert HWC uint8 frame to CHW float tensor in [0,1]."""
    tensor = torch.from_numpy(frame).to(dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).contiguous()
    tensor = tensor / 255.0
    return tensor.to(dtype=dtype)


class AvenueVideoDataset(Dataset):
    """
    Frame-level iterator over Avenue videos with optional GT labels.

    __getitem__ returns a dict with:
      - image: transformed frame (torch.Tensor if default transform)
      - frame: raw frame in RGB or BGR depending on return_rgb
      - video_path, video, frame_idx, label, orig_size
      - any extra metadata returned by transform (if dict or (img, meta) tuple)
    """

    def __init__(
        self,
        video_dir: str,
        gt_dir: str | None = None,
        split: str = "test",
        imgsz: int = 640,
        frame_stride: int = 1,
        max_frames: int | None = None,
        start_offset: int = 0,
        seed: int = 0,
        deterministic: bool = True,
        return_rgb: bool = True,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        video_paths: Optional[List[str]] = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if start_offset < 0:
            raise ValueError("start_offset must be >= 0")

        self.video_dir = str(Path(video_dir).expanduser().resolve())
        self.gt_dir = str(Path(gt_dir).expanduser().resolve()) if gt_dir else None
        self.split = split
        self.imgsz = int(imgsz)
        self.frame_stride = int(frame_stride)
        self.max_frames = None if max_frames is None or max_frames <= 0 else int(max_frames)
        self.start_offset = int(start_offset)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.return_rgb = bool(return_rgb)
        self.transform = transform

        if video_paths is None:
            self.videos: List[str] = list_videos(self.video_dir, ext=".avi")
        else:
            self.videos = [str(Path(v).expanduser().resolve()) for v in video_paths]
        self._gt_cache: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(self.seed if self.deterministic else None)
        self.samples: List[Dict[str, Any]] = []

        self._build_index()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        frame = self._read_frame(sample["video_path"], sample["frame_idx"])
        if self.return_rgb:
            frame_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_proc = frame

        transformed = (
            self.transform(frame_proc) if self.transform is not None else self._default_transform(frame_proc)
        )

        image_out = transformed
        extra: Dict[str, Any] = {}
        if isinstance(transformed, tuple) and len(transformed) == 2:
            image_out, extra = transformed
        elif isinstance(transformed, dict):
            extra = {k: v for k, v in transformed.items() if k != "image"}
            image_out = transformed.get("image", transformed)

        data: Dict[str, Any] = {
            "image": image_out,
            "frame": frame_proc,
            "video_path": sample["video_path"],
            "video": Path(sample["video_path"]).name,
            "video_id": Path(sample["video_path"]).stem,
            "frame_idx": sample["frame_idx"],
            "label": sample["label"],
            "gt": sample["label"],
            "orig_size": (frame.shape[0], frame.shape[1]),
        }
        data.update(extra)
        return data

    def _build_index(self) -> None:
        for video_path in self.videos:
            total_frames = self._count_frames(video_path)
            frame_indices = list(range(self.start_offset, total_frames, self.frame_stride))
            if self.max_frames is not None and len(frame_indices) > self.max_frames:
                chosen = self._rng.choice(frame_indices, size=self.max_frames, replace=False)
                frame_indices = sorted(int(i) for i in chosen)

            labels = None
            if self.gt_dir and self.split == "test":
                try:
                    gt_path = resolve_gt_mat_path(self.gt_dir, video_path)
                    labels = load_avenue_gt_per_frame(gt_path)
                    if len(labels) < total_frames:
                        labels = np.pad(labels, (0, total_frames - len(labels)), mode="constant")
                    elif len(labels) > total_frames:
                        labels = labels[:total_frames]
                    self._gt_cache[video_path] = labels
                except FileNotFoundError:
                    labels = None
                except Exception as e:
                    raise RuntimeError(f"Failed to load GT for {video_path}: {e}") from e

            for fi in frame_indices:
                lbl = -1
                if labels is not None and fi < len(labels):
                    lbl = int(labels[fi])
                self.samples.append(
                    {
                        "video_path": video_path,
                        "frame_idx": fi,
                        "label": lbl,
                    }
                )

    def _read_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        # First attempt: seek then read
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        finally:
            cap.release()

        if ok and frame is not None:
            return frame

        # Fallback: sequential read from start
        cap_fallback = cv2.VideoCapture(video_path)
        if not cap_fallback.isOpened():
            raise RuntimeError(f"Could not reopen video for fallback: {video_path}")
        current = -1
        frame_out = None
        try:
            while current < frame_idx:
                ok, frame_out = cap_fallback.read()
                if not ok or frame_out is None:
                    break
                current += 1
        finally:
            cap_fallback.release()

        if ok and frame_out is not None and current == frame_idx:
            return frame_out

        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    def _count_frames(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                total += 1
        cap.release()
        return total

    def _default_transform(self, frame: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        padded, scale, pad = letterbox_image(frame, new_shape=self.imgsz)
        tensor = frame_to_tensor(padded)
        meta = {"scale": scale, "pad": pad}
        return tensor, meta


def _cli() -> None:
    ap = argparse.ArgumentParser("Manual AvenueVideoDataset sanity check")
    ap.add_argument("--video_dir", required=True, help="Directory containing Avenue .avi videos")
    ap.add_argument("--gt_dir", default=None, help="Optional GT dir (contains *_label.mat)")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=8, help="Max frames sampled per video (0=all)")
    ap.add_argument("--start_offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    ds = AvenueVideoDataset(
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        split="test" if args.gt_dir else "train",
        frame_stride=args.frame_stride,
        max_frames=None if args.max_frames == 0 else args.max_frames,
        start_offset=args.start_offset,
        seed=args.seed,
        deterministic=bool(args.deterministic),
    )

    print(f"Dataset size: {len(ds)} samples from {len(ds.videos)} videos")
    gt_vals = [s["label"] for s in ds.samples]
    counts = {v: gt_vals.count(v) for v in sorted(set(gt_vals))}
    print(f"GT counts (label values): {counts}")

    n_show = min(5, len(ds))
    for i in range(n_show):
        sample = ds[i]
        print(
            f"[{i}] video_id={sample['video_id']} frame_idx={sample['frame_idx']} "
            f"label={sample['label']} image_shape={tuple(sample['image'].shape)}"
        )


if __name__ == "__main__":
    _cli()
