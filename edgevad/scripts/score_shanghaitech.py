#!/usr/bin/env python3
"""
EdgeVAD ShanghaiTech benchmark scorer.

Outputs JSON schema matching Avenue scorer:
  overall: auc_raw, auc_smooth, ap_raw, ap_smooth
  runtime: wall_seconds, proc_fps, latency_ms_p50, latency_ms_p95, frames_processed
  per_video: list with gt_counts and per-video auc/ap when possible
Also writes a scores.csv with raw/smoothed scores and GT per frame.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import cv2
import torch

try:
    from ultralytics import YOLO
    import ultralytics

    _ultralytics_import_error: Optional[Exception] = None
except Exception as e:  # pragma: no cover - import guard
    YOLO = None
    ultralytics = None
    _ultralytics_import_error = e

try:
    import torchvision

    _torchvision_import_error: Optional[Exception] = None
except Exception as e:  # pragma: no cover - import guard
    torchvision = None
    _torchvision_import_error = e

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover - optional dep
    roc_auc_score = None
    average_precision_score = None

from edgevad.data.shanghaitech_dataset import (
    list_videos,
    load_shanghaitech_gt,
)


# -------------------------
# Logging / Repro
# -------------------------


def utc_ts() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_git_hash(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL, text=True
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", out):
            return out
    except Exception:
        pass
    return None


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("score_shanghaitech")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def set_seed(seed: int, deterministic: bool, logger: logging.Logger) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:  # pragma: no cover - older torch
            logger.warning(f"torch.use_deterministic_algorithms(True) failed: {e}")


def percentile(x: List[float], p: float) -> Optional[float]:
    if not x:
        return None
    arr = np.asarray(x, dtype=np.float64)
    return float(np.percentile(arr, p))


# -------------------------
# Model helpers
# -------------------------


class MobileNetV3SmallEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if torchvision is None:
            raise RuntimeError(f"Failed to import torchvision. Error: {_torchvision_import_error}")
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        self.features = m.features
        self.avgpool = m.avgpool
        self.out_dim = 576

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._std
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def preprocess_crops_to_tensor(crops_bgr: List[np.ndarray], device: torch.device, fp16: bool) -> torch.Tensor:
    batch = []
    for c in crops_bgr:
        if c is None or c.size == 0:
            continue
        r = cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(r).to(torch.float32) / 255.0
        t = t.permute(2, 0, 1).contiguous()
        batch.append(t)
    if not batch:
        return torch.empty((0, 3, 224, 224), device=device, dtype=torch.float16 if fp16 else torch.float32)
    x = torch.stack(batch, dim=0).to(device=device)
    if fp16:
        x = x.half()
    return x


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


# -------------------------
# Metrics helpers
# -------------------------


def smooth_moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="reflect")
    ys = np.convolve(xp, kernel, mode="valid")
    return ys.astype(np.float32)


def compute_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if roc_auc_score is None or average_precision_score is None:
        return None, None
    y_true = y_true.astype(np.int32)
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None, None
    try:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
        return auc, ap
    except Exception:
        return None, None


@dataclass
class FrameResult:
    frame_idx: int
    score: float
    score_smooth: float
    num_dets: int
    gt: int
    latency_ms: float


@dataclass
class VideoResult:
    video: str
    frames: List[FrameResult]
    wall_time_s: float
    proc_fps: float
    auc_raw: Optional[float]
    ap_raw: Optional[float]
    auc_smooth: Optional[float]
    ap_smooth: Optional[float]
    gt_counts: Dict[str, int]


# -------------------------
# GT resolution
# -------------------------


def resolve_gt_path(gt_dir: str, video_path: str) -> Optional[Path]:
    """
    Best-effort resolver: looks for files matching the video stem with .npy or .mat.
    Examples:
      <gt_dir>/<stem>.npy
      <gt_dir>/<stem>.mat
    """
    if not gt_dir:
        return None
    root = Path(gt_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--gt_dir is not a directory: {root}")
    stem = Path(video_path).stem
    candidates = [root / f"{stem}.npy", root / f"{stem}.mat"]
    for c in candidates:
        if c.is_file():
            return c
    return None


# -------------------------
# Main scoring logic
# -------------------------


def score_videos(
    video_paths: List[str],
    frame_stride: int,
    max_frames: int,
    yolo: Any,
    embedder: MobileNetV3SmallEmbedder,
    bank: torch.Tensor,
    device: torch.device,
    imgsz: int,
    conf: float,
    classes: Optional[List[int]],
    max_dets_per_frame: int,
    fp16: bool,
    smooth: int,
    gt_dir: Optional[str],
    logger: logging.Logger,
) -> List[VideoResult]:
    results: List[VideoResult] = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        gt_labels = None
        if gt_dir:
            gt_path = resolve_gt_path(gt_dir, video_path)
            if gt_path:
                try:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_frames = total_frames if total_frames > 0 else None
                    gt_labels = load_shanghaitech_gt(gt_path, total_frames=total_frames)
                except Exception as e:
                    logger.warning(f"Failed to load GT for {video_path}: {e}")
            else:
                logger.warning(f"No GT file found for {Path(video_path).name} under {gt_dir}")

        frames: List[FrameResult] = []
        frame_idx = 0
        t_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            if max_frames > 0 and len(frames) >= max_frames:
                break

            t0 = time.time()
            preds = yolo.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                classes=classes,
                device=str(device),
                verbose=False,
            )
            boxes = preds[0].boxes
            dets = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                order = np.argsort(-confs)
                for j in order[:max_dets_per_frame]:
                    dets.append(tuple(xyxy[j].tolist()))

            crops = []
            h, w = frame.shape[:2]
            for (x1, y1, x2, y2) in dets:
                x1i = max(0, min(w - 1, int(x1)))
                y1i = max(0, min(h - 1, int(y1)))
                x2i = max(0, min(w, int(x2)))
                y2i = max(0, min(h, int(y2)))
                if x2i <= x1i or y2i <= y1i:
                    continue
                crops.append(frame[y1i:y2i, x1i:x2i].copy())

            score = 0.0
            nd = len(crops)
            if nd > 0:
                x = preprocess_crops_to_tensor(crops, device=device, fp16=fp16)
                with torch.no_grad():
                    emb = embedder(x)
                    emb = l2_normalize(emb.to(dtype=bank.dtype))
                    sim = emb @ bank.T
                    score = float(1.0 - sim.max().item())

            label = -1
            if gt_labels is not None and frame_idx < len(gt_labels):
                label = int(gt_labels[frame_idx])

            latency_ms = (time.time() - t0) * 1000.0
            frames.append(
                FrameResult(
                    frame_idx=frame_idx,
                    score=score,
                    score_smooth=0.0,  # filled later
                    num_dets=nd,
                    gt=label,
                    latency_ms=latency_ms,
                )
            )
            frame_idx += 1

        cap.release()

        # smoothing and metrics
        scores_np = np.asarray([f.score for f in frames], dtype=np.float32)
        scores_smooth = smooth_moving_average(scores_np, smooth)
        for f, s in zip(frames, scores_smooth):
            f.score_smooth = float(s)

        gt_np = np.asarray([f.gt for f in frames], dtype=np.int32)
        valid = gt_np >= 0
        y_true = gt_np[valid].astype(np.int32)
        y_raw = scores_np[valid]
        y_smooth = scores_smooth[valid]

        auc_raw, ap_raw = compute_auc_ap(y_true, y_raw)
        auc_s, ap_s = compute_auc_ap(y_true, y_smooth)

        gt_counts = {
            "-1": int((gt_np == -1).sum()),
            "0": int((gt_np == 0).sum()),
            "1": int((gt_np == 1).sum()),
        }

        wall = max(time.time() - t_start, 1e-9)
        proc_fps = len(frames) / wall if wall > 0 else 0.0

        results.append(
            VideoResult(
                video=Path(video_path).name,
                frames=frames,
                wall_time_s=float(wall),
                proc_fps=float(proc_fps),
                auc_raw=auc_raw,
                ap_raw=ap_raw,
                auc_smooth=auc_s,
                ap_smooth=ap_s,
                gt_counts=gt_counts,
            )
        )

    return results


# -------------------------
# Memory bank loader / env info
# -------------------------


def load_memory_bank_npz(path: str, device: torch.device, fp16: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    d = np.load(path, allow_pickle=True)
    if "embeddings" not in d:
        raise RuntimeError(f"Memory bank npz missing 'embeddings': {path}")
    E = d["embeddings"]
    E = E.astype(np.float16 if fp16 else np.float32, copy=False)
    meta = {}
    if "config" in d:
        try:
            meta["config"] = str(d["config"])
        except Exception:
            meta["config"] = None
    bank = torch.from_numpy(E).to(device=device)
    bank = l2_normalize(bank)
    return bank, meta


def env_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = sys.version.replace("\n", " ")
    info["platform"] = os.name
    info["torch"] = torch.__version__
    info["torchvision"] = getattr(torchvision, "__version__", None)
    info["ultralytics"] = getattr(ultralytics, "__version__", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available() and "cuda" in str(device):
        try:
            idx = int(str(device).split(":")[-1])
        except Exception:
            idx = 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
    else:
        info["gpu_name"] = None
    return info


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


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Score ShanghaiTech dataset with EdgeVAD")
    ap.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    ap.add_argument("--mb", required=True, help="Memory bank .npz (embeddings)")
    ap.add_argument("--test_dir", required=True, help="ShanghaiTech test videos directory")
    ap.add_argument("--gt_dir", default="", help="Optional GT directory with per-video labels (.npy/.mat)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="0", help="YOLO class ids, e.g. '0' for person. Empty disables filter.")
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--max_dets_per_frame", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--max_frames", type=int, default=0, help="Max processed frames per video (0 = no limit)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True, help="metrics_full.json output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_file", default="", help="Optional log file path")
    return ap.parse_args()


def parse_classes(s: str) -> Optional[List[int]]:
    s = str(s).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out if out else None


def main() -> None:
    args = parse_args()

    logger = setup_logger(args.log_file if args.log_file.strip() else None)
    set_seed(args.seed, args.deterministic, logger)

    device = torch.device(args.device)
    classes = parse_classes(args.classes)
    gt_dir = args.gt_dir.strip() or None

    repo_dir = str(Path(__file__).resolve().parents[2])
    git_hash = get_git_hash(repo_dir)

    test_dir = Path(args.test_dir).expanduser().resolve()
    if not test_dir.is_dir():
        raise RuntimeError(f"--test_dir is not a directory: {test_dir}")

    video_paths = list_videos(str(test_dir), ext=".avi")
    if len(video_paths) == 0:
        raise RuntimeError(f"No .avi videos found in: {test_dir}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    bank, mb_meta = load_memory_bank_npz(args.mb, device=device, fp16=args.fp16)

    if YOLO is None:
        raise RuntimeError(f"Failed to import ultralytics/YOLO. Error: {_ultralytics_import_error}")
    yolo = YOLO(args.yolo)
    embedder = MobileNetV3SmallEmbedder().to(device)
    embedder.eval()
    if args.fp16:
        embedder.half()

    t_all0 = time.time()
    logger.info(
        f"dataset videos={len(video_paths)} frame_stride={args.frame_stride} max_frames={args.max_frames} imgsz={args.imgsz}"
    )

    per_video = score_videos(
        video_paths=video_paths,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        yolo=yolo,
        embedder=embedder,
        bank=bank,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        classes=classes,
        max_dets_per_frame=args.max_dets_per_frame,
        fp16=bool(args.fp16),
        smooth=args.smooth,
        gt_dir=gt_dir,
        logger=logger,
    )

    wall_all = time.time() - t_all0

    # write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "score", "score_smooth", "num_dets", "gt"])
        for vr in per_video:
            for fr in vr.frames:
                w.writerow([vr.video, fr.frame_idx, fr.score, fr.score_smooth, fr.num_dets, fr.gt])

    all_scores = np.concatenate([np.asarray([fr.score for fr in v.frames], np.float32) for v in per_video], axis=0)
    all_scores_s = np.concatenate(
        [np.asarray([fr.score_smooth for fr in v.frames], np.float32) for v in per_video], axis=0
    )
    all_gt = np.concatenate([np.asarray([fr.gt for fr in v.frames], np.int32) for v in per_video], axis=0)

    valid = all_gt >= 0
    y_true = all_gt[valid].astype(np.int32)
    y_raw = all_scores[valid]
    y_smooth = all_scores_s[valid]

    overall_auc_raw, overall_ap_raw = compute_auc_ap(y_true, y_raw)
    overall_auc_s, overall_ap_s = compute_auc_ap(y_true, y_smooth)

    frames_processed = int(sum(len(v.frames) for v in per_video))
    proc_fps = float(frames_processed / max(wall_all, 1e-9))

    all_lat = []
    for v in per_video:
        all_lat.extend([fr.latency_ms for fr in v.frames])
    lat_p50 = percentile(all_lat, 50.0)
    lat_p95 = percentile(all_lat, 95.0)

    per_video_json = []
    for v in per_video:
        per_video_json.append(
            {
                "video": v.video,
                "num_rows": len(v.frames),
                "frames_processed": len(v.frames),
                "wall_time_s": v.wall_time_s,
                "proc_fps": v.proc_fps,
                "latency_ms_p50": percentile([fr.latency_ms for fr in v.frames], 50.0),
                "latency_ms_p95": percentile([fr.latency_ms for fr in v.frames], 95.0),
                "auc_raw": v.auc_raw,
                "ap_raw": v.ap_raw,
                "auc_smooth": v.auc_smooth,
                "ap_smooth": v.ap_smooth,
                "gt_counts": v.gt_counts,
            }
        )

    metrics_full = {
        "timestamp_utc": utc_ts(),
        "git_hash": git_hash,
        "args": vars(args),
        "env": env_info(device),
        "memory_bank": {
            "path": args.mb,
            "shape": [int(bank.shape[0]), int(bank.shape[1])],
            "dtype": str(bank.dtype),
            "meta": mb_meta,
        },
        "overall": {
            "auc_raw": overall_auc_raw,
            "auc_smooth": overall_auc_s,
            "ap_raw": overall_ap_raw,
            "ap_smooth": overall_ap_s,
        },
        "runtime": {
            "num_videos": len(per_video),
            "num_rows": frames_processed,
            "frames_processed": frames_processed,
            "wall_seconds": float(wall_all),
            "proc_fps": proc_fps,
            "latency_ms_p50": lat_p50,
            "latency_ms_p95": lat_p95,
        },
        "per_video": per_video_json,
    }

    validate_metrics_schema(metrics_full)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2)

    logger.info("=== ShanghaiTech Benchmark Summary ===")
    logger.info(json.dumps({"overall": metrics_full["overall"], "runtime": metrics_full["runtime"]}, indent=2))
    logger.info(f"CSV:  {args.out_csv}")
    logger.info(f"JSON: {args.out_json}")


if __name__ == "__main__":  # pragma: no cover
    main()
