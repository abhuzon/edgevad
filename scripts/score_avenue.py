#!/usr/bin/env python3
"""
EdgeVAD Avenue benchmark scorer (benchmark-grade).

Key guarantees:
- Robust GT loader for Avenue ground_truth_demo .mat (volLabel cell array parsing).
- Correct GT alignment: frame_idx is ORIGINAL frame index (0,3,6,...) and GT is indexed by that.
- Always writes CSV with gt column (0/1 or -1).
- Writes metrics_full JSON with: overall + per-video metrics, runtime, latency p50/p95, env versions, args, git hash.
- Stable logging + reproducible seed + optional deterministic mode.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Hard deps
import cv2
import torch
import scipy.io as sio

try:
    from ultralytics import YOLO
    import ultralytics
    _ultralytics_import_error: Optional[Exception] = None
except Exception as e:
    YOLO = None
    ultralytics = None
    _ultralytics_import_error = e

try:
    import torchvision
    _torchvision_import_error: Optional[Exception] = None
except Exception as e:
    torchvision = None
    _torchvision_import_error = e

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


# -------------------------
# Logging / Repro
# -------------------------

def utc_ts() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_git_hash(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", out):
            return out
    except Exception:
        pass
    return None


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("score_avenue")
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
        # Best-effort determinism (some ops may still be nondeterministic depending on kernels)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            logger.warning(f"torch.use_deterministic_algorithms(True) failed: {e}")


def percentile(x: List[float], p: float) -> Optional[float]:
    if not x:
        return None
    arr = np.asarray(x, dtype=np.float64)
    return float(np.percentile(arr, p))


# -------------------------
# GT Loader (FIXED)
# -------------------------

def volLabel_to_frame_labels(vol):
    """
    Convert Avenue volLabel into per-frame binary labels:
      1 = abnormal frame, 0 = normal frame
    vol can be:
      - MATLAB cell array loaded by scipy -> dtype=object, shapes (1,T), (T,1), (T,)
      - numeric array (H,W,T) or similar
    """
    vol = np.asarray(vol)

    # Case 1: cell array => object array, each element is a 2D mask
    if vol.dtype == object:
        cells = vol.ravel()  # works for (1,T), (T,1), (T,)
        labels = np.zeros(len(cells), dtype=np.uint8)
        for i, cell in enumerate(cells):
            m = np.asarray(cell)
            # if nested object, coerce again
            if m.dtype == object:
                m = np.asarray(m.tolist())
            labels[i] = 1 if (m > 0).any() else 0
        return labels

    # Case 2: numeric array
    if vol.ndim == 3:
        # assume last axis is time: (H,W,T)
        T = vol.shape[-1]
        return ((vol.reshape(-1, T).sum(axis=0) > 0).astype(np.uint8))
    if vol.ndim == 2:
        # ambiguous, but treat columns as time
        return ((vol.sum(axis=0) > 0).astype(np.uint8))

    # fallback
    flat = vol.ravel()
    return ((flat > 0).astype(np.uint8))


def label_path_for_video(gt_dir: str, video_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(video_path))[0]  # "01"
    m = re.search(r"(\d+)", stem)
    if not m:
        return None
    vid = int(m.group(1))  # 1..21
    candidates = [
        os.path.join(gt_dir, f"{vid}_label.mat"),
        os.path.join(gt_dir, f"{vid:02d}_label.mat"),
    ]
    for lp in candidates:
        if os.path.isfile(lp):
            return lp
    return None


def load_avenue_labels_for_video(gt_dir: str, video_path: str, total_frames: Optional[int] = None) -> Optional[np.ndarray]:
    lp = label_path_for_video(gt_dir, video_path)
    if lp is None:
        return None

    # DO NOT squeeze and then index [0] incorrectly
    mat = sio.loadmat(lp)
    if "volLabel" not in mat:
        return None

    labels = volLabel_to_frame_labels(mat["volLabel"])  # length = T
    if total_frames is not None:
        # match video length safely
        if len(labels) < total_frames:
            labels = np.pad(labels, (0, total_frames - len(labels)), constant_values=0)
        elif len(labels) > total_frames:
            labels = labels[:total_frames]
    return labels


# -------------------------
# Model components
# -------------------------

class MobileNetV3SmallEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if torchvision is None:
            raise RuntimeError(f"Failed to import torchvision. Error: {_torchvision_import_error}")
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        self.features = m.features
        self.avgpool = m.avgpool  # AdaptiveAvgPool2d(1)
        self.out_dim = 576

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,224,224) float in [0,1]
        x = (x - self._mean) / self._std
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def preprocess_crops_to_tensor(crops_bgr: List[np.ndarray], device: torch.device, fp16: bool) -> torch.Tensor:
    # Convert BGR uint8 -> RGB float tensor (B,3,224,224)
    batch = []
    for c in crops_bgr:
        if c is None or c.size == 0:
            continue
        r = cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(r).to(torch.float32) / 255.0  # (H,W,C)
        t = t.permute(2, 0, 1).contiguous()  # (C,H,W)
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
# Scoring / Metrics
# -------------------------

def smooth_moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    # pad reflect for stable edges
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="reflect")
    ys = np.convolve(xp, kernel, mode="valid")
    return ys.astype(np.float32)


def compute_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    # Requires both classes
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
class VideoResult:
    video: str
    frame_idxs: List[int]          # ORIGINAL frame idx (0,3,6,...)
    scores: List[float]
    scores_smooth: List[float]
    num_dets: List[int]
    gt: List[int]                  # 0/1 or -1
    wall_time_s: float
    proc_fps: float
    latency_ms: List[float]
    auc_raw: Optional[float]
    ap_raw: Optional[float]
    auc_smooth: Optional[float]
    ap_smooth: Optional[float]
    gt_counts: Dict[str, int]


def run_one_video(
    yolo: Any,
    embedder: MobileNetV3SmallEmbedder,
    bank: torch.Tensor,  # (N,D) normalized
    video_path: str,
    gt_dir: Optional[str],
    device: torch.device,
    imgsz: int,
    conf: float,
    classes: Optional[List[int]],
    frame_stride: int,
    max_dets_per_frame: int,
    fp16: bool,
    smooth: int,
    max_frames: int,
    logger: logging.Logger,
) -> VideoResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    labels = None
    if gt_dir:
        labels = load_avenue_labels_for_video(gt_dir, video_path, total_frames=total_frames)

    frame_idxs: List[int] = []
    scores: List[float] = []
    num_dets: List[int] = []
    gt_list: List[int] = []
    lat_ms: List[float] = []

    processed = 0
    t0 = time.time()
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue
        if max_frames > 0 and processed >= max_frames:
            break

        t_frame0 = time.time()

        # YOLO inference
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
            # sort by conf desc
            order = np.argsort(-confs)
            for j in order[:max_dets_per_frame]:
                x1, y1, x2, y2 = xyxy[j].tolist()
                dets.append((x1, y1, x2, y2, float(confs[j])))

        crops = []
        h, w = frame.shape[:2]
        for (x1, y1, x2, y2, _) in dets:
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
                # cosine sim (B,N)
                sim = emb @ bank.T
                sim_max = sim.max().item()
                score = float(1.0 - sim_max)  # anomaly score

        # GT alignment rule:
        # You store frame_idx as original frame index (0,3,6,…). GT must be indexed by that original index.
        gt_val = -1
        if labels is not None and frame_idx < len(labels):
            gt_val = int(labels[frame_idx])

        frame_idxs.append(frame_idx)
        scores.append(score)
        num_dets.append(nd)
        gt_list.append(gt_val)

        lat_ms.append((time.time() - t_frame0) * 1000.0)

        processed += 1
        if processed % 200 == 0:
            dt = time.time() - t0
            proc_fps = processed / max(dt, 1e-9)
            logger.info(f"{os.path.basename(video_path)}: processed={processed} frame_idx={frame_idx} proc_fps={proc_fps:.2f}")

    cap.release()
    wall = time.time() - t0
    proc_fps = processed / max(wall, 1e-9)

    scores_np = np.asarray(scores, dtype=np.float32)
    scores_s = smooth_moving_average(scores_np, smooth).tolist()

    # Metrics computed only where gt in {0,1}
    gt_np = np.asarray(gt_list, dtype=np.int32)
    valid = gt_np >= 0
    y_true = gt_np[valid].astype(np.int32)
    y_raw = scores_np[valid]
    y_smooth = np.asarray(scores_s, dtype=np.float32)[valid]

    auc_raw, ap_raw = compute_auc_ap(y_true, y_raw)
    auc_s, ap_s = compute_auc_ap(y_true, y_smooth)

    gt_counts = {"-1": int((gt_np == -1).sum()), "0": int((gt_np == 0).sum()), "1": int((gt_np == 1).sum())}
    # If gt is present but has no positives, warn loudly (this is exactly your bug symptom)
    if gt_counts["1"] == 0 and gt_counts["0"] > 0 and gt_counts["-1"] == 0:
        logger.warning(f"{os.path.basename(video_path)}: GT has 0 positives (all normal). If this happens for ALL videos, your GT parsing is wrong.")

    return VideoResult(
        video=os.path.basename(video_path),
        frame_idxs=frame_idxs,
        scores=scores,
        scores_smooth=scores_s,
        num_dets=num_dets,
        gt=gt_list,
        wall_time_s=float(wall),
        proc_fps=float(proc_fps),
        latency_ms=lat_ms,
        auc_raw=auc_raw,
        ap_raw=ap_raw,
        auc_smooth=auc_s,
        ap_smooth=ap_s,
        gt_counts=gt_counts,
    )


def parse_classes(s: str) -> Optional[List[int]]:
    s = str(s).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    # accepts: "0" or "0,1,2"
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out if out else None


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
    info["platform"] = platform.platform()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    ap.add_argument("--mb", required=True, help="Memory bank .npz (embeddings)")
    ap.add_argument("--test_dir", required=True, help="Avenue test videos directory")
    ap.add_argument("--gt_dir", default="", help="Avenue GT dir: ground_truth_demo/testing_label_mask")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="0", help="YOLO class ids, e.g. '0' for person. Empty disables filter.")
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--max_dets_per_frame", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)  # reserved (currently per-frame)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--max_frames", type=int, default=0, help="Max processed frames per video (0 = no limit)")
    ap.add_argument("--demo_video", default="", help="Optional: run one video and also save scored demo")
    ap.add_argument("--save_video", default="", help="Optional: path to save a scored demo mp4 (requires --demo_video)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True, help="metrics_full.json output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_file", default="", help="Optional log file path")

    args = ap.parse_args()

    logger = setup_logger(args.log_file if args.log_file.strip() else None)
    set_seed(args.seed, args.deterministic, logger)

    device = torch.device(args.device)
    classes = parse_classes(args.classes)
    gt_dir = args.gt_dir.strip() or None

    # Repo dir for git hash
    repo_dir = str(Path(__file__).resolve().parents[1])  # .../edgevad
    git_hash = get_git_hash(repo_dir)

    # Load memory bank
    bank, mb_meta = load_memory_bank_npz(args.mb, device=device, fp16=args.fp16)

    # Load models
    if YOLO is None:
        raise RuntimeError(f"Failed to import ultralytics/YOLO. Error: {_ultralytics_import_error}")
    yolo = YOLO(args.yolo)
    embedder = MobileNetV3SmallEmbedder().to(device)
    embedder.eval()
    if args.fp16:
        embedder.half()

    # Collect videos
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise RuntimeError(f"--test_dir is not a directory: {test_dir}")

    videos = sorted([str(p) for p in test_dir.glob("*.avi")])
    if not videos:
        raise RuntimeError(f"No .avi videos found in: {test_dir}")

    # Output dirs
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    if args.save_video.strip():
        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"memory_bank: {args.mb} embeddings={tuple(bank.shape)} dtype={bank.dtype}")
    logger.info(f"device={device} fp16={bool(args.fp16)} embedder_dim=576 seed={args.seed} deterministic={bool(args.deterministic)}")
    if git_hash:
        logger.info(f"git_hash: {git_hash}")

    # Run benchmark
    t_all0 = time.time()
    per_video: List[VideoResult] = []

    # If demo_video provided, run ONLY that + write demo mp4 later (video writing optional)
    if args.demo_video.strip():
        demo_path = args.demo_video.strip()
        vr = run_one_video(
            yolo=yolo,
            embedder=embedder,
            bank=bank,
            video_path=demo_path,
            gt_dir=gt_dir,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=classes,
            frame_stride=args.frame_stride,
            max_dets_per_frame=args.max_dets_per_frame,
            fp16=bool(args.fp16),
            smooth=args.smooth,
            max_frames=args.max_frames,
            logger=logger,
        )
        per_video.append(vr)
    else:
        for vp in videos:
            vr = run_one_video(
                yolo=yolo,
                embedder=embedder,
                bank=bank,
                video_path=vp,
                gt_dir=gt_dir,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=classes,
                frame_stride=args.frame_stride,
                max_dets_per_frame=args.max_dets_per_frame,
                fp16=bool(args.fp16),
                smooth=args.smooth,
                max_frames=args.max_frames,
                logger=logger,
            )
            per_video.append(vr)

    wall_all = time.time() - t_all0

    # Write CSV (always includes gt 0/1/-1 and ORIGINAL frame_idx)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "score", "score_smooth", "num_dets", "gt"])
        for vr in per_video:
            for fi, sc, ss, nd, gt in zip(vr.frame_idxs, vr.scores, vr.scores_smooth, vr.num_dets, vr.gt):
                w.writerow([vr.video, int(fi), float(sc), float(ss), int(nd), int(gt)])

    # Aggregate metrics
    all_scores = np.concatenate([np.asarray(v.scores, np.float32) for v in per_video], axis=0)
    all_scores_s = np.concatenate([np.asarray(v.scores_smooth, np.float32) for v in per_video], axis=0)
    all_gt = np.concatenate([np.asarray(v.gt, np.int32) for v in per_video], axis=0)

    valid = all_gt >= 0
    y_true = all_gt[valid].astype(np.int32)
    y_raw = all_scores[valid]
    y_smooth = all_scores_s[valid]

    overall_auc_raw, overall_ap_raw = compute_auc_ap(y_true, y_raw)
    overall_auc_s, overall_ap_s = compute_auc_ap(y_true, y_smooth)

    frames_processed = int(sum(len(v.scores) for v in per_video))
    proc_fps = float(frames_processed / max(wall_all, 1e-9))

    # latency stats across all processed frames
    all_lat = []
    for v in per_video:
        all_lat.extend(v.latency_ms)
    lat_p50 = percentile(all_lat, 50.0)
    lat_p95 = percentile(all_lat, 95.0)

    # Per-video list for JSON
    per_video_json = []
    for v in per_video:
        per_video_json.append({
            "video": v.video,
            "num_rows": len(v.scores),
            "frames_processed": len(v.scores),
            "wall_time_s": v.wall_time_s,
            "proc_fps": v.proc_fps,
            "latency_ms_p50": percentile(v.latency_ms, 50.0),
            "latency_ms_p95": percentile(v.latency_ms, 95.0),
            "auc_raw": v.auc_raw,
            "ap_raw": v.ap_raw,
            "auc_smooth": v.auc_smooth,
            "ap_smooth": v.ap_smooth,
            "gt_counts": v.gt_counts,
        })

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

    logger.info("=== Avenue Benchmark Summary ===")
    logger.info(json.dumps({"overall": metrics_full["overall"], "runtime": metrics_full["runtime"]}, indent=2))
    logger.info(f"CSV:  {args.out_csv}")
    logger.info(f"JSON: {args.out_json}")



if __name__ == "__main__":
    main()
