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
import json
import math
import os
import re
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

from edgevad.data import AvenueVideoDataset

try:
    from ultralytics import YOLO
    import ultralytics
    _ultralytics_import_error: Optional[Exception] = None
except Exception as e:
    YOLO = None
    ultralytics = None
    _ultralytics_import_error = e

# Shared core imports
from edgevad.core import (
    utc_ts, get_git_hash, setup_logger, env_info,
    set_seed, percentile,
    l2_normalize, sanitize_array,
    compute_auc_ap, validate_metrics_schema,
    smooth_moving_average, gaussian_smooth,
    MobileNetV3SmallEmbedder, preprocess_crops_to_tensor, get_embedder,
    load_memory_bank_npz,
    parse_classes,
)
from edgevad.core.memory_bank import score_knn


# -------------------------
# GT Loader (Avenue-specific)
# -------------------------

def volLabel_to_frame_labels(vol):
    """
    Convert Avenue volLabel into per-frame binary labels:
      1 = abnormal frame, 0 = normal frame
    """
    vol = np.asarray(vol)

    if vol.dtype == object:
        cells = vol.ravel()
        labels = np.zeros(len(cells), dtype=np.uint8)
        for i, cell in enumerate(cells):
            m = np.asarray(cell)
            if m.dtype == object:
                m = np.asarray(m.tolist())
            labels[i] = 1 if (m > 0).any() else 0
        return labels

    if vol.ndim == 3:
        T = vol.shape[-1]
        return ((vol.reshape(-1, T).sum(axis=0) > 0).astype(np.uint8))
    if vol.ndim == 2:
        return ((vol.sum(axis=0) > 0).astype(np.uint8))

    flat = vol.ravel()
    return ((flat > 0).astype(np.uint8))


def label_path_for_video(gt_dir: str, video_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    m = re.search(r"(\d+)", stem)
    if not m:
        return None
    vid = int(m.group(1))
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
    mat = sio.loadmat(lp)
    if "volLabel" not in mat:
        return None
    labels = volLabel_to_frame_labels(mat["volLabel"])
    if total_frames is not None:
        if len(labels) < total_frames:
            labels = np.pad(labels, (0, total_frames - len(labels)), constant_values=0)
        elif len(labels) > total_frames:
            labels = labels[:total_frames]
    return labels


# -------------------------
# Data structures
# -------------------------

@dataclass
class VideoResult:
    video: str
    frame_idxs: List[int]
    scores: List[float]
    scores_smooth: List[float]
    num_dets: List[int]
    gt: List[int]
    wall_time_s: float
    proc_fps: float
    latency_ms: List[float]
    auc_raw: Optional[float]
    ap_raw: Optional[float]
    auc_smooth: Optional[float]
    ap_smooth: Optional[float]
    gt_counts: Dict[str, int]


# -------------------------
# Scoring
# -------------------------

def score_dataset(
    dataset: AvenueVideoDataset,
    yolo: Any,
    embedder: torch.nn.Module,
    bank: torch.Tensor,
    device: torch.device,
    imgsz: int,
    conf: float,
    classes: Optional[List[int]],
    max_dets_per_frame: int,
    fp16: bool,
    smooth: int,
    max_frames: int,
    expand: float,
    topk: int,
    smooth_sigma: float,
    min_box_area: float,
    logger,
    feature_mode: str = "mobilenet",
) -> List[VideoResult]:
    per_video: Dict[str, Dict[str, Any]] = {}

    for sample in dataset:
        video = sample["video"]
        frame_idx = int(sample["frame_idx"])
        label = int(sample["gt"])

        vdata = per_video.setdefault(
            video,
            {
                "frame_idxs": [],
                "scores": [],
                "num_dets": [],
                "gt": [],
                "latency_ms": [],
                "t_start": time.time(),
                "t_last": None,
                "last_score": 0.5,
            },
        )

        if max_frames > 0 and len(vdata["frame_idxs"]) >= max_frames:
            continue

        frame = sample["image"]
        t_frame0 = time.time()

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
            confs_arr = boxes.conf.detach().cpu().numpy()
            order = np.argsort(-confs_arr)
            for j in order[:max_dets_per_frame]:
                x1, y1, x2, y2 = xyxy[j].tolist()
                dets.append((x1, y1, x2, y2))

        crops = []
        filtered_boxes = []
        h, w = frame.shape[:2]
        for (x1, y1, x2, y2) in dets:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = (x2 - x1) * expand
            bh = (y2 - y1) * expand
            x1e = cx - 0.5 * bw
            y1e = cy - 0.5 * bh
            x2e = cx + 0.5 * bw
            y2e = cy + 0.5 * bh
            x1i = max(0, min(w - 1, int(x1e)))
            y1i = max(0, min(h - 1, int(y1e)))
            x2i = max(0, min(w, int(x2e)))
            y2i = max(0, min(h, int(y2e)))
            if x2i <= x1i or y2i <= y1i:
                continue
            area = float((x2i - x1i) * (y2i - y1i))
            if area < min_box_area:
                continue
            if feature_mode == "mobilenet":
                crops.append(frame[y1i:y2i, x1i:x2i].copy())
            filtered_boxes.append([x1i, y1i, x2i, y2i])

        nd = len(filtered_boxes)
        if nd > 0:
            with torch.no_grad():
                if feature_mode == "fpn":
                    boxes_xyxy = np.array(filtered_boxes, dtype=np.float64)
                    emb = embedder(frame, boxes_xyxy, imgsz=imgsz)
                else:
                    x = preprocess_crops_to_tensor(crops, device=device, fp16=fp16)
                    emb = embedder(x)
                emb = l2_normalize(emb.to(dtype=bank.dtype))
                per_det_scores = score_knn(emb, bank, topk=topk)
                score = float(per_det_scores.max().item())
        else:
            score = vdata["last_score"]

        if not math.isfinite(score):
            logger.warning(f"{video} frame {frame_idx}: non-finite score -> carry-forward")
            score = vdata["last_score"]

        vdata["last_score"] = score
        vdata["frame_idxs"].append(frame_idx)
        vdata["scores"].append(score)
        vdata["num_dets"].append(nd)
        vdata["gt"].append(label)
        vdata["latency_ms"].append((time.time() - t_frame0) * 1000.0)
        vdata["t_last"] = time.time()

    results: List[VideoResult] = []
    for video, data in per_video.items():
        scores_np = sanitize_array(data["scores"], f"{video}.scores_raw", logger)
        if smooth_sigma > 0:
            scores_s_arr = gaussian_smooth(scores_np, smooth_sigma)
        else:
            scores_s_arr = smooth_moving_average(scores_np, smooth)
        scores_s = sanitize_array(scores_s_arr, f"{video}.scores_smooth", logger).tolist()

        gt_np = np.asarray(data["gt"], dtype=np.int32)
        valid = gt_np >= 0
        y_true = gt_np[valid].astype(np.int32)
        y_raw = scores_np[valid]
        y_smooth = np.asarray(scores_s, dtype=np.float32)[valid]

        auc_raw, ap_raw, _ = compute_auc_ap(y_true, y_raw)
        auc_s, ap_s, _ = compute_auc_ap(y_true, y_smooth)

        gt_counts = {
            "-1": int((gt_np == -1).sum()),
            "0": int((gt_np == 0).sum()),
            "1": int((gt_np == 1).sum()),
        }
        if gt_counts["1"] == 0 and gt_counts["0"] > 0 and gt_counts["-1"] == 0:
            logger.warning(f"{video}: GT has 0 positives (all normal).")

        t_start = data["t_start"]
        t_end = data["t_last"] if data["t_last"] is not None else t_start
        wall = max(t_end - t_start, 1e-9)
        proc_fps = len(data["scores"]) / wall

        results.append(
            VideoResult(
                video=video,
                frame_idxs=data["frame_idxs"],
                scores=scores_np.tolist(),
                scores_smooth=scores_s,
                num_dets=data["num_dets"],
                gt=data["gt"],
                wall_time_s=float(wall),
                proc_fps=float(proc_fps),
                latency_ms=data["latency_ms"],
                auc_raw=auc_raw,
                ap_raw=ap_raw,
                auc_smooth=auc_s,
                ap_smooth=ap_s,
                gt_counts=gt_counts,
            )
        )

    return results


# -------------------------
# CLI + main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    ap.add_argument("--mb", required=True, help="Memory bank .npz (embeddings)")
    ap.add_argument("--test_dir", required=True, help="Avenue test videos directory")
    ap.add_argument("--gt_dir", default="", help="Avenue GT dir: ground_truth_demo/testing_label_mask")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="0", help="YOLO class ids, e.g. '0' for person.")
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--max_dets_per_frame", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--expand", type=float, default=1.05, help="Box expansion factor (match build_memory_bank)")
    ap.add_argument("--max_frames", type=int, default=0, help="Max processed frames per video (0 = no limit)")
    ap.add_argument("--topk", type=int, default=5, help="k-NN neighbors for scoring (1=old max-sim)")
    ap.add_argument("--smooth_sigma", type=float, default=3.0, help="Gaussian smooth sigma. 0=use --smooth window.")
    ap.add_argument("--min_box_area", type=float, default=400.0, help="Min box area (px^2) to score")
    ap.add_argument("--feature_mode", choices=["mobilenet", "fpn"], default="mobilenet",
                    help="Feature extraction: mobilenet (crop+embed) or fpn (YOLO FPN RoI-Align)")
    ap.add_argument("--demo_video", default="")
    ap.add_argument("--save_video", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True, help="metrics_full.json output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_file", default="", help="Optional log file path")

    args = ap.parse_args()

    logger = setup_logger(name="score_avenue", log_file=args.log_file if args.log_file.strip() else None)
    set_seed(args.seed, args.deterministic, logger)

    device = torch.device(args.device)
    classes = parse_classes(args.classes)
    gt_dir = args.gt_dir.strip() or None

    repo_dir = str(Path(__file__).resolve().parents[2])
    git_hash = get_git_hash(repo_dir)

    bank, mb_meta = load_memory_bank_npz(args.mb, device=device, fp16=args.fp16)

    if YOLO is None:
        raise RuntimeError(f"Failed to import ultralytics/YOLO. Error: {_ultralytics_import_error}")
    yolo = YOLO(args.yolo)
    embedder = get_embedder(
        feature_mode=args.feature_mode,
        yolo_model=yolo if args.feature_mode == "fpn" else None,
        device=device,
    )
    if args.fp16 and args.feature_mode == "mobilenet":
        embedder.half()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise RuntimeError(f"--test_dir is not a directory: {test_dir}")

    video_paths = None
    if args.demo_video.strip():
        demo_path = str(Path(args.demo_video).resolve())
        video_paths = [demo_path]

    dataset = AvenueVideoDataset(
        video_dir=str(test_dir),
        gt_dir=gt_dir,
        split="test",
        imgsz=args.imgsz,
        frame_stride=args.frame_stride,
        max_frames=None,
        start_offset=0,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        return_rgb=False,
        transform=lambda x: x,
        video_paths=video_paths,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No .avi videos found in: {test_dir}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    if args.save_video.strip():
        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)

    t_all0 = time.time()
    logger.info(f"dataset videos={len(dataset.videos)} samples={len(dataset)} frame_stride={args.frame_stride} max_frames={args.max_frames} topk={args.topk} smooth_sigma={args.smooth_sigma}")

    per_video: List[VideoResult] = score_dataset(
        dataset=dataset,
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
        max_frames=args.max_frames,
        expand=args.expand,
        topk=args.topk,
        smooth_sigma=args.smooth_sigma,
        min_box_area=args.min_box_area,
        logger=logger,
        feature_mode=args.feature_mode,
    )

    wall_all = time.time() - t_all0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "score", "score_smooth", "num_dets", "gt"])
        for vr in per_video:
            scores_csv = sanitize_array(vr.scores, f"{vr.video}.scores_csv", logger)
            scores_s_csv = sanitize_array(vr.scores_smooth, f"{vr.video}.scores_smooth_csv", logger)
            assert len(scores_csv) == len(scores_s_csv) == len(vr.frame_idxs) == len(vr.num_dets) == len(vr.gt)
            for fi, sc, ss, nd, gt in zip(vr.frame_idxs, scores_csv, scores_s_csv, vr.num_dets, vr.gt):
                w.writerow([vr.video, int(fi), float(sc), float(ss), int(nd), int(gt)])

    all_scores = sanitize_array(
        np.concatenate([np.asarray(v.scores, np.float32) for v in per_video], axis=0),
        "all_scores_raw", logger,
    )
    all_scores_s = sanitize_array(
        np.concatenate([np.asarray(v.scores_smooth, np.float32) for v in per_video], axis=0),
        "all_scores_smooth", logger,
    )
    all_gt = np.concatenate([np.asarray(v.gt, np.int32) for v in per_video], axis=0)

    valid = all_gt >= 0
    y_true = all_gt[valid].astype(np.int32)
    y_raw = all_scores[valid]
    y_smooth = all_scores_s[valid]

    overall_auc_raw, overall_ap_raw, _ = compute_auc_ap(y_true, y_raw)
    overall_auc_s, overall_ap_s, _ = compute_auc_ap(y_true, y_smooth)

    frames_processed = int(sum(len(v.scores) for v in per_video))
    proc_fps = float(frames_processed / max(wall_all, 1e-9))

    all_lat = []
    for v in per_video:
        all_lat.extend(v.latency_ms)
    lat_p50 = percentile(all_lat, 50.0)
    lat_p95 = percentile(all_lat, 95.0)

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
