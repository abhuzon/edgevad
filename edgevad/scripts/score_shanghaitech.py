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
import json
import math
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

from edgevad.data.shanghaitech_dataset import (
    get_gt_per_frame,
    list_videos,
)

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
    extract_scene_id,
)
from edgevad.core.memory_bank import score_knn, build_scene_banks


# -------------------------
# ShanghaiTech-specific helpers
# -------------------------


def _natural_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.stem)
    key: List[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        elif p:
            key.append(p.lower())
    return key


def iter_frames(
    video_path_or_dir: str, max_frames: int = 0, stride: int = 1
) -> Iterator[tuple[int, np.ndarray]]:
    path = Path(video_path_or_dir)
    logger = logging.getLogger("score_shanghaitech")

    if path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_idx = 0
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % stride == 0:
                yield frame_idx, frame
                yielded += 1
                if max_frames > 0 and yielded >= max_frames:
                    break
            frame_idx += 1
        cap.release()
        return

    if path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png"}
        frames = [
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in image_exts
        ]
        frames = sorted(frames, key=_natural_key)
        if not frames:
            raise FileNotFoundError(f"No frame images found under: {path}")
        yielded = 0
        for frame_idx, frame_path in enumerate(frames):
            if frame_idx % stride != 0:
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"{path.name} frame {frame_idx}: failed to read {frame_path}")
                continue
            yield frame_idx, frame
            yielded += 1
            if max_frames > 0 and yielded >= max_frames:
                break
        return

    raise FileNotFoundError(f"Path is neither file nor directory: {path}")


# -------------------------
# Data structures
# -------------------------


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
# Main scoring logic
# -------------------------


def score_videos(
    video_paths: List[str],
    frame_stride: int,
    max_frames: int,
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
    expand: float,
    gt_dir: Optional[str],
    topk: int,
    smooth_sigma: float,
    min_box_area: float,
    logger: logging.Logger,
    scene_banks: Optional[Dict[str, torch.Tensor]] = None,
    feature_mode: str = "mobilenet",
) -> List[VideoResult]:
    results: List[VideoResult] = []

    for video_path in video_paths:
        path = Path(video_path)
        clip_id = path.stem if path.is_file() else path.name

        # Select effective bank: per-scene or global
        effective_bank = bank
        if scene_banks is not None:
            scene_id = extract_scene_id(clip_id)
            if scene_id is not None and scene_id in scene_banks:
                effective_bank = scene_banks[scene_id]
            else:
                logger.warning(
                    f"{clip_id}: scene '{scene_id}' not found in scene_banks, using global bank"
                )

        gt_labels = None
        if gt_dir:
            try:
                gt_labels = get_gt_per_frame(gt_dir, clip_id)
            except Exception as e:
                logger.warning(f"Failed to load GT for {clip_id}: {e}")

        frames: List[FrameResult] = []
        last_score = 0.5  # neutral fallback for no-detection frames
        t_start = time.time()

        for frame_idx, frame in iter_frames(
            video_path_or_dir=video_path,
            max_frames=max_frames,
            stride=frame_stride,
        ):
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
                confs_arr = boxes.conf.detach().cpu().numpy()
                order = np.argsort(-confs_arr)
                for j in order[:max_dets_per_frame]:
                    dets.append(tuple(xyxy[j].tolist()))

            crops = []
            filtered_boxes = []
            h, w = frame.shape[:2]
            for (x1, y1, x2, y2) in dets:
                # Expand box to match build-time expansion
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
                    emb = l2_normalize(emb.to(dtype=effective_bank.dtype))
                    per_det_scores = score_knn(emb, effective_bank, topk=topk)
                    score = float(per_det_scores.max().item())
            else:
                # No detections: carry forward last score (avoids false-normal bias)
                score = last_score

            if not math.isfinite(score):
                logger.warning(f"{clip_id} frame {frame_idx}: non-finite score -> carry-forward")
                score = last_score

            last_score = score

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

        # smoothing and metrics
        scores_np = sanitize_array(
            [f.score for f in frames], f"{clip_id}.scores_raw", logger, posinf=1.0
        )
        if smooth_sigma > 0:
            scores_smooth_arr = gaussian_smooth(scores_np, smooth_sigma)
        else:
            scores_smooth_arr = smooth_moving_average(scores_np, smooth)
        scores_smooth_arr = sanitize_array(
            scores_smooth_arr, f"{clip_id}.scores_smooth", logger, posinf=1.0
        )
        for f, s in zip(frames, scores_smooth_arr):
            f.score_smooth = float(s)

        gt_np = np.asarray([f.gt for f in frames], dtype=np.int32)
        valid = gt_np >= 0
        y_true = gt_np[valid].astype(np.int32)
        y_raw = scores_np[valid]
        y_smooth = scores_smooth_arr[valid]

        auc_raw, ap_raw, reason_raw = compute_auc_ap(y_true, y_raw)
        auc_s, ap_s, reason_s = compute_auc_ap(y_true, y_smooth)

        gt_counts = {
            "-1": int((gt_np == -1).sum()),
            "0": int((gt_np == 0).sum()),
            "1": int((gt_np == 1).sum()),
        }

        wall = max(time.time() - t_start, 1e-9)
        proc_fps = len(frames) / wall if wall > 0 else 0.0

        results.append(
            VideoResult(
                video=clip_id,
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
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Score ShanghaiTech dataset with EdgeVAD")
    ap.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    ap.add_argument("--mb", required=True, help="Memory bank .npz (embeddings)")
    ap.add_argument("--test_dir", required=True, help="ShanghaiTech testing/frames directory")
    ap.add_argument(
        "--gt_dir",
        default="",
        help="Optional GT root/testing directory (testing/test_frame_mask and/or testing/test_pixel_mask)",
    )
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="0", help="YOLO class ids, e.g. '0' for person. Empty disables filter.")
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--max_dets_per_frame", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--expand", type=float, default=1.05, help="Box expansion factor (match build_memory_bank)")
    ap.add_argument("--max_frames", type=int, default=0, help="Max processed frames per video (0 = no limit)")
    ap.add_argument("--topk", type=int, default=5, help="k-NN neighbors for scoring (1=old max-sim)")
    ap.add_argument("--smooth_sigma", type=float, default=3.0, help="Gaussian smooth sigma. 0=use --smooth window.")
    ap.add_argument("--min_box_area", type=float, default=400.0, help="Min box area (px^2) to score")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True, help="metrics_full.json output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log_file", default="", help="Optional log file path")
    ap.add_argument(
        "--per_scene",
        action="store_true",
        help="Use per-scene memory banks (requires bank built with --per_scene).",
    )
    ap.add_argument(
        "--feature_mode",
        choices=["mobilenet", "fpn"],
        default="mobilenet",
        help="Feature extraction: mobilenet (crop+embed) or fpn (YOLO FPN RoI-Align)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger(
        name="score_shanghaitech",
        log_file=args.log_file if args.log_file.strip() else None,
    )
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
        raise RuntimeError(f"No test videos or frame directories found in: {test_dir}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    bank, mb_meta = load_memory_bank_npz(args.mb, device=device, fp16=args.fp16)

    scene_banks: Optional[Dict[str, torch.Tensor]] = None
    if args.per_scene:
        if "scene_ids" not in mb_meta:
            raise RuntimeError(
                "--per_scene requires a memory bank built with --per_scene "
                "(must contain scene_ids). Rebuild with: build_memory_bank.py --per_scene"
            )
        scene_banks = build_scene_banks(bank, mb_meta["scene_ids"])
        logger.info(
            f"Per-scene banks: {len(scene_banks)} scenes, "
            + ", ".join(f"{k}={v.shape[0]}" for k, v in sorted(scene_banks.items()))
        )

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

    t_all0 = time.time()
    logger.info(
        f"dataset items={len(video_paths)} frame_stride={args.frame_stride} "
        f"max_frames={args.max_frames} imgsz={args.imgsz} topk={args.topk} "
        f"smooth_sigma={args.smooth_sigma} min_box_area={args.min_box_area}"
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
        expand=args.expand,
        gt_dir=gt_dir,
        topk=args.topk,
        smooth_sigma=args.smooth_sigma,
        min_box_area=args.min_box_area,
        logger=logger,
        scene_banks=scene_banks,
        feature_mode=args.feature_mode,
    )

    wall_all = time.time() - t_all0

    # write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "score", "score_smooth", "num_dets", "gt"])
        for vr in per_video:
            scores_csv = sanitize_array(
                [fr.score for fr in vr.frames], f"{vr.video}.scores_csv", logger, posinf=1.0
            )
            scores_s_csv = sanitize_array(
                [fr.score_smooth for fr in vr.frames],
                f"{vr.video}.scores_smooth_csv",
                logger,
                posinf=1.0,
            )
            assert len(scores_csv) == len(scores_s_csv) == len(vr.frames)
            for fr, sc, ss in zip(vr.frames, scores_csv, scores_s_csv):
                w.writerow([vr.video, fr.frame_idx, float(sc), float(ss), fr.num_dets, fr.gt])

    all_scores = sanitize_array(
        np.concatenate([np.asarray([fr.score for fr in v.frames], np.float32) for v in per_video], axis=0),
        "all_scores_raw",
        logger,
        posinf=1.0,
    )
    all_scores_s = sanitize_array(
        np.concatenate([np.asarray([fr.score_smooth for fr in v.frames], np.float32) for v in per_video], axis=0),
        "all_scores_smooth",
        logger,
        posinf=1.0,
    )
    all_gt = np.concatenate([np.asarray([fr.gt for fr in v.frames], np.int32) for v in per_video], axis=0)

    valid = all_gt >= 0
    y_true = all_gt[valid].astype(np.int32)
    y_raw = all_scores[valid]
    y_smooth = all_scores_s[valid]

    overall_auc_raw, overall_ap_raw, overall_reason_raw = compute_auc_ap(y_true, y_raw)
    overall_auc_s, overall_ap_s, overall_reason_s = compute_auc_ap(y_true, y_smooth)

    frames_processed = int(sum(len(v.frames) for v in per_video))
    proc_fps = float(frames_processed / max(wall_all, 1e-9))

    all_lat = []
    for v in per_video:
        all_lat.extend([fr.latency_ms for fr in v.frames])
    lat_p50 = percentile(all_lat, 50.0)
    lat_p95 = percentile(all_lat, 95.0)

    per_video_json = []
    for v in per_video:
        gt_np = np.asarray([fr.gt for fr in v.frames], np.int32)
        valid_v = gt_np >= 0
        y_true_v = gt_np[valid_v].astype(np.int32)
        y_raw_v = sanitize_array(
            [fr.score for fr in v.frames], f"{v.video}.scores_raw_metrics", logger, posinf=1.0
        )[valid_v]
        y_smooth_v = sanitize_array(
            [fr.score_smooth for fr in v.frames],
            f"{v.video}.scores_smooth_metrics",
            logger,
            posinf=1.0,
        )[valid_v]
        auc_raw_v, ap_raw_v, reason_raw_v = compute_auc_ap(y_true_v, y_raw_v)
        auc_s_v, ap_s_v, reason_s_v = compute_auc_ap(y_true_v, y_smooth_v)
        per_video_json.append(
            {
                "video": v.video,
                "num_rows": len(v.frames),
                "frames_processed": len(v.frames),
                "wall_time_s": v.wall_time_s,
                "proc_fps": v.proc_fps,
                "latency_ms_p50": percentile([fr.latency_ms for fr in v.frames], 50.0),
                "latency_ms_p95": percentile([fr.latency_ms for fr in v.frames], 95.0),
                "auc_raw": auc_raw_v,
                "ap_raw": ap_raw_v,
                "auc_smooth": auc_s_v,
                "ap_smooth": ap_s_v,
                "auc_raw_reason": reason_raw_v,
                "ap_raw_reason": reason_raw_v,
                "auc_smooth_reason": reason_s_v,
                "ap_smooth_reason": reason_s_v,
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
            "per_scene": scene_banks is not None,
            "scene_bank_sizes": (
                {k: int(v.shape[0]) for k, v in sorted(scene_banks.items())}
                if scene_banks is not None
                else None
            ),
        },
        "overall": {
            "auc_raw": overall_auc_raw,
            "auc_smooth": overall_auc_s,
            "ap_raw": overall_ap_raw,
            "ap_smooth": overall_ap_s,
            "auc_raw_reason": overall_reason_raw,
            "ap_raw_reason": overall_reason_raw,
            "auc_smooth_reason": overall_reason_s,
            "ap_smooth_reason": overall_reason_s,
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
