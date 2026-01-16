import argparse
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Deque, Tuple, Optional

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO


def pct(values, p):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def safe_fps(cap, default=25.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        return float(default)
    return float(fps)


def open_source(src: str):
    if src == "0":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(src)


def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    # ensure proper ordering
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_and_resize(frame, xyxy, out_size):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0",
                    help="0=webcam, /path/video.mp4, or rtsp://...")
    ap.add_argument("--yolo", type=str, default="yolo26n.pt",
                    help="detector weights path or name (e.g., yolo26n.pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=str, default="0",
                    help="comma-separated class ids. Use empty string '' for ALL classes.")
    ap.add_argument("--out_json", type=str, default="runs/events.jsonl",
                    help="per-frame heartbeat + optional per-det events (JSONL)")
    ap.add_argument("--out_clip_json", type=str, default="runs/clip_events.jsonl",
                    help="per-clip events (JSONL)")
    ap.add_argument("--save_video", type=str, default="runs/out.mp4",
                    help="output video path; set empty '' to disable saving")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI window (recommended on servers)")
    ap.add_argument("--max_frames", type=int, default=-1,
                    help="stop after N frames (-1=all)")

    # Clip/tracklet buffering
    ap.add_argument("--clip_len", type=int, default=16,
                    help="frames per clip per track")
    ap.add_argument("--clip_stride", type=int, default=2,
                    help="emit clip every N frames (per track), after buffer is full")
    ap.add_argument("--crop_size", type=int, default=224,
                    help="crop resize size for buffered clip frames")
    ap.add_argument("--track_ttl", type=int, default=50,
                    help="drop track buffer if not seen for N frames")

    # Logging control
    ap.add_argument("--log_dets", action="store_true",
                    help="also log per-detection lines to out_json (in addition to heartbeat).")
    args = ap.parse_args()

    # Parse classes: empty => None (means all classes)
    classes = None
    if args.classes.strip():
        classes = [int(x) for x in args.classes.split(",") if x.strip() != ""]

    Path("runs").mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_clip_json = Path(args.out_clip_json)
    out_clip_json.parent.mkdir(parents=True, exist_ok=True)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    model = YOLO(args.yolo)

    # writer (optional)
    writer = None
    save_path = args.save_video.strip()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # files
    f = open(out_json, "a", encoding="utf-8")
    cf = open(out_clip_json, "a", encoding="utf-8")

    # Track buffers: track_id -> deque of (frame_idx, crop_img)
    track_buf: Dict[int, Deque[Tuple[int, np.ndarray]]] = defaultdict(lambda: deque(maxlen=args.clip_len))
    last_seen: Dict[int, int] = {}

    # Stats
    lat_ms = []
    t0_wall = time.time()
    frame_idx = 0
    fps_ema = None
    alpha = 0.05

    src_fps = safe_fps(cap)

    # Main loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        t0 = time.perf_counter()

        # init writer on first frame
        if writer is None and save_path:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, src_fps, (w, h))

        # YOLO track call (conditionally pass classes)
        track_kwargs = dict(
            persist=True,
            verbose=False,
            imgsz=args.imgsz,
            conf=args.conf,
        )
        if classes is not None:
            track_kwargs["classes"] = classes

        results = model.track(frame, **track_kwargs)
        r = results[0]

        # Timing / FPS
        dt = time.perf_counter() - t0
        inst_fps = 1.0 / max(dt, 1e-9)
        fps_ema = inst_fps if fps_ema is None else (1 - alpha) * fps_ema + alpha * inst_fps

        # Extract boxes
        num_dets = 0
        det_boxes = None
        det_confs = None
        det_ids = None

        if r.boxes is not None and r.boxes.xyxy is not None:
            det_boxes = r.boxes.xyxy.detach().cpu().numpy()
            num_dets = len(det_boxes)
            det_confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.zeros((num_dets,))
            det_ids = r.boxes.id.detach().cpu().numpy().astype(int) if r.boxes.id is not None else None

        # HEARTBEAT LOG (always)
        f.write(json.dumps({
            "type": "frame",
            "ts": float(time.time()),
            "source": str(args.source),
            "frame_idx": int(frame_idx),
            "num_dets": int(num_dets),
            "fps": float(fps_ema),
        }) + "\n")

        # If we have detections, draw + (optional) per-det logs + tracklet buffering
        if num_dets > 0 and det_boxes is not None:
            for i, bbox in enumerate(det_boxes):
                cfv = float(det_confs[i]) if det_confs is not None and i < len(det_confs) else 0.0
                tid = int(det_ids[i]) if det_ids is not None else -1

                x1, y1, x2, y2 = bbox.tolist()

                # Draw (for saved video)
                ix1, iy1, ix2, iy2 = [int(v) for v in (x1, y1, x2, y2)]
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid} {cfv:.2f}", (ix1, max(0, iy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Optional per-detection log line
                if args.log_dets:
                    f.write(json.dumps({
                        "type": "det",
                        "ts": float(time.time()),
                        "source": str(args.source),
                        "frame_idx": int(frame_idx),
                        "track_id": int(tid),
                        "conf": float(cfv),
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "fps": float(fps_ema),
                    }) + "\n")

                # Tracklet buffering only if tracker provides a valid id
                if tid >= 0:
                    crop = crop_and_resize(frame, (x1, y1, x2, y2), args.crop_size)
                    if crop is not None:
                        track_buf[tid].append((frame_idx, crop))
                        last_seen[tid] = frame_idx

                        # Emit clip event when buffer is full and stride condition met
                        if len(track_buf[tid]) == args.clip_len and (frame_idx % args.clip_stride == 0):
                            start_f = int(track_buf[tid][0][0])
                            end_f = int(track_buf[tid][-1][0])

                            # Dummy score for now (replace later with real scoring)
                            score = 0.0

                            cf.write(json.dumps({
                                "type": "clip",
                                "ts": float(time.time()),
                                "source": str(args.source),
                                "track_id": int(tid),
                                "frame_idx": int(frame_idx),
                                "clip_len": int(args.clip_len),
                                "clip_stride": int(args.clip_stride),
                                "crop_size": int(args.crop_size),
                                "start_frame": start_f,
                                "end_frame": end_f,
                                "score": float(score),
                            }) + "\n")

        # Cleanup stale tracks (not seen recently)
        if args.track_ttl > 0:
            to_drop = [tid for tid, last in last_seen.items() if (frame_idx - last) > args.track_ttl]
            for tid in to_drop:
                last_seen.pop(tid, None)
                track_buf.pop(tid, None)

        # Overlay FPS
        cv2.putText(frame, f"FPS {fps_ema:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Write video
        if writer is not None:
            writer.write(frame)

        # Show if not headless
        if not args.headless:
            cv2.imshow("EdgeVAD (tracklet MVP)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    # Cleanup
    f.flush()
    f.close()
    cf.flush()
    cf.close()
    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # Summary
    elapsed = time.time() - t0_wall
    avg_fps = frame_idx / max(elapsed, 1e-9)

    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    gpu_mem_mb = None
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("=== EdgeVAD Tracklet MVP Summary ===")
    print(f"frames: {frame_idx}")
    print(f"avg_fps: {avg_fps:.2f}")
    print(f"latency_ms p50: {pct(lat_ms, 50):.2f}  p95: {pct(lat_ms, 95):.2f}")
    print(f"cpu_rss_mb: {rss_mb:.1f}")
    if gpu_mem_mb is not None:
        print(f"gpu_mem_alloc_mb: {gpu_mem_mb:.1f}")
    print(f"events_jsonl: {out_json}")
    print(f"clip_events_jsonl: {out_clip_json}")
    if save_path:
        print(f"saved_video: {save_path}")


if __name__ == "__main__":
    main()
