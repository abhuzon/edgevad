#!/usr/bin/env python3
"""EdgeVAD - Build a person-crop embedding memory bank from training videos.

What it does:
- Loads a YOLO detector.
- Iterates training_videos/*.avi.
- Every --frame_stride frames: detect persons -> crop -> embed -> store vector.
- Optionally applies coreset farthest-point selection to reduce bank size.
- Saves NPZ (embeddings + minimal metadata).

Multi-GPU (simple + robust): run 2 processes with sharding:
  --shard_id 0 --num_shards 2  (GPU0)
  --shard_id 1 --num_shards 2  (GPU1)
Then merge NPZ files.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

from edgevad.data import AvenueVideoDataset, list_videos
from edgevad.core import (
    get_embedder,
    preprocess_crops_to_tensor,
    set_seed,
    parse_classes,
    extract_scene_id,
)
from edgevad.core.memory_bank import coreset_farthest_point


def safe_mkdir(p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def clip_xyxy(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser(
        "EdgeVAD - build memory bank from training videos (person crops)."
    )

    ap.add_argument("--yolo", type=str, required=True, help="Path to YOLO weights (.pt).")
    ap.add_argument(
        "--train_dir", type=str, required=True, help="Directory containing training_videos/*.avi"
    )
    ap.add_argument("--out", type=str, required=True, help="Output .npz path (memory bank)")

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--classes",
        type=str,
        default="0",
        help="Comma-separated class ids. Default person=0.",
    )
    ap.add_argument(
        "--frame_stride", type=int, default=3, help="Process 1 frame every N frames."
    )
    ap.add_argument("--max_dets_per_frame", type=int, default=3)
    ap.add_argument("--max_embeddings", type=int, default=30000)

    ap.add_argument(
        "--min_box_area", type=float, default=20 * 20, help="Skip tiny boxes (px^2)."
    )
    ap.add_argument("--expand", type=float, default=1.05, help="Box expansion factor.")
    ap.add_argument("--batch_size", type=int, default=16, help="YOLO inference batch size (frames per batch). Higher = faster but more VRAM.")
    ap.add_argument(
        "--device", type=str, default="cuda", help="cuda / cpu / cuda:0 etc."
    )
    ap.add_argument("--fp16", action="store_true", help="Use autocast fp16 for embedder forward.")
    ap.add_argument(
        "--save_fp16", action="store_true", help="Store embeddings as float16 in NPZ."
    )
    ap.add_argument("--seed", type=int, default=1337)

    # Embedder options
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="Try torchvision pretrained weights (may download).",
    )
    ap.add_argument(
        "--embedder_weights",
        type=str,
        default="",
        help="Local path to embedder weights for offline use.",
    )

    # Coreset selection
    ap.add_argument(
        "--coreset_size",
        type=int,
        default=5000,
        help="Coreset size for farthest-point selection. 0=no coreset, keep all.",
    )
    ap.add_argument(
        "--per_scene",
        action="store_true",
        help="Apply coreset per-scene (ShanghaiTech). Each scene gets up to --coreset_size embeddings.",
    )
    ap.add_argument(
        "--feature_mode",
        choices=["mobilenet", "fpn"],
        default="mobilenet",
        help="Feature extraction: mobilenet (crop+embed) or fpn (YOLO FPN RoI-Align)",
    )

    # Sharding (for multi-GPU via multi-process)
    ap.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard index [0..num_shards-1]",
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total shards",
    )

    args = ap.parse_args()

    # When --per_scene, disable the global embedding cap: per-scene coreset handles size reduction.
    if args.per_scene:
        args.max_embeddings = 0  # 0 = unlimited

    if args.feature_mode == "fpn" and args.batch_size > 1:
        print("[INFO] FPN mode requires batch_size=1, overriding.", file=sys.stderr)
        args.batch_size = 1

    set_seed(args.seed, deterministic=True)

    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    safe_mkdir(args.out)

    # Resolve device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Detector
    det = YOLO(args.yolo)
    classes = parse_classes(args.classes)

    # Embedder (unified factory)
    embedder = get_embedder(
        feature_mode=args.feature_mode,
        yolo_model=det if args.feature_mode == "fpn" else None,
        device=device,
        pretrained=args.pretrained,
        weights_path=args.embedder_weights,
    )

    # Videos + shard (deterministic ordering)
    all_videos = list_videos(str(train_dir), ext=".avi")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("Invalid shard_id/num_shards")
    video_paths = all_videos[args.shard_id :: args.num_shards]
    if len(video_paths) == 0:
        raise RuntimeError(f"No .avi videos found under: {train_dir} for shard {args.shard_id}")

    print(
        f"[INFO] shard {args.shard_id}/{args.num_shards} videos={len(video_paths)} frame_stride={args.frame_stride} max_embeddings={args.max_embeddings}"
    )
    print(
        f"[INFO] device={device} fp16={bool(args.fp16)} save_fp16={bool(args.save_fp16)} embedder_dim=576 coreset_size={args.coreset_size} feature_mode={args.feature_mode}"
    )

    # Dataset driving frame sampling (BGR frames, identity transform)
    dataset = AvenueVideoDataset(
        video_dir=str(train_dir),
        gt_dir=None,
        split="train",
        frame_stride=args.frame_stride,
        max_frames=None,
        start_offset=0,
        seed=args.seed,
        deterministic=True,
        return_rgb=False,
        transform=lambda x: x,
        video_paths=video_paths,
    )

    embs = []
    meta_video = []
    meta_frame = []
    meta_xyxy = []

    t0 = time.time()
    total_frames_seen = 0
    total_frames_proc = 0
    total_boxes = 0

    # Batching state
    frame_batch = []
    frame_batch_meta = []  # (video_name, frame_idx, H, W)

    def process_yolo_batch():
        """Process accumulated batch of frames through YOLO."""
        nonlocal total_boxes, total_frames_proc

        if len(frame_batch) == 0:
            return

        # Run YOLO on batch
        with torch.no_grad():
            results = det.predict(
                source=frame_batch,  # List of frames
                imgsz=args.imgsz,
                conf=args.conf,
                classes=classes,
                device=str(device),
                verbose=False,
            )

        # Process each result
        for frame, (video_name, frame_idx, H, W), result in zip(frame_batch, frame_batch_meta, results):
            total_frames_proc += 1

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = (
                boxes.conf.detach().cpu().numpy()
                if boxes.conf is not None
                else np.ones((xyxy.shape[0],), dtype=np.float32)
            )

            order = np.argsort(-confs)
            xyxy = xyxy[order]
            confs = confs[order]

            crops = []
            crop_meta = []
            kept = 0
            for i in range(xyxy.shape[0]):
                if kept >= args.max_dets_per_frame:
                    break

                x1, y1, x2, y2 = xyxy[i].tolist()

                # Expand box
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                bw = (x2 - x1) * args.expand
                bh = (y2 - y1) * args.expand
                x1e = cx - 0.5 * bw
                y1e = cy - 0.5 * bh
                x2e = cx + 0.5 * bw
                y2e = cy + 0.5 * bh

                x1i, y1i, x2i, y2i = clip_xyxy((x1e, y1e, x2e, y2e), W, H)
                area = float((x2i - x1i) * (y2i - y1i))
                if area < args.min_box_area:
                    continue

                if args.feature_mode == "mobilenet":
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size == 0:
                        continue
                    crops.append(crop.copy())
                crop_meta.append((video_name, frame_idx, (x1i, y1i, x2i, y2i)))
                kept += 1

            if len(crop_meta) == 0:
                continue

            # Embed detections
            if args.feature_mode == "fpn":
                boxes_arr = np.array(
                    [(x1, y1, x2, y2) for _, _, (x1, y1, x2, y2) in crop_meta],
                    dtype=np.float64,
                )
                total_boxes += len(boxes_arr)
                with torch.no_grad():
                    feats = embedder(frame, boxes_arr, imgsz=args.imgsz)
                feats = feats.detach().cpu().numpy()
            else:
                batch_tensors = preprocess_crops_to_tensor(crops, device=device, fp16=args.fp16)
                total_boxes += len(crops)
                with torch.no_grad():
                    if args.fp16 and device.type == "cuda":
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            feats = embedder(batch_tensors)
                    else:
                        feats = embedder(batch_tensors)
                feats = feats.detach().cpu().numpy()

            for j in range(feats.shape[0]):
                embs.append(feats[j])
                meta_video.append(crop_meta[j][0])
                meta_frame.append(crop_meta[j][1])
                meta_xyxy.append(crop_meta[j][2])

                if args.max_embeddings > 0 and len(embs) >= args.max_embeddings:
                    break

            if args.max_embeddings > 0 and len(embs) >= args.max_embeddings:
                break

        # Clear batch
        frame_batch.clear()
        frame_batch_meta.clear()

    for sample in tqdm(dataset, desc=f"frames(shard={args.shard_id})"):
        frame = sample["image"]  # BGR frame (transform is identity)
        video_name = sample["video"]
        frame_idx = sample["frame_idx"]
        total_frames_seen += 1

        H, W = frame.shape[:2]

        # Add to batch
        frame_batch.append(frame)
        frame_batch_meta.append((video_name, frame_idx, H, W))

        # Process batch when full
        if len(frame_batch) >= args.batch_size:
            process_yolo_batch()

            # Check if done
            if args.max_embeddings > 0 and len(embs) >= args.max_embeddings:
                break

    # Process remaining frames in batch
    if len(frame_batch) > 0:
        process_yolo_batch()

    if len(embs) == 0:
        raise RuntimeError(
            "No embeddings collected. Check detection settings (conf/classes/imgsz) and videos."
        )

    emb_arr = np.stack(embs, axis=0)
    emb_arr = emb_arr.astype(np.float16 if args.save_fp16 else np.float32)

    meta_frame_arr = np.asarray(meta_frame, dtype=np.int32)
    meta_xyxy_arr = np.asarray(meta_xyxy, dtype=np.int32)
    meta_video_arr = np.asarray(meta_video, dtype=object)

    # Derive scene IDs from video names
    scene_ids = np.array(
        [extract_scene_id(str(v)) or "unknown" for v in meta_video_arr],
        dtype=object,
    )

    # Coreset selection
    if args.coreset_size > 0 and emb_arr.shape[0] > args.coreset_size:
        if args.per_scene:
            unique_scenes = sorted(set(scene_ids.tolist()))
            print(f"[INFO] Per-scene coreset: {len(unique_scenes)} scenes, up to {args.coreset_size} each")
            all_indices = []
            for sid in unique_scenes:
                mask = scene_ids == sid
                scene_orig_indices = np.where(mask)[0]
                n_scene = int(mask.sum())
                if n_scene > args.coreset_size:
                    _, local_idx = coreset_farthest_point(emb_arr[mask], args.coreset_size, seed=args.seed)
                    all_indices.append(scene_orig_indices[local_idx])
                    print(f"[INFO]   scene={sid}: {n_scene} -> {args.coreset_size}")
                else:
                    all_indices.append(scene_orig_indices)
                    print(f"[INFO]   scene={sid}: {n_scene} (kept all)")
            idx = np.concatenate(all_indices)
            idx.sort()
            emb_arr = emb_arr[idx]
            meta_frame_arr = meta_frame_arr[idx]
            meta_xyxy_arr = meta_xyxy_arr[idx]
            meta_video_arr = meta_video_arr[idx]
            scene_ids = scene_ids[idx]
            print(f"[INFO] Per-scene coreset done: {emb_arr.shape[0]} total embeddings")
        else:
            print(f"[INFO] Coreset selection: {emb_arr.shape[0]} -> {args.coreset_size}")
            emb_arr, idx = coreset_farthest_point(emb_arr, args.coreset_size, seed=args.seed)
            meta_frame_arr = meta_frame_arr[idx]
            meta_xyxy_arr = meta_xyxy_arr[idx]
            meta_video_arr = meta_video_arr[idx]
            scene_ids = scene_ids[idx]
            print(f"[INFO] Coreset done: {emb_arr.shape[0]} embeddings selected")

    np.savez_compressed(
        args.out,
        embeddings=emb_arr,
        video=meta_video_arr,
        frame=meta_frame_arr,
        xyxy=meta_xyxy_arr,
        scene_ids=scene_ids,
        config=json.dumps(vars(args), ensure_ascii=False),
    )

    dt = time.time() - t0
    print("\n=== Memory Bank Summary ===")
    print(f"out: {args.out}")
    print(f"embeddings: {emb_arr.shape} dtype={emb_arr.dtype}")
    print(f"videos_used: {len(set(meta_video_arr.tolist()))}")
    print(f"frames_seen: {total_frames_seen} frames_processed: {total_frames_proc}")
    print(f"boxes_embedded: {len(embs)} (before coreset)")
    print(f"coreset_size: {emb_arr.shape[0]} (after coreset)")
    print(f"throughput: {total_frames_proc / max(dt, 1e-9):.2f} processed_frames/s")
    print(f"wall_time_s: {dt:.1f}")


if __name__ == "__main__":
    main()
