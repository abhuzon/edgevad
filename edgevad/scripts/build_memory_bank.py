#!/usr/bin/env python3
"""EdgeVAD - Build a person-crop embedding memory bank from Avenue training videos.

What it does:
- Loads a YOLO detector (your YOLO26 weights).
- Iterates training_videos/*.avi.
- Every --frame_stride frames: detect persons -> crop -> embed -> store vector.
- Saves NPZ (embeddings + minimal metadata).

Multi-GPU (simple + robust): run 2 processes with sharding:
  --shard_id 0 --num_shards 2  (GPU0)
  --shard_id 1 --num_shards 2  (GPU1)
Then merge NPZ files.

Notes:
- Embedder is MobileNetV3-Small feature extractor (D=576).
- If --pretrained is used, torchvision may try to fetch weights (network dependent).
  For fully offline reproducibility, pass --embedder_weights PATH.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from ultralytics import YOLO

from edgevad.data import AvenueVideoDataset, list_videos


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_classes(s: str):
    s = (s or "").strip()
    if s == "" or s.lower() == "none":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


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


class MobileNetV3SmallEmbedder(torch.nn.Module):
    """Return pooled feature vector (D=576) from MobileNetV3-Small."""

    def __init__(self, pretrained: bool = False, weights_path: str = ""):
        super().__init__()

        if weights_path:
            backbone = torchvision.models.mobilenet_v3_small(weights=None)
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            cleaned = {}
            for k, v in sd.items():
                nk = k[7:] if k.startswith("module.") else k
                cleaned[nk] = v
            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"[WARN] Missing keys in embedder weights: {len(missing)}", file=sys.stderr)
            if unexpected:
                print(f"[WARN] Unexpected keys in embedder weights: {len(unexpected)}", file=sys.stderr)
            self.backbone = backbone
        else:
            if pretrained:
                try:
                    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                    self.backbone = torchvision.models.mobilenet_v3_small(weights=weights)
                except Exception as e:
                    print(f"[WARN] Could not load pretrained MobileNetV3 weights: {e}", file=sys.stderr)
                    print("[WARN] Falling back to random init. Use --embedder_weights for offline reproducibility.", file=sys.stderr)
                    self.backbone = torchvision.models.mobilenet_v3_small(weights=None)
            else:
                self.backbone = torchvision.models.mobilenet_v3_small(weights=None)

        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def build_preprocess():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def main():
    ap = argparse.ArgumentParser(
        "EdgeVAD - build memory bank from Avenue training videos (person crops)."
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
    ap.add_argument("--batch_size", type=int, default=32)
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

    set_seeds(args.seed)

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

    # Embedder
    embedder = (
        MobileNetV3SmallEmbedder(
            pretrained=args.pretrained, weights_path=args.embedder_weights
        )
        .to(device)
        .eval()
    )
    preprocess = build_preprocess()

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
        f"[INFO] device={device} fp16={bool(args.fp16)} save_fp16={bool(args.save_fp16)} embedder_dim=576"
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

    for sample in tqdm(dataset, desc=f"frames(shard={args.shard_id})"):
        frame = sample["image"]  # BGR frame (transform is identity)
        video_name = sample["video"]
        frame_idx = sample["frame_idx"]
        total_frames_seen += 1
        total_frames_proc += 1

        H, W = frame.shape[:2]

        with torch.no_grad():
            results = det.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=classes,
                device=str(device),
                verbose=False,
            )

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            if len(embs) >= args.max_embeddings:
                break
            continue

        boxes = r0.boxes
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

            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            crop_meta.append((video_name, frame_idx, (x1i, y1i, x2i, y2i)))
            kept += 1

        if len(crops) == 0:
            if len(embs) >= args.max_embeddings:
                break
            continue

        # Batch preprocess/embed
        batch_tensors = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
        total_boxes += len(crops)

        with torch.no_grad():
            if args.fp16 and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feats = embedder(batch_tensors)
            else:
                feats = embedder(batch_tensors)

        feats = feats.detach().cpu().numpy()  # [B, D]

        for j in range(feats.shape[0]):
            embs.append(feats[j])
            meta_video.append(crop_meta[j][0])
            meta_frame.append(crop_meta[j][1])
            meta_xyxy.append(crop_meta[j][2])

            if len(embs) >= args.max_embeddings:
                break

        if len(embs) >= args.max_embeddings:
            break

    if len(embs) == 0:
        raise RuntimeError(
            "No embeddings collected. Check detection settings (conf/classes/imgsz) and videos."
        )

    emb_arr = np.stack(embs, axis=0)
    emb_arr = emb_arr.astype(np.float16 if args.save_fp16 else np.float32)

    meta_frame = np.asarray(meta_frame, dtype=np.int32)
    meta_xyxy = np.asarray(meta_xyxy, dtype=np.int32)
    meta_video = np.asarray(meta_video, dtype=object)

    np.savez_compressed(
        args.out,
        embeddings=emb_arr,
        video=meta_video,
        frame=meta_frame,
        xyxy=meta_xyxy,
        config=json.dumps(vars(args), ensure_ascii=False),
    )

    dt = time.time() - t0
    print("\n=== Memory Bank Summary ===")
    print(f"out: {args.out}")
    print(f"embeddings: {emb_arr.shape} dtype={emb_arr.dtype}")
    print(f"videos_used: {len(set(meta_video.tolist()))}")
    print(f"frames_seen: {total_frames_seen} frames_processed: {total_frames_proc}")
    print(f"boxes_embedded: {len(embs)}")
    print(f"throughput: {total_frames_proc / max(dt, 1e-9):.2f} processed_frames/s")
    print(f"wall_time_s: {dt:.1f}")


if __name__ == "__main__":
    main()
