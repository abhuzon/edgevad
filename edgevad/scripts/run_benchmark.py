#!/usr/bin/env python3
"""
EdgeVAD benchmark runner (Avenue + ShanghaiTech).

Steps:
1) Build memory bank from training videos (with optional coreset selection).
2) Score test videos and write metrics/results.

Outputs: scores.csv, metrics_full.json, run_config.json inside --out_dir.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("EdgeVAD benchmark orchestrator")
    ap.add_argument(
        "--yolo",
        dest="yolo_weights",
        action="append",
        required=True,
        help="Path to YOLO weights (.pt). Repeat to run multiple weights.",
    )
    ap.add_argument(
        "--dataset",
        choices=["avenue", "shanghaitech"],
        default="avenue",
        help="Which dataset pipeline to run",
    )
    ap.add_argument("--train_dir", required=True, help="Training videos directory")
    ap.add_argument("--test_dir", required=True, help="Test directory")
    ap.add_argument(
        "--gt_dir",
        required=True,
        help=(
            "Ground-truth directory (Avenue: ground_truth_demo/testing_label_mask; "
            "ShanghaiTech: testing/test_frame_mask and/or testing/test_pixel_mask)"
        ),
    )
    ap.add_argument("--frame_stride", type=int, default=3, help="Frame stride for sampling")
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames per video during scoring (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda:0" if sys.platform != "win32" else "cuda")

    # Phase 1 pass-through args
    ap.add_argument("--coreset_size", type=int, default=5000, help="Coreset size for bank (0=keep all)")
    ap.add_argument("--topk", type=int, default=5, help="k-NN neighbors for scoring (1=old max-sim)")
    ap.add_argument("--smooth_sigma", type=float, default=3.0, help="Gaussian smooth sigma. 0=use moving avg.")
    ap.add_argument("--min_box_area", type=float, default=400.0, help="Min box area (px^2) to score")
    ap.add_argument("--feature_mode", choices=["mobilenet", "fpn"], default="mobilenet",
                    help="Feature extraction: mobilenet (crop+embed) or fpn (YOLO FPN RoI-Align)")
    ap.add_argument(
        "--per_scene",
        action="store_true",
        help="Per-scene memory banks (ShanghaiTech). Passes --per_scene to build and score.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    bench_root = project_root / "artifacts" / "benchmarks"

    for yolo in args.yolo_weights:
        yolo_path = Path(yolo).expanduser().resolve()
        yolo_tag = yolo_path.stem

        out_dir = (
            bench_root
            / args.dataset
            / f"yolo_{yolo_tag}"
            / f"fs{args.frame_stride}_seed{args.seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        mb_path = out_dir / f"{args.dataset}_{yolo_tag}_mb.npz"
        scores_path = out_dir / "scores.csv"
        metrics_path = out_dir / "metrics_full.json"
        run_config_path = out_dir / "run_config.json"

        # Step 1: build memory bank
        build_cmd = [
            sys.executable,
            "-m",
            "edgevad.scripts.build_memory_bank",
            "--yolo",
            str(yolo_path),
            "--train_dir",
            str(Path(args.train_dir).expanduser().resolve()),
            "--out",
            str(mb_path),
            "--frame_stride",
            str(args.frame_stride),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--pretrained",
            "--coreset_size",
            str(args.coreset_size),
            "--feature_mode",
            args.feature_mode,
        ]
        if args.per_scene:
            build_cmd.append("--per_scene")
        _run(build_cmd)

        # Step 2: score dataset-specific
        if args.dataset == "avenue":
            score_module = "edgevad.scripts.score_avenue"
        else:
            score_module = "edgevad.scripts.score_shanghaitech"

        score_cmd = [
            sys.executable,
            "-m",
            score_module,
            "--yolo",
            str(yolo_path),
            "--mb",
            str(mb_path),
            "--test_dir",
            str(Path(args.test_dir).expanduser().resolve()),
            "--gt_dir",
            str(Path(args.gt_dir).expanduser().resolve()),
            "--frame_stride",
            str(args.frame_stride),
            "--out_csv",
            str(scores_path),
            "--out_json",
            str(metrics_path),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--topk",
            str(args.topk),
            "--smooth_sigma",
            str(args.smooth_sigma),
            "--min_box_area",
            str(args.min_box_area),
            "--feature_mode",
            args.feature_mode,
        ]
        if args.max_frames and args.max_frames > 0:
            score_cmd += ["--max_frames", str(args.max_frames)]
        if args.deterministic:
            score_cmd.append("--deterministic")
        if args.fp16:
            score_cmd.append("--fp16")
        if args.per_scene and args.dataset == "shanghaitech":
            score_cmd.append("--per_scene")

        _run(score_cmd)

        # Save run configuration (resolved paths)
        config = {
            "dataset": args.dataset,
            "yolo": str(yolo_path),
            "mb": str(mb_path),
            "train_dir": str(Path(args.train_dir).expanduser().resolve()),
            "test_dir": str(Path(args.test_dir).expanduser().resolve()),
            "gt_dir": str(Path(args.gt_dir).expanduser().resolve()),
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "deterministic": bool(args.deterministic),
            "fp16": bool(args.fp16),
            "device": args.device,
            "coreset_size": args.coreset_size,
            "topk": args.topk,
            "smooth_sigma": args.smooth_sigma,
            "min_box_area": args.min_box_area,
            "feature_mode": args.feature_mode,
            "per_scene": bool(args.per_scene),
            "out_dir": str(out_dir),
            "scores_csv": str(scores_path),
            "metrics_json": str(metrics_path),
        }
        run_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"[OK] [{args.dataset}] {yolo_tag} -> {metrics_path}")


if __name__ == "__main__":
    main()
