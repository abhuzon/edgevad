#!/usr/bin/env python3
"""
EdgeVAD benchmark runner (Avenue only).

Steps:
1) Build memory bank from training videos.
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
    ap = argparse.ArgumentParser("EdgeVAD Avenue benchmark orchestrator")
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
    ap.add_argument("--train_dir", required=True, help="Avenue training_videos directory")
    ap.add_argument("--test_dir", required=True, help="Test videos directory")
    ap.add_argument(
        "--gt_dir",
        required=True,
        help="Ground-truth directory (Avenue: ground_truth_demo/testing_label_mask; ShanghaiTech: testing/gt)",
    )
    ap.add_argument("--frame_stride", type=int, default=3, help="Frame stride for sampling")
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames per video during scoring (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda:0" if sys.platform != "win32" else "cuda")
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
        ]
        _run(build_cmd)

        # Step 2: score dataset-specific
        if args.dataset == "avenue":
            score_module = "edgevad.scripts.score_avenue"
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
            ]
            if args.max_frames and args.max_frames > 0:
                score_cmd += ["--max_frames", str(args.max_frames)]
        else:  # shanghaitech
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
            ]
            if args.max_frames and args.max_frames > 0:
                score_cmd += ["--max_frames", str(args.max_frames)]

        if args.deterministic:
            score_cmd.append("--deterministic")
        if args.fp16:
            score_cmd.append("--fp16")

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
            "out_dir": str(out_dir),
            "scores_csv": str(scores_path),
            "metrics_json": str(metrics_path),
        }
        run_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"[OK] [{args.dataset}] {yolo_tag} -> {metrics_path}")


if __name__ == "__main__":
    main()
