#!/usr/bin/env python3
"""
Summarize EdgeVAD benchmark runs.

Scans artifacts/benchmarks/**/metrics_full.json and writes:
  - summary.csv
  - summary.md

Assumes directory layout:
  artifacts/benchmarks/{dataset}/yolo_{tag}/fs{frame_stride}_seed{seed}/metrics_full.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class RunSummary:
    dataset: str
    yolo_tag: str
    frame_stride: int
    seed: int
    auc_raw: Optional[float]
    auc_smooth: Optional[float]
    ap_raw: Optional[float]
    ap_smooth: Optional[float]
    proc_fps: Optional[float]
    latency_ms_p50: Optional[float]
    latency_ms_p95: Optional[float]
    frames_processed: Optional[int]


def parse_run_from_path(path: Path, root: Path) -> Optional[tuple[str, str, int, int]]:
    """Extract dataset, yolo_tag, frame_stride, seed from metrics path structure."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 3:
        return None

    dataset = parts[0]
    yolo_part = parts[1]
    run_part = parts[2]

    if not yolo_part.startswith("yolo_"):
        return None
    yolo_tag = yolo_part[len("yolo_") :]

    m = re.match(r"fs(?P<fs>-?\d+)_seed(?P<seed>-?\d+)", run_part)
    if not m:
        return None

    frame_stride = int(m.group("fs"))
    seed = int(m.group("seed"))
    return dataset, yolo_tag, frame_stride, seed


def collect_runs(root: Path) -> List[RunSummary]:
    runs: List[RunSummary] = []

    for metrics_path in sorted(root.rglob("metrics_full.json")):
        parsed = parse_run_from_path(metrics_path, root)
        if not parsed:
            continue
        dataset, yolo_tag, frame_stride, seed = parsed

        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            continue

        overall = metrics.get("overall", {})
        runtime = metrics.get("runtime", {})
        runs.append(
            RunSummary(
                dataset=dataset,
                yolo_tag=yolo_tag,
                frame_stride=frame_stride,
                seed=seed,
                auc_raw=_safe_float(overall.get("auc_raw")),
                auc_smooth=_safe_float(overall.get("auc_smooth")),
                ap_raw=_safe_float(overall.get("ap_raw")),
                ap_smooth=_safe_float(overall.get("ap_smooth")),
                proc_fps=_safe_float(runtime.get("proc_fps")),
                latency_ms_p50=_safe_float(runtime.get("latency_ms_p50")),
                latency_ms_p95=_safe_float(runtime.get("latency_ms_p95")),
                frames_processed=_safe_int(runtime.get("frames_processed")),
            )
        )

    runs.sort(key=lambda r: (r.dataset, r.yolo_tag, r.frame_stride, r.seed))
    return runs


def _safe_float(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        return None if x is None else int(x)
    except Exception:
        return None


def write_csv(rows: Iterable[RunSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "yolo_tag",
        "frame_stride",
        "seed",
        "auc_raw",
        "auc_smooth",
        "ap_raw",
        "ap_smooth",
        "proc_fps",
        "latency_ms_p50",
        "latency_ms_p95",
        "frames_processed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": r.dataset,
                    "yolo_tag": r.yolo_tag,
                    "frame_stride": r.frame_stride,
                    "seed": r.seed,
                    "auc_raw": r.auc_raw,
                    "auc_smooth": r.auc_smooth,
                    "ap_raw": r.ap_raw,
                    "ap_smooth": r.ap_smooth,
                    "proc_fps": r.proc_fps,
                    "latency_ms_p50": r.latency_ms_p50,
                    "latency_ms_p95": r.latency_ms_p95,
                    "frames_processed": r.frames_processed,
                }
            )


def fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def write_md(rows: Iterable[RunSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "yolo_tag",
        "frame_stride",
        "seed",
        "auc_raw",
        "auc_smooth",
        "ap_raw",
        "ap_smooth",
        "proc_fps",
        "latency_ms_p50",
        "latency_ms_p95",
        "frames_processed",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        vals = [
            r.dataset,
            r.yolo_tag,
            str(r.frame_stride),
            str(r.seed),
            fmt(r.auc_raw),
            fmt(r.auc_smooth),
            fmt(r.ap_raw),
            fmt(r.ap_smooth),
            fmt(r.proc_fps),
            fmt(r.latency_ms_p50),
            fmt(r.latency_ms_p95),
            fmt(r.frames_processed),
        ]
        lines.append("| " + " | ".join(vals) + " |")

    if len(lines) == 2:  # only header + divider -> add empty row placeholder
        lines.append("| (none) | - | - | - | - | - | - | - | - | - | - | - |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2] / "artifacts" / "benchmarks"
    ap = argparse.ArgumentParser("Summarize benchmark metrics_full.json files")
    ap.add_argument("--root", default=str(default_root), help="Benchmark root directory to scan")
    ap.add_argument("--csv", default="", help="Output CSV path (default: <root>/summary.csv)")
    ap.add_argument("--md", default="", help="Output Markdown path (default: <root>/summary.md)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else root / "summary.csv"
    md_path = Path(args.md).expanduser().resolve() if args.md else root / "summary.md"

    runs = collect_runs(root)
    write_csv(runs, csv_path)
    write_md(runs, md_path)

    print(f"[OK] Wrote {len(runs)} rows to {csv_path}")
    print(f"[OK] Wrote markdown to {md_path}")


if __name__ == "__main__":
    main()
