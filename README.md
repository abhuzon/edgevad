# edgevad

## Install

```bash
pip install -e .
```

## Avenue pipeline

Build the memory bank from training videos:

```bash
python -m edgevad.scripts.build_memory_bank --yolo weights/yolo.pt --train_dir /path/to/Avenue/training_videos --out artifacts/memory_bank.npz --frame_stride 3 --max_embeddings 30000
```

Score the test set (writes `artifacts/metrics_full.json`):

```bash
python -m edgevad.scripts.score_avenue --yolo weights/yolo.pt --mb artifacts/memory_bank.npz --test_dir /path/to/Avenue/testing_videos --gt_dir /path/to/Avenue/ground_truth_demo/testing_label_mask --out_csv artifacts/avenue_scores.csv --out_json artifacts/metrics_full.json --seed 0 --deterministic
```

For a quick smoke test, add `--max_frames 10` to limit per-video processing.

## ShanghaiTech pipeline

Build the memory bank from training videos:

```bash
python -m edgevad.scripts.build_memory_bank --yolo weights/yolo.pt --train_dir /path/to/ShanghaiTech/training/videos --out artifacts/shanghai_mb.npz --frame_stride 3 --max_embeddings 30000
```

Score the test set using frame folders (writes `artifacts/shanghai_metrics_full.json`):

```bash
python -m edgevad.scripts.score_shanghaitech --yolo weights/yolo.pt --mb artifacts/shanghai_mb.npz --test_dir /path/to/ShanghaiTech/testing/frames --gt_dir /path/to/ShanghaiTech/testing --out_csv artifacts/shanghai_scores.csv --out_json artifacts/shanghai_metrics_full.json --seed 0 --deterministic
```

Ground truth can live under `testing/test_frame_mask` and/or `testing/test_pixel_mask`.

## End-to-end benchmark runner

Run the full Avenue pipeline (memory bank + scoring) in one command:

```bash
python -m edgevad.scripts.run_benchmark --yolo weights/yolo.pt --mb artifacts/memory_bank.npz --train_dir /path/to/Avenue/training_videos --test_dir /path/to/Avenue/testing_videos --gt_dir /path/to/Avenue/ground_truth_demo/testing_label_mask --frame_stride 3 --max_frames 0 --seed 0 --out_dir artifacts/avenue_benchmark
```

Outputs: `scores.csv`, `metrics_full.json`, and `run_config.json` inside `--out_dir`.

## metrics_full.json keys

Top-level: `timestamp_utc`, `git_hash`, `args`, `env`, `memory_bank`, `overall`, `runtime`, `per_video`

Overall: `auc_raw`, `auc_smooth`, `ap_raw`, `ap_smooth`

Runtime: `wall_seconds`, `proc_fps`, optional `latency_ms_p50`, `latency_ms_p95`, plus counts
