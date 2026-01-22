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

## metrics_full.json keys

Top-level: `timestamp_utc`, `git_hash`, `args`, `env`, `memory_bank`, `overall`, `runtime`, `per_video`

Overall: `auc_raw`, `auc_smooth`, `ap_raw`, `ap_smooth`

Runtime: `wall_seconds`, `proc_fps`, optional `latency_ms_p50`, `latency_ms_p95`, plus counts
