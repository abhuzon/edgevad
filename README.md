# edgevad

## Evaluation

Example Avenue eval command (writes `artifacts/metrics_full.json`):

```bash
python scripts/score_avenue.py --yolo weights/yolo.pt --mb artifacts/memory_bank.npz --test_dir /path/to/Avenue/testing_videos --gt_dir /path/to/Avenue/ground_truth_demo/testing_label_mask --out_csv artifacts/avenue_scores.csv --out_json artifacts/metrics_full.json --seed 0 --deterministic
```

## metrics_full.json keys

Top-level: `timestamp_utc`, `git_hash`, `args`, `env`, `memory_bank`, `overall`, `runtime`, `per_video`

Overall: `auc_raw`, `auc_smooth`, `ap_raw`, `ap_smooth`

Runtime: `wall_seconds`, `proc_fps`, optional `latency_ms_p50`, `latency_ms_p95`, plus counts
