# EdgeVAD — Project Spec (MVP + Repro)

## One-liner
Edge-first video anomaly detection: person-centric detection + lightweight embedding + memory-bank scoring (Avenue benchmark today), with a clear path to real-time RTSP deployment and future optimization.

---

## Current Baseline (must remain working)
### Avenue benchmark pipeline
**Build (train split → memory bank)**
- Input: `Avenue_Dataset/training_videos/*.avi`
- Output: `artifacts/*.npz` memory bank (embeddings + metadata)
- Determinism controls: `--seed`, `--deterministic`, `--frame_stride`, `--max_frames`

**Score (test split + GT → metrics)**
- Input:
  - `Avenue_Dataset/testing_videos/*.avi`
  - GT: `ground_truth_demo/testing_label_mask/{1..21}_label.mat`  
    *(support both `1_label.mat` and `01_label.mat` naming)*
- Output:
  - `artifacts/*.csv` per-frame scores
  - `artifacts/*.json` full metrics dump (overall + per-video + runtime)

### Required outputs & schema
**Metrics JSON**
- Top-level keys: `timestamp_utc`, `git_hash`, `args`, `env`, `memory_bank`, `overall`, `runtime`, `per_video`
- `overall`: `auc_raw`, `auc_smooth`, `ap_raw`, `ap_smooth`
- `runtime`: `num_videos`, `num_rows`, `frames_processed`, `wall_seconds`, `proc_fps`, optional `latency_ms_p50`, `latency_ms_p95`

**Scores CSV**
- Columns: `video,frame_idx,score,score_smooth,num_dets,gt`

---

## MVP Acceptance Criteria (next 1–2 weeks)
### A) Dataset module (reusability + correctness)
Create `edgevad/data/avenue_dataset.py` with:
- Numeric-aware video ordering (`01.avi`..`21.avi`)
- Deterministic frame sampling (`frame_stride`, `max_frames`, `start_offset`, `seed`, `deterministic`)
- Robust GT loader for Avenue `.mat` (`volLabel` cell/array variants)
- Optional transforms aligned to detection/embedding (resize/letterbox, RGB conversion, dtype)

**Pass condition**
- `score_avenue.py` computes **non-null** AUC/AP when `--gt_dir` is provided and correct.
- Unit tests cover:
  - GT filename mapping (01 vs 1)
  - GT parsing edge cases
  - Numeric-aware sorting  
  *(tests should not require the Avenue dataset; use small synthetic `.mat` fixtures)*

### B) Repo hygiene (no large/binary dumps)
- `.gitignore` must cover: `artifacts/`, `data/`, `runs/`, `models/`, `*.npz`, `*.csv`, `*.json`
- Only commit *code* + *docs* + *tests* (optionally tiny example outputs if explicitly intended).

**Pass condition**
- `git status` is clean after running build/score locally.

### C) Repro “smoke test” (fast sanity check)
Add a quick command that runs in minutes:
- Use `--max_frames` small (e.g., 200) and small `--max_embeddings`
- Produces CSV+JSON, without requiring full evaluation time.

**Pass condition**
- A new user can follow README Quickstart and produce outputs without guessing paths.

### D) CI guardrail (dataset-free)
Add GitHub Actions workflow:
- Runs `pytest` on push/PR
- No dataset download in CI
- Optional linting if adopted (`ruff`, `black`)

**Pass condition**
- CI green on main; prevents future regressions.

---

## Stretch Goals (after MVP is stable)
### Real-time inference & logging (RTSP/webcam/file)
- `--source` supports: `0`, file path, `rtsp://...`
- Continuous `runs/events.jsonl` (JSONL) with:
  - `ts`, `source`, `frame_idx`, `track_id`, `conf`, `bbox_xyxy`, `fps`
- Optional rendered video output for headless servers
- Exit summary: avg FPS, latency p50/p95, CPU RSS, GPU mem (if CUDA)

### Optimization
- ONNX export + TensorRT (post-stability)
- kNN acceleration / approximate NN for memory bank
- Profiling + batch/stride tuning for edge devices

---

## Non-goals (for now)
- Training an end-to-end anomaly model (beyond memory-bank scoring baseline)
- Full-scale benchmarking across many datasets
- TensorRT optimization before the Python baseline is stable
