# edgevad

Fast, reproducible **Avenue** pipeline:

- **Build memory bank** from `training_videos/`
- **Score + metrics** on `testing_videos/` using **GT** in `ground_truth_demo/testing_label_mask/` (`1_label.mat ... 21_label.mat`)

---

## Quickstart

### 0) Set paths (copy/paste)

```bash
# YOLO weights (example: yolo26n)
export YOLO=/home/lize/edgevad/models/yolo26n.pt

# Avenue root (must contain training_videos/ and testing_videos/)
export AV_ROOT=/home/lize/edgevad/data/datasets/avenue/Avenue_Dataset
export AV_TRAIN=$AV_ROOT/training_videos
export AV_TEST=$AV_ROOT/testing_videos

# Avenue GT (note: NOT inside Avenue_Dataset in this setup)
export AV_GT=/home/lize/edgevad/data/datasets/avenue/ground_truth_demo/testing_label_mask
```

Sanity-check the layout:

```bash
ls -lah "$AV_TRAIN" | head
ls -lah "$AV_TEST"  | head
ls -lah "$AV_GT"    | head   # should show 1_label.mat ... 21_label.mat
```

---

### 1) Build memory bank (TRAINING videos ➜ `.npz`)

```bash
python scripts/build_memory_bank.py \
  --yolo "$YOLO" \
  --train_dir "$AV_TRAIN" \
  --out artifacts/avenue_mb_yolo26n_fp16.npz \
  --device cuda:0 \
  --imgsz 640 \
  --conf 0.25 \
  --classes 0 \
  --frame_stride 5 \
  --max_dets_per_frame 3 \
  --max_embeddings 50000 \
  --min_box_area 800 \
  --expand 1.2 \
  --batch_size 1 \
  --fp16 \
  --save_fp16 \
  --seed 0
```

---

### 2) Score Avenue (TESTING videos + GT ➜ `.csv` + `.json`)

```bash
python scripts/score_avenue.py \
  --yolo "$YOLO" \
  --mb artifacts/avenue_mb_yolo26n_fp16.npz \
  --test_dir "$AV_TEST" \
  --gt_dir "$AV_GT" \
  --device cuda:0 \
  --imgsz 640 \
  --conf 0.25 \
  --classes 0 \
  --frame_stride 5 \
  --max_dets_per_frame 3 \
  --batch_size 1 \
  --fp16 \
  --smooth 21 \
  --out_csv artifacts/avenue_yolo26n_fp16.csv \
  --out_json artifacts/metrics_yolo26n_fp16.json \
  --seed 0 \
  --deterministic
```

---

## Reproducibility

### Environment

Minimum expectations:
- Python 3.10+
- PyTorch + CUDA (if using `--device cuda:0`)
- `ultralytics`, `opencv-python`, `numpy`, `scipy`, `pytest`

Example (conda):

```bash
conda create -n edgevad python=3.10 -y
conda activate edgevad
pip install -r requirements.txt
```

### Dataset placement

This repo expects:

- `training_videos/*.avi` for memory-bank building
- `testing_videos/*.avi` for scoring
- `ground_truth_demo/testing_label_mask/{1..21}_label.mat` for GT

If your GT files are named `1_label.mat` (not `01_label.mat`), that is supported.

### Fast smoke test (quick sanity run)

Runs only a small number of frames per test video:

```bash
python scripts/score_avenue.py \
  --yolo "$YOLO" \
  --mb artifacts/avenue_mb_yolo26n_fp16.npz \
  --test_dir "$AV_TEST" \
  --gt_dir "$AV_GT" \
  --device cuda:0 \
  --imgsz 640 \
  --conf 0.25 \
  --classes 0 \
  --frame_stride 5 \
  --max_dets_per_frame 3 \
  --batch_size 1 \
  --fp16 \
  --smooth 21 \
  --max_frames 200 \
  --out_csv artifacts/smoke_scores.csv \
  --out_json artifacts/smoke_metrics.json \
  --seed 0 \
  --deterministic
```

---

## Outputs

### CSV (`--out_csv`)

Row-level scores per frame:

- `video`
- `frame_idx`
- `score`
- `score_smooth`
- `num_dets`
- `gt` (if GT provided)

### JSON (`--out_json`)

Top-level keys:

- `timestamp_utc`, `git_hash`, `args`, `env`
- `memory_bank` (path + embedding metadata)
- `overall` (AUC/AP)
- `runtime` (fps/latency + counts)
- `per_video` (video-wise summaries)

**overall**
- `auc_raw`, `auc_smooth`
- `ap_raw`, `ap_smooth`

**runtime**
- `wall_seconds`, `proc_fps`
- optional: `latency_ms_p50`, `latency_ms_p95`
- plus counts: `num_videos`, `frames_processed`, etc.

---

## Tests

Unit tests require **no dataset**:

```bash
pytest -q
```
