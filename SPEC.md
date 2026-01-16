# EdgeVAD — Project Spec (MVP)

## One-liner
Real-time anomaly detection on RTSP/webcam/file with person tracking, event logs, and deployment benchmarks (ONNX/TensorRT + kNN later).

## MVP Acceptance Criteria (Week 1)
### Inputs
- `--source` supports:
  - `0` (webcam)
  - local video file path
  - `rtsp://...`

### Outputs
- `runs/events.jsonl` written continuously (JSON Lines)
- Optional rendered video saved to `runs/` (for headless servers)
- Overlay includes: bbox, track_id, conf, FPS

### Event schema (JSONL)
Each line is one detection event:
- `ts` (unix time, float)
- `source` (string)
- `frame_idx` (int)
- `track_id` (int)
- `conf` (float)
- `bbox_xyxy` ([x1,y1,x2,y2] floats)
- `fps` (float)

### Performance reporting
At exit, print:
- FPS avg
- latency p50/p95 per frame (ms)
- CPU RSS (MB)
- GPU mem allocated (MB) if CUDA

## Targets (initial)
- Headless demo on server must run without GUI.
- Produce a saved output video + JSONL logs.

## Non-goals (MVP)
- No anomaly model yet (score is not included in MVP).
- No TensorRT yet (comes after pipeline is stable).
