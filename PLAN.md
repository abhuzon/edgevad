# EdgeVAD Plan (v0)

## Goal
Make EdgeVAD recruiter-grade and research paper grade: reproducible eval, metrics JSON, benchmarks, export path, demo.

## Milestones
### M1 — Benchmark-grade Avenue evaluation (Week 1)
- [ ] `scripts/score_avenue.py` produces `artifacts/metrics_full.json`
- [ ] JSON schema: overall + runtime + per_video
- [ ] Deterministic mode (`--seed`, `--deterministic`)
- [ ] Lightweight unit tests (no dataset needed)
- [ ] README: one eval command + JSON keys

### M2 — Inference benchmark harness (Week 2)
- [ ] `scripts/bench_infer.py` reports FPS + latency p50/p95
- [ ] Saves benchmark JSON to `artifacts/bench.json`
- [ ] README: benchmark table

### M3 — Export + parity (Week 3–4)
- [ ] ONNX export script
- [ ] Parity check (PyTorch vs ONNX within tolerance)
- [ ] Deployment notes (device, fp16, batch)

## Definition of Done
- `python -m pytest -q` passes
- No secrets committed
- Commands in README are copy-paste runnable
