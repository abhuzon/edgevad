## Repo DoD (definition of done)
- `python -m pytest -q` passes
- Evaluation writes `metrics_full.json` to `artifacts/metrics_full.json`
- CLI supports: --seed, --deterministic, --log_file
- Runtime stats recorded: wall time, proc_fps, optional latency p50/p95
- README includes exact eval command + example output keys

## Commands to run before finishing
- python -m pytest -q


- Ensure `.codex/` and `auth.json` are gitignored (never commit).

