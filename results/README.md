# Results provenance

| File | Produced by | Notes |
|---|---|---|
| `../data/hitoyoshi/processed/metrics.json` | `python scripts/run_pipeline.py` | All model/CV/calibration/importance figures quoted in the top-level README. Committed from the original pipeline run; not re-run for this release. |
| `portfolio_hindcast.json` | `python scripts/hindcast.py --out results/portfolio_hindcast.json` | Served (out-of-fold) scores vs the 2020 GSI extent for the 120 simulated assets. |
| `demo_downtown.txt` | `python scripts/demo.py --place downtown` | Output of the CLI demo. |
| `test_run.txt` | `python -m pytest -q` and `python -m pyflakes src scripts dashboard tests` | Fresh venv from `requirements.txt`. |
| `../benchmarks/api_latency_{before,after}.json` | `python scripts/bench_api.py --lat 32.2158 --lon 130.7535` | In-process timings from the round-1 to round-4 work; single machine, not re-measured here. |

Environment of the runs made for this release: macOS (Darwin 25.2, Apple Silicon), Python 3.12.11, dependencies pinned in `requirements.txt`. The machine was under heavy load from unrelated jobs, so the 449 s test duration is not representative.
