# Improvements log

## Round 1 (2026-11-05): agent tool-call cap, LLM failure fallback, ingestion validation, compose without .env

### What changed

1. **The LLM agent's "8 tool call" cap was really an 8-turn cap.** `llm_report` looped `for _ in range(8)` over
   model turns, and one turn can request many tools, so a single turn asking for 12 tools ran all 12. The loop
   now counts tool calls. Requests past the budget get an error `tool_result`, so every `tool_use` still has a
   result as the API requires, and the model gets one turn to write the report. If it asks for tools again
   after that, the loop stops with `mode: llm_agent_truncated`.
2. **LLM failures no longer turn `/agent/analyze` into a 500.** The Anthropic client now has a 60 s timeout and
   2 retries (the SDK default timeout is 10 minutes). Any `anthropic.APIError` (connection error, timeout, bad key,
   5xx) falls back to the deterministic report, which uses the same tools and the same numbers, with
   `mode: deterministic_fallback_llm_error` and the reason in the report.
3. **Ingestion validation.**
   - A CSV with out-of-range coordinates (for example lat/lon swapped, `lat=130.75`) used to come back
     `ok: true`. Now it is an error that names the likely cause.
   - Non-numeric coordinates used to give a 400 with a leaked numpy message
     (`could not convert string to float: np.str_('abc')`). Now those rows are dropped with a warning, the same
     way missing coordinates already were.
   - Rasters whose nodata is NaN reported 0 nodata pixels, because `NaN == NaN` is always false. Fixed.
   - `load_raster` left its dataset open and returned a `(ds, report)` tuple. Every other loader returns an
     `IngestResult`, so the API and the dashboard needed a `hasattr` check. It now closes the file and returns
     an `IngestResult`. `/ingest` also copes with an upload that has no filename.
4. **`docker compose up` from a clean clone failed** with `env file .../.env not found`, even though the README
   says `.env` is optional. `env_file` is now marked `required: false`.

### Why it matters

The README promises a bounded agent loop and an API that answers when the LLM is unavailable. Neither was true
or tested. Bad coordinates in a client's asset register are the most common ingestion problem, and the validator
passed them as OK. The Docker quick start did not work from a clean clone.

### How it was verified

```
$ python -m pyflakes src scripts dashboard tests && echo "pyflakes: clean"
pyflakes: clean

$ python -m pytest -q -p no:warnings          # before (main)
58 passed

$ python -m pytest -q -p no:warnings          # after
71 passed in 2.56s
```

The 13 new tests run on `main`'s source in a git worktree: 10 fail there (the tool-call cap, the API fallback,
the client timeout, out-of-range and non-numeric coordinates, NaN and integer nodata, an unreadable raster).
`main` already passes the other 3: the happy path of the loop, refusing a tool that is not whitelisted, and
raster upload through `/ingest`.

The LLM loop is tested with a scripted fake client (`tests/test_llm_loop.py`), so no API key or network is
needed.

```
$ docker compose config --quiet   # main, no .env
env file .../.env not found: stat .../.env: no such file or directory
$ docker compose config --quiet && echo ok   # this branch, no .env
ok
```

The README's deterministic example query was re-run and still gives the documented figures: HIGH (0.56),
53 assets (¥2,323,681,950), 25 HIGH (¥1,117,086,343), 37 sites, 17 HIGH. The README's test count went from
58/58 to 71/71, re-measured with the command above. No model metric was touched.

### Not verified

- The real Claude path: there is no `ANTHROPIC_API_KEY` in this environment. Only the fake client was run.
  I also could not confirm that the default `LLM_MODEL=claude-sonnet-5` resolves on the API.
- The Docker image build. Only `docker compose config` was run.

### Files touched

`src/agents/agent.py`, `src/agents/llm.py`, `src/api/main.py`, `src/ingestion/csv_source.py`,
`src/ingestion/raster_source.py`, `src/ingestion/validate.py`, `dashboard/app.py`, `docker-compose.yml`,
`README.md`, `docs/design_notes.md`, `tests/test_llm_loop.py` (new), `tests/test_ingestion.py`, `tests/test_api.py`,
`IMPROVEMENTS.md` (new)

## Round 2 (2026-11-05): coordinate and radius validation across the API, tools and agent

Stacked on round 1 (`improve/round-1`).

### What changed

1. **`/predict?lat=1000` returned a 500.** NaN or inf coordinates returned a 404 whose message leaked an
   internal error (`cannot convert float NaN to integer`). `/assets/exposed?lat=95` returned 200 with an empty
   result. The point endpoints now declare `lat` in [-90, 90] and `lon` in [-180, 180], with no NaN or inf, so
   any of these inputs gets a 422.
2. **`Grid.cell_of` is safe for any input.** Non-finite or out-of-range coordinates are reported as outside
   the grid (row/col -1) and never reach PROJ, which raised `utm: Invalid latitude` or returned inf for them.
   An agent query such as "risk at 91.5, 130.7" now says the point is outside the analysis area instead of
   showing the PROJ error.
3. **One radius limit everywhere.** The API capped `radius_km` at 20 km, but the agent's tools accepted any
   value: 5000 km scanned the whole portfolio, and -3 km silently returned nothing. `tools.MAX_RADIUS_KM = 20`
   is now enforced by the tools, which return an error result outside (0, 20]. The same constant is used by
   the API's `Query` bound and is advertised in the tool JSON schema the LLM sees.
4. **The deterministic agent clamps parsed radii.** "within 300 km" or "within 0 km" are clamped to
   0.05 to 20 km, and the report says so. Before, "300 km" was passed straight to the tools.

### Why it matters

A public API should never answer bad input with a 500, or with a "not found" that leaks internal error
text. The LLM path is only as safe as its tools' own argument checks, because the model can send any number.

### How it was verified

```
$ python -m pyflakes src scripts dashboard tests        # clean
$ python -m pytest -q -p no:warnings                    # round 1 -> round 2
71 passed  ->  86 passed in 2.73s
```

The 15 new tests are in `test_api.py`, `test_agent.py` and `test_geo.py`. The API and geo tests were also run
on round 1's code. All 7 behavioural ones fail there, starting with these two:

```
FAILED tests/test_api.py::test_predict_rejects_invalid_coordinates[params0]          # lat=1000 -> 500
FAILED tests/test_api.py::test_exposure_rejects_invalid_coordinates_and_radius[...]  # lat=95 -> 200
```

`test_agent.py` cannot be imported on round 1, because `MAX_RADIUS_KM` does not exist there.

`cell_of` is on the `/predict` path, so the API benchmark was run twice on each branch, on the same machine:

```
$ python scripts/bench_api.py --lat 32.2158 --lon 130.7535 --n 100
p50 ms         /predict   /assets/exposed   /agent/analyze
round 1, run 1   3.5          6.3              16.2
round 1, run 2   3.6          7.4              16.4
round 2, run 1   3.8          6.3              18.3
round 2, run 2   3.5          6.6              16.5
```

There is no difference beyond run-to-run noise. The README's latency figures were measured on a different
machine and are left unchanged. The README's test count is now 86/86, re-measured.

### Not verified

- The LLM path still has not been run against the real API (no key). The new radius limit in the tool schema
  is covered only by a test that inspects the schema.

### Files touched

`src/geo/grid.py`, `src/agents/tools.py`, `src/agents/agent.py`, `src/api/main.py`, `README.md`,
`tests/test_api.py`, `tests/test_agent.py`, `tests/test_geo.py`, `IMPROVEMENTS.md`
