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

## Round 3 (2026-11-05): bounded, non-blocking `/ingest`

Stacked on round 2 (`improve/round-2`).

### What changed

1. **Upload size limit.** `/ingest` did `tmp.write(await file.read())`, which reads the whole upload into memory
   with no limit. It now streams to the temp file in 1 MB chunks and stops with a **413** past 50 MB
   (`MAX_UPLOAD_BYTES`). The partial temp file is deleted.
2. **Unsupported formats are rejected before the body is read.** Before, a `.docx` upload was written to disk
   in full before the loader lookup failed. The response was already 400, but now it comes from the filename
   alone, and the message lists the supported extensions.
3. **Parsing runs off the event loop.** `ingest` is an `async def`, but geopandas and rasterio parsing is
   blocking, so a large upload stalled every other request the server was handling. Parsing now goes through
   `run_in_threadpool`.

### Why it matters

`/ingest` is the only endpoint that takes arbitrary client data. Without a size bound, one request can exhaust
memory or disk. Blocking the event loop turns one slow upload into latency for every caller of `/predict`.

### How it was verified

```
$ python -m pyflakes src scripts dashboard tests        # clean
$ python -m pytest -q -p no:warnings                    # round 2 -> round 3
86 passed  ->  88 passed in 3.08s
```

The 2 new tests (`tests/test_api.py`) were also run on round 2's code. `test_ingest_rejects_oversized_upload`
fails there: a 2.6 KB upload against a 1000-byte limit, patched for the test, is accepted. The test also checks
that no temp file is left behind. `test_ingest_rejects_unsupported_format_before_reading` passes on both
branches, because the status was already 400 and only the point where the request stops changed.

### Not verified

- The event-loop change was not measured. I did not build a concurrent-request benchmark. It is the standard
  FastAPI pattern for blocking work inside `async def`, and the existing tests pass through it.
- 50 MB is a judgement call (the committed client register is 120 rows), not a measured requirement.

### Files touched

`src/api/main.py`, `tests/test_api.py`, `README.md`, `IMPROVEMENTS.md`

## Round 4 (2026-11-05): review of rounds 1–3, and the one fix it found

Rounds 1–3 (https://github.com/athish-shubhan/Inundata/pull/1, https://github.com/athish-shubhan/Inundata/pull/2,
https://github.com/athish-shubhan/Inundata/pull/3) are merged. This round is a second pass over their merged
diff, and its PR is a fresh branch from `main`.

### What changed

1. **`/ingest` reports no longer expose the server's temp path.** The loaders quote the file they read, and in
   the API that file is the temp copy. A bad upload came back as
   `failed to read geojson: '/tmp/tmpkrjyditw.geojson' not recognized…`, an internal path the client never sent
   that tells them nothing. Errors and warnings now name the uploaded file (`'sites.geojson'`). The leak was
   older than round 1, but rounds 1 and 3 both reworked this endpoint without catching it.

### What else the review checked (no change needed)

- **Tool-call budget loop:** it terminates on every path, counts unknown tools against the budget, and gives
  every `tool_use` a `tool_result`.
- **`radius_km` on the API:** NaN, `inf` and `-inf` all return 422, because the `gt`/`le` bounds reject them.
  Only lat/lon needed `allow_inf_nan=False`.
- **Upload limit:** the temp file is removed on 413, 400 and success alike.
- **The 36 s test run seen after pulling `main`:** a one-off cold start. Repeat runs take 3.03 s and 2.79 s,
  and the slowest test is 0.39 s.
- **`/grid?step=1`:** returns 21 MB of JSON in about 2 s (all 153k cells). That is heavy, but the endpoint
  allows it by design, so it is left alone.

### How it was verified

```
$ python -m pyflakes src scripts dashboard tests        # clean
$ python -m pytest -q -p no:warnings
89 passed in 3.20s                                      # was 88
```

The new test, `test_ingest_errors_name_the_upload_not_the_server_temp_file`, fails with the fix stashed:

```
E  assert 'sites.geojson' in "failed to read geojson: '/tmp/tmpkrjyditw.geojson' not recognized as being in a supported file format. ..."
```

### Not verified

- The real Claude API path (no key here). `main` now defaults to `claude-sonnet-5-5` (PR #4), which this
  round did not exercise.

### Files touched

`src/api/main.py`, `tests/test_api.py`, `README.md`, `IMPROVEMENTS.md`
