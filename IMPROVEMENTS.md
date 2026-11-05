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
