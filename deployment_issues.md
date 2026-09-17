# Hireable — Render Deployment: Issues & Fixes Log

**Date:** 2026-09-17  
**Platform:** Render.com free tier (512MB RAM, no persistent disk)  
**Stack:** FastAPI + React (pre-built dist) + OpenRouter (LLM judges) + SQLite

---

## Issue 1 — HTTP 500 on every `/analyze` call (slowapi runtime error)

**Symptom:** Every POST to `/analyze` returned 500 immediately. Render logs showed:
```
parameter 'request' must be an instance of starlette.requests.Request
```

**Root cause:** slowapi rate limiter scans function parameters by name to find the Starlette `Request` object. The Pydantic request body was named `request: AnalyzeRequest` — slowapi found it first, type-checked it, failed.

**Fix:** Renamed all Pydantic body parameters from `request` to `body` in the 4 rate-limited endpoints (`analyze_resume_unified`, `analyze_stream`, `summarize_batch`, `summarize_candidate`). Updated all internal references from `request.field` to `body.field`.

**File:** `routers/analysis.py`

---

## Issue 2 — App startup crash: "No 'request' or 'websocket' argument"

**Symptom:** After renaming body params to `body`, the app crashed at startup before accepting any requests.

**Root cause:** slowapi checks at decoration time for a parameter named exactly `request` (type `starlette.requests.Request`). After the body rename, the Starlette request param had been renamed to `http_request: Request` — slowapi couldn't find it and refused to register the route.

**Fix:** Renamed `http_request: Request` → `request: Request` across all 4 rate-limited endpoints.

**File:** `routers/analysis.py`

---

## Issue 3 — "No complete event received" — analysis hanging at ~49s

**Symptom:** ATS pre-flight succeeded (no LLM), but all 5 judges hung and the SSE stream timed out at ~49s with "No complete event received."

**Root cause:** The pre-built frontend committed to the repo was sending `api_base_url=http://localhost:11434` (Ollama) in every request. `get_api_config()` had:
```python
if request_api_key or request_api_base_url:
    # use caller-supplied credentials
```
This short-circuited before the OpenRouter env var check, sending all LLM calls to localhost:11434 (Ollama not running on Render → connection refused → judges hang).

**Fix:** Added a `_local_ollama` check to detect and ignore localhost URLs when no API key is provided, so the OpenRouter env var path is taken instead.

**File:** `routers/deps.py`

---

## Issue 4 — OOM kill: "Web Service hireable exceeded its memory limit"

**Symptom:** Render killed the service with an OOM email. Occurred after setting `ANALYSER_API_SEM=5` (parallel judges).

**Root cause:** SBERT (`all-MiniLM-L6-v2`) loads ~400MB at first request. Combined with FastAPI baseline (~100MB) and 5 parallel judge threads, this blew past the 512MB Render free tier limit. The Dockerfile pre-warms SBERT at build time (cached in image), but each new process still loads it into RAM at runtime.

**Fix:** Added `DISABLE_SEMANTIC=1` env var to `resume_jd_similarity()`. When set, the function returns `None` immediately without loading the model.

**File:** `analysers/semantic.py`  
**Render env var:** `DISABLE_SEMANTIC=1`

---

## Issue 5 — `response_format: {"type": "json_object"}` causing 400 on OpenRouter free models

**Symptom:** All judges returning `success=False` after 4 retries (~93s each). OpenRouter returning 400 Bad Request.

**Root cause:** `response_format: json_object` is an OpenAI-specific parameter. Most OpenRouter free models (including nemotron variants) don't support it and return 400.

**Fix:** First attempt sends with `response_format`. On 400 at attempt 0, retries without it. Subsequent attempts skip `response_format` entirely.

**File:** `analysers/base.py`

---

## Issue 6 — 401 Unauthorized on every OpenRouter call

**Symptom:** All judges returning 401 Unauthorized after 3×4 retries (~92s each per judge). Every run produced 0% scores.

**Root cause:** `OPENROUTER_API_KEY` set in Render env vars was invalid/expired. The per-model keys (`OPENROUTER_KEY_N`) from local `.env` were never added to Render, so all calls fell back to the bad global key.

**Fix (code):** None required — purely a credentials problem.  
**Fix (ops):** Added `OPENROUTER_MODEL_1`/`OPENROUTER_KEY_1` and `OPENROUTER_MODEL_4`/`OPENROUTER_KEY_4` to Render env vars. Updated `OPENROUTER_DEFAULT_MODEL` to match the model that had a valid key (`nvidia/nemotron-3-super-120b-a12b:free`).

---

## Issue 7 — Fast model 404 (model not found on OpenRouter)

**Symptom:** After fixing 401, the 3 "fast" judges (seniority, relevancy, impact) returned 404 from OpenRouter. The 2 "slow" judges (skills, experience) returned 200.

**Root cause:** `OPENROUTER_FAST_MODEL` was set to `nvidia/nemotron-nano-9b-v2:free` which doesn't exist as a valid model slug on OpenRouter. 404 from OpenRouter = model not found (not auth).

**Fix:** Updated `OPENROUTER_FAST_MODEL` in Render env to `nvidia/nemotron-3-super-120b-a12b:free` (same as default — slower but confirmed working). Both model slots now use the same model and same key.

---

## Issue 8 — `/summarize_batch` and `/summarize_candidate` returning 500

**Symptom:** After analysis completed (or failed), the Results/Analyzer page called `/summarize_batch`, which returned HTTP 500.

**Root cause:** `execute_llm_call()` returns `{"success": False, "error": "..."}` on LLM failure. The endpoint then did `result["data"]` → `KeyError: 'data'` → caught by `except Exception as e: raise HTTPException(500)`.

**Fix:** Added `if not result.get("success"): return {"summary": "Summary unavailable — LLM call failed."}` before accessing `result["data"]`.

**File:** `routers/analysis.py`

---

## Issue 9 — All 5 judges showing 0% scores despite 200 OK responses (the real bug)

**Symptom:** All 5 judges logged "OK" with real scores in Render logs (skills 100%, experience 33%, relevancy 100%, seniority 38%, impact 36%). But the UI showed 0% for every judge and 0% overall.

**Root cause:** `_run_judge_timed()` returns a 5-tuple `(status, raw, scores, err, latency)`. All 5 `run_*_judge()` methods unpacked only 4 variables:
```python
status, raw, _, err = await _run_judge_timed(...)  # BUG: ValueError
```
Python raises `ValueError: too many values to unpack (expected 4)` — silently caught by the stream handler's `try/except`:
```python
try:
    result = task.result()
except Exception as e:
    result = {"success": False, "data": {}, "error": str(e)}
```
The engine logged "OK" before the ValueError because the log fires inside `_run_judge_timed` before the return. The caller raising ValueError on unpack was invisible in logs.

**Why it was hard to spot:** The engine log looked completely healthy. No error log appeared. The only symptom was silent 0% scores.

**Fix:** Added the missing 5th discard variable across all 5 methods:
```python
status, raw, _, err, _ = await _run_judge_timed(...)  # fixed
```

**File:** `analysers/engine.py`

---

---

## Issue 10 — AI Summary always "unavailable" (`summarize_batch`, `summarize_candidate`, `summarize_result`)

**Symptom:** After analysis completed, the Results/Analyzer page showed "Summary unavailable — LLM call failed" or "Summary failed: HTTP 404" in every run.

**Root cause:** All three summarize endpoints called `execute_llm_call(prompt, body.model)` or `call_ollama_async(...)` directly, passing the model name from the frontend (`cogito:14b` or `cogito` — an Ollama model). `execute_llm_call` never passes `api_key`/`api_base_url` to `call_llm_async`, so all calls routed to Ollama (localhost), which isn't running on Render → connection refused / 404.

**Fix:** Replaced `execute_llm_call` and `call_ollama_async` in all three endpoints with `call_llm_async` called directly using `get_api_config()` credentials, so they route through OpenRouter on Render and Ollama locally.

**File:** `routers/analysis.py`  
**Commit:** `e1c0cad`

---

## Issue 11 — System Health page shows "LLM Service: Offline" even when OpenRouter is working

**Symptom:** The top-right card on the System Health page always showed "Offline" with "[Errno 111] Connection refused", even after all 5 judges were producing valid scores via OpenRouter.

**Root cause:** The deployed `frontend_dist` was built from an older version of `Health.tsx` that displayed Ollama status as the "LLM Service" card. The current source already checks `openrouterOk || ollamaOk` and shows "Online" if either is up — but the old JS bundle was being served.

**Fix:** Rebuilt the React frontend from the updated source and replaced `frontend_dist/` with the new build.

**File:** `frontend/src/pages/Health.tsx` (source already correct), `frontend_dist/` (stale bundle replaced)  
**Commit:** `2031664`

---

## Issue 12 — Scores vary on every run (LLM non-determinism)

**Symptom:** Running the same resume + job multiple times produced different scores each time (e.g. skills 100% one run, 0% another). No bug — inherent stochasticity of the free OpenRouter model tier.

**Root cause:** Free-tier models (`nvidia/nemotron-3-super-120b-a12b:free`) don't guarantee deterministic output even at `temperature=0`. Each run is a fresh LLM call.

**Fix:** Added a score cache keyed on `SHA-256(user_id + resume_text + job_description)`. First run hits the LLM and stores the result; subsequent identical runs return the cached result instantly (stream endpoint replays judge events from stored data). A "Force Re-analyze" checkbox in the Analyzer UI bypasses the cache when a fresh run is actually wanted. A `cached` badge appears on the score panel when results come from cache.

**Files:** `db.py` (new `score_cache` table + `get_cached_run_id`/`set_cached_run`), `routers/analysis.py` (cache check + store in both `/analyze` and `/analyze/stream`), `frontend/src/pages/Analyzer.tsx` (Force Re-analyze checkbox + cache badge)  
**Commit:** `9366ed8`

---

## Summary of Commits

| Commit | Fix |
|--------|-----|
| `44fa8ee` | slowapi param naming — rename `http_request` → `request` |
| `18e24bc` | ignore localhost `api_base_url` so OpenRouter env is used |
| `3ade26a` | `DISABLE_SEMANTIC=1` env var to skip SBERT, save ~400MB RAM |
| `a1861dd` | fall back to plain payload on 400 (response_format unsupported) |
| `f0e34e0` | graceful failure on summarize_batch/candidate when LLM call fails |
| `ec12808` | **core bug** — unpack 5-tuple correctly in all `run_*_judge` methods |
| `e1c0cad` | route summarize endpoints through OpenRouter instead of Ollama |
| `2031664` | rebuild frontend dist — LLM health shows OpenRouter status, not Ollama |
| `9366ed8` | cache analysis results by (user, resume, jd) hash — Force Re-analyze to bypass |

## Render Env Vars Required

```
OPENROUTER_API_KEY=<any truthy value — triggers OpenRouter routing>
OPENROUTER_DEFAULT_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_FAST_MODEL=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_MODEL_1=nvidia/nemotron-3-super-120b-a12b:free
OPENROUTER_KEY_1=<your key for model 1>
DISABLE_SEMANTIC=1
ANALYSER_API_SEM=5
```
