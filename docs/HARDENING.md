# Hardening changes (audit follow-up)

Everything here was verified by running the code. `pytest tests` (305 tests), `ruff`, `bandit -ll` and
`python -m benchmark.run --backend fake` all pass. New tests were also run against the original code to
confirm they detect the defects (23 of 29 scoring/schema/weight tests fail there).

## Corrections to the first-pass audit
The README-only review was wrong in places; the source review corrected it:
* **`sanitize_chunk` IS on the live path** (via `chunker_compat`). Tag break-out was handled, but it
  truncated *everything* after a forged closing tag, and zero-width variants surfaced as five failed judges
  rather than a clean rejection. Fullwidth and paraphrased injections passed both layers.
* The judge timeout envelope, quorum flag, weight renormalisation, skills gate, WAL + pooled SQLite
  and citation downgrade already existed. A global `API_KEY` middleware also exists (off by default, and
  incompatible with JWT because both use the `Authorization` header).
* The weighted-sum "Skills=0 still passes" concern was already covered by the gate; the real flaw is the
  *cliff* at 0.40 (see below).

## What changed
### Endpoints / network
| Area | Change |
|---|---|
| `/benchmark-*` (11 routes) | Admin-only (`ADMIN_EMAILS`); **fails closed** when unset |
| `/fetch-job-url` | Login required; `net_safety`: scheme/port/host rules, all resolved IPs must be public, pinned connection (DNS-rebinding), manual redirect re-validation, 2 MB cap; Playwright tier validates every sub-request |
| `/extract-pdf` | Login required; 5 MB / 30 page caps, `%PDF` check, extraction off the event loop with 45 s limit |
| `api_base_url` (analyze, evals) | https + public host only, optional `ALLOWED_API_HOSTS`; `localhost` substring trick removed |
| `/record_outcome`, `/outcomes`, `/summarize_*`, `/results/{id}/evals`, `/judge_*` | Login required, ownership-checked, `result_id` pattern-validated (path traversal gone), score taken from the stored run |
| `/interview_prep`, `/edit_resume`, `/ats-*`, `/skill-match`, `/parse-jd-text` | Login required (they spent your LLM key / CPU for anyone) |
| `/auth/register`, `/auth/login` | Rate-limited; bcrypt off the event loop; dummy hash so unknown emails take the same time; **admin addresses reserved** unless `ADMIN_BOOTSTRAP_TOKEN` is supplied (`ADMIN_EMAILS` trusts an unverified email string, so it could be pre-registered by anyone) |
| Rate limiting | One shared limiter keyed by user id, else client IP; `TRUSTED_PROXY_COUNT` for proxies |
| Errors | No `str(e)` / regex / exception text returned to clients; `/health` no longer echoes exceptions |

### Scoring correctness
* Schemas can no longer "succeed" on `{}`: nothing-to-score is a parse failure; failed or unscorable judges are
  *excluded and weights renormalised* instead of counting as a zero.
* Run status is explicit: `complete | degraded | insufficient | not_scored`. `insufficient` → HTTP 503 (+`Retry-After`),
  not saved; `not_scored` → 422; `degraded` → returned, flagged, saved, **never cached**.
* ATS gate: skip LLM judges only for genuinely unreadable text (and return `not_scored`, never `0.0`). A low ATS score
  on readable text is a warning (a clean Spanish resume scored exactly 40.0 before).
* Learned weights are passed *into* the engine, so renormalisation/quorum are no longer overwritten afterwards.
* The streaming endpoint (the one the UI uses) previously ran its own copy of the pipeline without timeouts, quorum,
  renormalisation or arbitration. It now drives `engine.run()` through a progress queue. One pipeline.
* Cache key covers user, text, **model, weights and pipeline version**, has a TTL (`SCORE_CACHE_TTL_DAYS`) and only
  serves `complete` runs. Bump `PIPELINE_VERSION` when changing prompts or scoring.
* Synthesized skills earn ≤ partial credit; frequency weighting uses the real job count and is per-tenant
  (it was `sum(all counts)`, ≈1.0 weights, across *all* users); precedence bug in `post_validate`; dead `low_fit_cap` removed.
* Learned weights: priors scaled by shrunken accuracy (no more mixing accuracy with priors), one outcome per user/run,
  only complete runs and successful judges train, atomic appends.
* Retry/arbitration prompts are two-sided and no longer reveal the confidence threshold (they previously nudged scores up).
* Optional `ANALYSER_SKILLS_GATE_RAMP_UPPER` removes the 0.39→0.41 cliff. **Off by default**: it changes scores, so
  re-run the benchmark first.

### Concurrency
* LLM calls are native async (`httpx`), so timeout/cancel really aborts the request. Reproduced before: 6 timed-out calls
  put 4 requests in flight against a limit of 2. API back-off is capped (30 s) and never blocks a thread.
* The fallback model is now actually reachable (it was dead code), structured-output schemas are sent to the backend
  with automatic degradation, and a client disconnect cancels the running pipeline.
* `BEGIN IMMEDIATE` for write transactions; `concurrency.run_blocking` dedicated pool for PDF/ATS/file work.
* `sem_acquire_timeout` 3600 s → 120 s with `Retry-After`.

### Injection handling
`prepare_user_text` (live path): NFKC + invisible/bidi strip + look-alike folding, then neutralise the whole
instruction-like *sentence*, flag it in `security_flags`, never reject, never reveal the rule. Strict
`sanitize_user_content` remains for callers wanting reject-on-detect. Result: 0% attack success on five variants in the benchmark.

### Deployment
* Persistent disk now mounts at `/app/state` (`HIREABLE_STATE_DIR`). Mounting it over `/app/data` hid the baked-in taxonomy and sample CSVs.
  **Existing deployments:** copy `runs.db` and `judge_feedback.jsonl` to the new location once.
* `.gitignore` `lib/` excluded `frontend/src/lib/` (imported as `@/lib/api`); a fresh clone can't build the frontend. Rule anchored
  to `/lib/`; run `git add frontend/src/lib` from your local copy.
* `render.yaml`: `plan: free` has no persistent disk on Render (verify), so state resets per deploy.

## New configuration
`ADMIN_EMAILS`, `ADMIN_BOOTSTRAP_TOKEN`, `TRUSTED_PROXY_COUNT`, `ALLOWED_API_HOSTS`, `HIREABLE_STATE_DIR`, `MAX_PDF_BYTES`, `MAX_PDF_PAGES`,
`SCORE_CACHE_TTL_DAYS`, `BLOCKING_POOL_SIZE`, `ANALYSER_LLM_HTTP_TIMEOUT`, `ANALYSER_API_MAX_RETRY_WAIT`,
`ANALYSER_STRUCTURED_OUTPUT`, `ANALYSER_SKILLS_GATE_RAMP_UPPER`, `ANALYSER_FEEDBACK_PRIOR_STRENGTH`,
`ANALYSER_MIN_JOBS_FOR_FREQUENCY` (see `.env.example`).

## Behaviour changes to be aware of
* The Benchmark page needs an admin account (register it once with `ADMIN_BOOTSTRAP_TOKEN`); `/fetch-job-url`, `/extract-pdf` and the summary/outcome routes need a token.
* `ANALYSER_MIN_FEEDBACK_SAMPLES` 3 → 5; calibration text needs that many outcomes before it enters prompts.
* The UI ignores the new `status` / `degraded` / `failed_judges` / `security_flags` fields; showing them is a small frontend change.

## Not done (and why)
* **Two-stage "frozen requirement list" extraction.** Needs prompt redesign validated against a real model; I can't do that here.
  Mitigations that shipped: synthesized-skill cap, sentence-level neutralisation, strict citations.
* **Per-judge checkpoint/resume.** Runs now have an explicit status and failed runs are never cached, but a retry recomputes all judges.
* **Per-tenant learned weights / calibration** are still global (now auth'd, ownership-checked, de-duplicated and shrunk).
* **Hidden-text detection in PDFs** (white text, tiny fonts), **keyword-stuffing defence**, **JWT revocation** (7-day tokens), moving to Postgres, and a
  multi-process-safe rate limiter (slowapi counters are per process) are unchanged.
* The synthetic benchmark validates the pipeline, not real hiring accuracy; build a human-labelled holdout.
