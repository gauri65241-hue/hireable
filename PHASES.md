# Portfolio Build-Out — Phase Roadmap

---

## Phase 1 — Quick Wins ✓ DONE

All completed automatically:
- MIT LICENSE added to all 16 projects
- Mermaid architecture diagrams added to all SDDs
- `slowapi` rate limiting added to all 9 FastAPI projects

---

## Phase 2 — Make Hireable Publicly Demonstrable

### Step 1: Push to GitHub
```bash
cd /Users/gauricapuri/Work/Projects/linkedin-agent-assistant
git init
git add .
git commit -m "Add Docker, Render deploy, frontend serving, rate limiting"
git remote add origin https://github.com/gauri65241-hue/hireable.git
git push -u origin main
```

### Step 2: Get OpenRouter API key
- Go to openrouter.ai → sign up → API Keys → create key
- Free tier works for demo purposes

### Step 3: Deploy on Render
1. render.com → sign up with GitHub
2. New → Web Service → connect `hireable` repo
3. Render detects `render.yaml` automatically → click Apply
4. Environment tab → add:
   - `OPENROUTER_API_KEY` = your key
   - `ALLOWED_ORIGINS` = `https://hireable.onrender.com`
5. Click Deploy — first build ~5 min

### Step 4: Verify
- `https://hireable.onrender.com/health` → `{"status": "ok"}`
- `https://hireable.onrender.com/` → React UI loads
- Register account, upload resume + JD, run full analysis

### Step 5: Record walkthrough video (2-3 min)
Script:
1. Open the live URL (30s) — show the landing page
2. Upload a resume PDF + paste a job description (30s)
3. Hit Analyze — watch the SSE stream judges arriving in real time (45s)
4. Show the score breakdown + coach recommendations (30s)
5. Show the ATS checker (15s)
6. Show the eval harness / kappa results (15s)

Tools: Loom (free), QuickTime screen record, or OBS
Upload to YouTube (unlisted is fine) → link in README

### Step 6: Add demo section to README
Add to the top of the README:
- Live demo URL
- YouTube walkthrough link
- One screenshot of the analysis results UI
- One screenshot of the judge score breakdown

---

## Phase 3 — Engineering Credibility Signals

### Step 1: GitHub Actions CI on hireable

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: ruff format --check .
      - run: pytest tests/ -v --tb=short
      - name: Type check
        run: pip install pyright && pyright linkedin_api.py routers/ analysers/
```

Then commit and push. The CI badge in the README will show green.

To add the badge to README, add this line at the top:
```
![CI](https://github.com/gauri65241-hue/hireable/actions/workflows/ci.yml/badge.svg)
```

### Step 2: Structured logging on hireable

Replace `print()` calls in the FastAPI app with Python `logging`. Do this:

1. Create `logger.py` in the project root:
```python
import logging, json, sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **({"exc": self.formatException(record.exc_info)} if record.exc_info else {})
        })

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JSONFormatter())
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger
```

2. In `linkedin_api.py`, replace:
```python
print("Starting hireable API...")
```
with:
```python
from logger import get_logger
log = get_logger("hireable")
log.info("Starting hireable API")
```

3. In each router, replace `print()` with `log = get_logger(__name__)` and `log.info()` / `log.warning()` / `log.error()`

4. Add request ID middleware to `linkedin_api.py`:
```python
import uuid
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
```

### Step 3: Write tests for ai-agents, ai-rag, ai-llmops

These three have empty `tests/` directories but SDDs claim coverage.

**ai-agents** — create `tests/test_agent.py`:
- Test `run_research_agent()` with mocked Ollama (returns fallback profile when Ollama unreachable)
- Test `scrape_url()` with a static HTML fixture
- Test `search_web()` returns non-empty string

**ai-rag** — create `tests/test_rag.py`:
- Test `FixedSizeChunker` produces correct chunk sizes
- Test `/ingest/text` adds documents to ChromaDB (use test collection name)
- Test `/query` returns citations when documents exist
- Test `/query` returns empty when collection empty

**ai-llmops** — create `tests/test_logger.py`:
- Test `LLMLogger.trace()` context manager writes a span to DB
- Test `compute_cost()` returns correct value for known model
- Test alert threshold fires when metric exceeds threshold

For each: `cd <project> && conda activate ai-engineer && pytest tests/ -v`

---

## Phase 4 — Technical Depth (hireable only)

### Step 1: RAG layer for evidence-grounded scoring

This is the most important technical upgrade. Goal: each judge cites the specific resume/JD text it scored from, not free-form LLM claims.

**Architecture:**
```
Resume + JD
    ↓
Section-aware chunking (chunker.py already exists — extend it)
    ↓
Embed chunks → ChromaDB collection per run_id
    ↓
Each judge retrieves top-3 relevant chunks before scoring
    ↓
Judge prompt = "Score based ONLY on this evidence: {chunks}"
    ↓
Score + evidence_quotes[] in JSON output
```

**Steps:**
1. Add `chromadb` and `sentence-transformers` to requirements (already there)
2. Create `analysers/retrieval.py`:
   - `index_run(run_id, resume_text, job_text)` → chunks → embeddings → ChromaDB
   - `retrieve(run_id, query, k=3)` → top-k chunks with scores
3. Update each judge prompt to receive retrieved context
4. Update judge output schema to include `evidence: list[str]`
5. Update `db.py` to store evidence per judge in `run_summaries`
6. Update frontend to show evidence quotes under each score
7. Clean up ChromaDB collection after run completes (or keep for 24h)

**File to start with:** `analysers/skills/judge.py` — add retrieval call before the LLM prompt

### Step 2: Multi-labeller evaluation harness

Goal: make the kappa harness employer-ready with proper inter-rater agreement.

**Steps:**
1. Export 50 resume-JD pairs from the DB:
   ```bash
   python scripts/export_for_labelling.py --n 50 --out data/label_batch_01.csv
   ```
   Strip candidate names, email addresses, phone numbers from the export

2. Label them yourself twice (one week apart, blind second pass) using `scripts/label_ui.py` or a Google Sheet with columns: `run_id | fit_score_1_5 | labeller_id | notes`

3. Calculate Cohen's kappa between your two passes:
   ```bash
   python scripts/kappa.py --file data/label_batch_01.csv
   ```
   Target κ > 0.6 before claiming "inter-rater agreement"

4. Add Fleiss' kappa support to `scripts/kappa.py` for 3+ labellers

5. Add calibration plot: scatter LLM 0-100 score vs human 1-5 rating, fit a line, report R²

6. Document actual κ value in README once measured

### Step 3: Public demo account

1. Create anonymised sample data:
   - Take `data/resumes.sample.csv` — add 3-4 realistic (fake) resumes across different seniority levels
   - Add 5-6 job descriptions covering ML engineer, data scientist, backend engineer
   - Pre-run analysis on all combinations, store results in a `demo.db`

2. Add demo login to the app:
   - Username: `demo@hireable.app` / Password: `demo1234`
   - Demo account is read-only (can run analysis, cannot delete data)
   - Add `is_demo: bool` flag to user table, block destructive operations

3. Show pre-computed results on load for the demo account so it works even when Ollama/OpenRouter is slow

4. Add to README:
   ```
   Live demo: https://hireable.onrender.com
   Demo login: demo@hireable.app / demo1234
   ```

---

## Priority Order Summary

| # | Task | Time | Impact |
|---|------|------|--------|
| 1 | ✓ LICENSE, diagrams, rate limiting | Done | Medium |
| 2 | Deploy to Render + walkthrough video | 2-3h | High |
| 3 | GitHub Actions CI | 1h | High |
| 4 | Structured logging | 2h | Medium |
| 5 | Tests for ai-agents/rag/llmops | 4h | Medium |
| 6 | RAG layer in hireable | 2-3 days | Very high |
| 7 | Multi-labeller kappa harness | 1 day | High |
| 8 | Demo account + anonymised data | 3h | High |

The deployment + video (Phase 2) is the highest single ROI item remaining — it turns hireable from a local project into something you can share in 10 seconds.
