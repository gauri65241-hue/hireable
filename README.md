# hireable

![CI](https://github.com/gauri65241-hue/hireable/actions/workflows/ci.yml/badge.svg)

LLM-as-judge pipeline that scores resume-job fit across 5 independent dimensions. Each judge runs a structured prompt, returns labelled JSON, and all scoring arithmetic happens in Python — not delegated to the LLM. A coach agent synthesises recommendations across judges and rewrites existing bullets without fabricating experience.

---

## What it does

### 5-Judge Scoring Pipeline

Each judge evaluates one dimension independently. All five run concurrently before the coach synthesises.

```
Resume + Job Description
         │
         ├─► ATS Pre-flight (instant, no LLM)
         │
         ▼
┌────────────────────────────────────────┐
│         5 Parallel LLM Judges          │
│                                        │
│  Skills     (25%) → skill gap analysis │
│  Experience (25%) → depth + relevance  │
│  Seniority  (20%) → level match        │
│  Relevancy  (15%) → domain alignment   │
│  Impact     (15%) → quantified outcomes│
└────────────────────────────────────────┘
         │
         ▼
   Weighted score (0–100%)
         │
         ▼
┌────────────────────────────────────────┐
│            Coach Judge                 │
│  bullet_rewrite → reframe existing     │
│  deletion       → remove irrelevant    │
│  gap_warning    → flag real gaps       │
└────────────────────────────────────────┘
         │
         ├─► Tailored resume
         └─► Interview prep
```

### Scoring internals

The LLM extracts and labels; Python scores. This keeps scoring deterministic and auditable.

| Judge | Formula |
|-------|---------|
| Skills, Experience | `(present + 0.5 × partial) / total` |
| Seniority, Relevancy | `binary: demonstrated / total` |
| Impact | `(quantified + 0.5 × directional) / total` |
| Overall | `skills×0.35 + experience×0.30 + seniority×0.10 + relevancy×0.15 + impact×0.10` |

### Evaluation harness

- Spearman correlation, Cohen's Kappa, F1, precision/recall against human-labelled ground truth
- Adversarial test suite: injection attempts, keyword stuffing, tag breakout, empty resume
- Benchmark scripts for model selection across judge parse success rate and discrimination accuracy

### ATS Checker

9-criterion heuristic scan with no LLM — results in under a second. PDF-aware: handles pypdf/pdfminer stripping bullet characters and collapsing section headings.

### Career Paths

Scores resume skills against 20 income paths (full-time, freelance, consulting, content). Pure token overlap — no LLM.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn (async, concurrent judge calls) |
| LLM providers | Ollama (local, default `cogito:14b`) + OpenRouter (optional API fallback) |
| Auth | JWT (HS256, 7-day expiry) + bcrypt password hashing |
| Persistence | SQLite — run history, scores, tailored text |
| Frontend | React + TypeScript + Vite + Tailwind |
| Deployment | Docker + docker-compose + nginx |

---

## Project structure

```
hireable/
├── linkedin_api.py          # FastAPI app entry point
├── analysers/               # One module per judge dimension
│   ├── skills/
│   ├── experience/
│   ├── seniority/
│   ├── relevancy/
│   ├── impact/
│   ├── coach/
│   └── engine.py            # Runs all judges concurrently
├── routers/                 # FastAPI route handlers
│   ├── analysis.py
│   ├── benchmark.py
│   ├── career.py
│   ├── data.py
│   ├── tools.py
│   └── auth.py
├── specialized_judges.py    # Judge prompts + LLM calls
├── ats_checker.py           # ATS heuristic scanner (no LLM)
├── chunker.py               # Text parsing + prompt formatting
├── db.py                    # SQLite persistence
├── frontend/                # React + TypeScript source
├── scripts/                 # Eval, benchmark, calibration scripts
├── tests/                   # Unit + adversarial benchmarks
├── data/
│   ├── resumes.sample.csv   # Format reference (fake data)
│   ├── jobs.sample.csv      # Format reference (fake data)
│   ├── skills_taxonomy.json # Skill domain → token mapping
│   └── income_paths.json    # 20 income paths
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Clone and configure environment**

```bash
git clone https://github.com/gauri65241-hue/hireable.git
cd hireable
cp .env.example .env
# Add your OpenRouter API key to .env
```

**2. Install dependencies**

```bash
# Conda
conda env create -f environment.yml
conda activate hireable

# Or pip
pip install -r requirements.txt
```

**3. Run**

```bash
uvicorn linkedin_api:app --reload
```

Open `http://localhost:8000`. The React frontend is served at `/` by FastAPI from `frontend_dist/`. To rebuild the frontend after changes:

```bash
cd frontend && npm install && npm run build
```

**Or with Docker (local, includes Ollama)**

```bash
# Copy and fill in your secrets
cp .env.example .env

# Build and start — first run downloads Ollama + SBERT model (~2 GB)
docker-compose up --build

# Pull the LLM model into Ollama (one-time)
docker-compose exec ollama ollama pull llama3.1:8b
```

Open `http://localhost`.

---

## Deploy to Render

One-service deploy — FastAPI serves both the API and the built React frontend.

1. Fork the repo on GitHub
2. Create a new **Web Service** on [render.com](https://render.com), connect the repo
3. Render auto-detects `render.yaml` — review and confirm
4. Set these environment variables in the Render dashboard:
   - `OPENROUTER_API_KEY` — your OpenRouter key
   - `ALLOWED_ORIGINS` — your Render URL, e.g. `https://hireable.onrender.com`
5. Deploy. First build takes ~5 min (installs deps + pre-warms SBERT)

The `render.yaml` uses the sample data files (`data/*.sample.csv`) by default. To use real data, upload via the Render disk or the `/resumes` and `/jobs` API endpoints.

---

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /auth/register` | Create account, return JWT |
| `POST /auth/login` | Verify credentials, return JWT |
| `GET /auth/me` | Return current user id + email |
| `POST /analyze` | Run all 5 judges + coach |
| `POST /ats-check` | Instant ATS scan, no LLM |
| `POST /ats-deep-audit` | 4-dimension LLM deep audit |
| `POST /skill-match` | Score against 20 income paths |
| `POST /edit_resume` | Apply coach → tailored resume |
| `POST /interview_prep` | Role-specific questions + answers |
| `GET /list_results` | Saved pipeline results |
| `POST /fetch-job-url` | Scrape JD from URL |
| `POST /parse-jd-text` | Parse pasted JD via LLM |
| `POST /extract-pdf` | Extract + structure PDF resume |
| `GET /health` | API + Ollama + DB status |
| `GET /ollama-models` | List installed Ollama models |
| `GET /openrouter-models` | List configured OpenRouter model pool |
| `POST /benchmark-ollama` | Synthetic Ollama model benchmark |
| `POST /benchmark-ollama-real` | Real-data Ollama benchmark |
| `POST /record_outcome` | Log interview/job result for weight learning |

Full docs at `http://localhost:8000/docs`.

---

## Evaluation

```bash
# Run eval harness against human-labelled ground truth
python scripts/run_eval.py

# Cohen's Kappa inter-rater agreement
python scripts/kappa.py

# Model benchmark across judge parse success + discrimination
python scripts/run_benchmarks.py
```

---

## Cost and latency

Numbers from `data/model_benchmark_ollama_real.json` — 8 models, 3 jobs, real resume/JD pairs, local Ollama on M-series Mac.

### Latency

Measured via `scripts/run_benchmarks.py`, 8 models × 3 real resume/JD pairs, local Ollama on M-series Mac.

| Stage | cogito:14b | qwen2.5:14b | mistral-nemo |
|-------|-----------|-------------|--------------|
| ATS check (no LLM) | <1s | <1s | <1s |
| 5 judges concurrent (wall-clock) | 15.3s avg | 16.7s avg | 10.8s avg |
| Coach/audit | 32.9s | 41.6s | 22.5s |
| **End-to-end full analysis** | **~48s** | **~58s** | **~34s** |

`cogito:14b` best composite score (accuracy 0.80, composite 0.391). `mistral-nemo` is fastest (~34s) but accuracy drops to 0.30 — not usable. `qwen2.5:14b` comparable accuracy to cogito at slightly higher latency.

### Cost

| Setup | Per full analysis | Per ATS-only check |
|-------|------------------|--------------------|
| Local Ollama | $0 | $0 |
| OpenRouter, 14B (e.g. ~$0.20/M in, $0.60/M out) | ~$0.003–0.005 | $0 |
| OpenRouter, GPT-4o class | ~$0.08–0.15 | $0 |

Token budget per full analysis: ~8k input (5 judges + coach prompts) + ~2k output (structured JSON responses). ATS check is pure heuristic — zero LLM tokens.

---

## Data

Real resume/job data is gitignored. Only fake sample files ship with the repo.

| File | Description |
|------|-------------|
| `data/resumes.sample.csv` | Format reference — fake data |
| `data/jobs.sample.csv` | Format reference — fake data |
| `data/skills_taxonomy.json` | Skill domain token mapping |
| `data/income_paths.json` | 20 income paths with required skills |
