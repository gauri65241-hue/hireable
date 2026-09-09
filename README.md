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
| Overall | `skills×0.25 + experience×0.25 + seniority×0.20 + relevancy×0.15 + impact×0.15` |

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
| LLM provider | OpenRouter (configurable model pool) |
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

Open `http://localhost:8000`.

**Or with Docker**

```bash
docker-compose up --build
```

---

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /analyze` | Run all 5 judges + coach |
| `POST /ats-check` | Instant ATS scan, no LLM |
| `POST /skill-match` | Score against 20 income paths |
| `POST /edit_resume` | Apply coach → tailored resume |
| `POST /interview_prep` | Role-specific questions + answers |
| `GET /list_results` | Saved pipeline results |
| `POST /fetch-job-url` | Scrape JD from URL |
| `POST /parse-jd-text` | Parse pasted JD via LLM |
| `POST /extract-pdf` | Extract + structure PDF resume |

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

| Stage | cogito:14b | qwen2.5:14b | mistral-nemo |
|-------|-----------|-------------|--------------|
| ATS check (no LLM) | <1s | <1s | <1s |
| Per-judge call | 15.3s avg | 16.7s avg | 10.8s avg |
| 5 judges concurrent | ~16s wall-clock | ~18s wall-clock | ~15s wall-clock |
| Coach/audit | ~33s | ~42s | ~23s |
| **End-to-end full analysis** | **~50s** | **~60s** | **~38s** |

`cogito:14b` wins on composite (accuracy × speed): score accuracy 0.80, best overall. `mistral-nemo` is fastest but accuracy drops to 0.30.

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
