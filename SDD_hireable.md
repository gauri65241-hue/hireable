# Software Design Document — Hireable

## 1. System Overview

Hireable is a resume-to-job-description scoring engine that evaluates candidate fit using a multi-judge LLM pipeline. Users upload a resume, paste job descriptions, and receive a composite score plus structured coaching feedback. A batch scorer processes large JD sets overnight. A kappa evaluation harness measures inter-rater reliability between the pipeline and human labellers.

**Runtime modes:**
- Interactive (FastAPI + React frontend, single JD)
- Batch (CLI, N JDs against one resume)
- Kappa eval (offline blind-sheet export + agreement calculation)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React Frontend (Vite + TypeScript)                     │
│  Pages: Dashboard, Analyzer, History, Settings          │
└───────────────────┬─────────────────────────────────────┘
                    │ REST + SSE
┌───────────────────▼─────────────────────────────────────┐
│  FastAPI Backend (Python)                               │
│  ┌──────────────┐ ┌────────────┐ ┌────────────────────┐ │
│  │ /api/jobs    │ │ /api/auth  │ │ /api/analyze (SSE) │ │
│  └──────────────┘ └────────────┘ └────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Scoring Engine                                   │   │
│  │  ATS Checker → 5 Judges → Coach Judge → Composite│   │
│  └───────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
┌───────▼──────┐     ┌─────────▼────────┐
│  SQLite DB   │     │  LLM Backends    │
│  runs.db     │     │  Ollama (local)  │
│              │     │  OpenRouter      │
└──────────────┘     └──────────────────┘
```

---

## 3. Components

### 3.1 ATS Checker

Pre-filter before any LLM call. Parses resume for formatting red flags: missing contact info, tables detected via whitespace heuristics, columns (two-column layouts), images, missing section headers, non-standard fonts. Outputs an ATS score (0-100).

**Gate:** If ATS score < 40, emits `early_exit` SSE event and halts. No judge calls made.

### 3.2 Five-Judge Pipeline

Each judge is an independent LLM prompt returning a structured JSON score. Judges run sequentially per JD.

| Judge | Weight | What it evaluates |
|-------|--------|-------------------|
| Skills | 25% | Skill keyword overlap, proficiency signals, missing critical skills |
| Experience | 25% | Years of experience, domain depth, progression quality |
| Seniority | 20% | Level alignment (IC vs manager, junior vs senior) |
| Relevancy | 15% | Domain and industry match |
| Impact | 15% | Quantified achievements, business outcomes, scope |

Each judge returns: `{ score: 0-100, reasoning: string, gaps: string[] }`.

### 3.3 Coach Judge

Synthesises all five judge outputs into actionable feedback. Not scored — output only. Generates: top 3 strengths, top 3 gaps, 3 specific resume edits ranked by impact.

### 3.4 Composite Scorer

```
composite = (skills × 0.25) + (experience × 0.25) + (seniority × 0.20)
          + (relevancy × 0.15) + (impact × 0.15)
```

Result stored in `runs` table with full judge breakdown JSON.

### 3.5 Semantic Similarity (SBERT)

Sentence-BERT embeddings computed for resume summary + JD summary. Cosine similarity used as a signal fed to the relevancy judge, not as a standalone score. Model loaded once at startup, cached in memory.

### 3.6 Learned Weights System

Stores per-user weight overrides in `user_weights` table. Admins or power users can shift judge weights based on hiring domain (e.g., weight skills higher for technical roles). Applied at composite calculation time.

### 3.7 Batch Scorer

CLI entry: `python scripts/batch_score.py --resume <path> --jobs <sqlite-db> --user-id <uuid>`

- Reads all JDs from `jobs` table for the given user
- Runs full pipeline on each JD sequentially
- Writes results to `runs` table
- Prints progress to stdout, writes summary CSV on completion
- PID and run parameters logged to allow resume on failure

### 3.8 Kappa Evaluation Harness

`scripts/kappa.py` — measures inter-rater reliability between pipeline scores and human labels.

**Export:** `python scripts/kappa.py export --jobs 1-12 --out data/kappa_java.csv`
- Produces blind CSV with job text, no scores — for human labellers

**Calculate:** `python scripts/kappa.py calculate --human data/kappa_java_labelled.csv --pipeline data/runs.db`
- Computes Cohen's kappa across label categories (strong fit / possible fit / no fit)

---

## 4. Data Structures

### 4.1 SQLite Schema (`runs.db`)

**`users`**
```
id TEXT PK (UUID)
email TEXT UNIQUE
password_hash TEXT
created_at TEXT
```

**`jobs`**
```
id INTEGER PK AUTOINCREMENT
user_id TEXT → users (was NULL for 31 legacy rows — migration applied)
title TEXT
company TEXT
description TEXT
url TEXT
source TEXT
created_at TEXT
```

**`resumes`**
```
id INTEGER PK AUTOINCREMENT
user_id TEXT → users
filename TEXT
content TEXT         -- raw extracted text
parsed_sections JSONB -- {contact, summary, experience[], education[], skills[]}
created_at TEXT
```

**`runs`**
```
id INTEGER PK AUTOINCREMENT
user_id TEXT → users
job_id INTEGER → jobs
resume_id INTEGER → resumes
ats_score REAL
skills_score REAL
experience_score REAL
seniority_score REAL
relevancy_score REAL
impact_score REAL
composite_score REAL
judge_outputs JSONB   -- full per-judge {score, reasoning, gaps}
coach_output JSONB    -- {strengths[], gaps[], edits[]}
model TEXT            -- which LLM was used
created_at TEXT
```

**`user_weights`**
```
id INTEGER PK
user_id TEXT → users
skills_weight REAL DEFAULT 0.25
experience_weight REAL DEFAULT 0.25
seniority_weight REAL DEFAULT 0.20
relevancy_weight REAL DEFAULT 0.15
impact_weight REAL DEFAULT 0.15
```

**`skills_taxonomy`**
```
id INTEGER PK
canonical_name TEXT
aliases TEXT[]       -- "JS", "JavaScript", "javascript"
category TEXT        -- frontend | backend | ml | data | devops | etc.
```

### 4.2 SSE Event Stream (Analysis)

```
event: progress
data: { stage: "ats" | "skills" | "experience" | "seniority" | "relevancy" | "impact" | "coach", percent: 0-100 }

event: early_exit
data: { detail: "ATS score too low — fix resume formatting before running AI analysis." }

event: result
data: { ats_score, composite_score, judge_scores: {}, coach: {}, run_id }

event: error
data: { message: string }
```

---

## 5. API Interfaces

All routes require `Authorization: Bearer <JWT>` unless noted.

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/auth/signup` | Create account, return JWT |
| POST | `/api/auth/login` | Verify credentials, return JWT |
| GET | `/api/jobs` | List JDs for authenticated user |
| POST | `/api/jobs` | Add JD (title, company, description) |
| DELETE | `/api/jobs/{id}` | Remove JD |
| GET | `/api/resumes` | List uploaded resumes |
| POST | `/api/resumes` | Upload resume (multipart/form-data) |
| GET | `/api/analyze` | SSE stream — full pipeline for job+resume pair |
| GET | `/api/runs` | History of scored runs |
| GET | `/api/runs/{id}` | Full run detail including judge outputs |
| GET | `/api/weights` | Current user weight config |
| PATCH | `/api/weights` | Update judge weights |

---

## 6. Key Algorithms

### Resume Parsing
1. Extract text via `pdfminer` or `python-docx`
2. Split into sections via regex on header keywords (EXPERIENCE, EDUCATION, SKILLS, etc.)
3. Extract contact block (email, phone, LinkedIn) from top ~10 lines
4. Skills extracted by matching against `skills_taxonomy` aliases (case-insensitive)

### ATS Scoring
- Each red flag carries a penalty weight
- Table detected: -30
- Missing contact: -20
- Two-column layout: -20
- Missing standard headers: -10 each (capped at -20)
- Score = 100 - sum(penalties), floor 0

### Semantic Similarity
```python
resume_embedding = sbert_model.encode(resume_summary)
jd_embedding = sbert_model.encode(jd_summary)
similarity = cosine_similarity(resume_embedding, jd_embedding)  # 0.0 – 1.0
```
Passed to relevancy judge as `context.semantic_similarity`.

### Composite with Custom Weights
```python
weights = get_user_weights(user_id)  # falls back to defaults
composite = sum([
    skills_score * weights.skills,
    experience_score * weights.experience,
    seniority_score * weights.seniority,
    relevancy_score * weights.relevancy,
    impact_score * weights.impact,
])
```

---

## 7. Auth Flow

1. Signup: email + password → `bcrypt` hash stored → UUID generated → JWT signed with `SECRET_KEY`
2. Login: verify hash → issue JWT (expires 24h)
3. All API routes: `Authorization: Bearer <token>` → decoded via `python-jose` → `user_id` extracted
4. Frontend stores JWT in `localStorage`, attaches to every request header

---

## 8. LLM Backend Configuration

Controlled via env vars:

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openrouter` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3` | Model tag |
| `OPENROUTER_API_KEY` | — | OpenRouter key (if provider = openrouter) |
| `OPENROUTER_MODEL` | `anthropic/claude-3-haiku` | Model string |

All judge prompts are provider-agnostic — same prompt template regardless of backend. Provider abstracted behind `llm_client.complete(prompt) → str`.

---

## 9. Data Flow — Single Analysis

```
User selects resume + JD
        │
        ▼
POST /api/analyze (SSE)
        │
        ▼
ATS Checker → score < 40? → emit early_exit → close stream
        │ score ≥ 40
        ▼
Skills Judge → emit progress(skills)
        │
        ▼
Experience Judge → emit progress(experience)
        │
        ▼
Seniority Judge → emit progress(seniority)
        │
        ▼
Relevancy Judge (+ semantic_similarity) → emit progress(relevancy)
        │
        ▼
Impact Judge → emit progress(impact)
        │
        ▼
Coach Judge (all 5 outputs as context)
        │
        ▼
Composite calculation (apply user weights)
        │
        ▼
Write to runs table
        │
        ▼
emit result → close stream
```

---

## 10. Frontend Architecture

**Pages:**
- `Dashboard` — resume list + JD list, trigger analysis
- `Analyzer` — SSE progress UI, score display, coach feedback
- `History` — paginated run history, filter by date/score
- `Settings` — weight sliders, LLM provider config

**State management:** React Context for auth (JWT, user). Local component state for analysis stream. No global store.

**SSE handling in Analyzer.tsx:**
```typescript
const es = new EventSource(`/api/analyze?job_id=${jobId}&resume_id=${resumeId}`, { headers })
es.addEventListener('progress', handler)
es.addEventListener('result', handler)
es.addEventListener('early_exit', handler)  // ← added fix
es.addEventListener('error', handler)
```

---

## 11. What Can Be Built Next

**High-value, near-term:**
- **Multi-resume support per run**: score same JD against 3 resume versions simultaneously — shows which version wins and by how much
- **Resume diff view**: when user edits resume, show delta in composite score with attribution (which judge moved)
- **JD keyword extractor**: highlight which JD terms are present/absent in resume, visible inline
- **Export**: PDF report per run (score card + coach notes), shareable link

**Medium-term:**
- **Kappa dashboard**: once labelling session done, visualise pipeline vs human agreement per judge, identify weakest judge
- **Calibration mode**: when kappa < 0.6 on a judge, surface the disagreed-on examples for prompt tuning
- **Application tracker**: add status column to jobs (applied / screened / interview / offer / rejected), pipeline graph view
- **Cover letter generator**: given resume + JD + coach gaps, generate targeted cover letter para per gap

**Architectural:**
- **Async batch via queue**: replace synchronous batch_score.py with a task queue (Celery/RQ) — allows concurrent batch runs, progress via websocket
- **Embedding cache**: store SBERT embeddings per resume/JD chunk, skip recomputation on re-score
- **Prompt versioning**: track which prompt version produced each run — enables A/B comparison when prompts change
