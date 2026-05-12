# LinkedIn Agent Assistant

Resume-to-Job fit scorer. Week 2 priority project.

## What it does

Scores how well a resume matches a job description using 3 signals:
- **TF-IDF (40%)** — text similarity between resume and JD
- **Skills (40%)** — Jaccard overlap via `skills_taxonomy.json`
- **Experience (20%)** — stub (Week 4 replaces with LLM extraction)

Prints final score, component breakdown, and missing skills list.

## Setup

```bash
conda activate ai-engineer
pip install -r requirements.txt
cp .env.example .env
```

## Usage

```bash
# Score resume against raw job text
python linkedin_cli_v1.py score --resume-id 1 --job-text "Senior NLP Engineer Python LangChain RAG..."

# Score resume against a job in jobs.csv
python linkedin_cli_v1.py score --resume-id 1 --job-id 16

# Run spec test cases
python linkedin_cli_v1.py test
```

## Example output

```
───────────────────────────────────────────────────────
  Resume : Gaurica P. — AI/MLOps Engineer - RAG & LLM Evaluation
  Job    : job_id=16 (NLP Engineer @ LegalTech P)
───────────────────────────────────────────────────────
  FINAL SCORE    : 0.7812  (78.1%)
  TF-IDF (40%)   : 0.6234
  Skills (40%)   : 0.8900
  Experience(20%): 0.8000  [stub]
───────────────────────────────────────────────────────
  Missing skills : faiss
  Latency        : 3.2 ms
───────────────────────────────────────────────────────
```

## Data

| File | Description |
|------|-------------|
| `data/resumes.csv` | 5 resumes (resume_id 1–5) |
| `data/jobs.csv` | 25 job descriptions (job_id 1–25) |
| `data/labels.csv` | 100 (resume, job) pairs — 60 fits, 40 non-fits, 20 golden |
| `data/skills_taxonomy.json` | 17 skill categories → canonical skill tokens |

## Tests

```bash
pytest tests/ -v   # 19 tests
```

## Evaluation

The project now has three eval tracks in `eval_framework.py`:

| Track | Command | What it measures |
|-------|---------|------------------|
| Scoring | `python eval_framework.py --track scoring` | Precision, recall, F1, AUC, confusion matrix, and best score threshold against `labels.csv` |
| Classifier | `python eval_framework.py --track classifier` | Cross-validated Logistic Regression on `[tfidf_score, skills_score, experience_score]` and feature importance |
| Analysis | `python eval_framework.py --track analysis --sample 5` | LLM fit-label accuracy, JSON validity, required field coverage, groundedness, calibration, actionability, completeness, honesty, hallucinations, and critical misses |

Run all non-LLM and LLM evals:

```bash
python eval_framework.py --track all
```

LLM evals require an API key:

```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_BASE_URL="https://openrouter.ai/api"
export OPENROUTER_MODEL="openrouter/free"
python eval_framework.py --track analysis --sample 5
```

Generated reports are written to `eval_results/`.

What should be evaluated:
- **Numeric ranking quality**: do good resume/JD pairs score above bad pairs?
- **Threshold quality**: what score cutoff gives the best F1 for "fit" vs "not fit"?
- **Skill extraction quality**: are matched and missing skills correct?
- **Seniority calibration**: does the system penalize junior/senior mismatches?
- **LLM output reliability**: does the analysis return valid JSON with all required fields?
- **Groundedness**: does every strength/gap cite evidence from the resume/JD?
- **Hallucination rate**: does the LLM invent experience not present in the resume?
- **Actionability**: are recommendations specific enough to edit the resume?
- **Honesty/calibration**: does the recommendation match the human label and computed score?
- **Latency and cost**: how long and how expensive is analysis per pair?

## Project structure

```
linkedin-agent-assistant/
├── linkedin_cli_v1.py      # main CLI
├── data/
│   ├── resumes.csv
│   ├── jobs.csv
│   ├── labels.csv
│   └── skills_taxonomy.json
├── tests/
│   ├── conftest.py
│   └── test_linkedin_cli.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Week 3: FastAPI REST API

### New Endpoints
- **POST /match_job** - Score resume against job description
- **POST /match_job_by_id** - Score resume against job from database
- **GET /health** - Health check
- **GET /resumes** - List available resumes
- **GET /jobs** - List available jobs
- **GET /stats** - Dataset statistics

### API Usage
```bash
# Install API dependencies
pip install -r requirements_api.txt

# Start API server
python linkedin_api.py
# or: uvicorn linkedin_api:app --reload --port 8001

# Score resume against job text
curl -X POST "http://localhost:8001/match_job" \
  -H "Content-Type: application/json" \
  -d '{"resume_id": 1, "job_description": "Senior ML Engineer Python..."}'

# Score resume against job from database
curl -X POST "http://localhost:8001/match_job_by_id" \
  -H "Content-Type: application/json" \
  -d '{"resume_id": 1, "job_id": 16}'

# Health check
curl http://localhost:8001/health
```

### API Response Example
```json
{
  "final_score": 0.7812,
  "tfidf_score": 0.6234,
  "skills_score": 0.8900,
  "experience_score": 0.8,
  "missing_skills": ["faiss"],
  "latency_ms": 3.2,
  "resume_id": 1,
  "job_description": "Senior ML Engineer..."
}
```

## Roadmap

| Week | Feature |
|------|---------|
| 2 | TF-IDF + skills scoring (CLI) |
| 3 | FastAPI REST API conversion ✅ |
| 4 | LLM-based experience extraction |
| 6 | Fine-tuned classifier on labels.csv |
| 8 | Full agent with LinkedIn API |
