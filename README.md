# LinkedIn Agent Assistant

LLM-as-judge pipeline that scores resume-job fit across 5 dimensions, generates a tailored resume, and produces role-specific interview prep — all running locally via Ollama.

## What it does

Runs a resume and job description through 5 specialised LLM judges, each scoring a different dimension:

| Judge | Weight | What it evaluates |
|-------|--------|-------------------|
| Skills | 25% | Required vs present skills against a taxonomy |
| Experience | 25% | Relevance and depth of past work |
| Seniority | 20% | Level match (junior/mid/senior) |
| Relevancy | 15% | Domain and industry alignment |
| Impact | 15% | Quantified results and business outcomes |

After scoring, a **Coach judge** produces honest improvement recommendations — rewrites of existing bullets (`bullet_rewrite`), removals of irrelevant content (`deletion`), and flagged genuine gaps (`gap_warning`). No fabricated metrics, no invented experience.

From any scored result you can generate:
- **Tailored resume** — applies coach rewrites to the original, cached to `results/tailored_resumes/`
- **Interview prep** — role-specific questions with model answers for each real gap, cached to `results/interview_prep/`

## Stack

- **Backend**: FastAPI + Uvicorn
- **LLM**: Ollama (local) — benchmarked across multiple models, `deepseek-r1:14b` selected as best judge
- **Frontend**: Single-file dashboard (`dashboard.html`)
- **Data**: CSV-based resume and job store

## Setup

```bash
conda env create -f environment.yml
conda activate linkedin-agent-assistant

# Or pip:
pip install -r requirements_api.txt

cp .env.example .env
```

Requires [Ollama](https://ollama.com) running locally with at least one model pulled:

```bash
ollama pull deepseek-r1:14b
ollama serve
```

## Run

```bash
uvicorn linkedin_api:app --reload
```

Open `dashboard.html` in a browser. Add your resume and a job description, then run the pipeline.

## Data files

Add your own data — files are gitignored after the initial empty commit:

| File | Description |
|------|-------------|
| `data/resumes.csv` | Resumes — see `resumes.sample.csv` for format |
| `data/jobs.csv` | Job descriptions — see `jobs.sample.csv` for format |
| `data/labels.csv` | Optional human labels for evaluation |
| `data/skills_taxonomy.json` | 10 skill domains → canonical tokens used by the skills judge |

## Results structure

All outputs are saved locally and never committed:

```
results/
├── pipeline/          # Full judge pipeline output per run
├── skills_judge/      # Individual judge results
├── experience_judge/
├── seniority_judge/
├── relevancy_judge/
├── impact_judge/
├── coach_judge/
├── tailored_resumes/  # Generated tailored resumes (.txt)
├── interview_prep/    # Generated interview prep (.json)
└── summary/           # Result summaries
```

## Model benchmarking

Run `benchmark_models.py` to evaluate which local model performs best as a judge on your hardware:

```bash
python benchmark_models.py
```

Results written to `benchmark_results.json`. The pipeline automatically uses the top-ranked model.

## Project structure

```
linkedin-agent-assistant/
├── linkedin_api.py         # FastAPI backend — all endpoints
├── specialized_judges.py   # Judge prompts and parsing logic
├── dashboard.html          # Frontend
├── benchmark_models.py     # Model selection benchmark
├── chunker.py              # Text chunking utilities
├── data/
│   ├── resumes.csv
│   ├── jobs.csv
│   ├── labels.csv
│   └── skills_taxonomy.json
├── tests/
├── requirements_api.txt
├── environment.yml
└── .env.example
```
