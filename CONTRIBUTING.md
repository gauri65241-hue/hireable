# Contributing

Contributions welcome. Read this before opening a PR.

---

## What's in scope

- **New judges** — additional scoring dimensions (e.g. culture fit signals, keyword density, ATS formatting checks)
- **Skills taxonomy** — extend `data/skills_taxonomy.json` with missing domains or tokens
- **Scraper coverage** — fix or add job board scrapers in `linkedin_api.py` (currently fails on LinkedIn auth-gated pages and JS-heavy SPAs)
- **Model benchmarks** — run `benchmark_models.py` against a new Ollama model and open a PR with results appended to `benchmark_results.json`
- **Eval improvements** — better faithfulness metrics in `eval.py` (current relevancy grounding is keyword overlap — embedding similarity is an obvious upgrade)
- **Bug fixes** — broken parsing, wrong scores, coach generating fabricated metrics

## What's out of scope

- Cloud API integrations (OpenAI, Anthropic, etc.) — this project stays local-only
- UI framework rewrites — `dashboard.html` is intentionally a single file with no build step
- Adding dependencies without a clear reason — check `requirements.txt` before proposing new packages
- Resume fabrication features — the coach must never invent experience or metrics; PRs that weaken this constraint will be closed

---

## How to contribute

1. Open an issue first for anything non-trivial. State what you're fixing and why.
2. Fork, branch off `main`, make your change.
3. Tests live in `tests/`. Run `pytest` before opening the PR. New behaviour needs a test.
4. Keep PRs small and single-concern. One fix or feature per PR.
5. PR description: what changed, why, and how you tested it.

---

## Local setup

```bash
ollama pull deepseek-r1:14b
ollama serve

conda env create -f environment.yml
conda activate linkedin-agent-assistant

cp .env.example .env
uvicorn linkedin_api:app --reload
```

Open `dashboard.html` in a browser.

---

## Code style

`ruff` is configured — run `ruff check .` before pushing. Pre-commit hooks handle this automatically if you run `pre-commit install`.
