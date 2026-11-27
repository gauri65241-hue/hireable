# Evaluation harness

Continuously measures the *pipeline* (not just the model) against a synthetic dataset whose gold
labels are known by construction.

| Run | Command | When | Needs |
|---|---|---|---|
| Plumbing gate | `python -m benchmark.run --backend fake` | every PR (CI `eval-gate`) | nothing |
| Real model | `python -m benchmark.run --backend real --model cogito:14b --repeats 3` | nightly / manual | Ollama or `OPENROUTER_API_KEY` |

`--backend fake` swaps the five LLM judges for deterministic rule-based ones that are **gullible**
(they obey any instruction-like text that reaches them). Whatever the sanitiser lets through shows
up as attack success. It validates sanitisation, the ATS gate, scoring maths, gates, quorum, weights
and status handling — *not* LLM quality.

## Metrics
| Axis | Metrics |
|---|---|
| Format adherence | complete / degraded / insufficient / not_scored / crash rates, first-try schema pass (real) |
| Classification | precision, recall, F1 at 0.75; **knockout recall** (missing most must-haves but senior prose — must never pass); adjacent-domain recall |
| Ranking | pairwise accuracy inside each JD (+ bootstrap 95% CI), Spearman vs gold tier |
| Robustness | mean abs score delta and decision-flip rate under merged lines, flattened tables, non-English headings, emoji, keyword stuffing, 50-page logs |
| Injection | attack success rate: share of poor/partial fits whose decision flips (or score rises > 0.05) when one of 5 injection variants is added |
| Guards | unreadable / empty input must be **not scored** — never a numeric (zero) score |
| Stability | per-case std-dev and decision-flip rate over `--repeats` runs (real) |
| Ops | latency p50 / p95 |

## Gating
`thresholds.json` holds absolute limits per backend. `--baseline path.json` additionally fails on a
regression of the tracked metrics (pairwise accuracy, F1, knockout recall, complete rate, injection ASR)
beyond a 0.02 tolerance. Commit `benchmark_results.json` from a good nightly run as
`benchmark/baseline.real.json` to enable it.

## Things the synthetic set cannot tell you
* Real-world hiring accuracy. Keep a small **human-labelled holdout** (≥ 50 resume/JD pairs, double-annotated)
  and report it separately; a few hundred pairs are needed before a 2-point difference is meaningful.
* Keyword-stuffing robustness for a real LLM: the fake judges are keyword matchers, so that metric is
  informational in the fake run and gated only in the real run.
* Use a different model family to *generate* data than the one you *judge* with, to avoid self-preference.

## Existing `benchmark_suite.py`
The pre-existing `python -m benchmark.benchmark_suite` (hand-written cases, real LLM) is unchanged and complements this harness; it is not wired into CI.
