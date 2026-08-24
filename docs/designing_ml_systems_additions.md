# Designing Machine Learning Systems — Chip Huyen
## Additions for hireable (linkedin-agent-assistant)

Source: `/Users/gauricapuri/Desktop/books/Designing Machine Learning Systems.pdf`
Chapters reviewed: 1 (ML Systems Overview), 2 (System Design), 3 (Data Collection), 4 (Feature Engineering), 5 (Model Development), 6 (Evaluation), 7 (Deployment), 8 (Distribution Shifts & Monitoring), 9 (Continuous Learning), 10 (Infrastructure & MLOps)

Note: CoT, few-shot, LoRA/DPO, JSON parsing, YAML, hybrid search, SetFit, SBERT, constrained sampling, quantization, and batching are covered in prior books. Items below are net-new.

---

## Ch 3 — Data Collection and Preparation

### 1. Judge Disagreement as Active Learning Signal
**What:** Book recommends focusing human labeling effort on uncertain examples, not random ones. Uncertainty = samples where the model is least confident. For ensemble models, high disagreement between sub-models = high uncertainty.

**For hireable:** When two judges produce wildly different scores for the same candidate (e.g., seniority=80%, impact=15%), that candidate is uncertain. These are exactly the cases where the pipeline is most likely wrong — and where human annotation effort is most valuable.

**Implementation:**
```python
# In analysers/engine.py — after aggregating judge scores
judge_scores = [skills_score, experience_score, seniority_score, relevancy_score, impact_score]
disagreement = max(judge_scores) - min(judge_scores)
if disagreement > 0.50:
    result["_flag"] = "high_judge_disagreement"
    result["_disagreement"] = round(disagreement, 3)
    # Log to a separate file for human review
    with open("data/uncertain_candidates.jsonl", "a") as f:
        json.dump({"id": candidate_id, "scores": judge_scores, "disagreement": disagreement}, f)
```

**Where:** `analysers/engine.py`. Also log these to a `data/uncertain_candidates.jsonl` file. Review this file periodically to find the highest-value items to add to the eval set.

**Expected impact:** Instead of randomly expanding the eval set, you're adding the items most likely to catch judge failures. 10 curated uncertain items > 30 random items for improving coverage.

---

### 2. Edge Case Test Set — Separate from Main Eval
**What:** Book warns that models trained on typical data fail silently on outliers. Standard eval sets reflect the middle of the distribution — they won't catch systematic failures on unusual inputs.

**For hireable — create `data/eval_edge_cases.json` with these types:**

| Edge case type | Why it's hard |
|---|---|
| CEO/VP applying for IC role | Seniority judge should score fit vs absolute level — may still fail |
| Fresh grad with strong open-source | Impact judge may underweight non-commercial contributions |
| Career switcher (e.g., teacher → PM) | Relevancy judge may score 0 despite strong transferable skills |
| Academic CV (publications, no bullets) | Chunker section patterns may not parse correctly |
| Non-English resume (French/German) | All 5 judges trained on English patterns |
| Single-page minimal resume | May not have enough text for confidence > threshold, always retries |
| 8-page exhaustive resume | Truncation risk, scores later experience sections lower |

**Action:** Create 7-10 annotated items of each type. Run a separate eval: `python scripts/run_eval.py --eval data/eval_edge_cases.json`. Report separately — don't fold into main accuracy number.

**Where:** New file `data/eval_edge_cases.json`. Add `--eval` flag to `run_eval.py` if it doesn't already accept alternative eval files.

---

## Ch 4 — Feature Engineering

### 3. Chunk Validation — Monitor Parsing Quality
**What:** Book says feature engineering is only as good as feature quality. Validate that features have stable distributions, low null rates, and expected value ranges. Silent data corruption (e.g., a regex change breaks date parsing) will degrade model performance without any error.

**For hireable:** `chunker.py` produces structured chunks for all 5 judges. If the skills section fails to parse (null), `_extract_tech_mentions` runs as fallback — but that fallback is incomplete (hardcoded keyword list). These silent fallbacks degrade judge inputs without any warning.

**Add to `chunker.py`:**
```python
def validate_resume_chunks(chunks) -> dict:
    issues = []
    if not chunks.skills:
        issues.append("skills_empty")
    if len(chunks.skills) < 3:
        issues.append(f"skills_sparse_{len(chunks.skills)}")
    if not chunks.work_history:
        issues.append("work_history_empty")
    if len(chunks.achievement_bullets) == 0:
        issues.append("no_achievement_bullets")
    if not chunks.summary and not chunks.work_history:
        issues.append("minimal_content")
    return {"issues": issues, "skills_count": len(chunks.skills),
            "experience_count": len(chunks.work_history),
            "bullets_count": len(chunks.achievement_bullets)}
```

**Where:** Call `validate_resume_chunks` in `analysers/engine.py` before passing to judges. Attach validation result to the response as `"_parse_quality"`. If `skills_empty` is flagged on a candidate who scores low, the score is likely invalid — not the candidate's fault.

---

### 4. Two-Stage Pipeline — Fast Filter Before Deep Eval
**What:** Book covers feature interaction and cascade models: use a fast, cheap first stage to filter obvious cases, reserve expensive computation for uncertain ones.

**For hireable:** 5 judges run in parallel (~160s total). But for obvious Weak candidates (e.g., skills judge returns 8%, experience returns 15%), there's no need to run seniority, impact, and relevancy to exhaustion. The composite can't reach the Moderate threshold regardless.

**Early-exit logic:**
```python
# In analysers/engine.py — after skills + experience judges complete
FAST_JUDGES = ["skills", "experience"]
DEEP_JUDGES = ["seniority", "relevancy", "impact"]

fast_results = await asyncio.gather(*[run_judge(j, ...) for j in FAST_JUDGES])
fast_composite = weighted_sum(fast_results, weights={"skills": 0.35, "experience": 0.30})

# Skills gate already handles 0% case. This handles "obviously weak" case.
max_possible = fast_composite + sum(cfg.default_weights[j] for j in DEEP_JUDGES)
if max_possible < cfg.moderate_threshold:
    # Even if deep judges score 100%, candidate can't reach Moderate
    return build_weak_result(fast_results, reason="fast_exit_below_threshold")

deep_results = await asyncio.gather(*[run_judge(j, ...) for j in DEEP_JUDGES])
```

**Expected impact:** For clear Weak candidates (~30% of candidate pool), cuts pipeline time from ~160s to ~60s. The `max_possible` calculation guarantees no Strong/Moderate candidate is misclassified as Weak by the fast exit.

**Where:** `analysers/engine.py` — restructure the `asyncio.gather` call.

---

## Ch 5 — Model Development and Offline Evaluation

### 5. Weight Hyperparameter Search — Don't Guess the Weights
**What:** Book says hyperparameters should be tuned on a validation set, not set by intuition. Random search finds good configurations 2-3× faster than grid search for most parameter spaces.

**For hireable:** Current weights (skills=0.35, experience=0.30, relevancy=0.15, seniority=0.10, impact=0.10) were set by judgment. Main weakness is Moderate recall at 60%. Weights should be tuned to maximise Moderate F1 on the development set.

**Add `scripts/tune_weights.py`:**
```python
import random, json
from analysers.config import AnalyserConfig
from scripts.run_eval import evaluate_all

SEARCH_SPACE = {
    "skills":     (0.20, 0.45),
    "experience": (0.20, 0.40),
    "relevancy":  (0.10, 0.25),
    "seniority":  (0.05, 0.20),
    "impact":     (0.05, 0.20),
}

def random_weights() -> dict:
    w = {k: random.uniform(*v) for k, v in SEARCH_SPACE.items()}
    total = sum(w.values())
    return {k: round(v / total, 3) for k, v in w.items()}

best_moderate_f1 = 0
for trial in range(50):
    weights = random_weights()
    results = evaluate_all(weights=weights, eval_file="data/eval_set.json")
    moderate_f1 = results["per_class"]["Moderate"]["f1"]
    if moderate_f1 > best_moderate_f1:
        best_moderate_f1 = moderate_f1
        print(f"Trial {trial}: Moderate F1={moderate_f1:.3f} weights={weights}")
```

**Important:** Run this on the 20-item development set only, NOT the held-out 10-item benchmark. Goodhart's Law applies — if you optimise weights on the eval set, accuracy numbers become meaningless.

---

### 6. Threshold Tuning — Calibrate to Real Candidate Distribution
**What:** Book covers precision-recall threshold tuning. Fixed thresholds (65%, 28%) assume a specific operating point. Tuning thresholds shifts the recall/precision trade-off for each class.

**For hireable's Moderate recall problem:** The 28% Moderate threshold may be too low — Moderate candidates who score 35% composite should be Moderate, but the gap between their score and the 65% Strong threshold may be causing some to be misclassified. OR the gap between 28% and a weak candidate at 26% is too narrow.

**Plot the precision-recall curve:**
```python
# In scripts/eval_metrics.py
thresholds = [(s_thresh, m_thresh) for s_thresh in range(50, 80, 5)
                                    for m_thresh in range(15, 45, 5)]
for s_thresh, m_thresh in thresholds:
    preds = [classify(score, s_thresh/100, m_thresh/100) for score in composite_scores]
    moderate_recall = recall_score(y_true, preds, labels=["Moderate"], average="macro")
    print(f"Strong≥{s_thresh}% Moderate≥{m_thresh}%: Moderate recall={moderate_recall:.2f}")
```

**Where:** `scripts/eval_metrics.py` — add threshold sweep. Report the Pareto frontier: threshold pairs where Moderate recall improves without Strong recall dropping below 85%.

---

## Ch 8 — Data Distribution Shifts and Monitoring

### 7. Three Types of Drift — What to Monitor for Each
**What:** Book formally defines three distribution shift types. Each requires a different response.

| Shift type | Definition | Hireable example | Fix |
|---|---|---|---|
| Covariate shift | P(X) changes, P(Y\|X) same | LLM/AI boom → more ML resumes in 2025 than 2022, but a strong ML candidate is still strong | Rebalance eval set to include AI-era resumes |
| Label shift | P(Y) changes, P(X\|Y) same | Hot job market → more candidates are Strong, same evidence standards | Recalibrate thresholds, not prompts |
| Concept drift | P(Y\|X) changes | "Python experience" now implies Python 3.10+, LLM familiarity; 2019 standards don't apply | Update judge prompts, retrain |

**For hireable:** Concept drift is the biggest risk. Skills decay and evolve — "AWS experience" from 2020 means something different from 2024. Monitor:

```python
# In a new scripts/monitor_drift.py
def monitor_score_distributions(results_file="data/eval_results.jsonl"):
    import json, numpy as np
    from collections import defaultdict

    by_month = defaultdict(list)
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            month = r["timestamp"][:7]   # "2026-06"
            by_month[month].append(r)

    baseline_month = sorted(by_month.keys())[0]
    baseline_scores = {j: [] for j in ["skills","experience","seniority","relevancy","impact"]}
    for r in by_month[baseline_month]:
        for j, s in r["judge_scores"].items():
            baseline_scores[j].append(s)

    for month in sorted(by_month.keys())[1:]:
        for judge in baseline_scores:
            current = [r["judge_scores"][judge] for r in by_month[month]]
            baseline_mean = np.mean(baseline_scores[judge])
            current_mean = np.mean(current)
            shift = abs(current_mean - baseline_mean) / (baseline_mean + 1e-9)
            if shift > 0.15:
                print(f"DRIFT: {judge} judge mean shifted {shift:.1%} in {month}")
```

**Where:** New `scripts/monitor_drift.py`. Already exists in the project as a stub — verify its current implementation and extend.

---

### 8. Degenerate Feedback Loop Detection
**What:** Book warns this is one of the most dangerous production ML failure modes. A model's predictions influence future training data, amplifying any existing bias. If hireable recommends candidates and those recommendations influence hiring decisions, then using hiring outcomes as training labels will encode the model's biases into the next version.

**For hireable:** If you build a feedback loop (recruiter accepts/rejects → retraining):
- Track acceptance rate by predicted tier: `acceptance_rate["Strong"]`, `acceptance_rate["Moderate"]`, `acceptance_rate["Weak"]`
- Track acceptance rate by demographic proxies in the resume (university prestige, company name tier, nationality indicators)
- If acceptance rate correlates with features that shouldn't matter (university name ≠ job performance), the feedback loop is amplifying bias

**Add to feedback collection:**
```python
def check_feedback_loop_health(feedback_log: list) -> dict:
    by_tier = defaultdict(list)
    for f in feedback_log:
        by_tier[f["predicted_tier"]].append(f["recruiter_accepted"])
    acceptance_rates = {tier: sum(v)/len(v) for tier, v in by_tier.items()}
    # Flag if Moderate acceptance rate < 20% — model may be systematically excluding good candidates
    if acceptance_rates.get("Moderate", 1.0) < 0.20:
        return {"warning": "low_moderate_acceptance", "rates": acceptance_rates}
    return {"status": "ok", "rates": acceptance_rates}
```

**Where:** New `scripts/feedback_health.py`. Run before each retraining cycle if feedback is collected.

---

## Ch 6 — Evaluation

### 9. Judge Ablation — Which Judge Drives Moderate Failures
**What:** Book recommends ablation studies: remove one component at a time and measure impact. Tells you which components matter most.

**For hireable — judge ablation on Moderate recall:**
```python
# In scripts/eval_metrics.py — add ablation mode
def ablation_study(results: list) -> None:
    judges = ["skills", "experience", "seniority", "relevancy", "impact"]
    base_weights = {"skills": 0.35, "experience": 0.30, "seniority": 0.10,
                    "relevancy": 0.15, "impact": 0.10}

    for dropped_judge in judges:
        # Redistribute dropped judge's weight proportionally
        ablated_weights = {j: w for j, w in base_weights.items() if j != dropped_judge}
        total = sum(ablated_weights.values())
        ablated_weights = {j: w/total for j, w in ablated_weights.items()}

        moderate_items = [r for r in results if r["human_tier"] == "Moderate"]
        correct = 0
        for r in moderate_items:
            composite = sum(r["judge_scores"][j] * ablated_weights[j]
                           for j in ablated_weights) / 100
            pred = "Strong" if composite >= 0.65 else ("Moderate" if composite >= 0.28 else "Weak")
            if pred == "Moderate":
                correct += 1
        recall = correct / len(moderate_items)
        print(f"Drop {dropped_judge}: Moderate recall = {recall:.1%}")
```

**Expected finding:** If dropping seniority improves Moderate recall from 60% → 75%, seniority judge is the culprit. If dropping impact doesn't change recall, impact is not the bottleneck for Moderate candidates.

**Where:** `scripts/eval_metrics.py`. Run this on the existing 30-item eval results — no new LLM calls needed, just re-score with different weights.

---

## Ch 9 — Continuous Learning

### 10. Champion/Challenger for Prompt Changes
**What:** Book describes champion/challenger as the standard pattern for safe model updates. Don't replace the current system with a new version until the new version is proven better.

**For hireable — apply to prompt iteration:** Every time a judge's system prompt is changed, the new prompt is the Challenger. Before fully rolling it out, run both prompts on the 30-item eval set and compare Moderate recall specifically.

**Lightweight implementation — no new infrastructure needed:**
```python
# In each analyser's prompts.py — maintain two versions
SYSTEM_v1 = """...(current prompt)..."""
SYSTEM_v2 = """...(candidate prompt with changes)..."""

# In scripts/prompt_ab_test.py
async def champion_challenger(judge_class, eval_items, champion_prompt, challenger_prompt):
    champ_results = await run_judge_with_prompt(judge_class, eval_items, champion_prompt)
    chall_results = await run_judge_with_prompt(judge_class, eval_items, challenger_prompt)
    print("Champion Moderate recall:", moderate_recall(champ_results))
    print("Challenger Moderate recall:", moderate_recall(chall_results))
    print("Promote challenger?", moderate_recall(chall_results) > moderate_recall(champ_results))
```

**Rule:** Only promote challenger if Moderate recall improves AND Strong recall doesn't drop by more than 3%. Never tune on the full eval set — use the 20-item dev split.

---

### 11. Natural Labels from User Behaviour
**What:** Book defines natural labels as labels generated automatically by system usage — no human annotation cost. These are the gold standard for feedback loops.

**For hireable — potential natural labels if deployed:**
- Recruiter views candidate profile after Strong recommendation → weak positive signal
- Recruiter downloads CV → stronger positive signal
- Recruiter moves candidate to interview → strong positive signal
- Recruiter skips recommendation without viewing → negative signal

**Even before deployment, collect proxy labels:**
- When the dashboard is used, log which candidates the user actually clicks into
- If a user overrides the tier (manually marks Strong as Moderate), that's a strong correction label
- Store all corrections in `data/human_corrections.jsonl`

**Schema:**
```json
{
    "candidate_id": "eval_006",
    "model_predicted_tier": "Weak",
    "human_corrected_tier": "Strong",
    "corrected_by": "recruiter_1",
    "timestamp": "2026-06-07T10:30:00",
    "note": "Fatima has exactly the skills required, model missed statsmodels"
}
```

**Where:** Add a correction endpoint to the FastAPI app (`POST /candidates/{id}/correct-tier`). Store corrections to `data/human_corrections.jsonl`. Use these as high-quality training examples.

---

## Ch 10 — Infrastructure

### 12. Eval Result Versioning — Track Model and Prompt Versions
**What:** Book says model versioning is mandatory for production. Without versioning, you can't know which model produced which result, can't roll back, and can't run meaningful A/B comparisons.

**For hireable:** `eval_results.jsonl` currently stores scores but not which prompt version was used for each judge. If a prompt changes between eval runs, results are not comparable.

**Add to every eval result entry:**
```python
import hashlib

def prompt_version(system_prompt: str) -> str:
    return hashlib.md5(system_prompt.encode()).hexdigest()[:8]

result_entry = {
    "id": item_id,
    "model": model_name,
    "prompt_versions": {
        "skills": prompt_version(skills_system_prompt),
        "experience": prompt_version(experience_system_prompt),
        # ...
    },
    "judge_scores": {...},
    "timestamp": datetime.now().isoformat(),
}
```

**Where:** `scripts/run_eval.py` — add `prompt_versions` dict to each result. This makes every eval reproducible and comparable. When you see Moderate recall improve from 60% to 70%, you can trace it to the specific prompt change.

---

## Priority Order

| Priority | Item | Effort | Expected Gain |
|----------|------|--------|---------------|
| 1 | Judge ablation study — find which judge drives Moderate failures (#9) | 1 hour | Pinpoints root cause without new LLM calls |
| 2 | Threshold tuning sweep — find optimal Strong/Moderate cutoffs (#6) | 2 hours | Direct Moderate recall improvement |
| 3 | Weight hyperparameter search (#5) | 3 hours | Potentially higher Moderate F1 |
| 4 | Two-stage pipeline — fast exit for obvious Weak (#4) | 3 hours | 40% latency reduction on Weak candidates |
| 5 | Chunk validation metrics in engine output (#3) | 2 hours | Surfaces silent parse failures like eval_006 |
| 6 | Judge disagreement → active learning log (#1) | 1 hour | Focuses eval set expansion on uncertain cases |
| 7 | Eval result versioning with prompt hashes (#12) | 1 hour | Makes results comparable across prompt changes |
| 8 | Champion/Challenger for prompt changes (#10) | 2 hours | Safe prompt iteration |
| 9 | Distribution drift monitoring (#7) | 3 hours | Catches silent performance degradation |
| 10 | Edge case test set — 7 types (#2) | 4 hours | Systematic coverage of parser/judge blind spots |
| 11 | Human corrections endpoint (#11) | 3 hours | Starts natural label collection |
| 12 | Degenerate feedback loop check (#8) | 2 hours | Safety guard before any retraining cycle |

Items 1–3 address the current Moderate recall weakness directly. Items 4–8 improve system reliability. Items 9–12 are for when hireable reaches production scale.
