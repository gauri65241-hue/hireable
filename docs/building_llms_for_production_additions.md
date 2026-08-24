# Building LLMs for Production
## Additions for hireable (linkedin-agent-assistant)

Source: `/Users/gauricapuri/Desktop/books/Building LLMs for production.pdf`
Chapters reviewed: I (Intro/Prompting), II (Architectures), III (Hallucinations/Bias), IV (Prompting Techniques), V (RAG), VI–VII (LangChain), VIII–IX (Indexes/Advanced RAG), X (Agents), XI (Fine-Tuning), XII (Deployment)

Note: CoT prompting, few-shot examples, JSON parsing robustness, LoRA/DPO finetuning, hybrid BM25+semantic, self-consistency, quantization, and parallel batching are already covered in `hands_on_llms_additions.md` and `ai_engineering_additions.md`. Items below are net-new.

---

## Ch III — Hallucinations & Bias

### 1. Bias Audit — Permute Candidate Demographics, Check Score Variance
**What:** Book shows LLMs amplify training-data bias in occupational contexts (WinoGender benchmark). A model scoring a resume may unconsciously penalise non-traditional names, career gaps, or unconventional education paths. The test: swap demographic signals in a resume and verify the score doesn't change.

**For hireable:**
```python
# scripts/bias_audit.py
PERMUTATIONS = [
    ("Maria Garcia", "John Smith"),
    ("Priya Patel", "James Wilson"),
    ("2018-2020 career gap", "2018-2020 freelance consulting"),
]

async def audit_bias(resume: str, jd: str, engine: AnalyserEngine):
    base_result = await engine.run(resume, jd)
    base_score = base_result["composite_score"]
    for original, replacement in PERMUTATIONS:
        modified = resume.replace(original, replacement)
        result = await engine.run(modified, jd)
        delta = abs(result["composite_score"] - base_score)
        if delta > 0.05:
            print(f"BIAS FLAG: '{original}' → '{replacement}' changed score by {delta:.3f}")
```

**Where:** New `scripts/bias_audit.py`. Run once on the 30-item eval set to surface any demographic sensitivity in the judges.

**Expected finding:** Skills and experience judges should be immune (they look at tech evidence). Seniority and relevancy judges are most at risk — language about "leadership" is culturally loaded.

---

### 2. Benchmark Regression Suite — Protect Against Prompt Drift
**What:** Book recommends a fixed golden dataset evaluated before every prompt change. Without it, prompt iterations are flying blind — you can't tell if a change helped or hurt.

**For hireable:** `eval_set.json` already exists as a 30-item benchmark but it's run manually and results aren't compared automatically against a baseline.

**Add to `scripts/run_eval.py`:**
```python
# Load previous best result from disk
BASELINE_FILE = "data/eval_baseline.json"

def check_regression(current_accuracy: float, per_tier: dict):
    if not os.path.exists(BASELINE_FILE):
        return  # First run — no baseline yet
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    delta = current_accuracy - baseline["accuracy"]
    if delta < -0.03:   # >3% drop = regression
        print(f"REGRESSION: accuracy dropped {delta:.1%} vs baseline ({baseline['accuracy']:.1%})")
    elif delta > 0.02:
        print(f"IMPROVEMENT: +{delta:.1%} vs baseline. Update baseline? (y/n)")

def save_baseline(accuracy: float, per_tier: dict):
    with open(BASELINE_FILE, "w") as f:
        json.dump({"accuracy": accuracy, "per_tier": per_tier,
                   "timestamp": datetime.now().isoformat()}, f)
```

**Where:** `scripts/run_eval.py` — call `check_regression()` at the end of every eval run. Only update the baseline when explicitly confirming improvement.

---

## Ch IV — Prompting Techniques

### 3. Section-Separated Input — Guide Attention by Labelling Resume Sections
**What:** Book covers how decoder-only models weight tokens via attention. Multi-head attention processes different aspects independently — but only if the input gives it structure to latch onto. Unstructured text blocks make attention diffuse.

**Current state:** Judge user prompts dump the entire resume as a flat text block. The model has to infer where skills end and experience begins.

**Fix — label every resume section explicitly:**
```
CANDIDATE RESUME:
[SUMMARY]
Data engineer with 6 years...

[EXPERIENCE]
• Senior Data Engineer, Acme Corp (2021–present)
  Led migration of ETL pipelines...

[SKILLS]
Python, Spark, Kafka, PostgreSQL, AWS

[EDUCATION]
BSc Computer Science, University of Melbourne
```

**Where:** `analysers/chunker_compat.py` — `format_for_*_judge` functions. Parse resume sections and emit them with `[SECTION]` labels before handing to any judge.

**Expected impact:** Skills judge stops confusing experience bullets with skills section. Relevancy judge better separates signal (experience) from noise (education). Low effort — purely a formatting change in the chunker.

---

### 4. Prompt Variant A/B Testing Against Benchmark
**What:** Book: small wording changes significantly influence model performance. Without a testing framework, prompt iteration is subjective.

**For hireable:** Build a lightweight A/B test runner that scores two prompt variants on the same eval set and reports which is better.

```python
# scripts/prompt_ab_test.py
async def ab_test(variant_a_prompt: str, variant_b_prompt: str,
                  eval_items: list, judge_class, model: str):
    results_a, results_b = [], []
    for item in eval_items:
        # Monkey-patch SYSTEM prompt, run judge, record tier
        ...
    acc_a = accuracy(results_a, eval_items)
    acc_b = accuracy(results_b, eval_items)
    print(f"Variant A: {acc_a:.1%}  Variant B: {acc_b:.1%}  Winner: {'A' if acc_a > acc_b else 'B'}")
    return acc_a, acc_b
```

**Usage:** When changing any judge's system prompt, run A/B test on the 20-item development set before committing. Document variant + result in `docs/prompt_history.md`.

**Where:** New `scripts/prompt_ab_test.py`. Uses existing `run_eval.py` infrastructure.

---

## Ch VIII — Data Preparation

### 5. Per-Judge Input Token Budget — Warn Before Context Overflow
**What:** Book covers context window management: if prompt + resume + JD exceeds context, the model silently truncates input. This explains mysterious low scores for long resumes — the model never saw the relevant evidence.

**Current risk:** `num_ctx` is set dynamically in `call_llm` based on character count divided by 4, but there's no check on whether the actual content fits. A 20,000-char resume + 10,000-char JD + 2,000-char system prompt = ~32,000 tokens — right at the edge for mistral-nemo.

**Fix — add token estimation and warning:**
```python
# In analysers/base.py — call_llm()
CHARS_PER_TOKEN = 3.5  # Conservative estimate

def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)

# Before LLM call:
total_estimated = estimate_tokens(system_prompt + user_prompt)
if total_estimated > 28_000:
    logger.warning(
        "Input may exceed context: ~%d tokens estimated. "
        "Resume/JD may be silently truncated. Consider reducing input size.",
        total_estimated
    )
```

**Where:** `analysers/base.py` — `call_llm()` before the `ChatOllama` invocation.

**Expected impact:** Surfaces the silent truncation bug. If a candidate scores 0.2 on all judges but has a long resume, this warning would explain it.

---

## Ch X — Agents

### 6. Execution Trace Logging — Debug Low-Confidence Scores
**What:** Book: agents are hard to evaluate because they take variable paths. Fix is structured trace logging — log which path was taken, not just the final output.

**For hireable:** Currently when a candidate scores unexpectedly low, there's no way to know which judge failed and why without re-running manually.

**Add structured trace to engine output:**
```python
# In analysers/engine.py — build into the result dict
trace = {
    "skills":     {"score": 0.82, "confidence": 88, "attempts": 1, "latency_ms": 4200},
    "experience": {"score": 0.41, "confidence": 71, "attempts": 2, "latency_ms": 8100},
    "seniority":  {"score": 0.65, "confidence": 80, "attempts": 1, "latency_ms": 3900},
    "relevancy":  {"score": 0.55, "confidence": 77, "attempts": 1, "latency_ms": 4100},
    "impact":     {"score": 0.30, "confidence": 68, "attempts": 2, "latency_ms": 7800},
}
```

The trace should surface: which judges needed retries, which had low confidence, which were slowest. This is the diagnostic layer missing from the current result format.

**Where:** `analysers/engine.py` — the final result dict already has some of this but not structured. Make the trace a first-class field in the API response.

**Also expose in UI:** The frontend should show a collapsed "Judge Detail" panel that expands to show per-judge scores + attempts + confidence. Currently only the composite score is visible.

---

## Ch XII — Deployment & Optimization

### 7. JD Analysis Cache — Reuse Across Multiple Resumes
**What:** Book covers caching for repeated LLM inputs. When screening a pool of candidates (same JD, many resumes), the JD analysis is identical for every candidate but re-computed from scratch each time.

**For hireable:** If the experience judge extracts "3 core JD responsibilities" on every call, that extraction is pure redundant compute when the JD hasn't changed.

**Fix — cache JD-only analysis:**
```python
# In analysers/base.py or a new analysers/jd_cache.py
import hashlib
from functools import lru_cache

_jd_cache: dict[str, dict] = {}

def jd_cache_key(jd_text: str, judge: str) -> str:
    return hashlib.md5(f"{judge}:{jd_text}".encode()).hexdigest()

def get_cached_jd_analysis(jd_text: str, judge: str) -> dict | None:
    return _jd_cache.get(jd_cache_key(jd_text, judge))

def cache_jd_analysis(jd_text: str, judge: str, result: dict):
    _jd_cache[jd_cache_key(jd_text, judge)] = result
```

**Where:** `analysers/engine.py` — before each judge call, check cache. On bulk screening runs (`run_eval.py` loops over 30 items with the same JD), this would skip 25-29 redundant JD extractions per judge.

**Expected impact:** On a 30-item eval with 10 unique JDs, saves ~20 × 5 judges = 100 LLM calls. At ~4s each = ~400s saved per eval run.

---

### 8. Exponential Backoff in Retry Loop
**What:** Book covers production reliability: retrying immediately after a failure puts pressure on an already struggling inference server. Exponential backoff gives the server time to recover.

**Current state:** `run_with_retry` in `base.py` retries immediately with `continue` — no delay between attempts.

**Fix:**
```python
# In analysers/base.py — run_with_retry()
import random

for attempt in range(max_retries):
    # ... existing logic ...
    
    if attempt < max_retries - 1:
        backoff = (2 ** attempt) + random.uniform(0, 1)   # 1s, 2s, 4s + jitter
        logger.info("Retry backoff: %.1fs", backoff)
        await asyncio.sleep(backoff)
```

**Where:** `analysers/base.py` — `run_with_retry()`, after a failed parse or low-confidence result. Add `await asyncio.sleep(backoff)` before the `continue`.

**Expected impact:** Reduces retry pile-up when Ollama is under load (e.g., 5 judges all retrying simultaneously). Jitter prevents thundering herd.

---

### 9. Per-Judge Model Config — Allow Different Models Per Judge
**What:** Book covers task-appropriate model selection: smaller models for simple tasks (binary classification), larger for complex reasoning. Not every judge needs the same model.

**Current state:** All 5 judges use the same `cfg.default_model`. The skills judge (binary present/absent) is simpler than the experience judge (nuanced scope comparison).

**Add to `config.py`:**
```python
@dataclass(frozen=True)
class AnalyserConfig:
    ...
    # Per-judge model overrides — falls back to default_model if not set
    skills_model: str     = _str("ANALYSER_SKILLS_MODEL", "")
    experience_model: str = _str("ANALYSER_EXPERIENCE_MODEL", "")
    seniority_model: str  = _str("ANALYSER_SENIORITY_MODEL", "")
    relevancy_model: str  = _str("ANALYSER_RELEVANCY_MODEL", "")
    impact_model: str     = _str("ANALYSER_IMPACT_MODEL", "")

    def model_for(self, judge: str) -> str:
        override = getattr(self, f"{judge}_model", "")
        return override or self.default_model
```

**Where:** `analysers/config.py` + each `analyser.py` — pass `cfg.model_for("skills")` instead of `cfg.default_model`.

**Use case:** Run skills judge on `phi4-mini:latest` (fast, simple binary task) and experience judge on the full `mistral-nemo` (needs nuanced reasoning). Could cut total pipeline time by 30-40% if the 2 fast judges (skills, relevancy) use a smaller model.

---

## Priority Order

| Priority | Item | Effort | Expected Gain |
|----------|------|--------|---------------|
| 1 | Token budget warning in call_llm (#5) | 1 hour | Surfaces silent truncation bugs |
| 2 | Execution trace as first-class result field (#6) | 2 hours | Debuggability of low scores |
| 3 | Section-separated resume input in chunker (#3) | 2 hours | Better judge attention on right evidence |
| 4 | Exponential backoff in retry loop (#8) | 1 hour | Reduces retry pile-up under load |
| 5 | Regression suite with baseline comparison (#2) | 2 hours | Prevents silent accuracy regressions |
| 6 | JD analysis cache for bulk screening (#7) | 3 hours | ~400s saved per 30-item eval run |
| 7 | Per-judge model config (#9) | 2 hours | 30-40% pipeline speedup potential |
| 8 | Prompt A/B test runner (#4) | 3 hours | Structured prompt iteration |
| 9 | Bias audit script (#1) | 3 hours | Validates fairness of scoring |

Items 1–4 are infrastructure/reliability fixes. Items 5–7 are performance wins. Items 8–9 are diagnostic tools for ongoing quality work.
