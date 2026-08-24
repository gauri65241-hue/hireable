# AI Engineering: Building Applications with Foundation Models — Chip Huyen
## Additions for hireable (linkedin-agent-assistant)

Source: `/Users/gauricapuri/Desktop/books/AI Engineering_ Building Applications With Foundation Models by Chip Huyen (1).pdf`
Chapters reviewed: 1 (Intro/LM Landscape), 2 (Understanding Foundation Models), 3 (Evaluation Methodology), 4 (Evaluate AI Systems), 5 (Prompt Engineering), 6 (RAG and Agents), 7 (Finetuning), 8 (Dataset Engineering), 9 (Inference Optimization)

---

## Ch 2 — Understanding Foundation Models

### 1. Fix Sampling Parameters for Deterministic Judge Scoring
**What:** Temperature, top-p, top-k, and seed together control output variance. Book states that fixing all four is the minimum requirement for reproducible results. Seed alone is not enough — top-p and top-k sampling introduce stochasticity independent of temperature.

**Required fix for hireable:** Every judge call must set all four:
```python
# In analysers/base.py — call_llm_async / call_llm_sync
options = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "seed": 42,
}
```

**Where:** `analysers/base.py` — wherever Ollama `options` dict is built. Currently temperature may be set but top_k and seed are unconfirmed.

**Expected impact:** Currently the same candidate can score ±8% across runs (observed in eval). Fixing all four should reduce run-to-run variance to near-zero, making the eval set meaningful.

---

### 2. YAML Output Instead of JSON — Token Reduction
**What:** Book explicitly compares YAML vs JSON output schemas and shows YAML uses 10-30% fewer tokens for the same structured data (no quotes, no colons on values, no closing braces). Fewer output tokens = faster inference, lower cost, and fewer hallucination opportunities.

**Current state:** All 5 judges output JSON with double-quoted keys and values, nested braces.

**Proposed change:** Switch judge output format to YAML, parse with `yaml.safe_load()` instead of `json.loads()`.

**Example — current JSON output size vs YAML:**
```yaml
# YAML equivalent of skills judge output
required_skills: [python, sql, aws]
missing_critical_skills: [aws]
skill_analysis:
  - skill: python
    is_present: 1
    evidence_strength: strong
    resume_evidence: "5 years Python development"
confidence: 85
```

**Where:** `analysers/*/prompts.py` (rewrite output schema in each SYSTEM prompt), `analysers/*/analyser.py` (swap `json.loads` for `yaml.safe_load`), `analysers/base.py` (`parse_and_validate` to accept YAML string).

**Tradeoff:** More complex parsing logic. `yaml.safe_load` is safe (no code execution). Worth it if inference time is a bottleneck.

---

### 3. Post-Processing Defensive Parser — LinkedIn's 90% → 99.99% Fix
**What:** Book covers LinkedIn's approach to structured output reliability: they built a multi-stage defensive parser that handles partial JSON, truncated output, extra markdown fences, and whitespace corruption. Result: correct parse rate went from 90% → 99.99%.

**Current state in hireable:** `parse_and_validate` in `base.py` likely does basic `json.loads` with a try/except. If the LLM outputs ```json\n{...}\n``` (with markdown fences), it fails.

**Defensive parser to add:**
```python
import re, json

def defensive_parse(raw: str) -> dict:
    # Strip markdown fences
    raw = re.sub(r"^```(?:json|yaml)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    # Find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in output: {raw[:200]}")
```

**Where:** `analysers/base.py` — replace or wrap existing `json.loads` call in `parse_and_validate`.

**Expected impact:** Reduces the retry rate (currently judges retry 1-3× when output is unparseable). Every retry adds 30-40 seconds. This fix alone could cut total pipeline time by 20%.

---

### 4. Constrained Sampling — Force Valid JSON at Generation Time
**What:** Book describes constrained decoding (also called structured generation) as the strongest reliability guarantee: the logit vector is masked at each token step to only allow tokens that could legally appear next in a valid JSON document. The model literally cannot output invalid JSON.

**Tools:** `outlines` (Python library), `llama.cpp` grammar-based sampling, or Ollama's built-in `format: "json"` option.

**For hireable:** Ollama already supports `format: "json"` in the API options. This enables constrained sampling natively — no new dependency.

```python
# In analysers/base.py — add to Ollama options
payload = {
    "model": model,
    "messages": messages,
    "format": "json",   # <-- constrained JSON sampling
    "options": {"temperature": 0.0, "top_k": 1, "seed": 42}
}
```

**Warning:** `format: "json"` guarantees valid JSON syntax but not schema compliance. Must still validate required keys. Combine with defensive parser (#3) for full coverage.

**Expected impact:** Eliminates "invalid JSON" class of retries entirely. This is the single highest-ROI structural fix in this list.

---

## Ch 3 — Evaluation Methodology

### 5. Calibration: Judge Confidence Should Match Accuracy
**What:** Book defines calibration formally: a model is calibrated if its stated confidence matches its empirical accuracy. A judge that says confidence=90 should be right 90% of the time. Currently there's no check whether confidence scores in the 5 judges are calibrated.

**Measurement:**
```python
# After running full eval, compute calibration
bins = [(0,50), (50,70), (70,85), (85,100)]
for lo, hi in bins:
    items = [r for r in results if lo <= r["judge_confidence"] < hi]
    accuracy = sum(r["correct"] for r in items) / len(items) if items else 0
    print(f"Confidence {lo}-{hi}: n={len(items)}, accuracy={accuracy:.2f}")
```

**Expected finding:** If confidence=90 items have accuracy=60%, the judge is overconfident and its confidence scores are meaningless. Fix: add calibration instruction to system prompt: "Your confidence should reflect your uncertainty — if you are unsure, state 50-60 not 85+."

**Where:** Add calibration analysis to `scripts/run_eval.py`. Add calibration instruction to all 5 system prompts.

---

### 6. Consistency Score — Same Input, Same Output
**What:** Book distinguishes accuracy from consistency. Accuracy = right answer. Consistency = same answer on repeated runs. A judge can be accurate on average but inconsistent per-candidate — which makes it unreliable for individual hiring decisions.

**Measurement:**
```python
# Run the same 5 eval items 3× each, measure output variance
for item in sample_5:
    scores = [judge.run(item) for _ in range(3)]
    variance = np.std([s["score"] for s in scores])
    print(f"Item {item['id']}: mean={np.mean(...):.2f}, std={variance:.3f}")
```

**Target:** Std dev < 0.03 per judge per candidate. Currently unknown — needs measurement.

**Fix if inconsistent:** See #1 (fix sampling params) and #4 (constrained sampling). Both reduce variance without changing accuracy.

---

### 7. Inter-Judge Agreement Rate as a Quality Signal
**What:** Book recommends measuring agreement between multiple evaluators as a reliability proxy. When two judges strongly disagree, the result is suspect.

**For hireable:** Already have 5 judges. When skills=0.85 and relevancy=0.25 for same candidate, that's a red flag — either skills judge is inflating or relevancy is deflating. Add agreement monitoring:

```python
# In analysers/engine.py — post-scoring
scores = {"skills": 0.85, "experience": 0.72, "relevancy": 0.25, ...}
score_values = list(scores.values())
if max(score_values) - min(score_values) > 0.50:
    warnings.append("high_judge_disagreement")
```

**Where:** `analysers/engine.py` or `post_validate` in `base.py`. Add `"judge_disagreement_flag"` to the result JSON.

---

## Ch 5 — Prompt Engineering

### 8. Instruction Positioning — Critical Content at the End
**What:** Book cites the "lost in the middle" problem (Liu et al. 2023) and recommends placing the most critical instruction at the very end of the prompt, not the middle. Models have primacy bias (beginning) and recency bias (end) — middle content is systematically underweighted.

**For hireable:** Current judge prompts structure: system instructions → JD → resume → output format instruction. The output format instruction (most critical for reliable JSON) is at the end — correct. But the JD requirements (second most critical) are buried in the middle between system instructions and the resume.

**Fix:** Move JD to the bottom of user prompt, just above the output format line. Resume (supporting evidence) goes above JD.

```
[system: judge instructions]
[user: task framing]
[CANDIDATE RESUME: ...]       <-- evidence first
[JOB DESCRIPTION: ...]        <-- requirements second-to-last
[output format instruction]   <-- most critical at the end
```

**Where:** `_USER_TEMPLATE` in all 5 `analysers/*/prompts.py`. Swap the ordering of JD and resume blocks.

---

### 9. Explicit Output Length Constraint Reduces Hallucination
**What:** Book: fewer output tokens = fewer hallucination opportunities. Hallucinations compound — an initial wrong assumption causes the model to generate supporting fabrications to be consistent (snowballing). Shorter outputs have less space for this to occur.

**Current state:** Judge prompts say "Output JSON only" but don't constrain verbosity of reasoning fields like `overall_reasoning` or `resume_evidence`.

**Fix:** Add explicit token budget to each prompt:
- `overall_reasoning`: max 2 sentences
- `resume_evidence`: max 1 direct quote (no paraphrasing)
- `reasoning` per bullet: max 1 sentence

**Where:** All 5 `SYSTEM` prompts — add to the CRITICAL RULES section: `"Keep overall_reasoning under 40 words. Keep per-statement reasoning under 15 words. Use direct quotes only for resume_evidence — never paraphrase."`

---

### 10. Meta-Prompt: Ask the Model What's Unclear
**What:** Book describes a prompt debugging technique: before finalizing a prompt, ask the model to identify ambiguities. "Given this system prompt and task, what is unclear or could be misinterpreted?"

**Application for hireable:** Run this once on each of the 5 judge system prompts to surface blind spots. Not a production change — a one-time diagnostic.

```python
# One-time diagnostic
meta_prompt = f"""
Given this judge system prompt:

{SYSTEM}

What parts of these instructions are ambiguous? What would cause you to 
produce inconsistent scores across similar candidates? List the top 3 ambiguities.
"""
```

**Expected output:** The model will surface things like "what counts as 'comparable scale' for seniority" — actionable rubric improvements.

---

## Ch 4 — Evaluate AI Systems

### 11. Reference Set Scoring — Anchor Judges with Gold Standard
**What:** Book recommends calibrating AI judges using a reference set: 3-5 examples with known correct scores, embedded in the system prompt. These examples anchor the model's interpretation of the scoring rubric.

**For hireable:** From the 30-item eval set, select 2 gold-standard items:
- 1 confirmed Strong (human_tier=Strong, agreed by >2 annotators)
- 1 confirmed Weak

Embed in each judge system prompt as reference examples:
```
REFERENCE EXAMPLES (do not score these — use them to calibrate your rubric):

Example 1 — STRONG FIT:
Resume: "Led migration of monolithic Python service to microservices, reducing p99 latency by 40%..."
JD Requirement: "Experience with distributed systems at scale"
Correct Score: 0.9 | Reasoning: Direct evidence of scale + measurable outcome

Example 2 — WEAK FIT:
Resume: "Familiar with Python scripting for automation tasks..."
JD Requirement: "5+ years Python in production distributed systems"
Correct Score: 0.1 | Reasoning: 'Familiar' signals shallow exposure, no scale evidence
```

**Where:** `analysers/skills/prompts.py` and `analysers/experience/prompts.py` first (highest impact). Add to other 3 after verifying improvement.

---

### 12. Functional Correctness Over Surface Similarity
**What:** Book argues evaluation should measure functional correctness (does the output achieve the intended outcome?) not surface-level text matching. For hireable's tier classification, this means measuring whether the tier decision is correct — not whether the score is exactly 0.87 vs 0.83.

**Current eval gap:** `run_eval.py` likely checks exact tier match. It should also track:
- Was the Strong candidate scored above the Moderate threshold? (functional correctness)
- Was the Weak candidate scored below the Strong threshold? (functional correctness)
- How far off was the score when wrong? (error magnitude)

**Add to `run_eval.py`:**
```python
# Beyond accuracy: measure error magnitude on misclassifications
misclassified = [r for r in results if r["pred_tier"] != r["true_tier"]]
for r in misclassified:
    error_magnitude = abs(r["composite_score"] - r["boundary_score"])
    # boundary_score = nearest tier threshold (0.65 for Strong, 0.28 for Moderate)
    print(f"Miss by: {error_magnitude:.3f} | pred={r['pred_tier']} true={r['true_tier']}")
```

**Expected finding:** If 70% of Moderate→Strong misclassifications miss the boundary by <0.05, a small threshold adjustment fixes them without changing prompts. If misses are >0.15, the judge is fundamentally wrong, not just threshold-sensitive.

---

## Ch 6 — RAG and Agents

### 13. Semantic Chunking Over Fixed-Size Chunking
**What:** Book covers semantic chunking: split documents at natural semantic boundaries (paragraph breaks, topic shifts) rather than fixed token counts. Fixed chunking cuts sentences mid-thought and destroys context.

**For hireable:** `chunker.py` and `chunker_compat.py` parse resumes and JDs. If these split on fixed character counts or line counts, semantic signal is lost mid-bullet.

**Better approach:** Split resume at section headers (Experience, Education, Skills) and then at bullet-point boundaries. Never split mid-bullet.

```python
import re

def split_resume_semantically(text: str) -> list[str]:
    # Split at section headers
    sections = re.split(r'\n(?=(?:EXPERIENCE|EDUCATION|SKILLS|SUMMARY|WORK HISTORY))', 
                        text, flags=re.IGNORECASE)
    chunks = []
    for section in sections:
        # Within each section, split at bullet points
        bullets = re.split(r'\n[•\-\*]\s+', section)
        chunks.extend([b.strip() for b in bullets if b.strip()])
    return chunks
```

**Where:** `analysers/chunker_compat.py` — `parse_resume` function. Affects all 5 judges' input quality.

---

### 14. Query Rewriting for JD Requirement Extraction
**What:** Book covers query rewriting in RAG — instead of using the raw query, rewrite it to better match the retrieval target. For hireable's experience judge, the "query" is a JD responsibility and the "document" is the resume.

**For hireable:** The experience judge extracts 3 responsibilities from the JD verbatim. But JD language ("Leads cross-functional alignment across engineering and product") is often abstract. Rewriting to a concrete search query ("direct evidence of managing cross-team projects with engineering + product stakeholders") would surface more relevant resume evidence.

**One-time LLM call to rewrite JD responsibilities:**
```python
REWRITE_PROMPT = """Convert this JD responsibility into a concrete search query 
for finding resume evidence. Be specific about what evidence would prove this.

Responsibility: {responsibility}
Search query:"""
```

**Where:** `analysers/experience/prompts.py` `build_prompt` — add a preprocessing step before formatting the user prompt.

**Cost:** 1 additional LLM call per analysis. ~2 seconds overhead.

---

## Ch 7 — Finetuning

### 15. Finetuning Decision Framework — When to Finetune vs Prompt
**What:** Book provides explicit criteria for when finetuning beats prompting:
1. Prompt is >1000 tokens and cost is a concern → finetune on shorter format
2. Output format is complex and retry rate is >5% → finetune to internalize format
3. Domain vocabulary is highly specialized → finetune to embed domain knowledge
4. Need consistency across >100 calls/day → finetune for stable behavior

**For hireable's current state:**
- Prompts are ~600-800 tokens each: below finetuning threshold
- Retry rate: unknown — measure first (#5's calibration analysis will surface this)
- Domain: technical recruiting — specialized but not exotic
- Volume: low (evaluation runs, not production at scale)

**Verdict:** Do NOT finetune yet. Fix prompts + constrained sampling first. Revisit after reaching >500 labeled examples and consistent retry rate >10%.

---

### 16. PEFT / LoRA — Adapter-Based Finetuning for Judge Models
**What:** Book covers LoRA (Low-Rank Adaptation) as the standard approach for finetuning LLMs with limited GPU/MPS memory. LoRA adds trainable low-rank matrices to attention layers — freezes base model, trains only adapters (~0.1% of parameters).

**For hireable (future):** When finetuning becomes appropriate (#15 criteria met), use LoRA adapters — one per judge. Each adapter is ~50MB vs full model copy at 7GB.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
```

**Key point:** With LoRA, can maintain 5 separate judge adapters, switch between them with `model.load_adapter()`. Much cheaper than 5 separate model copies.

**Prerequisite:** 200+ high-quality (prompt, output) examples per judge. Accumulate from eval runs.

---

## Ch 9 — Inference Optimization

### 17. Prompt Caching — Reuse System Prompt Prefix
**What:** Book covers KV-cache prompt prefix reuse: if the same prefix (system prompt) is sent repeatedly, modern inference servers cache the KV state and skip recomputing it. Cuts TTFT (time to first token) by 40-60% when system prompt is large.

**For hireable:** Each of the 5 judges sends a ~500-token system prompt on every call. With Ollama, the system prompt KV cache is reused if the model and system prompt are identical between calls. This is automatic — but only works if the system prompt is NOT modified between calls.

**Current risk:** `SYSTEM + calibration_note` concatenation in `build_prompt` adds variable content to the system prompt. If `calibration_note` changes between calls, cache is busted.

**Fix:** Move `calibration_note` from system prompt to user prompt. System prompt stays constant → cache always hits.

```python
# In all analysers/*/prompts.py build_prompt functions
# Current:
return SYSTEM + calibration_note, user

# Fix:
return SYSTEM, calibration_note + "\n\n" + user
```

**Where:** All 5 `build_prompt` functions. One-line change per file, potentially 40-60% faster on repeat calls.

---

### 18. Batching Judge Calls — Parallel Execution is Already Right
**What:** Book covers inference batching: processing multiple requests simultaneously using the same loaded model to amortize overhead. Most valuable when running many calls sequentially.

**Current state in hireable:** `engine.py` already runs 5 judges via `asyncio.gather` (parallel). This is correct. The book validates this architectural decision.

**Remaining opportunity:** The retry loop (`run_with_retry`) runs retries sequentially. If 3 judges need retries simultaneously, they run one after another.

**Fix:** Gather retries in parallel too — if confidence < threshold for multiple judges, retry all simultaneously:
```python
# In engine.py or base.py
retry_tasks = [judge.retry() for judge in low_confidence_judges]
retry_results = await asyncio.gather(*retry_tasks)
```

**Expected impact:** When 2+ judges need retries (common for weak candidates), saves 30-60 seconds.

---

### 19. Quantization Awareness — Q4_K_M is the Right Choice
**What:** Book covers quantization tradeoffs: Q8 ≈ full precision (minimal degradation), Q4 ≈ 10-15% degradation on reasoning tasks, Q2 ≈ significant degradation. For judge tasks (classification, structured output), Q4_K_M hits the right cost/quality tradeoff.

**Validation of current setup:** `mistral-nemo:12b-instruct-2407-q4_K_M` is the correct model for this task. Book's guidance aligns: 12B at Q4 gives ~10B effective parameters worth of reasoning, fits in 8GB VRAM, inference speed ~15 tok/s on Apple Silicon.

**When to upgrade:** If judge retry rate >15% after applying prompt fixes, try `q6_K` or `q8_0`. These are 2× the file size but materially better at following complex JSON schemas.

**When to downgrade:** If latency is the bottleneck and accuracy is already >90%, try `mistral-nemo:12b-instruct-2407-q2_K` as a faster pre-filter (#13 in hands_on_llms notes) — full model only for second-pass scoring.

---

### 20. Speculative Decoding — Worth Testing with Ollama
**What:** Book covers speculative decoding: a small draft model generates tokens, the large model verifies them in parallel. Net result: 2-3× throughput with near-identical quality. Works best for structured/predictable output (JSON with fixed keys).

**For hireable:** JSON output with known schema is highly predictable — speculative decoding would work well. Ollama supports speculative decoding via draft model configuration.

**Test setup:**
```bash
# Pull a small draft model
ollama pull mistral:7b-instruct-q4_K_M

# Configure in Ollama model file
PARAMETER draft mistral:7b-instruct-q4_K_M
```

**Expected impact:** If judge outputs are 200-400 tokens and highly predictable (JSON), speculative decoding could reduce inference time by 40-60%.

**Risk:** Requires testing that draft model doesn't corrupt JSON structure. Run eval before/after.

---

## Ch 8 — Dataset Engineering

### 21. Data Augmentation for Eval Set Expansion
**What:** Book covers LLM-based data augmentation: use a model to generate synthetic (resume, JD, tier) pairs that expand the eval set without manual labeling. Key constraint: augmented data must be verified before use in training (but can be used for eval development set).

**For hireable — concrete augmentation:**
```python
AUGMENT_PROMPT = """Generate a realistic resume snippet (150 words) and matching 
job description (100 words) for a {tier} fit candidate. 
The candidate should be a {role} applying to a {industry} company.
Format: RESUME: ... JD: ... TIER: {tier}"""
```

Generate 20× augmented items per tier → 60 new development set items at near-zero cost. Use these for prompt iteration (#15 in hands_on_llms), keep real eval set clean.

**Quality check:** Run LLM-generated items through the current pipeline. Items where all 5 judges agree strongly = likely high quality. Items with high disagreement = likely ambiguous/poor quality, discard.

---

### 22. Data Flywheel — Every Eval Run Generates Training Data
**What:** Book describes the data flywheel: every inference run produces (input, output) pairs that can become training data. The system improves itself over time.

**For hireable:** `eval_results.jsonl` already accumulates after each eval run. This is the start of a data flywheel. Structure to make it usable:

```python
# In scripts/run_eval.py — ensure each result is stored with full prompt
result = {
    "id": item["id"],
    "system_prompt": system_prompt,    # <-- add this
    "user_prompt": user_prompt,        # <-- add this
    "judge_output": raw_response,      # <-- add this (raw, pre-parse)
    "parsed_score": parsed["score"],
    "human_tier": item["human_tier"],
    "correct": pred_tier == human_tier,
    "timestamp": datetime.now().isoformat()
}
```

**Use cases of accumulated data:**
- Prompt fine-tuning (item #16 / LoRA)
- DPO rejection pairs (wrong judge outputs as `rejected`)
- Calibration analysis (item #5)
- Augmentation quality verification (item #21)

---

## Priority Order

| Priority | Item | Effort | Expected Gain |
|----------|------|--------|---------------|
| 1 | Constrained JSON sampling via Ollama `format: "json"` (#4) | 30 min | Eliminates invalid JSON retries |
| 2 | Fix all sampling params: temperature=0, top_k=1, seed=42 (#1) | 30 min | Run-to-run consistency |
| 3 | Move `calibration_note` to user prompt for KV cache (#17) | 30 min | 40-60% faster on repeat calls |
| 4 | Defensive parser — strip markdown fences + extract JSON (#3) | 1 hour | Reduces retry rate |
| 5 | Explicit output length constraints in all SYSTEM prompts (#9) | 1 hour | Reduces hallucination in reasoning fields |
| 6 | Move JD below resume in user prompt (recency bias fix) (#8) | 1 hour | JD requirements better attended to |
| 7 | Judge disagreement flag in engine.py (#7) | 1 hour | Surfaces unreliable scores |
| 8 | Calibration measurement in run_eval.py (#5) | 2 hours | Know if confidence scores are meaningful |
| 9 | Consistency measurement — same input 3× (#6) | 2 hours | Know actual variance per judge |
| 10 | Error magnitude analysis in run_eval.py (#12) | 2 hours | Threshold vs prompt problem diagnosis |
| 11 | Reference set examples in skills + experience prompts (#11) | 3 hours | Anchors judge rubric calibration |
| 12 | Store full prompts in eval_results.jsonl (#22) | 2 hours | Enables data flywheel |
| 13 | Semantic resume chunking in chunker_compat.py (#13) | 3 hours | Better evidence extraction per bullet |
| 14 | JD requirement query rewriting in experience judge (#14) | 3 hours | Better evidence retrieval |
| 15 | Parallel retry gather in engine.py (#18) | 2 hours | Saves 30-60s when multiple judges retry |
| 16 | Augmented eval set generation (#21) | 3 hours | Larger dev set for prompt iteration |
| 17 | Speculative decoding test (#20) | 4 hours | 2-3× throughput (test carefully) |
| 18 | YAML output format (#2) | 1 day | 10-30% token reduction (high migration cost) |
| 19 | LoRA judge fine-tuning (#16) | 3-5 days | Domain-specialized judges (post 200+ examples) |

Items 1–4 are infrastructure fixes — no prompt changes, immediate reliability improvement. Do these first.
Items 5–10 are measurement additions — understand the system before fixing it further.
Items 11–16 are prompt + code improvements — requires eval runs to verify.
Items 17–19 are optional optimizations — only after core accuracy is stable.
