# Hands-On Large Language Models — Alammar & Grootendorst
## Additions for hireable (linkedin-agent-assistant)

Source: `/Users/gauricapuri/Desktop/books/250153.pdf`
Chapters reviewed: 4 (Text Classification), 5 (Clustering/Topic Modeling), 6 (Prompt Engineering), 7 (Agents), 8 (Semantic Search/RAG), 10 (Embedding Models), 11 (Fine-Tuning Representation), 12 (Fine-Tuning Generation + DPO)

---

## Ch 6 — Prompt Engineering

### 1. Zero-Shot Chain-of-Thought on all judges
**What:** Append `"Think step-by-step before producing your JSON."` before the output schema in each judge system prompt. Forces the model to distribute compute over a reasoning trace before committing to a score. Book shows this alone fixes math errors that one-shot prompts get wrong.

**Where:** `analysers/*/prompts.py` — add one sentence to each system prompt, before the JSON schema block.

**Expected impact:** Skills and relevancy judges are the two that score inconsistently on the same candidate across runs. CoT stabilises them.

**Cost:** Zero latency impact (reasoning tokens are within the same call). One-line change per prompt.

---

### 2. Few-Shot Examples in Judge Prompts
**What:** Add 1–2 concrete (resume snippet, JD snippet, expected score) examples directly in the system prompt. Book: "an example is worth a thousand words" — few-shot prompting outperforms instruction-only prompts on classification tasks.

**Where:** `analysers/skills/prompts.py`, `analysers/relevancy/prompts.py` (the two weakest judges). Add as structured message pairs in `build_prompt`.

**Data source:** Pull 2 items from `data/eval_set.json` — one Strong with score=0.9, one Weak with score=0.1. Embed them as literal examples in the prompt.

**Expected impact:** Reduces the relevancy judge's tendency to score Strong candidates at 55% instead of 70%+.

---

### 3. "Lost in the Middle" — Prompt Position Fix
**What:** Book cites research showing LLMs focus on beginning (primacy) and end (recency) of prompts; content in the middle is forgotten. Current prompts put JD requirements in the middle and candidate evidence after.

**Where:** All judge prompts. Reorder to: JD requirements at TOP → scoring rubric → candidate evidence at BOTTOM. Never bury the most critical content in the middle.

**Expected impact:** Skills judge misses explicit JD requirements buried mid-prompt. Moving them to top should reduce false negatives.

---

### 4. Self-Consistency for the Overall Score (Optional / Expensive)
**What:** Run the overall scoring call (coach) 3× at different temperatures (0.0, 0.3, 0.6), take majority vote on tier. Book shows self-consistency improves accuracy on hard classification tasks.

**Where:** `analysers/base.py` — modify the coach call in `run_with_retry` to run 3x if confidence < threshold.

**Cost:** 3× coach latency (~60s extra). Only worth it when confidence < 60%. Gate it: `if confidence < 60: run_self_consistency()`.

---

## Ch 8 — Semantic Search and RAG

### 5. Cross-Encoder Reranking for Skills Matching
**What:** Current SBERT (all-MiniLM-L6-v2) is a bi-encoder — fast but less accurate. Book shows cross-encoders achieve significantly higher accuracy by jointly encoding both texts. Two-stage pipeline: bi-encoder retrieves candidates, cross-encoder reranks.

**Where:** `analysers/skills/` — after the LLM judge produces a skills list, use a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` — 80MB, already on HuggingFace) to score each claimed skill against JD evidence. Replace or supplement the current `tech_in_experience` cosine check.

**Expected impact:** Reduces skills judge false negatives where bi-encoder misses semantic equivalents (e.g. "AWS Lambda" ≠ "serverless functions" in cosine space, but cross-encoder knows they're the same).

---

### 6. Hybrid Search: BM25 + Semantic for Skills
**What:** Book shows BM25 (keyword) + semantic = better than either alone. Skills matching benefits from exact keyword hits (specific tool names) combined with semantic similarity.

**Where:** `chunker.py` / skills pre-filtering. Before calling the LLM skills judge, run a BM25 pass over JD tokens against resume tokens to surface exact tech keyword matches. Feed both BM25 hits and SBERT matches into the judge context.

**Library:** `rank_bm25` (already shown in book, 1 dependency).

**Expected impact:** Catches exact tool names that semantic search dilutes (e.g. "Terraform" → high BM25 score, but SBERT may conflate it with "infrastructure").

---

### 7. Resume Chunking Strategy for SBERT
**What:** Currently embedding the full resume as one vector in `SentenceTransformer.encode()`. Book explicitly warns: one vector per long document = high information loss. Better: embed resume sections (skills block, each work experience bullet, summary) as separate chunks, take max cosine sim.

**Where:** wherever SBERT cosine similarity is computed at the end of `AnalyserEngine.run()`. Split resume into sentences/sections, embed each, take `max(cosine_sim(chunk, jd))` instead of whole-resume sim.

**Expected impact:** The current sim score averages Strong candidates at 0.75. Per-chunk max would surface better signal — a candidate whose work history is highly relevant but whose education section is generic wouldn't be dragged down.

---

## Ch 10 — Creating Text Embedding Models

### 8. Domain-Specific SBERT Fine-Tuning (Contrastive Learning)
**What:** all-MiniLM-L6-v2 trained on general web text. Resume/JD domain has specialized vocabulary (role titles, tech stacks, seniority signals). Fine-tune using contrastive learning: Strong label = positive pair (resume should be close to JD in embedding space), Weak = negative pair.

**Data:** `train.csv` (the source CSV) already has (resume, JD, label) triplets. Map: Strong/Good Fit → entailment (positive), Weak/No Fit → contradiction (negative), Moderate → neutral.

**How:**
```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer

model = SentenceTransformer('all-MiniLM-L6-v2')
train_loss = losses.SoftmaxLoss(model=model, ...)
# Train on (resume, jd, label) pairs from train.csv
```

**Expected impact:** Replaces generic SBERT with a resume-domain-aware model. Cosine similarity scores become more meaningful as a pre-filter and as a diagnostic metric.

**Prerequisite:** Need ~500+ labeled pairs from train.csv. Check count before starting.

---

### 9. Augmented SBERT — Cross-Encoder Pseudo-Labels
**What:** When labeled pairs are scarce, use a cross-encoder to generate pseudo-labels for unlabeled (resume, JD) pairs, then fine-tune the bi-encoder on those labels. Book calls this "Augmented SBERT."

**Where:** Applicable if train.csv has <500 clean labels. Use `cross-encoder/ms-marco-MiniLM-L-6-v2` to score 1000 unlabeled pairs, use scores as soft labels for bi-encoder training.

**Cost:** One-time compute ~2 hours on CPU/MPS.

---

## Ch 11 — Fine-Tuning Representation Models

### 10. SetFit for Tier Classification (Few-Shot)
**What:** SetFit (Sentence Transformer Fine-Tuning) achieves competitive classification with only 8 examples per class. Uses contrastive fine-tuning of SBERT + classification head. No GPU required.

**Application:** Train a 3-class classifier (Strong / Moderate / Weak) directly from the 30-item eval set — 10 per class. Replace or supplement the current weighted-average scoring with a direct classification head.

**How:**
```python
from setfit import SetFitModel, Trainer, TrainingArguments

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# 10 examples per tier from eval_set.json
trainer = Trainer(model=model, train_dataset=eval_df, ...)
trainer.train()
```

**Expected impact:** SetFit at 8-shot typically achieves 85%+ on 3-class text classification. Could replace the noisy weighted-average tier decision with a learned boundary.

**Risk:** 30 items is the floor for SetFit. Accuracy will be unstable. Needs more eval data first — target 100 items before training.

---

## Ch 12 — Fine-Tuning Generation Models

### 11. QLoRA Instruction Fine-Tuning of mistral-nemo
**What:** Fine-tune mistral-nemo (or a smaller model) on (resume, JD, judge_output) tuples using QLoRA. The judge would then natively produce structured JSON at correct confidence without retries. Book shows QLoRA runs on a single GPU (or Apple Silicon MPS) with 4-bit quantization.

**Data:** Every eval run produces `eval_results.jsonl` with judge outputs. Accumulate 200+ runs → extract (system_prompt, user_prompt, judge_json_output) → instruction tuples. Each of the 5 judges gets its own adapter.

**Format:**
```
Instruction: [judge system prompt]
Input: RESUME: ... JOB DESCRIPTION: ...
Output: {"score": 0.87, "confidence": 90, ...}
```

**Library:** `transformers` + `peft` + `bitsandbytes`. Runs on MPS (Mac) with 4-bit.

**Expected impact:** Eliminates retries (currently 1-3 retries per judge on complex inputs). Removes the 30-40s retry overhead. Judge accuracy should improve as it learns domain-specific scoring patterns.

**Prerequisite:** Need 200+ high-quality judge outputs. Run full 30-item eval 7+ times first.

---

### 12. DPO (Preference Tuning) for Score Calibration
**What:** After SFT, apply Direct Preference Optimization: collect preferred vs rejected judge outputs (where preferred = correct tier, rejected = wrong tier from existing evals). DPO aligns the model to human-preferred scoring without a reward model.

**Data:** From eval runs, identify where the judge scored wrong tier vs correct tier. These become (prompt, rejected_output, preferred_output) triples.

**When:** After collecting 50+ rejection pairs (i.e., after running ~10 full evals).

---

## Priority Order

| Priority | Item | Effort | Expected Gain |
|----------|------|--------|---------------|
| 1 | Zero-shot CoT on all prompts | 1 hour | Score stability |
| 2 | Prompt position fix (lost in the middle) | 2 hours | Skills/relevancy accuracy |
| 3 | Few-shot examples in skills + relevancy prompts | 3 hours | Relevancy calibration |
| 4 | Resume chunking for SBERT | 2 hours | Sim score signal quality |
| 5 | Hybrid BM25 + semantic for skills | 4 hours | Skills false-negative reduction |
| 6 | Cross-encoder reranking for skills | 4 hours | Skills precision |
| 7 | SetFit tier classifier | 1 day | Replaces weighted average |
| 8 | SBERT domain fine-tuning | 2 days | Embedding quality |
| 9 | QLoRA judge fine-tuning | 3-5 days | Eliminates retries, improves accuracy |
| 10 | DPO preference tuning | 2 days | Score calibration |

Items 1–3 are prompt-only changes. No new dependencies, no infra changes, high ROI. Do these first.
Items 4–6 require code changes to chunker + base but no training.
Items 7–10 require training infrastructure — do after eval set grows to 100+ items.

---

## Ch 4 — Text Classification

### 13. Zero-Shot Tier Classification via Label Embeddings
**What:** Embed descriptive label sentences and rank candidates by cosine similarity — no LLM call, no labeled data required. Book achieves F1=0.78 zero-shot. The trick: label descriptions must be specific, not just class names.

**For hireable:** Instead of just "Strong/Moderate/Weak", embed:
- `"A strong candidate with directly relevant experience, matching technical skills, and proven impact in this role's domain"`
- `"A moderate candidate with partial skill overlap and transferable experience but missing key requirements"`
- `"A weak candidate whose background does not align with the role's core technical or domain requirements"`

Compute cosine sim between (resume+JD concatenated embedding) and each label embedding. Take argmax.

**Where:** New `scripts/zero_shot_classifier.py` as a fast pre-filter before running the full 5-judge pipeline. Filter out obvious Weak candidates (sim < 0.3 to Strong label) without burning 160 seconds of LLM time.

**Expected impact:** ~80% of true Weak candidates filterable in <1 second. Use as a gate: only run full pipeline if candidate clears a minimum similarity threshold.

---

### 14. SBERT + Logistic Regression Tier Classifier
**What:** Freeze SBERT, embed (resume+JD) pairs, train logistic regression on tier labels. Book achieves F1=0.85 with this approach — comparable to the LLM pipeline at a fraction of the cost.

**Data:** `eval_set.json` (30 items — too small) + accumulate from eval runs. Need ~200 labeled (resume, JD, tier) pairs. Each full 30-item eval run produces 30. After 7 runs, have enough.

**How:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([r["resume_text"][:1000] + " " + r["job_description"][:500] 
                           for r in eval_items])
clf = LogisticRegression(random_state=42)
clf.fit(embeddings, labels)
```

**Use case:** Replace or complement the weighted-average tier decision. Run LR classifier in parallel with judges, flag disagreements (`post_validate` warning: `"lr_disagrees_with_judges"`).

**Prerequisite:** 200+ labeled pairs. Start after 7 full eval runs.

---

### 15. Specific Label Descriptions Improve Zero-Shot Accuracy
**What:** Book shows that vague label names ("positive"/"negative") give ~78% F1 but more descriptive phrases improve it. This applies to how judges describe scoring criteria.

**For hireable:** Audit all judge system prompts for vague rubric language. Replace:
- "Score high if candidate has relevant experience" → "Score 0.8+ if candidate has ≥3 years directly working with the required technology stack in a production environment and can demonstrate measurable outcomes"

**Where:** `analysers/*/prompts.py` — every scoring rubric. One pass through all 5 prompts, tighten the criteria language to be concrete and measurable.

---

### 16. TF-IDF + LogReg as Diagnostic Baseline
**What:** Book explicitly recommends always comparing against TF-IDF + logistic regression as a strong baseline. If the LLM pipeline can't beat TF-IDF+LR, something is wrong.

**For hireable:** Add to `scripts/run_eval.py` a parallel TF-IDF+LR baseline score. Track if the 5-judge LLM pipeline is adding value over this cheap baseline.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

tfidf_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000)), 
                        ("clf", LogisticRegression())])
```

**Expected impact:** Diagnostic only. If LLM pipeline accuracy < TF-IDF+LR, the prompts are the problem not the architecture.

---

## Ch 5 — Text Clustering and Topic Modeling

### 17. BERTopic on Job Descriptions — Discover JD Categories
**What:** Run BERTopic (SBERT → UMAP → HDBSCAN → c-TF-IDF) on train.csv job descriptions to discover latent JD clusters. Reveals which JD types the pipeline handles well vs poorly.

**Pipeline:**
```python
from bertopic import BERTopic

topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=True)
topics, probs = topic_model.fit_transform(jd_texts)
topic_model.get_topic_info()
```

**Application:** After running the full eval, group failures by their BERTopic cluster. If 80% of Moderate→Weak misclassifications belong to cluster 3 ("contract/part-time engineering roles"), that's a systematic prompt failure, not a random one. Fix the prompt for that JD type.

**Data:** Use the 30 JDs from `eval_set.json` to start. Run on train.csv JDs for full picture.

---

### 18. HDBSCAN Outlier Detection for Anomalous Resumes
**What:** HDBSCAN assigns outlier label (-1) to points that don't fit any cluster. Book explicitly covers this. Outlier resumes = candidates the pipeline will likely score inconsistently because they don't match any recognizable pattern.

**Application:** Before running the 5-judge pipeline, embed the resume and check if it falls in an HDBSCAN outlier region (trained on train.csv resumes). If outlier: add `post_validate` warning `"anomalous_resume_no_cluster"` and treat the score as lower-confidence.

**Expected impact:** Flags exotic/unusual resume formats (e.g., career changers, academics moving to industry) before the judges see them. Surfaces cases where human review is warranted.

---

### 19. BERTopic on Eval Failures — Systematic Error Analysis
**What:** After accumulating 100+ eval results, cluster the wrong predictions using BERTopic. The topic keywords of the failure cluster reveal WHY the judges are failing.

**For hireable:** After each batch of eval runs, run:
```python
failures = [r for r in results if r["predicted_tier"] != r["human_tier"]]
failure_jds = [f["job_description"] for f in failures]
topic_model.fit_transform(failure_jds)
topic_model.visualize_barchart()
```

If failures cluster around "leadership/management JDs" → seniority judge needs better C-suite criteria. If they cluster around "startup/early-stage JDs" → impact judge's metric calibration is off.

---

### 20. Semi-Supervised Topic Modeling for Tier-Correlated JD Types
**What:** BERTopic supports guided/semi-supervised mode where you provide seed labels. Use eval tier labels to guide clustering and find JD topic clusters that correlate with high/low pipeline accuracy.

**Application:** Run BERTopic in semi-supervised mode with Strong/Moderate/Weak as seeds on the (JD, tier) pairs. This surfaces: "JDs of type X are systematically classified wrong." More targeted than looking at per-judge scores alone.

---

## Ch 12 Addendum — DPO Implementation Details

### 21. DPO Data Format and Training (Concrete Implementation)
**What:** Book shows exact DPO data format and training config. Dataset format: `{prompt, chosen, rejected}` where `chosen` = correct judge output (right tier) and `rejected` = wrong judge output (wrong tier). Filter: only use pairs where `chosen_score >= 8` (high-confidence correct) and not ties.

**DPO training config for judge fine-tuning:**
```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./judge_dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,          # Lower than SFT (2e-4) — critical
    lr_scheduler_type="cosine",
    max_steps=200,
    warmup_ratio=0.1,
    fp16=True,
    beta=0.1,                    # KL divergence penalty
)

dpo_trainer = DPOTrainer(
    model, args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=512,
    max_length=512,
)
```

**Data collection:** From eval runs, collect pairs where the judge produced wrong tier. Wrong tier output = `rejected`. Run same prompt again (or use a higher-quality model response) to get `chosen`. Need ~50+ rejection pairs minimum.

**When:** After 10+ full eval runs — enough wrong predictions to build a preference dataset.

---

### 22. ORPO — Single-Loop SFT + Preference Tuning
**What:** Book covers ORPO (Odds Ratio Preference Optimization) which combines SFT and DPO into a single training pass. Eliminates the two-stage SFT→DPO pipeline complexity.

**For hireable:** When doing judge fine-tuning, prefer ORPO over SFT+DPO if compute is limited. One training loop, one LoRA adapter. Library: `trl` (same as DPO).

**When:** Same prerequisite as DPO — needs 200+ (resume, JD, judge_output) pairs with preference labels.

---

### 23. LLM-as-a-Judge for Automated Eval Quality
**What:** Book describes using a third LLM to evaluate the quality of another LLM's output. "Pairwise comparison" variant: two judge outputs for same candidate, third LLM decides which is better.

**For hireable:** Use as a cheap calibration check. For any candidate where judges disagree significantly (e.g., skills=90%, impact=17%), pipe both scores to a meta-evaluator LLM that asks: "Given this resume and JD, is 90% skills plausible alongside 17% impact?" Flag inconsistencies.

**Where:** `analysers/base.py` — add optional meta-evaluation step when judge score variance is high.

---

### 24. Goodhart's Law — Protect the Eval Set
**What:** Book quotes Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure." Explicitly warns that benchmark overfitting is a real risk.

**For hireable:** The 30-item eval set must remain a true held-out benchmark. Do NOT tune thresholds, judge prompts, or weights specifically to improve eval set scores — that's overfitting. Keep a separate "development set" for tuning. Eval set accuracy is only valid if you haven't touched it during iteration.

**Action:** Split the eval set: 20 items for development tuning, 10 items held out as true benchmark. Only run the 10-item set when reporting final accuracy numbers.

---

## Updated Priority Order

| Priority | Item | Effort | Expected Gain |
|----------|------|--------|---------------|
| 1 | Zero-shot CoT on all prompts (#1) | 1 hour | Score stability |
| 2 | Prompt position fix — lost in the middle (#3) | 2 hours | Skills/relevancy accuracy |
| 3 | Specific label descriptions in rubrics (#15) | 2 hours | Judge calibration |
| 4 | Few-shot examples in skills + relevancy (#2) | 3 hours | Relevancy calibration |
| 5 | Resume chunking for SBERT (#7) | 2 hours | Sim score signal |
| 6 | Zero-shot label embedding classifier (#13) | 3 hours | Fast pre-filter |
| 7 | Hybrid BM25 + semantic for skills (#6) | 4 hours | Skills false-negative reduction |
| 8 | Cross-encoder reranking for skills (#5) | 4 hours | Skills precision |
| 9 | TF-IDF+LR baseline in eval (#16) | 2 hours | Diagnostic |
| 10 | BERTopic on JD clusters (#17) | 4 hours | Failure pattern analysis |
| 11 | HDBSCAN outlier flagging (#18) | 4 hours | Confidence calibration |
| 12 | Goodhart's Law — split eval set (#24) | 30 min | Eval integrity |
| 13 | SetFit tier classifier (#10) | 1 day | Replaces weighted average |
| 14 | SBERT domain fine-tuning (#8) | 2 days | Embedding quality |
| 15 | QLoRA judge fine-tuning (#11) | 3-5 days | Eliminates retries |
| 16 | DPO / ORPO preference tuning (#21, #22) | 2 days | Score calibration |

Items 1–4 are prompt-only. Items 5–12 are code changes, no training. Items 13–16 require training data — wait until 200+ labeled pairs accumulated.
