"""
Evaluation harness for the LLM judge pipeline.

Measures whether judge scores correlate with real hiring outcomes (Correctness),
and whether judge reasoning is grounded in source documents (Faithfulness).

Three faithfulness checks (from paper: Gan et al. 2025, §3.1):
  1. Coach faithfulness   — bullet_rewrite original_content must exist in resume text
  2. Impact factuality    — extracted metrics must appear in resume text
  3. Relevancy grounding  — relevant statements must share keywords with mapped JD requirement

Usage:
    python eval.py                    # run evaluation
    python eval.py --threshold 0.65   # custom pass/fail threshold
    python eval.py --faithfulness     # faithfulness audit only (no labels required)

Requires:
    - data/labels.csv populated with resume_id, job_id, fit_label (1=good fit, 0=bad fit)
    - data/resumes.csv and data/jobs.csv to resolve IDs to text
    - data/runs.db (SQLite) populated by prior pipeline runs

How to add labels:
    Option A: POST /feedback with resume_id, job_id, got_interview=true/false
    Option B: Edit data/labels.csv directly — resume_id,job_id,fit_label,golden
"""

import argparse
import csv
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
LABELS_CSV = BASE_DIR / "data" / "labels.csv"
RESUMES_CSV = BASE_DIR / "data" / "resumes.csv"
JOBS_CSV = BASE_DIR / "data" / "jobs.csv"
# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pipeline_results() -> list[dict]:
    """Load all pipeline runs from SQLite (data/runs.db)."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from db import list_runs
    rows = list_runs(limit=1000)
    results = []
    for row in rows:
        pr = row.get("pipeline_data") or {}
        results.append({
            "result_id":       row["result_id"],
            "pipeline_result": pr,
            "resume_text":     row.get("resume_snippet", ""),
            "job_description": row.get("job_snippet", ""),
            "_file":           row["result_id"],
        })
    return results


# ── Faithfulness checks ───────────────────────────────────────────────────────

_STOP = {"a", "an", "the", "and", "or", "of", "in", "to", "for", "with",
         "at", "by", "on", "is", "was", "are", "were", "it", "this", "that"}

def _content_words(text: str) -> set[str]:
    """Lowercase alpha tokens, stop-words removed, min length 3."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOP}


def coach_faithfulness(pipeline_result: dict) -> dict | None:
    """
    For every bullet_rewrite, check original_content has word-overlap with resume_text.
    Returns None if no rewrites present (gap_warning-only run is not applicable).

    Paper ref: Generation Faithfulness — Response ↔ Relevant Documents.
    """
    resume_text = pipeline_result.get("resume_text", "")
    resume_words = _content_words(resume_text)
    coach_data = (pipeline_result.get("pipeline_result", {})
                                 .get("coach", {})
                                 .get("data", {}))
    rewrites = [r for r in coach_data.get("improvement_analysis", [])
                if r.get("improvement_type") == "bullet_rewrite"
                and r.get("original_content", "").strip()]

    if not rewrites:
        return None

    grounded, ungrounded = [], []
    for r in rewrites:
        orig_words = _content_words(r["original_content"])
        overlap = orig_words & resume_words
        # grounded if >50% of original_content words found in resume
        ratio = len(overlap) / len(orig_words) if orig_words else 0
        bucket = grounded if ratio >= 0.5 else ungrounded
        bucket.append({
            "original_content": r["original_content"][:80],
            "overlap_ratio": round(ratio, 2),
            "grounded": ratio >= 0.5,
        })

    score = len(grounded) / len(rewrites)
    return {
        "score": round(score, 3),
        "n_rewrites": len(rewrites),
        "n_grounded": len(grounded),
        "n_ungrounded": len(ungrounded),
        "ungrounded_items": ungrounded,
    }


def impact_factuality(pipeline_result: dict) -> dict | None:
    """
    For each impact statement the judge marked as quantified (is_quantified=1),
    verify its extracted_metrics actually appear in resume_text.

    Paper ref: FactScore — atomic facts must be supported by source document.
    """
    resume_text = pipeline_result.get("resume_text", "").lower()
    impact_data = (pipeline_result.get("pipeline_result", {})
                                  .get("impact", {})
                                  .get("judge_result", {})
                                  .get("data", {}))
    quantified = [s for s in impact_data.get("impact_analysis", [])
                  if s.get("is_quantified") == 1
                  and s.get("extracted_metrics")]

    if not quantified:
        return None

    supported, unsupported = [], []
    for s in quantified:
        metrics = s.get("extracted_metrics", [])
        found = [m for m in metrics if str(m).lower() in resume_text]
        all_found = len(found) == len(metrics)
        bucket = supported if all_found else unsupported
        bucket.append({
            "statement": s.get("resume_statement", "")[:60],
            "metrics": metrics,
            "found_in_resume": found,
        })

    score = len(supported) / len(quantified)
    return {
        "score": round(score, 3),
        "n_quantified": len(quantified),
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "unsupported_items": unsupported,
    }


def _sbert_model():
    """Lazy-load SBERT model once. Returns None if sentence-transformers not installed."""
    if not hasattr(_sbert_model, "_cache"):
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model._cache = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _sbert_model._cache = None
    return _sbert_model._cache


def relevancy_grounding(pipeline_result: dict) -> dict | None:
    """
    For each statement the relevancy judge marked relevant (is_relevant=1),
    verify semantic similarity to its mapped_jd_requirement using SBERT cosine
    similarity (falls back to keyword overlap if sentence-transformers not installed).

    SBERT threshold 0.4: below this the judge is calling things relevant without
    semantic evidence. Paper ref: Retrieval Relevance — Relevant Docs ↔ Query.
    Semantic similarity approach from: Reimers & Gurevych 2019 (1908.10084).
    """
    rel_data = (pipeline_result.get("pipeline_result", {})
                               .get("relevancy", {})
                               .get("judge_result", {})
                               .get("data", {}))
    relevant = [s for s in rel_data.get("statement_analysis", [])
                if s.get("is_relevant") == 1
                and s.get("mapped_jd_requirement", "").strip()]

    if not relevant:
        return None

    model = _sbert_model()
    similarity_scores = []
    low_similarity = []

    if model is not None:
        import numpy as np
        statements = [s.get("resume_statement", "") for s in relevant]
        requirements = [s.get("mapped_jd_requirement", "") for s in relevant]
        stmt_embs = model.encode(statements, convert_to_numpy=True)
        req_embs = model.encode(requirements, convert_to_numpy=True)
        # cosine similarity per pair
        norms_s = np.linalg.norm(stmt_embs, axis=1, keepdims=True)
        norms_r = np.linalg.norm(req_embs, axis=1, keepdims=True)
        sims = (stmt_embs * req_embs).sum(axis=1) / (norms_s.squeeze() * norms_r.squeeze() + 1e-8)
        threshold = 0.4
        method = "sbert-cosine"
        for s, sim in zip(relevant, sims):
            similarity_scores.append(float(sim))
            if sim < threshold:
                low_similarity.append({
                    "statement": s.get("resume_statement", "")[:60],
                    "mapped_to": s.get("mapped_jd_requirement", ""),
                    "similarity": round(float(sim), 3),
                })
    else:
        # keyword overlap fallback
        method = "keyword-overlap"
        threshold = 0.2
        for s in relevant:
            stmt_words = _content_words(s.get("resume_statement", ""))
            req_words = _content_words(s.get("mapped_jd_requirement", ""))
            overlap = stmt_words & req_words
            ratio = len(overlap) / len(req_words) if req_words else 0
            similarity_scores.append(ratio)
            if ratio < threshold:
                low_similarity.append({
                    "statement": s.get("resume_statement", "")[:60],
                    "mapped_to": s.get("mapped_jd_requirement", ""),
                    "similarity": round(ratio, 3),
                })

    avg = sum(similarity_scores) / len(similarity_scores)
    return {
        "score": round(avg, 3),
        "method": method,
        "threshold": threshold,
        "n_relevant_statements": len(relevant),
        "n_low_similarity": len(low_similarity),
        "low_similarity_items": low_similarity,
    }


def run_faithfulness_audit(pipeline_results: list[dict]) -> list[dict]:
    """Run all three faithfulness checks across all pipeline results."""
    audit = []
    for r in pipeline_results:
        rid = r.get("result_id", r.get("_file", "unknown"))
        cf = coach_faithfulness(r)
        imf = impact_factuality(r)
        rg = relevancy_grounding(r)
        audit.append({
            "result_id": rid,
            "coach_faithfulness": cf,
            "impact_factuality": imf,
            "relevancy_grounding": rg,
        })
    return audit


# ── Matching ──────────────────────────────────────────────────────────────────

def _name_from_resume_text(text: str) -> str:
    """Extract candidate name from first non-empty line of resume text."""
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) < 60:
            return line.upper()
    return ""


def match_pipeline_to_labels(
    pipeline_results: list[dict],
    resumes: list[dict],
    jobs: list[dict],
    labels: list[dict],
) -> list[dict]:
    """
    Returns matched rows: {result_file, resume_name, job_title, score, fit_label, golden}.
    Matches pipeline result → resume by name comparison, then looks up label.
    """
    # Build lookup maps
    resume_by_id = {r["resume_id"]: r for r in resumes}
    job_by_id = {j["job_id"]: j for j in jobs}
    label_map = {(lb["resume_id"], lb["job_id"]): lb for lb in labels}

    # Build name → resume_id map for matching pipeline results to labels
    name_to_id = {}
    for r in resumes:
        key = r.get("name", "").upper().strip()
        if key:
            name_to_id[key] = r["resume_id"]

    matched = []
    unmatched = []

    for result in pipeline_results:
        resume_text = result.get("resume_text", "")
        job_desc = result.get("job_description", "")
        overall = (result.get("pipeline_result", {})
                         .get("overall", {})
                         .get("overall_score", None))
        if overall is None:
            continue

        candidate_name = _name_from_resume_text(resume_text)
        resume_id = name_to_id.get(candidate_name)

        if not resume_id:
            unmatched.append(result["_file"])
            continue

        # Match job by description prefix (first 100 chars)
        desc_prefix = job_desc[:100].strip()
        job_id = None
        for j in jobs:
            if j.get("description", "")[:100].strip() == desc_prefix:
                job_id = j["job_id"]
                break

        if not job_id:
            unmatched.append(result["_file"])
            continue

        label = label_map.get((resume_id, job_id))
        if not label:
            unmatched.append(result["_file"])
            continue

        resume = resume_by_id.get(resume_id, {})
        job = job_by_id.get(job_id, {})

        matched.append({
            "result_file": result["_file"],
            "resume_name": resume.get("name", candidate_name),
            "job_title": job.get("title", ""),
            "score": overall,
            "fit_label": int(label["fit_label"]),
            "golden": int(label.get("golden", 0)),
        })

    if unmatched:
        print(f"\n  Note: {len(unmatched)} pipeline result(s) could not be matched to labels:")
        for f in unmatched:
            print(f"    {f}")

    return matched


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(matched: list[dict], threshold: float = 0.60) -> dict:
    if not matched:
        return {}

    scores = [r["score"] for r in matched]
    labels = [r["fit_label"] for r in matched]
    n = len(matched)

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        corr, pvalue = spearmanr(scores, labels)
    except ImportError:
        corr, pvalue = None, None

    # Binary predictions at threshold
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, lb in zip(preds, labels) if p == 1 and lb == 1)
    fp = sum(1 for p, lb in zip(preds, labels) if p == 1 and lb == 0)
    fn = sum(1 for p, lb in zip(preds, labels) if p == 0 and lb == 1)
    tn = sum(1 for p, lb in zip(preds, labels) if p == 0 and lb == 0)

    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Score distribution by label
    pos_scores = [r["score"] for r in matched if r["fit_label"] == 1]
    neg_scores = [r["score"] for r in matched if r["fit_label"] == 0]
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "n": n,
        "threshold": threshold,
        "spearman_r": round(corr, 3) if corr is not None else "n/a (install scipy)",
        "spearman_p": round(pvalue, 4) if pvalue is not None else "n/a",
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "avg_score_positive": round(avg(pos_scores), 3),
        "avg_score_negative": round(avg(neg_scores), 3),
        "n_positive": len(pos_scores),
        "n_negative": len(neg_scores),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(matched: list[dict], metrics: dict, threshold: float):
    print("\n" + "=" * 60)
    print("  JUDGE PIPELINE EVALUATION REPORT")
    print("=" * 60)

    if not matched:
        print("\n  No matched pairs found.")
        print("\n  To run evaluation:")
        print("  1. Add resumes and jobs via the dashboard")
        print("  2. Run the pipeline on resume-job pairs")
        print("  3. Mark outcomes: POST /feedback with got_interview=true/false")
        print("     or edit data/labels.csv directly")
        print("  4. Re-run: python eval.py")
        return

    print(f"\n  Matched pairs:  {metrics['n']}")
    print(f"  Positive (fit): {metrics['n_positive']}")
    print(f"  Negative (fit): {metrics['n_negative']}")
    print(f"  Threshold:      {threshold:.0%}")

    print("\n── Correlation ──────────────────────────────────────────")
    print(f"  Spearman r:     {metrics['spearman_r']}")
    print(f"  p-value:        {metrics['spearman_p']}")
    if isinstance(metrics['spearman_r'], float):
        if abs(metrics['spearman_r']) >= 0.5:
            print("  Interpretation: strong rank correlation with outcomes ✓")
        elif abs(metrics['spearman_r']) >= 0.3:
            print("  Interpretation: moderate correlation — judges directionally correct")
        else:
            print("  Interpretation: weak correlation — review judge prompts or label quality")

    print(f"\n── Classification at {threshold:.0%} threshold ──────────────────────")
    print(f"  Accuracy:       {metrics['accuracy']:.1%}")
    print(f"  Precision:      {metrics['precision']:.1%}  (of predicted fits, how many were real?)")
    print(f"  Recall:         {metrics['recall']:.1%}  (of real fits, how many did we catch?)")
    print(f"  F1:             {metrics['f1']:.3f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")

    print("\n── Score distribution ───────────────────────────────────")
    print(f"  Avg score (fit=1):  {metrics['avg_score_positive']:.1%}")
    print(f"  Avg score (fit=0):  {metrics['avg_score_negative']:.1%}")

    print("\n── Per-pair results ─────────────────────────────────────")
    header = f"  {'Resume':<20} {'Job':<30} {'Score':>6} {'Label':>6} {'Pred':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(matched, key=lambda x: x["score"], reverse=True):
        pred = "✓" if r["score"] >= threshold else "✗"
        actual = "fit" if r["fit_label"] == 1 else "no"
        mark = "  " if (r["score"] >= threshold) == bool(r["fit_label"]) else "⚠ "
        print(f"{mark} {r['resume_name']:<20} {r['job_title'][:28]:<30} {r['score']:>5.1%} {actual:>6} {pred:>5}")

    print()


# ── Faithfulness report ───────────────────────────────────────────────────────

def print_faithfulness_report(audit: list[dict]):
    print("\n" + "=" * 60)
    print("  FAITHFULNESS AUDIT REPORT")
    print("=" * 60)
    print("  (Paper: Gan et al. 2025 — Internal Evaluation §3.1)\n")

    for entry in audit:
        print(f"  Result: {entry['result_id']}")

        cf = entry["coach_faithfulness"]
        if cf:
            status = "✓" if cf["score"] >= 0.8 else "⚠"
            print(f"  {status} Coach Faithfulness:    {cf['score']:.1%}  "
                  f"({cf['n_grounded']}/{cf['n_rewrites']} rewrites grounded in resume)")
            for item in cf["ungrounded_items"]:
                print(f"       ✗ LOW OVERLAP ({item['overlap_ratio']:.0%}): \"{item['original_content']}\"")
        else:
            print("  — Coach Faithfulness:    n/a (no bullet rewrites in this result)")

        imf = entry["impact_factuality"]
        if imf:
            status = "✓" if imf["score"] >= 0.8 else "⚠"
            print(f"  {status} Impact Factuality:     {imf['score']:.1%}  "
                  f"({imf['n_supported']}/{imf['n_quantified']} claimed metrics found in resume)")
            for item in imf["unsupported_items"]:
                print(f"       ✗ NOT IN RESUME: {item['metrics']} in \"{item['statement']}\"")
        else:
            print("  — Impact Factuality:     n/a (no fully-quantified statements)")

        rg = entry["relevancy_grounding"]
        if rg:
            method = rg.get("method", "keyword-overlap")
            threshold = rg.get("threshold", 0.3)
            status = "✓" if rg["score"] >= threshold else "⚠"
            label = "avg cosine sim" if method == "sbert-cosine" else "avg keyword overlap"
            print(f"  {status} Relevancy Grounding:   {rg['score']:.3f} {label} [{method}]  "
                  f"({rg['n_low_similarity']} low-similarity statements)")
            for item in rg["low_similarity_items"][:3]:
                print(f"       ⚠ \"{item['statement']}\"")
                print(f"         mapped to: {item['mapped_to']}  (sim={item['similarity']:.3f})")
        else:
            print("  — Relevancy Grounding:   n/a")

        print()

    # Aggregate
    cf_scores = [e["coach_faithfulness"]["score"]
                 for e in audit if e["coach_faithfulness"]]
    imf_scores = [e["impact_factuality"]["score"]
                  for e in audit if e["impact_factuality"]]
    rg_scores = [e["relevancy_grounding"]["score"]
                 for e in audit if e["relevancy_grounding"]]

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    print("── Aggregate ────────────────────────────────────────────")
    cf_avg = avg(cf_scores)
    imf_avg = avg(imf_scores)
    rg_avg = avg(rg_scores)
    print(f"  Coach faithfulness:   {f'{cf_avg:.1%}' if cf_avg is not None else 'n/a'}")
    print(f"  Impact factuality:    {f'{imf_avg:.1%}' if imf_avg is not None else 'n/a'}")
    print(f"  Relevancy grounding:  {f'{rg_avg:.1%}' if rg_avg is not None else 'n/a'}")
    print()
    print("  Interpretation:")
    print("  Coach faithfulness <80%  → coach is rewriting bullets not in the resume")
    print("  Impact factuality  <80%  → impact judge is hallucinating metrics")
    print("  Relevancy grounding <0.4 (sbert) / <30% (keyword) → relevancy judge not grounding decisions in text")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM judge pipeline accuracy")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Score threshold for pass/fail classification (default: 0.60)")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Run faithfulness audit only (no labels required)")
    args = parser.parse_args()

    print(f"Loading data from {BASE_DIR}...")
    pipeline_results = load_pipeline_results()
    print(f"  Pipeline files:  {len(pipeline_results)}")

    if not pipeline_results:
        print("\n  No pipeline results found in data/runs.db — run the analyzer first.")
        print("  Run the pipeline first via the dashboard.")
        sys.exit(0)

    # Faithfulness audit runs on all pipeline results, no labels needed
    audit = run_faithfulness_audit(pipeline_results)
    print_faithfulness_report(audit)

    if args.faithfulness:
        sys.exit(0)

    labels = load_csv(LABELS_CSV)
    resumes = load_csv(RESUMES_CSV)
    jobs = load_csv(JOBS_CSV)

    print(f"  Labels:          {len(labels)} rows")
    print(f"  Resumes:         {len(resumes)} rows")
    print(f"  Jobs:            {len(jobs)} rows")

    if not labels:
        print("\n  labels.csv is empty — skipping correctness evaluation.")
        print("  Run the pipeline, then mark outcomes via POST /feedback or edit labels.csv.")
        sys.exit(0)

    matched = match_pipeline_to_labels(pipeline_results, resumes, jobs, labels)
    metrics = compute_metrics(matched, threshold=args.threshold)
    print_report(matched, metrics, threshold=args.threshold)


if __name__ == "__main__":
    main()
