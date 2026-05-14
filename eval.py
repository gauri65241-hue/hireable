"""
Evaluation harness for the LLM judge pipeline.

Measures whether judge scores correlate with real hiring outcomes.

Usage:
    python eval.py                    # run evaluation
    python eval.py --threshold 0.65   # custom pass/fail threshold

Requires:
    - data/labels.csv populated with resume_id, job_id, fit_label (1=good fit, 0=bad fit)
    - data/resumes.csv and data/jobs.csv to resolve IDs to text
    - results/pipeline/*.json from prior pipeline runs

How to add labels:
    Option A: POST /feedback with resume_id, job_id, got_interview=true/false
    Option B: Edit data/labels.csv directly — resume_id,job_id,fit_label,golden
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
LABELS_CSV = BASE_DIR / "data" / "labels.csv"
RESUMES_CSV = BASE_DIR / "data" / "resumes.csv"
JOBS_CSV = BASE_DIR / "data" / "jobs.csv"
PIPELINE_DIR = BASE_DIR / "results" / "pipeline"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pipeline_results() -> list[dict]:
    results = []
    if not PIPELINE_DIR.exists():
        return results
    for p in sorted(PIPELINE_DIR.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = p.name
            results.append(data)
        except Exception as e:
            print(f"  Warning: could not read {p.name}: {e}")
    return results


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
    label_map = {(l["resume_id"], l["job_id"]): l for l in labels}

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
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)

    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Score distribution by label
    pos_scores = [r["score"] for r in matched if r["fit_label"] == 1]
    neg_scores = [r["score"] for r in matched if r["fit_label"] == 0]
    avg = lambda lst: sum(lst) / len(lst) if lst else 0

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

    print(f"\n── Score distribution ───────────────────────────────────")
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM judge pipeline accuracy")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Score threshold for pass/fail classification (default: 0.60)")
    args = parser.parse_args()

    print(f"Loading data from {BASE_DIR}...")
    labels = load_csv(LABELS_CSV)
    resumes = load_csv(RESUMES_CSV)
    jobs = load_csv(JOBS_CSV)
    pipeline_results = load_pipeline_results()

    print(f"  Labels:          {len(labels)} rows")
    print(f"  Resumes:         {len(resumes)} rows")
    print(f"  Jobs:            {len(jobs)} rows")
    print(f"  Pipeline files:  {len(pipeline_results)}")

    if not labels:
        print("\n  labels.csv is empty — no evaluation data yet.")
        print("  Run the pipeline, then mark outcomes via POST /feedback or edit labels.csv.")
        sys.exit(0)

    if not pipeline_results:
        print("\n  No pipeline results found in results/pipeline/.")
        print("  Run the pipeline first via the dashboard.")
        sys.exit(0)

    matched = match_pipeline_to_labels(pipeline_results, resumes, jobs, labels)
    metrics = compute_metrics(matched, threshold=args.threshold)
    print_report(matched, metrics, threshold=args.threshold)


if __name__ == "__main__":
    main()
