"""
Custom evaluation metrics for the 5-judge pipeline.
Run after run_eval.py has populated data/eval_results.jsonl.

Metrics:
  1. Tier accuracy          — already in run_eval.py, reproduced here for completeness
  2. Evidence groundedness  — are evidence quotes actually in the resume? (hallucination check)
  3. Score-reasoning gap    — judges that return high confidence but low/zero scores
  4. Cross-run consistency  — re-runs a sample and measures score variance (optional, slow)
  5. Judge ablation         — leave one judge out, recompute tier metrics
  6. Threshold sweep        — find Strong/Moderate cutoffs that improve macro F1
  7. Error magnitude        — how far misses are from labels / expected ranges
  8. Calibration            — score/confidence bins vs actual correctness

Usage:
    python scripts/eval_metrics.py                    # metrics from saved results
    python scripts/eval_metrics.py --consistency 10  # also runs metric 4 on 10 random items
"""
import argparse
import asyncio
import json
import os
import sys
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_FILE = os.path.join(BASE_DIR, "data", "eval_results.jsonl")
EVAL_FILE    = os.path.join(BASE_DIR, "data", "eval_set.json")

JUDGES = ["skills", "experience", "seniority", "relevancy", "impact"]
TIERS = ["Strong", "Moderate", "Weak"]
TIER_TO_ORDINAL = {"Weak": 0, "Moderate": 1, "Strong": 2}
DEFAULT_WEIGHTS = {
    "skills": 0.30,
    "experience": 0.25,
    "seniority": 0.15,
    "relevancy": 0.15,
    "impact": 0.15,
}


def score_to_tier(score_pct: float, strong_cutoff: float = 70.0, moderate_cutoff: float = 18.0) -> str:
    if score_pct >= strong_cutoff:
        return "Strong"
    if score_pct >= moderate_cutoff:
        return "Moderate"
    return "Weak"


def _dedupe_latest_by_id(results: list[dict]) -> list[dict]:
    """
    eval_results.jsonl is append-only. For diagnostics, use the latest row per ID
    so repeated runs do not overweight older outputs.
    """
    latest = {}
    for r in results:
        rid = r.get("id")
        if not rid:
            continue
        latest[rid] = r
    return list(latest.values())


def _ok(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("success")]


def _classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    total = len(y_true)
    if not total:
        return {"error": "no labels"}

    by_tier = {}
    f1s = []
    for tier in TIERS:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == tier and yp == tier)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != tier and yp == tier)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == tier and yp != tier)
        support = sum(1 for yt in y_true if yt == tier)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        by_tier[tier] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / total
    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3),
        "by_tier": by_tier,
    }


def _weighted_score_pct(r: dict, judges: list[str] | None = None, weights: dict | None = None) -> float | None:
    judges = judges or JUDGES
    weights = weights or _weights_from_result(r)
    scores = r.get("judge_scores") or {}
    available = [j for j in judges if j in scores and scores[j] is not None]
    if not available:
        return None
    total_weight = sum(weights.get(j, 0) for j in available)
    if total_weight <= 0:
        return None
    score = sum(float(scores[j]) * weights.get(j, 0) for j in available) / total_weight
    return round(score, 1)


def _weights_from_result(r: dict) -> dict:
    pipeline_weights = (
        r.get("pipeline_data", {})
        .get("overall", {})
        .get("weights_used")
    )
    if isinstance(pipeline_weights, dict) and pipeline_weights:
        return pipeline_weights
    return DEFAULT_WEIGHTS


def _expected_ranges(eval_items: list[dict]) -> dict[str, list[float]]:
    return {
        item["id"]: item.get("expected_score_range", [0, 100])
        for item in eval_items
        if item.get("id")
    }


def _confidence_from_pipeline(r: dict, judge: str | None = None) -> float | None:
    pipeline = r.get("pipeline_data", {})
    if judge:
        confidence = (
            pipeline.get(judge, {})
            .get("analyser_result", {})
            .get("data", {})
            .get("confidence")
        )
        return float(confidence) if confidence is not None else None

    confidences = []
    for j in JUDGES:
        confidence = _confidence_from_pipeline(r, j)
        if confidence is not None:
            confidences.append(confidence)
    return round(sum(confidences) / len(confidences), 1) if confidences else None


# ── Metric 1: Tier accuracy (summary) ────────────────────────────────────────

def metric_tier_accuracy(results: list[dict]) -> dict:
    ok = _ok(results)
    if not ok:
        return {"error": "no successful results"}
    correct = sum(r["correct"] for r in ok)
    by_tier = {}
    for tier in ["Strong", "Moderate", "Weak"]:
        t = [r for r in ok if r["human_tier"] == tier]
        by_tier[tier] = {
            "n": len(t),
            "correct": sum(r["correct"] for r in t),
            "accuracy": round(sum(r["correct"] for r in t) / len(t) * 100, 1) if t else None,
        }
    return {
        "total": len(ok),
        "correct": correct,
        "accuracy_pct": round(correct / len(ok) * 100, 1),
        "by_tier": by_tier,
    }


# ── Metric 2: Threshold sweep ────────────────────────────────────────────────

def metric_threshold_sweep(results: list[dict]) -> dict:
    """
    Sweeps tier cutoffs over saved overall scores. This identifies whether
    Moderate/Strong errors are mainly threshold calibration rather than judge output.
    """
    ok = _ok(results)
    if not ok:
        return {"error": "no successful results"}

    baseline_true = [r["human_tier"] for r in ok]
    baseline_pred = [r["predicted_tier"] for r in ok]
    baseline = _classification_metrics(baseline_true, baseline_pred)

    candidates = []
    for moderate_cutoff in range(10, 46, 2):
        for strong_cutoff in range(max(50, moderate_cutoff + 10), 91, 2):
            y_pred = [
                score_to_tier(float(r["overall_pct"]), strong_cutoff, moderate_cutoff)
                for r in ok
            ]
            metrics = _classification_metrics(baseline_true, y_pred)
            candidates.append({
                "moderate_cutoff": moderate_cutoff,
                "strong_cutoff": strong_cutoff,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "moderate_f1": metrics["by_tier"]["Moderate"]["f1"],
                "moderate_recall": metrics["by_tier"]["Moderate"]["recall"],
            })

    candidates.sort(
        key=lambda c: (c["macro_f1"], c["accuracy"], c["moderate_f1"], c["moderate_recall"]),
        reverse=True,
    )
    best = candidates[0]
    best_pred = [
        score_to_tier(float(r["overall_pct"]), best["strong_cutoff"], best["moderate_cutoff"])
        for r in ok
    ]

    return {
        "baseline_thresholds": {"moderate_cutoff": 18, "strong_cutoff": 70},
        "baseline": baseline,
        "best_thresholds": {
            "moderate_cutoff": best["moderate_cutoff"],
            "strong_cutoff": best["strong_cutoff"],
        },
        "best": _classification_metrics(baseline_true, best_pred),
        "top_5": candidates[:5],
        "interpretation": (
            "If the best metrics are much higher than baseline, tune thresholds first. "
            "If they barely move, judge prompts/weights are the likely bottleneck."
        ),
    }


# ── Metric 3: Judge ablation ─────────────────────────────────────────────────

def metric_judge_ablation(results: list[dict]) -> dict:
    """
    Leave one judge out, renormalize remaining weights, recompute tiers, and
    compare against the saved full-pipeline prediction.
    """
    ok = _ok(results)
    if not ok:
        return {"error": "no successful results"}

    y_true = [r["human_tier"] for r in ok]
    full_pred = [r["predicted_tier"] for r in ok]
    full = _classification_metrics(y_true, full_pred)

    rows = []
    for removed in JUDGES:
        remaining = [j for j in JUDGES if j != removed]
        y_pred = []
        changed = []
        fixed = []
        broken = []
        for r in ok:
            score = _weighted_score_pct(r, remaining)
            pred = score_to_tier(score) if score is not None else "Weak"
            y_pred.append(pred)
            if pred != r["predicted_tier"]:
                changed.append({
                    "id": r["id"],
                    "human": r["human_tier"],
                    "full_pred": r["predicted_tier"],
                    "ablated_pred": pred,
                    "full_score": r["overall_pct"],
                    "ablated_score": score,
                })
                if r["predicted_tier"] != r["human_tier"] and pred == r["human_tier"]:
                    fixed.append(r["id"])
                if r["predicted_tier"] == r["human_tier"] and pred != r["human_tier"]:
                    broken.append(r["id"])

        metrics = _classification_metrics(y_true, y_pred)
        rows.append({
            "removed_judge": removed,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "delta_accuracy": round(metrics["accuracy"] - full["accuracy"], 3),
            "delta_macro_f1": round(metrics["macro_f1"] - full["macro_f1"], 3),
            "moderate_f1": metrics["by_tier"]["Moderate"]["f1"],
            "prediction_changes": len(changed),
            "fixes": fixed,
            "breaks": broken,
            "examples": changed[:8],
        })

    rows.sort(key=lambda r: (r["delta_macro_f1"], r["delta_accuracy"]), reverse=True)
    return {
        "baseline": full,
        "leave_one_out": rows,
        "interpretation": (
            "Positive delta means removing that judge improved saved-result metrics. "
            "Negative delta means the judge is helping. Many prediction changes indicate high influence."
        ),
    }


# ── Metric 4: Error magnitude ────────────────────────────────────────────────

def metric_error_magnitude(results: list[dict], eval_items: list[dict]) -> dict:
    """
    Measures severity beyond right/wrong: tier distance and distance outside the
    human expected score range.
    """
    ok = _ok(results)
    ranges = _expected_ranges(eval_items)
    if not ok:
        return {"error": "no successful results"}

    rows = []
    by_tier = {tier: [] for tier in TIERS}
    for r in ok:
        expected = ranges.get(r["id"], [0, 100])
        score = float(r["overall_pct"])
        if score < expected[0]:
            range_error = expected[0] - score
        elif score > expected[1]:
            range_error = score - expected[1]
        else:
            range_error = 0.0

        tier_error = abs(
            TIER_TO_ORDINAL.get(r["human_tier"], 0)
            - TIER_TO_ORDINAL.get(r["predicted_tier"], 0)
        )
        signed_tier_error = (
            TIER_TO_ORDINAL.get(r["predicted_tier"], 0)
            - TIER_TO_ORDINAL.get(r["human_tier"], 0)
        )
        row = {
            "id": r["id"],
            "human": r["human_tier"],
            "predicted": r["predicted_tier"],
            "overall_pct": score,
            "expected_range": expected,
            "range_error": round(range_error, 1),
            "tier_error": tier_error,
            "direction": "over-scored" if signed_tier_error > 0 else "under-scored" if signed_tier_error < 0 else "same-tier",
            "judge_scores": r.get("judge_scores", {}),
        }
        rows.append(row)
        by_tier[r["human_tier"]].append(row)

    misses = [r for r in rows if r["tier_error"] > 0]
    out_of_range = [r for r in rows if r["range_error"] > 0]
    rows.sort(key=lambda r: (r["tier_error"], r["range_error"]), reverse=True)

    return {
        "tier_misses": len(misses),
        "mean_tier_error": round(sum(r["tier_error"] for r in rows) / len(rows), 2),
        "out_of_expected_range": len(out_of_range),
        "mean_range_error": round(sum(r["range_error"] for r in rows) / len(rows), 1),
        "by_human_tier": {
            tier: {
                "n": len(items),
                "mean_range_error": round(sum(i["range_error"] for i in items) / len(items), 1) if items else 0,
                "tier_misses": sum(1 for i in items if i["tier_error"] > 0),
                "under_scored": sum(1 for i in items if i["direction"] == "under-scored"),
                "over_scored": sum(1 for i in items if i["direction"] == "over-scored"),
            }
            for tier, items in by_tier.items()
        },
        "worst_10": rows[:10],
        "interpretation": (
            "Large range error with correct tier means score calibration is off. "
            "Large tier error means thresholds, weights, or judge evidence are changing decisions."
        ),
    }


# ── Metric 5: Calibration ───────────────────────────────────────────────────

def _bin_label(start: int, end: int) -> str:
    return f"{start:02d}-{end:02d}"


def _score_bins(rows: list[dict], value_key: str, bucket_size: int = 10) -> list[dict]:
    bins = []
    for start in range(0, 100, bucket_size):
        end = start + bucket_size
        bucket = [
            r for r in rows
            if r.get(value_key) is not None
            and start <= float(r[value_key]) < (end if end < 100 else 101)
        ]
        if not bucket:
            continue
        correct = sum(1 for r in bucket if r["correct"])
        in_range = sum(1 for r in bucket if r.get("in_range"))
        avg_value = sum(float(r[value_key]) for r in bucket) / len(bucket)
        observed_accuracy = correct / len(bucket) * 100
        bins.append({
            "bin": _bin_label(start, min(end, 100)),
            "n": len(bucket),
            f"avg_{value_key}": round(avg_value, 1),
            "accuracy_pct": round(observed_accuracy, 1),
            "in_expected_range_pct": round(in_range / len(bucket) * 100, 1),
            "calibration_gap_pct": round(avg_value - observed_accuracy, 1),
        })
    return bins


def metric_calibration(results: list[dict]) -> dict:
    """
    Overall score calibration: do higher scores actually correspond to more
    correct tier decisions and more in-range scores?

    Confidence calibration: do average judge confidence values behave like
    correctness probabilities? This is approximate because judge confidence is
    not a calibrated probability, but it quickly shows overconfidence.
    """
    ok = _ok(results)
    if not ok:
        return {"error": "no successful results"}

    rows = []
    judge_rows = []
    for r in ok:
        rows.append({
            "id": r["id"],
            "overall_pct": float(r["overall_pct"]),
            "avg_judge_confidence": _confidence_from_pipeline(r),
            "correct": bool(r["correct"]),
            "in_range": bool(r.get("in_range")),
            "human_tier": r["human_tier"],
            "predicted_tier": r["predicted_tier"],
        })

        for judge in JUDGES:
            confidence = _confidence_from_pipeline(r, judge)
            score = (r.get("judge_scores") or {}).get(judge)
            if confidence is None or score is None:
                continue
            judge_rows.append({
                "id": r["id"],
                "judge": judge,
                "confidence": confidence,
                "score": float(score),
                "item_correct": bool(r["correct"]),
                "human_tier": r["human_tier"],
            })

    overall_bins = _score_bins(rows, "overall_pct")
    confidence_bins = _score_bins(rows, "avg_judge_confidence")

    by_judge = {}
    for judge in JUDGES:
        jr = [r for r in judge_rows if r["judge"] == judge]
        if not jr:
            continue
        high_conf = [r for r in jr if r["confidence"] >= 80]
        by_judge[judge] = {
            "n": len(jr),
            "avg_score": round(sum(r["score"] for r in jr) / len(jr), 1),
            "avg_confidence": round(sum(r["confidence"] for r in jr) / len(jr), 1),
            "high_confidence_count": len(high_conf),
            "high_confidence_item_accuracy_pct": round(
                sum(1 for r in high_conf if r["item_correct"]) / len(high_conf) * 100, 1
            ) if high_conf else None,
        }

    return {
        "overall_score_bins": overall_bins,
        "avg_judge_confidence_bins": confidence_bins,
        "by_judge_confidence": by_judge,
        "interpretation": (
            "Positive calibration_gap_pct means scores/confidence are higher than observed accuracy. "
            "Use this to decide whether to adjust score ranges, confidence prompts, or thresholds."
        ),
    }


# ── Metric 2: Evidence groundedness ──────────────────────────────────────────

def _fuzzy_in(quote: str, source: str, threshold: float = 0.75) -> bool:
    """
    Returns True if `quote` appears in `source` with ≥ threshold similarity.
    Uses sliding window so short quotes match substrings, not the whole text.
    """
    if not quote or not source:
        return False
    quote = quote.lower().strip()
    source = source.lower()
    # Exact substring — fast path
    if quote in source:
        return True
    # Sliding window fuzzy match over source chunks of similar length
    qlen = len(quote)
    step = max(1, qlen // 2)
    for i in range(0, len(source) - qlen + 1, step):
        window = source[i:i + qlen + 20]
        ratio = SequenceMatcher(None, quote, window[:qlen]).ratio()
        if ratio >= threshold:
            return True
    return False


def metric_evidence_groundedness(eval_items: list[dict], results: list[dict] | None = None) -> dict:
    """
    Checks whether evidence quotes produced by judges are grounded in the resume text.
    Requires the full pipeline_data — reads from eval_results.jsonl if available there,
    otherwise falls back to checking what's in each result's judge_scores (limited).

    Full groundedness requires re-running the pipeline and capturing raw judge outputs.
    This function checks what it can from saved results and flags what's missing.
    """
    # Load full pipeline data if available. Prefer caller-provided results so
    # append-only historical rows do not skew groundedness counts.
    if results is not None:
        results_with_pipeline = [r for r in results if r.get("pipeline_data")]
    else:
        results_with_pipeline = []
    if not results_with_pipeline and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("pipeline_data"):
                        results_with_pipeline.append(r)

    if not results_with_pipeline:
        return {
            "status": "skipped",
            "reason": "pipeline_data not saved in eval_results.jsonl — re-run with --save-pipeline flag (not yet implemented)",
            "recommendation": "Add pipeline_data to eval results to enable groundedness checks",
        }

    # Build resume lookup from eval_set.json
    resume_lookup = {item["id"]: item["resume_text"] for item in eval_items}

    grounded, ungrounded, missing = 0, 0, 0
    failures = []

    EVIDENCE_PATHS = {
        "skills":     ("skill_analysis", "evidence"),
        "experience": ("responsibility_analysis", "resume_evidence"),
        "relevancy":  ("statement_analysis", "statement"),   # statements ARE the resume text
        "impact":     ("impact_analysis", "statement"),
    }

    for r in results_with_pipeline:
        resume = resume_lookup.get(r["id"], "")
        if not resume:
            missing += 1
            continue
        pipeline = r.get("pipeline_data", {})
        for judge, (list_key, quote_key) in EVIDENCE_PATHS.items():
            judge_data = pipeline.get(judge, {}).get("analyser_result", {}).get("data", {})
            items = judge_data.get(list_key, [])
            for item in items:
                quote = str(item.get(quote_key, "")).strip()
                if not quote or quote in ("N/A", "unknown", ""):
                    missing += 1
                    continue
                if _fuzzy_in(quote, resume):
                    grounded += 1
                else:
                    ungrounded += 1
                    failures.append({
                        "eval_id": r["id"],
                        "judge": judge,
                        "quote": quote[:120],
                    })

    total = grounded + ungrounded
    return {
        "total_quotes_checked": total,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "missing_or_empty": missing,
        "groundedness_pct": round(grounded / total * 100, 1) if total else None,
        "failures": failures[:20],  # cap at 20 examples
    }


# ── Metric 3: Score-reasoning gap ────────────────────────────────────────────

def metric_score_reasoning_gap(results: list[dict]) -> dict:
    """
    Flags judges where confidence is high (>=80) but score is very low (<20%).
    These are the most dangerous outputs — model is certain but likely wrong.
    """
    ok = [r for r in results if r.get("success") and r.get("pipeline_data")]
    if not ok:
        return {"status": "skipped", "reason": "no pipeline_data in results"}

    flags = []
    for r in ok:
        pipeline = r.get("pipeline_data", {})
        for judge in ["skills", "experience", "seniority", "relevancy", "impact"]:
            ar = pipeline.get(judge, {}).get("analyser_result", {})
            confidence = ar.get("data", {}).get("confidence", None)
            score = r["judge_scores"].get(judge)
            if confidence is None or score is None:
                continue
            if confidence >= 80 and score < 20:
                flags.append({
                    "eval_id": r["id"],
                    "judge": judge,
                    "score_pct": score,
                    "confidence": confidence,
                    "human_tier": r["human_tier"],
                })

    return {
        "total_results": len(ok),
        "high_conf_low_score_flags": len(flags),
        "flag_rate_pct": round(len(flags) / (len(ok) * 5) * 100, 1),  # 5 judges per result
        "examples": flags[:15],
    }


# ── Metric 4: Cross-run consistency ──────────────────────────────────────────

async def metric_consistency(eval_items: list[dict], model: str, n: int = 10) -> dict:
    """
    Re-runs n random eval items twice and measures score variance.
    temperature=0 should give near-zero variance; JSON repair or model
    instability can cause drift. Flag if std dev > 5 points.
    """
    import random
    import numpy as np
    from analysers import AnalyserEngine

    sample = random.sample(eval_items, min(n, len(eval_items)))
    engine = AnalyserEngine()

    results = []
    for item in sample:
        scores_across_runs = []
        for run in range(2):
            try:
                pipeline = await engine.run(
                    resume_text=item["resume_text"],
                    job_description=item["job_description"],
                    llm_model=model,
                )
                overall = pipeline.get("overall", {}).get("overall_score", 0) * 100
                scores_across_runs.append(overall)
            except Exception as e:
                scores_across_runs.append(None)
        if all(s is not None for s in scores_across_runs):
            delta = abs(scores_across_runs[0] - scores_across_runs[1])
            results.append({
                "id": item["id"],
                "run1": round(scores_across_runs[0], 1),
                "run2": round(scores_across_runs[1], 1),
                "delta": round(delta, 1),
                "unstable": delta > 5,
            })

    unstable = [r for r in results if r["unstable"]]
    deltas = [r["delta"] for r in results]
    return {
        "n_pairs": len(results),
        "mean_delta": round(np.mean(deltas), 2) if deltas else None,
        "max_delta": round(max(deltas), 2) if deltas else None,
        "unstable_count": len(unstable),
        "unstable_pct": round(len(unstable) / len(results) * 100, 1) if results else None,
        "unstable_examples": unstable,
        "verdict": "STABLE" if not unstable else f"UNSTABLE — {len(unstable)} pairs varied >5 points",
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_metric(name: str, data: dict):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(json.dumps(data, indent=2))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consistency", type=int, default=0, metavar="N",
                    help="Re-run N eval items twice to measure score variance (slow)")
    ap.add_argument("--model", default="mistral-nemo:latest")
    ap.add_argument("--all-results", action="store_true",
                    help="Use every JSONL row instead of the latest result per eval ID")
    args = ap.parse_args()

    # Load eval set
    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_items = json.load(f)

    # Load results
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    raw_result_count = len(results)
    if not args.all_results:
        results = _dedupe_latest_by_id(results)

    print(f"\n{'='*60}")
    print("  JUDGE PIPELINE — CUSTOM EVAL METRICS")
    print(f"{'='*60}")
    print(f"  Eval set : {len(eval_items)} items")
    print(f"  Results  : {len(results)} used ({raw_result_count} JSONL rows)")

    print_metric("1. Tier Accuracy", metric_tier_accuracy(results))
    print_metric("2. Threshold Sweep", metric_threshold_sweep(results))
    print_metric("3. Judge Ablation", metric_judge_ablation(results))
    print_metric("4. Error Magnitude", metric_error_magnitude(results, eval_items))
    print_metric("5. Calibration", metric_calibration(results))
    print_metric("6. Evidence Groundedness", metric_evidence_groundedness(eval_items, results))
    print_metric("7. High-Confidence Low-Score Flags", metric_score_reasoning_gap(results))

    if args.consistency > 0:
        print(f"\nRunning consistency check on {args.consistency} items (2 runs each)…")
        consistency = await metric_consistency(eval_items, args.model, args.consistency)
        print_metric("8. Cross-Run Consistency", consistency)

    print(f"\n{'='*60}")
    print("  Recommended next step:")
    print("  Compare threshold sweep vs ablation deltas:")
    print("  - big threshold gain → tune cutoffs first")
    print("  - big ablation gain/loss → inspect that judge's prompt and scoring")
    print("  - big range error with correct tiers → calibrate score scale")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
