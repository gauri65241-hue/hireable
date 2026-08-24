"""
Run the judge pipeline against data/eval_set.json and report calibration metrics.

Usage:
    python scripts/run_eval.py [--model mistral-nemo:latest] [--limit 60] [--resume]

Results saved to data/eval_results.jsonl (one JSON line per eval item).
Re-running with --resume skips already-completed IDs.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from analysers import AnalyserEngine
from analysers.semantic import resume_jd_similarity

RESULTS_FILE = os.path.join(BASE_DIR, "data", "eval_results.jsonl")
EVAL_FILE    = os.path.join(BASE_DIR, "data", "eval_set.json")

# Score → tier mapping. Must mirror expected_score_range in build_eval_set.py.
TIER_THRESHOLDS = {"Strong": 0.70, "Moderate": 0.18}


def score_to_tier(overall_pct: float) -> str:
    if overall_pct >= TIER_THRESHOLDS["Strong"] * 100:
        return "Strong"
    if overall_pct >= TIER_THRESHOLDS["Moderate"] * 100:
        return "Moderate"
    return "Weak"


def in_expected_range(score_pct: float, expected_range: list[int]) -> bool:
    return expected_range[0] <= score_pct <= expected_range[1]


async def run_one(engine: AnalyserEngine, item: dict, model: str) -> dict:
    t0 = time.monotonic()
    try:
        pipeline = await engine.run(
            resume_text=item["resume_text"],
            job_description=item["job_description"],
            llm_model=model,
        )
    except Exception as e:
        return {"id": item["id"], "error": str(e), "success": False}

    # engine.run already computes scores; pull them from the "scores" sub-dict
    s   = pipeline.get("skills",     {}).get("scores", {})
    e   = pipeline.get("experience", {}).get("scores", {})
    sr  = pipeline.get("seniority",  {}).get("scores", {})
    rel = pipeline.get("relevancy",  {}).get("scores", {})
    imp = pipeline.get("impact",     {}).get("scores", {})

    # overall is also pre-computed by the engine
    overall_score = pipeline.get("overall", {}).get("overall_score", 0)
    overall_pct   = round(overall_score * 100, 1)
    predicted_tier = score_to_tier(overall_pct)

    sim = resume_jd_similarity(item["resume_text"][:2000], item["job_description"][:2000])

    return {
        "id":              item["id"],
        "success":         True,
        "human_tier":      item["human_tier"],
        "predicted_tier":  predicted_tier,
        "correct":         predicted_tier == item["human_tier"],
        "in_range":        in_expected_range(overall_pct, item.get("expected_score_range", [0, 100])),
        "overall_pct":     overall_pct,
        "judge_scores": {
            "skills":     round(s.get("score", 0) * 100, 1),
            "experience": round(e.get("score", 0) * 100, 1),
            "seniority":  round(sr.get("score", 0) * 100, 1),
            "relevancy":  round(rel.get("score", 0) * 100, 1),
            "impact":     round(imp.get("score", 0) * 100, 1),
        },
        "semantic_similarity": sim,
        "pipeline_data":  pipeline,   # saved for groundedness + reasoning-gap metrics
        "latency_s":  round(time.monotonic() - t0, 1),
        "model":      model,
        "timestamp":  datetime.utcnow().isoformat(),
    }


def load_completed() -> set[str]:
    done = set()
    if not os.path.exists(RESULTS_FILE):
        return done
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def append_result(result: dict):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def print_report(results: list[dict]):
    ok = [r for r in results if r.get("success")]
    if not ok:
        print("No successful results to report.")
        return

    y_true = [r["human_tier"]     for r in ok]
    y_pred = [r["predicted_tier"] for r in ok]
    tiers  = ["Strong", "Moderate", "Weak"]

    print("\n" + "="*60)
    print("EVAL REPORT")
    print("="*60)
    print(f"Evaluated : {len(ok)} / {len(results)} items")
    print(f"Errors    : {len(results) - len(ok)}")
    acc = sum(r["correct"] for r in ok) / len(ok) * 100
    print(f"Accuracy  : {acc:.1f}%")
    in_range = sum(r["in_range"] for r in ok) / len(ok) * 100
    print(f"In range  : {in_range:.1f}%  (score within expected_score_range)")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=tiers, zero_division=0))

    print("--- Confusion Matrix (rows=actual, cols=predicted) ---")
    cm = confusion_matrix(y_true, y_pred, labels=tiers)
    col_w = 12
    print(" " * col_w + "".join(t.rjust(col_w) for t in tiers))
    for i, tier in enumerate(tiers):
        print(tier.rjust(col_w) + "".join(str(v).rjust(col_w) for v in cm[i]))

    print("\n--- Per-Judge Score by Tier ---")
    judges = ["skills", "experience", "seniority", "relevancy", "impact"]
    header = f"{'Judge':12s}" + "".join(f"{t:>12s}" for t in tiers)
    print(header)
    for judge in judges:
        row = f"{judge:12s}"
        for tier in tiers:
            vals = [r["judge_scores"][judge] for r in ok if r["human_tier"] == tier]
            avg = np.mean(vals) if vals else 0
            row += f"{avg:>11.1f}%"
        print(row)

    print("\n--- Semantic Similarity by Tier ---")
    for tier in tiers:
        vals = [r["semantic_similarity"] for r in ok
                if r["human_tier"] == tier and r["semantic_similarity"] is not None]
        if vals:
            print(f"  {tier:10s}: mean={np.mean(vals):.3f}  "
                  f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    print("\n--- Worst Failures (Strong↔Weak swaps) ---")
    swaps = [r for r in ok
             if (r["human_tier"] == "Strong" and r["predicted_tier"] == "Weak") or
                (r["human_tier"] == "Weak"   and r["predicted_tier"] == "Strong")]
    if swaps:
        for r in swaps:
            print(f"  {r['id']}: actual={r['human_tier']} predicted={r['predicted_tier']} "
                  f"score={r['overall_pct']}% sim={r['semantic_similarity']}")
    else:
        print("  None — no Strong↔Weak swaps.")

    print("\n--- Latency ---")
    lats = [r["latency_s"] for r in ok]
    print(f"  Mean: {np.mean(lats):.1f}s  "
          f"P50: {np.median(lats):.1f}s  "
          f"P95: {np.percentile(lats, 95):.1f}s  "
          f"Total: {sum(lats):.0f}s")
    print("="*60)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval",   default=EVAL_FILE)
    ap.add_argument("--model",  default="mistral-nemo:latest")
    ap.add_argument("--limit",  type=int, default=0, help="0 = run all")
    ap.add_argument("--resume", action="store_true", help="skip already-completed IDs")
    ap.add_argument("--report-only", action="store_true", help="print report from existing results, no new runs")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        items = json.load(f)

    if args.report_only:
        all_results = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        print_report(all_results)
        return

    completed = load_completed() if args.resume else set()
    todo = [it for it in items if it["id"] not in completed]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Model     : {args.model}")
    print(f"Total     : {len(items)} items in eval set")
    print(f"Completed : {len(completed)} (skipped)")
    print(f"Running   : {len(todo)}")
    if not todo:
        print("Nothing to run — use --report-only to see results.")
        return

    engine = AnalyserEngine()
    all_results = []

    for i, item in enumerate(todo, 1):
        tier = item["human_tier"]
        print(f"[{i:3d}/{len(todo)}] {item['id']} ({tier}) ...", end=" ", flush=True)
        result = await run_one(engine, item, args.model)
        append_result(result)
        all_results.append(result)

        if result.get("success"):
            correct = "✓" if result["correct"] else "✗"
            print(f"{correct} predicted={result['predicted_tier']:8s} "
                  f"score={result['overall_pct']:5.1f}% "
                  f"sim={result['semantic_similarity'] or 0:.2f} "
                  f"({result['latency_s']}s)")
        else:
            print(f"ERROR: {result.get('error', '?')[:80]}")

    # Merge with any previously completed results for the full report
    if args.resume and completed:
        prev = []
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["id"] not in {x["id"] for x in all_results}:
                        prev.append(r)
        all_results = prev + all_results

    print_report(all_results)
    print(f"\nResults appended to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
