"""
Compare two eval runs — shows accuracy improvement/regression between runs.
Useful after prompt changes, model swaps, or threshold adjustments.

Usage:
    python scripts/eval_compare.py --baseline data/eval_results_v1.jsonl \
                                   --candidate data/eval_results.jsonl

    # Compare by model within same file:
    python scripts/eval_compare.py --model-a mistral-nemo:latest \
                                   --model-b phi4-mini:latest

Results printed to console and saved to data/eval_compare_report.json.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_FILE   = os.path.join(BASE_DIR, "data", "eval_results.jsonl")
COMPARE_REPORT = os.path.join(BASE_DIR, "data", "eval_compare_report.json")
TIERS  = ["Strong", "Moderate", "Weak"]
JUDGES = ["skills", "experience", "seniority", "relevancy", "impact"]


def load_file(path: str) -> list[dict]:
    results = []
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("success"):
                        results.append(r)
                except Exception:
                    pass
    return results


def load_by_model(path: str, model: str) -> list[dict]:
    all_r = load_file(path)
    filtered = [r for r in all_r if r.get("model", "").startswith(model)]
    if not filtered:
        print(f"No results found for model '{model}' in {path}")
        sys.exit(1)
    return filtered


def accuracy(results: list[dict]) -> float:
    correct = sum(1 for r in results if r.get("correct"))
    return round(correct / len(results) * 100, 1) if results else 0.0


def per_tier_accuracy(results: list[dict]) -> dict:
    out = {}
    for tier in TIERS:
        subset = [r for r in results if r["human_tier"] == tier]
        out[tier] = {
            "n": len(subset),
            "accuracy": accuracy(subset),
        }
    return out


def judge_means(results: list[dict]) -> dict:
    out = {}
    for judge in JUDGES:
        vals = [r["judge_scores"][judge] for r in results if judge in r.get("judge_scores", {})]
        out[judge] = round(np.mean(vals), 1) if vals else None
    return out


def strong_weak_swaps(results: list[dict]) -> list[dict]:
    return [
        r for r in results
        if (r["human_tier"] == "Strong" and r["predicted_tier"] == "Weak") or
           (r["human_tier"] == "Weak"   and r["predicted_tier"] == "Strong")
    ]


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    # Match by ID where both have results
    base_map = {r["id"]: r for r in baseline}
    cand_map = {r["id"]: r for r in candidate}
    shared_ids = set(base_map) & set(cand_map)

    if not shared_ids:
        print("WARNING: No shared IDs between baseline and candidate — comparing overall stats only.")

    base_shared = [base_map[i] for i in shared_ids] if shared_ids else baseline
    cand_shared = [cand_map[i] for i in shared_ids] if shared_ids else candidate

    base_acc  = accuracy(base_shared)
    cand_acc  = accuracy(cand_shared)
    acc_delta = round(cand_acc - base_acc, 1)

    # Per-item regressions (correct → incorrect) and improvements (incorrect → correct)
    regressions  = []
    improvements = []
    for id_ in shared_ids:
        b, c = base_map[id_], cand_map[id_]
        if b["correct"] and not c["correct"]:
            regressions.append({
                "id": id_, "tier": b["human_tier"],
                "base_predicted": b["predicted_tier"], "base_score": b["overall_pct"],
                "cand_predicted": c["predicted_tier"], "cand_score": c["overall_pct"],
            })
        elif not b["correct"] and c["correct"]:
            improvements.append({
                "id": id_, "tier": b["human_tier"],
                "base_predicted": b["predicted_tier"], "base_score": b["overall_pct"],
                "cand_predicted": c["predicted_tier"], "cand_score": c["overall_pct"],
            })

    # Judge score deltas
    base_judges = judge_means(base_shared)
    cand_judges = judge_means(cand_shared)
    judge_deltas = {
        j: round((cand_judges[j] or 0) - (base_judges[j] or 0), 1)
        for j in JUDGES
    }

    return {
        "n_shared":         len(shared_ids),
        "n_baseline":       len(baseline),
        "n_candidate":      len(candidate),
        "accuracy": {
            "baseline":  base_acc,
            "candidate": cand_acc,
            "delta":     acc_delta,
            "verdict":   "IMPROVED" if acc_delta > 0 else ("REGRESSED" if acc_delta < 0 else "UNCHANGED"),
        },
        "per_tier": {
            tier: {
                "baseline":  per_tier_accuracy(base_shared)[tier]["accuracy"],
                "candidate": per_tier_accuracy(cand_shared)[tier]["accuracy"],
                "delta":     round(
                    per_tier_accuracy(cand_shared)[tier]["accuracy"] -
                    per_tier_accuracy(base_shared)[tier]["accuracy"], 1
                ),
            }
            for tier in TIERS
        },
        "judge_score_deltas": judge_deltas,
        "regressions":        regressions,
        "improvements":       improvements,
        "swap_changes": {
            "baseline_swaps":  len(strong_weak_swaps(base_shared)),
            "candidate_swaps": len(strong_weak_swaps(cand_shared)),
        },
    }


def print_report(report: dict, label_a: str, label_b: str):
    print(f"\n{'='*65}")
    print("  EVAL COMPARISON REPORT")
    print(f"{'='*65}")
    print(f"  Baseline  : {label_a} ({report['n_baseline']} results)")
    print(f"  Candidate : {label_b} ({report['n_candidate']} results)")
    print(f"  Shared IDs: {report['n_shared']}\n")

    acc = report["accuracy"]
    verdict_icon = "✓" if acc["verdict"] == "IMPROVED" else ("✗" if acc["verdict"] == "REGRESSED" else "─")
    delta_str = f"{acc['delta']:+.1f}%"
    print(f"  {verdict_icon} Overall accuracy: {acc['baseline']}% → {acc['candidate']}%  ({delta_str})")

    print(f"\n  Per-tier accuracy:")
    for tier, v in report["per_tier"].items():
        delta_str = f"{v['delta']:+.1f}%"
        icon = "↑" if v["delta"] > 0 else ("↓" if v["delta"] < 0 else "─")
        print(f"    {tier:10s}: {v['baseline']:5.1f}% → {v['candidate']:5.1f}%  {icon} {delta_str}")

    print(f"\n  Judge score deltas (candidate − baseline):")
    for judge, delta in report["judge_score_deltas"].items():
        icon = "+" if delta > 0 else ("" if delta == 0 else "")
        print(f"    {judge:12s}: {delta:+.1f}pts")

    regressions  = report["regressions"]
    improvements = report["improvements"]
    print(f"\n  Regressions  (correct → wrong): {len(regressions)}")
    for r in regressions[:5]:
        print(f"    {r['id']} ({r['tier']}): {r['base_predicted']}@{r['base_score']}% → {r['cand_predicted']}@{r['cand_score']}%")

    print(f"  Improvements (wrong → correct): {len(improvements)}")
    for r in improvements[:5]:
        print(f"    {r['id']} ({r['tier']}): {r['base_predicted']}@{r['base_score']}% → {r['cand_predicted']}@{r['cand_score']}%")

    swaps = report["swap_changes"]
    print(f"\n  Strong↔Weak swaps: {swaps['baseline_swaps']} → {swaps['candidate_swaps']}")
    print(f"\n{'='*65}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",  help="Baseline .jsonl results file")
    ap.add_argument("--candidate", help="Candidate .jsonl results file (defaults to data/eval_results.jsonl)")
    ap.add_argument("--model-a",   help="Baseline model name (filter within single file)")
    ap.add_argument("--model-b",   help="Candidate model name (filter within single file)")
    args = ap.parse_args()

    if args.model_a and args.model_b:
        src = args.baseline or RESULTS_FILE
        baseline  = load_by_model(src, args.model_a)
        candidate = load_by_model(args.candidate or src, args.model_b)
        label_a, label_b = args.model_a, args.model_b
    elif args.baseline:
        baseline  = load_file(args.baseline)
        candidate = load_file(args.candidate or RESULTS_FILE)
        label_a   = os.path.basename(args.baseline)
        label_b   = os.path.basename(args.candidate or RESULTS_FILE)
    else:
        ap.print_help()
        sys.exit(1)

    report = compare(baseline, candidate)
    print_report(report, label_a, label_b)

    with open(COMPARE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {COMPARE_REPORT}")


if __name__ == "__main__":
    main()
