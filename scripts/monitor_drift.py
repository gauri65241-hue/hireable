"""
Score distribution drift monitor.
Based on Designing ML Systems Ch.8 — data distribution monitoring.

Reads eval_results.jsonl, computes rolling statistics per judge per tier,
and alerts if the latest window mean falls outside historical 2σ band.

Usage:
    python scripts/monitor_drift.py [--window 10] [--sigma 2.0]

Output: drift_report.json + console summary.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_FILE  = os.path.join(BASE_DIR, "data", "eval_results.jsonl")
DRIFT_REPORT  = os.path.join(BASE_DIR, "data", "drift_report.json")

JUDGES = ["skills", "experience", "seniority", "relevancy", "impact"]
TIERS  = ["Strong", "Moderate", "Weak"]


def load_results() -> list[dict]:
    results = []
    if not os.path.exists(RESULTS_FILE):
        return results
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("success") and r.get("judge_scores"):
                        results.append(r)
                except Exception:
                    pass
    return results


def compute_drift(results: list[dict], window: int, sigma: float) -> dict:
    # Group scores by (judge, tier) in chronological order
    series: dict[tuple, list[float]] = defaultdict(list)
    for r in results:
        tier = r["human_tier"]
        for judge in JUDGES:
            score = r["judge_scores"].get(judge)
            if score is not None:
                series[(judge, tier)].append(score)

    alerts = []
    stats = {}

    for (judge, tier), scores in series.items():
        if len(scores) < window + 1:
            continue

        historical = np.array(scores[:-window])
        recent     = np.array(scores[-window:])

        hist_mean = float(np.mean(historical))
        hist_std  = float(np.std(historical))
        rec_mean  = float(np.mean(recent))
        delta     = rec_mean - hist_mean
        z_score   = delta / hist_std if hist_std > 0 else 0

        key = f"{judge}/{tier}"
        stats[key] = {
            "n_total":    len(scores),
            "n_window":   window,
            "hist_mean":  round(hist_mean, 2),
            "hist_std":   round(hist_std, 2),
            "recent_mean": round(rec_mean, 2),
            "delta":      round(delta, 2),
            "z_score":    round(z_score, 2),
        }

        if abs(z_score) >= sigma:
            direction = "↑" if delta > 0 else "↓"
            alerts.append({
                "key":        key,
                "direction":  direction,
                "delta":      round(delta, 2),
                "z_score":    round(z_score, 2),
                "hist_mean":  round(hist_mean, 2),
                "recent_mean": round(rec_mean, 2),
                "severity":   "HIGH" if abs(z_score) >= sigma * 1.5 else "MEDIUM",
            })

    alerts.sort(key=lambda a: abs(a["z_score"]), reverse=True)

    return {
        "window":        window,
        "sigma":         sigma,
        "n_results":     len(results),
        "alerts":        alerts,
        "stats":         stats,
    }


def print_report(report: dict):
    print(f"\n{'='*60}")
    print("  SCORE DRIFT MONITOR")
    print(f"{'='*60}")
    print(f"  Results  : {report['n_results']}")
    print(f"  Window   : last {report['window']} runs vs historical")
    print(f"  Threshold: ±{report['sigma']}σ\n")

    alerts = report["alerts"]
    if not alerts:
        print("  ✓ No drift detected — all judges within normal bands.\n")
    else:
        print(f"  ⚠ {len(alerts)} drift alert(s):\n")
        for a in alerts:
            sev = "🔴" if a["severity"] == "HIGH" else "🟡"
            print(f"  {sev} {a['key']}")
            print(f"     {a['direction']} {abs(a['delta']):.1f}pts  "
                  f"(hist={a['hist_mean']:.1f}% → recent={a['recent_mean']:.1f}%)  "
                  f"z={a['z_score']:.1f}")

    print(f"\n  Stats by judge/tier:")
    for key, s in sorted(report["stats"].items()):
        flag = " ⚠" if any(a["key"] == key for a in alerts) else ""
        print(f"  {key:30s}  hist={s['hist_mean']:5.1f}%±{s['hist_std']:4.1f}  "
              f"recent={s['recent_mean']:5.1f}%  Δ={s['delta']:+.1f}{flag}")

    print(f"\n{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10,
                    help="Number of most-recent results to treat as 'current window'")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Alert threshold in standard deviations")
    args = ap.parse_args()

    results = load_results()
    if not results:
        print("No successful results found in eval_results.jsonl.")
        sys.exit(1)

    report = compute_drift(results, args.window, args.sigma)
    print_report(report)

    with open(DRIFT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {DRIFT_REPORT}")


if __name__ == "__main__":
    main()
