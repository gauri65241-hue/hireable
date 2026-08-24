"""
Calibration drift checker.

Reads the last N pipeline runs from the DB, runs all 6 eval judges against each,
computes per-judge calibration drift (mean predicted score vs eval quality score),
and logs results to data/calibration_log.jsonl.

Usage:
    conda run -n linkedin-agent-assistant python scripts/check_calibration.py
    conda run -n linkedin-agent-assistant python scripts/check_calibration.py --runs 20 --model phi4-mini:latest
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "calibration_log.jsonl"
)

DRIFT_WARN_THRESHOLD = 0.15  # flag judge if mean drift exceeds this


async def eval_run(engine, run: dict, model: str) -> dict | None:
    pr = run.get("pipeline_result", {})
    if not pr:
        return None
    try:
        evals = await engine.run_evals(
            pipeline_result=pr,
            resume_text=run.get("resume_text", ""),
            job_description=run.get("job_description", ""),
            llm_model=model,
        )
        return evals
    except Exception as e:
        print(f"  eval failed for {run['result_id']}: {e}")
        return None


def extract_eval_score(eval_result: dict, judge: str) -> float | None:
    r = eval_result.get(judge, {})
    if not r or not r.get("success"):
        return None
    data = r.get("data", {})
    return data.get("quality_score") or data.get("score") or data.get("overall_score")


def extract_predicted_score(pr: dict, judge: str) -> float | None:
    section = pr.get(judge, {})
    scores = section.get("scores", {})
    return scores.get("score") or scores.get("overall_score")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of recent runs to evaluate")
    parser.add_argument("--model", default="phi4-mini:latest")
    args = parser.parse_args()

    import db as run_db
    from analysers.engine import AnalyserEngine

    engine = AnalyserEngine(confidence_threshold=70, max_retries=1)

    print(f"Loading last {args.runs} runs from DB...")
    runs = await asyncio.to_thread(run_db.list_runs, None, args.runs, 0)

    if not runs:
        print("No runs found in DB.")
        return

    judges = ["skills", "experience", "seniority", "relevancy", "impact", "coach"]
    drift_by_judge: dict[str, list[float]] = {j: [] for j in judges}
    flagged = []

    for i, run_summary in enumerate(runs):
        run_id = run_summary["result_id"]
        print(f"[{i+1}/{len(runs)}] Evaluating {run_id}...")

        full_run = await asyncio.to_thread(run_db.get_run, run_id)
        if not full_run:
            print(f"  Could not fetch full run — skipping")
            continue

        evals = await eval_run(engine, full_run, args.model)
        if not evals:
            continue

        pr = full_run.get("pipeline_result", {})
        run_drifts = {}
        for judge in judges:
            predicted = extract_predicted_score(pr, judge)
            eval_score = extract_eval_score(evals, judge)
            if predicted is None or eval_score is None:
                continue
            drift = abs(predicted - eval_score)
            drift_by_judge[judge].append(drift)
            run_drifts[judge] = {
                "predicted": round(predicted, 3),
                "eval_score": round(eval_score, 3),
                "drift": round(drift, 3),
            }

        print(f"  Drifts: { {j: round(v['drift'], 3) for j, v in run_drifts.items()} }")

    # Compute mean drift per judge
    summary = {}
    for judge in judges:
        drifts = drift_by_judge[judge]
        if not drifts:
            summary[judge] = {"mean_drift": None, "n": 0, "flagged": False}
            continue
        mean = round(sum(drifts) / len(drifts), 3)
        flag = mean > DRIFT_WARN_THRESHOLD
        summary[judge] = {"mean_drift": mean, "n": len(drifts), "flagged": flag}
        if flag:
            flagged.append(judge)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs_evaluated": len(runs),
        "model": args.model,
        "judge_drift": summary,
        "flagged_judges": flagged,
    }

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\n{'='*50}")
    print(f"Calibration check complete — {len(runs)} runs evaluated")
    print(f"Results logged to {LOG_FILE}")
    print()
    for judge, stats in summary.items():
        if stats["mean_drift"] is None:
            print(f"  {judge:12s}  N/A")
            continue
        flag = " ⚠ DRIFT" if stats["flagged"] else ""
        print(f"  {judge:12s}  mean_drift={stats['mean_drift']:.3f}  n={stats['n']}{flag}")

    if flagged:
        print(f"\nFlagged judges (drift > {DRIFT_WARN_THRESHOLD}): {flagged}")
        print("Consider retraining prompts or adjusting calibration notes for these judges.")
    else:
        print("\nAll judges within drift threshold.")


if __name__ == "__main__":
    asyncio.run(main())
