"""
Shadow eval — compare current vs variant prompt for a single judge.
Based on Designing ML Systems Ch.9 shadow deployment pattern.

Run BEFORE switching a prompt to production. Flags regressions on eval set.
Runs both prompts in parallel per item; measures score delta and tier impact.

Usage:
    python scripts/shadow_eval.py --judge skills \
                                  --variant-prompt path/to/new_prompt.txt \
                                  [--model mistral-nemo:latest] [--limit 20]

    # The variant-prompt file should contain only the system prompt text.
    # Current prompt is read from the live analyser module.

NOTE: Do NOT run while run_eval.py is running — both use Ollama concurrently.
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from analysers.base import call_llm_async, parse_and_validate

EVAL_FILE    = os.path.join(BASE_DIR, "data", "eval_set.json")
RESULTS_FILE = os.path.join(BASE_DIR, "data", "eval_results.jsonl")
SHADOW_FILE  = os.path.join(BASE_DIR, "data", "shadow_eval_results.jsonl")

TIER_THRESHOLDS = {"Strong": 70, "Moderate": 18}


def score_to_tier(pct: float) -> str:
    if pct >= TIER_THRESHOLDS["Strong"]: return "Strong"
    if pct >= TIER_THRESHOLDS["Moderate"]: return "Moderate"
    return "Weak"


def _get_current_prompt(judge: str) -> str:
    """Pull the live system prompt from the analyser module for the given judge."""
    try:
        mod = __import__(f"analysers.{judge}", fromlist=["SYSTEM_PROMPT"])
        return getattr(mod, "SYSTEM_PROMPT", "")
    except Exception:
        pass
    try:
        mod = __import__(f"analysers.{judge}_analyser", fromlist=["SYSTEM_PROMPT"])
        return getattr(mod, "SYSTEM_PROMPT", "")
    except Exception:
        pass
    # Fallback: try to read from the analyser file directly
    for fname in [f"{judge}.py", f"{judge}_analyser.py"]:
        path = os.path.join(BASE_DIR, "analysers", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            import re
            m = re.search(r'SYSTEM_PROMPT\s*=\s*["\']([^"\']+)["\']|'
                          r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', content, re.DOTALL)
            if m:
                return (m.group(1) or m.group(2) or "").strip()
    return ""


async def _run_prompt(system: str, user: str, model: str) -> float | None:
    """Run a single prompt pair and return the extracted score (0–100) or None."""
    result = await call_llm_async(system, user, model, temperature=0.0)
    if not result.get("success"):
        return None
    try:
        data = parse_and_validate(result["data"])
        score = data.get("score", data.get("overall_score", data.get("match_score")))
        if score is not None:
            score = float(score)
            return score * 100 if score <= 1.0 else score
    except Exception:
        pass
    return None


async def shadow_one(item: dict, current_sys: str, variant_sys: str, model: str) -> dict:
    resume = item["resume_text"]
    jd     = item["job_description"]
    user   = f"RESUME:\n{resume[:3000]}\n\nJOB DESCRIPTION:\n{jd[:1500]}"

    current_score, variant_score = await asyncio.gather(
        _run_prompt(current_sys, user, model),
        _run_prompt(variant_sys, user, model),
    )

    current_tier = score_to_tier(current_score) if current_score is not None else None
    variant_tier = score_to_tier(variant_score) if variant_score is not None else None
    human_tier   = item["human_tier"]

    current_correct = current_tier == human_tier if current_tier else None
    variant_correct = variant_tier == human_tier if variant_tier else None

    delta = None
    if current_score is not None and variant_score is not None:
        delta = round(variant_score - current_score, 1)

    tier_changed = current_tier != variant_tier

    return {
        "id":              item["id"],
        "human_tier":      human_tier,
        "current_score":   round(current_score, 1) if current_score is not None else None,
        "variant_score":   round(variant_score, 1) if variant_score is not None else None,
        "delta":           delta,
        "current_tier":    current_tier,
        "variant_tier":    variant_tier,
        "current_correct": current_correct,
        "variant_correct": variant_correct,
        "tier_changed":    tier_changed,
        "regression":      current_correct and not variant_correct,
        "improvement":     not current_correct and variant_correct,
    }


def print_report(results: list[dict], judge: str):
    ok = [r for r in results if r["current_score"] is not None and r["variant_score"] is not None]
    if not ok:
        print("No valid paired results.")
        return

    deltas      = [r["delta"] for r in ok]
    regressions = [r for r in ok if r["regression"]]
    improvements= [r for r in ok if r["improvement"]]
    tier_changes= [r for r in ok if r["tier_changed"]]

    curr_acc = sum(1 for r in ok if r["current_correct"]) / len(ok) * 100
    var_acc  = sum(1 for r in ok if r["variant_correct"]) / len(ok) * 100

    print(f"\n{'='*65}")
    print(f"  SHADOW EVAL — judge: {judge}")
    print(f"{'='*65}")
    print(f"  Items evaluated : {len(ok)}")
    print(f"  Current accuracy: {curr_acc:.1f}%")
    print(f"  Variant accuracy: {var_acc:.1f}%   ({var_acc - curr_acc:+.1f}%)")
    print(f"  Score delta     : mean={np.mean(deltas):+.1f}  "
          f"std={np.std(deltas):.1f}  max_abs={max(abs(d) for d in deltas):.1f}")
    print(f"  Tier changes    : {len(tier_changes)} / {len(ok)}")
    print(f"  Regressions     : {len(regressions)}  (correct→wrong)")
    print(f"  Improvements    : {len(improvements)}  (wrong→correct)")

    verdict = "✓ SAFE" if not regressions else f"⚠ RISKY — {len(regressions)} regression(s)"
    print(f"\n  Verdict: {verdict}\n")

    if regressions:
        print("  Regressions:")
        for r in regressions[:5]:
            print(f"    {r['id']} ({r['human_tier']}): "
                  f"current={r['current_tier']}@{r['current_score']}% → "
                  f"variant={r['variant_tier']}@{r['variant_score']}%  Δ={r['delta']:+.1f}")

    if improvements:
        print(f"\n  Improvements:")
        for r in improvements[:5]:
            print(f"    {r['id']} ({r['human_tier']}): "
                  f"current={r['current_tier']}@{r['current_score']}% → "
                  f"variant={r['variant_tier']}@{r['variant_score']}%  Δ={r['delta']:+.1f}")

    print(f"\n{'='*65}\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True,
                    choices=["skills", "experience", "seniority", "relevancy", "impact"])
    ap.add_argument("--variant-prompt", required=True,
                    help="Path to text file containing new system prompt")
    ap.add_argument("--model",  default="mistral-nemo:latest")
    ap.add_argument("--limit",  type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.variant_prompt):
        print(f"Variant prompt file not found: {args.variant_prompt}")
        sys.exit(1)

    with open(args.variant_prompt, encoding="utf-8") as f:
        variant_sys = f.read().strip()

    current_sys = _get_current_prompt(args.judge)
    if not current_sys:
        print(f"WARNING: Could not extract current system prompt for judge '{args.judge}'.")
        print("Proceeding with empty baseline — all deltas will be vs blank system prompt.")

    with open(EVAL_FILE, encoding="utf-8") as f:
        all_items = json.load(f)

    items = all_items[:args.limit] if args.limit else all_items

    print(f"Shadow eval: judge={args.judge}  model={args.model}  n={len(items)}")
    print(f"Variant prompt: {args.variant_prompt}  ({len(variant_sys)} chars)")

    all_results = []
    for i, item in enumerate(items, 1):
        print(f"  [{i:3d}/{len(items)}] {item['id']} ...", end=" ", flush=True)
        t0 = time.monotonic()
        result = await shadow_one(item, current_sys, variant_sys, args.model)
        all_results.append(result)
        reg = "⚠" if result["regression"] else ("+" if result["improvement"] else "·")
        print(f"{reg} delta={result['delta'] or 'err':+}  ({round(time.monotonic()-t0, 1)}s)")

    with open(SHADOW_FILE, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print_report(all_results, args.judge)
    print(f"Results saved to: {SHADOW_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
