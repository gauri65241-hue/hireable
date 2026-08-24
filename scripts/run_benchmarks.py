"""
Benchmark runner — scores each case in data/benchmarks/cases.json through the
full pipeline and checks whether the result falls within expected ranges.

By default runs each case against BOTH OpenRouter and Ollama (if both available)
and reports pass/fail per backend plus a side-by-side score comparison.

Usage:
    conda run -n linkedin-agent-assistant python scripts/run_benchmarks.py
    conda run -n linkedin-agent-assistant python scripts/run_benchmarks.py --case strong_match
    conda run -n linkedin-agent-assistant python scripts/run_benchmarks.py --backend openrouter
    conda run -n linkedin-agent-assistant python scripts/run_benchmarks.py --backend ollama
    conda run -n linkedin-agent-assistant python scripts/run_benchmarks.py --model cogito:14b
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmarks", "cases.json")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m~\033[0m"


def _check(label: str, value: float | bool | None, lo: float, hi: float) -> bool:
    if value is None:
        print(f"  {WARN} {label}: N/A (skipped judge)")
        return True
    ok = lo <= value <= hi
    symbol = PASS if ok else FAIL
    print(f"  {symbol} {label}: {value:.3f}  (expected [{lo:.2f}, {hi:.2f}])")
    return ok


async def run_case(case: dict, model: str, fast_model: str, api_key: str = "", api_base_url: str = "") -> dict:
    from analysers.engine import AnalyserEngine
    from analysers.semantic import resume_jd_similarity

    engine = AnalyserEngine(confidence_threshold=75, max_retries=2)
    resume = case["resume"]
    jd = case["job_description"]

    sim = resume_jd_similarity(resume, jd)
    return await engine.run(
        resume_text=resume,
        job_description=jd,
        llm_model=model,
        fast_model=fast_model,
        api_key=api_key,
        api_base_url=api_base_url,
        semantic_similarity=sim,
    )


def evaluate_case(case: dict, result: dict, backend_label: str) -> bool:
    overall = result.get("overall", {})
    scores = overall.get("component_scores", {})
    overall_score = overall.get("overall_score", 0)
    skills_gate = overall.get("skills_gate_applied", False)

    print(f"\n  [{backend_label}]  overall={overall_score:.3f}  gate={skills_gate}")
    print(f"  statuses: { {k: v['status'] for k, v in overall.get('judge_statuses', {}).items()} }")

    checks = []
    checks.append(_check("  overall", overall_score, *case["expected_overall_range"]))

    if "expected_skills_range" in case:
        checks.append(_check("  skills", scores.get("skills"), *case["expected_skills_range"]))
    if "expected_experience_range" in case:
        checks.append(_check("  experience", scores.get("experience"), *case["expected_experience_range"]))
    if "expected_seniority_range" in case:
        checks.append(_check("  seniority", scores.get("seniority"), *case["expected_seniority_range"]))
    if "expected_skills_gate_applied" in case:
        gate_ok = skills_gate == case["expected_skills_gate_applied"]
        symbol = PASS if gate_ok else FAIL
        print(f"  {symbol}   skills_gate_applied: {skills_gate}  (expected {case['expected_skills_gate_applied']})")
        checks.append(gate_ok)

    passed = all(checks)
    print(f"  → {backend_label}: {'PASS' if passed else 'FAIL'}")
    return passed


def _ollama_available() -> bool:
    import httpx
    from analysers.config import cfg
    try:
        r = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["both", "openrouter", "ollama"], default="both",
                        help="Which backend(s) to run (default: both)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run every OPENROUTER_MODEL_N from .env (and Ollama if reachable)")
    parser.add_argument("--model", default=None, help="Override model for the chosen backend")
    parser.add_argument("--fast-model", default=None, help="Override fast model")
    parser.add_argument("--case", default=None, help="Run only this case ID")
    args = parser.parse_args()

    from routers.deps import get_api_config
    from analysers.config import cfg

    # Build backend configs to run
    backends: list[dict] = []

    want_openrouter = args.backend in ("both", "openrouter")
    want_ollama     = args.backend in ("both", "ollama")

    fast_model_override = args.fast_model or cfg.openrouter_fast_model

    if want_openrouter and cfg.openrouter_api_key:
        if args.all_models:
            # One backend entry per OPENROUTER_MODEL_N
            i = 1
            while True:
                model_name = os.getenv(f"OPENROUTER_MODEL_{i}")
                if model_name is None:
                    break
                key = os.getenv(f"OPENROUTER_KEY_{i}", cfg.openrouter_api_key)
                backends.append({
                    "api_key":      key,
                    "api_base_url": cfg.openrouter_base_url,
                    "llm_model":    model_name,
                    "fast_model":   fast_model_override,
                    "label":        f"OR-{i}: {model_name.split('/')[-1]}",
                })
                i += 1
        else:
            or_cfg = get_api_config()
            if args.model:      or_cfg = {**or_cfg, "llm_model": args.model}
            if args.fast_model: or_cfg = {**or_cfg, "fast_model": args.fast_model}
            backends.append({**or_cfg, "label": f"OpenRouter ({or_cfg['llm_model'].split('/')[-1]})"})
    elif want_openrouter:
        print("OpenRouter: OPENROUTER_API_KEY not set — skipping")

    if want_ollama:
        if _ollama_available():
            backends.append({
                "api_key":      "",
                "api_base_url": "",
                "llm_model":    args.model or cfg.default_model,
                "fast_model":   args.fast_model or cfg.fast_model,
                "label":        f"Ollama ({args.model or cfg.default_model})",
            })
        else:
            print("Ollama: not reachable — skipping")

    if not backends:
        print("No backends available. Check OPENROUTER_API_KEY or Ollama connection.")
        sys.exit(1)

    print("\nBackends to run:")
    for b in backends:
        print(f"  {b['label']:12s}  model={b['llm_model']}  fast={b['fast_model']}")

    with open(CASES_FILE) as f:
        cases = json.load(f)

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id '{args.case}'")
            sys.exit(1)

    # { case_id: { backend_label: passed } }
    all_results: dict[str, dict[str, bool]] = {}

    # Gap between backend runs to respect free-tier rate limits (req/min per key)
    inter_model_delay = 30 if len(backends) > 1 else 0

    for case in cases:
        print(f"\n{'='*60}")
        print(f"Case: {case['id']}")
        print(f"  {case['description']}")
        all_results[case["id"]] = {}

        for i, b in enumerate(backends):
            if i > 0 and inter_model_delay:
                print(f"  (waiting {inter_model_delay}s between models…)")
                await asyncio.sleep(inter_model_delay)
            t0 = time.monotonic()
            try:
                result = await run_case(
                    case, b["llm_model"], b["fast_model"],
                    api_key=b["api_key"], api_base_url=b["api_base_url"],
                )
                elapsed = round(time.monotonic() - t0, 1)
                passed = evaluate_case(case, result, b["label"])
                print(f"  elapsed: {elapsed}s")
            except Exception as e:
                print(f"  {FAIL} [{b['label']}] ERROR: {e}")
                passed = False
            all_results[case["id"]][b["label"]] = passed

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    col_w = 14
    header = f"  {'Case':<30}" + "".join(f"{b['label']:<{col_w}}" for b in backends)
    print(header)
    print("  " + "-" * (30 + col_w * len(backends)))

    all_passed = True
    for cid, backend_results in all_results.items():
        row = f"  {cid:<30}"
        for b in backends:
            p = backend_results.get(b["label"])
            row += f"{'PASS' if p else 'FAIL':<{col_w}}"
            if not p:
                all_passed = False
        print(row)

    totals_row = f"  {'TOTAL':<30}"
    for b in backends:
        n_pass = sum(1 for r in all_results.values() if r.get(b["label"]))
        totals_row += f"{n_pass}/{len(all_results):<{col_w - 2}}    "
    print(totals_row)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
