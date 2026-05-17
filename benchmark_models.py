"""
Benchmark all 8 Ollama models against resume/job pairs.
Runs 3 models concurrently, tracks per-judge success, scores, timing.
"""

import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from specialized_judges import SpecializedJudgeEngine

# ── Data ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

RESUMES = load_csv(os.path.join(BASE_DIR, "data", "resumes.csv"))
JOBS    = load_csv(os.path.join(BASE_DIR, "data", "jobs.csv"))

# GAURICA (ML) vs ML-adjacent jobs; RISHABH (Java) vs Java + one wrong-fit job
PAIRS = [
    ("GAURICA PURI",   "AI Operations Engineer - Part Time at VendueTech"),
    ("GAURICA PURI",   "AI Engineer - Pentesting Agent"),
    ("GAURICA PURI",   "Data Scientist - Remote"),
    ("GAURICA PURI",   "Data Scientist"),          # White Light
    ("RISHABH RAGHAV", "Java Software Engineer"),
    ("RISHABH RAGHAV", "Data Scientist - Remote"),  # wrong fit — tests discrimination
]

MODELS = [
    "deepseek-r1:14b",
]

# Embedding models are excluded — they don't support chat and score 0 on all judges
EMBEDDING_MODELS = {"nomic-embed-text", "nomic-embed", "mxbai-embed-large", "all-minilm", "bge-m3"}

CONCURRENCY = 3  # models tested in parallel


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_resume(name):
    for r in RESUMES:
        if r["name"].strip().upper() == name.strip().upper():
            return r
    raise ValueError(f"Resume not found: {name}")

def find_job(title):
    for j in JOBS:
        if j["title"].strip() == title.strip():
            return j
    raise ValueError(f"Job not found: {title}")

def judge_ok(result: dict) -> bool:
    return bool(result and result.get("success") and result.get("data"))

def extract_scores(pipeline: dict) -> dict:
    return {
        "skills":     pipeline["skills"]["scores"].get("score", 0),
        "experience": pipeline["experience"]["scores"].get("score", 0),
        "seniority":  pipeline["seniority"]["scores"].get("score", 0),
        "relevancy":  pipeline["relevancy"]["scores"].get("score", 0),
        "impact":     pipeline["impact"]["scores"].get("score", 0),
        "overall":    pipeline["overall"].get("overall_score", 0),
    }


# ── Core benchmark task ───────────────────────────────────────────────────────
async def benchmark_one(model: str, resume_name: str, job_title: str) -> dict:
    base = model.split(":")[0].lower()
    if base in EMBEDDING_MODELS:
        raise ValueError(f"Skipping embedding model: {model}")
    resume = find_resume(resume_name)
    job    = find_job(job_title)
    engine = SpecializedJudgeEngine()

    t0 = time.time()
    try:
        result = await engine.run_all_judges(
            resume_text=resume["text"],
            job_description=job["description"],
            llm_model=model,
        )
        elapsed = time.time() - t0
        scores  = extract_scores(result)

        judge_results = {
            "skills":     judge_ok(result["skills"]["judge_result"]),
            "experience": judge_ok(result["experience"]["judge_result"]),
            "seniority":  judge_ok(result["seniority"]["judge_result"]),
            "relevancy":  judge_ok(result["relevancy"]["judge_result"]),
            "impact":     judge_ok(result["impact"]["judge_result"]),
            "coach":      judge_ok(result["coach"]),
        }
        judges_passed = sum(judge_results.values())

        return {
            "model":         model,
            "resume":        resume_name,
            "job":           job_title,
            "success":       True,
            "elapsed_s":     round(elapsed, 1),
            "judges_passed": judges_passed,
            "judges":        judge_results,
            "scores":        {k: round(v, 3) for k, v in scores.items()},
            "error":         None,
        }

    except Exception as e:
        return {
            "model":         model,
            "resume":        resume_name,
            "job":           job_title,
            "success":       False,
            "elapsed_s":     round(time.time() - t0, 1),
            "judges_passed": 0,
            "judges":        {},
            "scores":        {},
            "error":         str(e)[:200],
        }


# ── Run all pairs for one model ───────────────────────────────────────────────
async def benchmark_model(model: str) -> list:
    print(f"\n{'='*60}")
    print(f"  MODEL: {model}")
    print(f"{'='*60}")
    tasks = [benchmark_one(model, r, j) for r, j in PAIRS]
    results = await asyncio.gather(*tasks)
    for r in results:
        status = "✅" if r["success"] else "❌"
        score  = r["scores"].get("overall", 0) if r["success"] else 0
        print(f"  {status} {r['resume'][:15]} × {r['job'][:35]} | "
              f"overall={score:.0%} | {r['elapsed_s']}s | judges={r['judges_passed']}/6")
    return list(results)


# ── Aggregate per-model stats ─────────────────────────────────────────────────
def aggregate(model_results: list) -> dict:
    if not model_results:
        return {}
    n = len(model_results)
    successes = [r for r in model_results if r["success"]]

    def avg(vals): return sum(vals) / len(vals) if vals else 0

    return {
        "runs":          n,
        "run_success":   len(successes),
        "success_rate":  len(successes) / n,
        "avg_elapsed_s": avg([r["elapsed_s"] for r in model_results]),
        "avg_judges":    avg([r["judges_passed"] for r in successes]) if successes else 0,
        "avg_overall":   avg([r["scores"].get("overall", 0) for r in successes]) if successes else 0,
        "avg_skills":    avg([r["scores"].get("skills", 0) for r in successes]) if successes else 0,
        "avg_experience":avg([r["scores"].get("experience", 0) for r in successes]) if successes else 0,
        "avg_seniority": avg([r["scores"].get("seniority", 0) for r in successes]) if successes else 0,
        "avg_relevancy": avg([r["scores"].get("relevancy", 0) for r in successes]) if successes else 0,
        "avg_impact":    avg([r["scores"].get("impact", 0) for r in successes]) if successes else 0,
    }


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(all_results: dict):
    print("\n\n" + "=" * 80)
    print("  BENCHMARK REPORT")
    print("=" * 80)

    rows = []
    for model, results in all_results.items():
        a = aggregate(results)
        rows.append((model, a))

    # Sort by: success_rate desc, avg_overall desc, avg_elapsed_s asc
    rows.sort(key=lambda x: (-x[1].get("success_rate", 0),
                              -x[1].get("avg_overall", 0),
                               x[1].get("avg_elapsed_s", 999)))

    header = f"{'MODEL':<30} {'OK':>3} {'RATE':>6} {'TIME':>6} {'JUDGES':>7} {'OVERALL':>8} {'SKILLS':>7} {'EXP':>5} {'SEN':>5} {'REL':>5} {'IMP':>5}"
    print(header)
    print("-" * len(header))

    for model, a in rows:
        if not a:
            print(f"{model:<30}  no data")
            continue
        print(
            f"{model:<30} "
            f"{a['run_success']:>2}/{a['runs']:<1} "
            f"{a['success_rate']:>5.0%} "
            f"{a['avg_elapsed_s']:>5.0f}s "
            f"{a['avg_judges']:>6.1f}/6 "
            f"{a['avg_overall']:>7.0%} "
            f"{a['avg_skills']:>6.0%} "
            f"{a['avg_experience']:>4.0%} "
            f"{a['avg_seniority']:>4.0%} "
            f"{a['avg_relevancy']:>4.0%} "
            f"{a['avg_impact']:>4.0%}"
        )

    print("\n── RECOMMENDATION ───────────────────────────────────────────────")
    passed = [(m, a) for m, a in rows if a.get("success_rate", 0) >= 0.8 and a.get("avg_judges", 0) >= 5]
    if passed:
        print(f"  KEEP:  {', '.join(m for m, _ in passed)}")
    marginal = [(m, a) for m, a in rows if 0.5 <= a.get("success_rate", 0) < 0.8]
    if marginal:
        print(f"  MAYBE: {', '.join(m for m, _ in marginal)}")
    skip = [(m, a) for m, a in rows if a.get("success_rate", 0) < 0.5]
    if skip:
        print(f"  SKIP:  {', '.join(m for m, _ in skip)}")
    print("  NOTE:  nomic-embed-text is an embedding model — not for chat tasks")


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    print(f"Benchmark started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODELS)} | Pairs: {len(PAIRS)} | Concurrency: {CONCURRENCY} models at a time")
    print(f"Total runs: {len(MODELS) * len(PAIRS)}")

    all_results = {}

    # Batch models CONCURRENCY at a time
    for i in range(0, len(MODELS), CONCURRENCY):
        batch = MODELS[i:i + CONCURRENCY]
        print(f"\n\n>>> Batch {i//CONCURRENCY + 1}: {batch}")
        batch_results = await asyncio.gather(*[benchmark_model(m) for m in batch])
        for model, results in zip(batch, batch_results):
            all_results[model] = results

    # Save raw results
    out_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved → {out_path}")

    print_report(all_results)


if __name__ == "__main__":
    asyncio.run(main())
