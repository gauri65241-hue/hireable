"""
Batch scorer — calls analyser engine directly, no HTTP, no auth, no semaphore.

  Rishabh's resume (269d5b10...) → jobs 1–12  (Java roles)
  Gaurica's resume  (64777726...) → jobs 13–31 (ML/AI roles)

Smoke test (always runs first): job 1 alone, prints raw output + parsed scores.
Resumable: skips jobs that already have a run_id in runs.job_id.
Parse failures: raw output written to /tmp/batch_parse_failures.log.

Usage:
  python scripts/batch_score.py          # smoke test then full batch
  python scripts/batch_score.py --smoke  # job 1 only

Pre-registration block [DO NOT MODIFY after labeling begins]:
  Model of record:  mistral-nemo:12b-instruct-2407-q4_K_M (local Ollama)
                    chosen for local batch feasibility; r1:14b failed 3/3 attempts
                    (semaphore deadlock × 1, 600s judge timeout × 2, 31-min hang × 1)
  Full taxonomy:    restored (r1-specific prompt-length constraint does not apply to nemo)
  Score bins:       ≥0.65 fit, 0.45–0.65 maybe, <0.45 no  [pre-registered 2026-07-28]
  Taxonomy note:    skills judge run with full taxonomy [2026-07-31]
  Both composites:  gated overall AND raw weighted average both reported [2026-07-31]
                    gate cliff (skills 0.399 → 0.16 vs 0.401 → 0.46) documented, not fixed
  Scoring fix:      skills scorer synthesizes missing skill_analysis entries from
                    present_skills when LLM output truncates before completing per-skill
                    breakdown [2026-07-31]; jobs 21 and 28 rescored under fixed scorer;
                    jobs 9, 10, 20 verified genuine zeros (present_skills=[])
  Known limitation: skill name normalization mismatch in job 20 — required_skills held
                    "MLOps pipelines" while skill_analysis emitted "mlops_pipelines";
                    5/6 required skills failed string match [2026-08-04]. Because
                    present_skills was also empty, synthesis scored all as absent and
                    the overall zero is correct. However, the same mismatch on a job
                    with non-empty present_skills would silently zero out real matches.
                    Post-study fix: normalize both sides (lowercase + underscore) before
                    comparison in synthesis and score(). If job 20 shows large
                    model-vs-human disagreement, this is the likely explanation.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Ensure project root on path so analysers/ and db.py are importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.chdir(ROOT)  # load_dotenv in deps picks up .env relative to cwd

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Belt-and-suspenders: clear the master key so config.openrouter_api_key stays ""
# and get_api_config falls through to local Ollama.
os.environ.pop("OPENROUTER_API_KEY", None)

from analysers import AnalyserEngine
from analysers.semantic import resume_jd_similarity
from ats_checker import score_ats, ats_to_dict
import db as run_db

TAXONOMY_PATH     = ROOT / "data" / "skills_taxonomy.json"
FAILURE_LOG       = Path("/tmp/batch_parse_failures.log")
MODEL             = "mistral-nemo:12b-instruct-2407-q4_K_M"

with open(TAXONOMY_PATH) as _f:
    _taxonomy = json.load(_f)

RISHABH_RESUME_ID = "269d5b10-ca17-4851-b9ad-84dfad87179a"
GAURICA_RESUME_ID = "64777726-05a6-4d6a-ac66-a98165500441"
JAVA_JOB_RANGE    = (1, 12)
ML_JOB_RANGE      = (13, 31)


def load_resume(resume_id: str) -> str:
    pool = run_db._get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute("SELECT text FROM resumes WHERE resume_id=?", (resume_id,)).fetchone()
    finally:
        pool.release(conn)
    if not row:
        raise RuntimeError(f"Resume not found: {resume_id}")
    return row[0]


def load_jobs(lo: int, hi: int) -> list[tuple]:
    pool = run_db._get_pool()
    conn = pool.acquire()
    try:
        rows = conn.execute(
            "SELECT job_id, title, description FROM jobs WHERE job_id BETWEEN ? AND ?",
            (lo, hi),
        ).fetchall()
    finally:
        pool.release(conn)
    return rows


def already_scored(job_id: int) -> bool:
    pool = run_db._get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE job_id=?", (job_id,)
        ).fetchone()
    finally:
        pool.release(conn)
    return row[0] > 0


def stamp_job_id(run_id: str, job_id: int) -> None:
    run_db._get_pool().execute_tx(
        lambda c: c.execute("UPDATE runs SET job_id=? WHERE id=?", (job_id, run_id))
    )


def get_user_id() -> str:
    pool = run_db._get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute("SELECT user_id FROM users WHERE email='sniffer@gmail.com'").fetchone()
    finally:
        pool.release(conn)
    return row[0] if row else "batch_runner"


def log_parse_failure(job_id: int, title: str, judge: str, raw: str) -> None:
    with open(FAILURE_LOG, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"job_id={job_id} '{title}' judge={judge} ts={datetime.utcnow().isoformat()}\n")
        f.write(raw[:3000])
        f.write("\n")


def print_smoke_report(job_id: int, title: str, result: dict, ats_score: float, sim: float, elapsed: float) -> None:
    overall  = result.get("overall", {})
    cs       = overall.get("component_scores", {})
    statuses = overall.get("judge_statuses", {})
    weights  = overall.get("weights_used", {})

    # raw weighted average (un-gated) for sanity check
    raw_avg = sum(cs.get(j, 0) * weights.get(j, 0) for j in cs) if weights else None

    print("\n" + "="*60)
    print(f"SMOKE TEST — job_id={job_id} '{title}'")
    print("="*60)
    print(f"ats_score          : {ats_score}")
    print(f"semantic_similarity: {sim:.3f}")
    print(f"overall_score      : {overall.get('overall_score', 'N/A'):.3f}  (gated={overall.get('skills_gate_applied')})")
    if raw_avg is not None:
        print(f"raw_weighted_avg   : {raw_avg:.3f}  (un-gated weighted average)")
    print(f"elapsed_total      : {elapsed:.0f}s")
    print()
    print(f"{'judge':<12} {'score':>6}  {'status':<10}  {'elapsed_ms':>10}")
    print("-" * 45)
    for judge in ("skills", "experience", "seniority", "relevancy", "impact"):
        score   = cs.get(judge, "N/A")
        st      = statuses.get(judge, {})
        status  = st.get("status", "?")
        ms      = st.get("elapsed_ms", "?")
        score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
        print(f"  {judge:<10} {score_s:>6}  {status:<10}  {ms:>10}")

    print("\n--- RAW OUTPUT PER JUDGE (first 400 chars) ---")
    for judge in ("skills", "experience", "seniority", "relevancy", "impact"):
        raw = result.get(judge, {}).get("analyser_result", {}).get("raw_response", "")
        if raw:
            print(f"\n[{judge.upper()}]:\n{raw[:400]}")
        else:
            print(f"\n[{judge.upper()}]: (no raw output)")
    print("\n" + "="*60 + "\n")


async def score_job(
    engine: AnalyserEngine,
    resume_text: str,
    jd: str,
    title: str,
    candidate_name: str,
    job_id: int,
    user_id: str,
) -> float | None:
    t0 = time.time()

    ats_raw    = await asyncio.to_thread(score_ats, resume_text, jd)
    ats_result = ats_to_dict(ats_raw) if ats_raw else {}
    ats_score  = ats_result.get("score", 100)

    sim = await asyncio.to_thread(resume_jd_similarity, resume_text, jd)

    result = await engine.run(
        resume_text=resume_text,
        job_description=jd,
        llm_model=MODEL,
        skills_taxonomy=_taxonomy,
        api_key="",
        api_base_url="",
        ats_score=ats_score,
        semantic_similarity=sim,
    )

    # Augment result with pipeline inputs so they're queryable in DB
    result["ats_preflight"]      = ats_result
    result["semantic_similarity"] = sim

    overall       = result.get("overall", {})
    overall_score = overall.get("overall_score")
    elapsed       = time.time() - t0

    # Log raw output for any judge that returned empty/failed
    for judge in ("skills", "experience", "seniority", "relevancy", "impact"):
        raw = result.get(judge, {}).get("analyser_result", {}).get("raw_response", "")
        status = overall.get("judge_statuses", {}).get(judge, {}).get("status", "unknown")
        if status != "success" or not raw:
            log_parse_failure(job_id, title, judge, raw or "(empty)")

    run_id = f"analysis_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S-%f')}"
    run_db.save_run(
        run_id=run_id,
        model=MODEL,
        resume_text=resume_text,
        job_desc=jd,
        pipeline_result=result,
        ats_result=ats_result,
        candidate_name=candidate_name,
        job_title=title,
        user_id=user_id,
    )
    stamp_job_id(run_id, job_id)

    score_str = f"{overall_score:.3f}" if overall_score is not None else "None"
    print(f"  ✓ overall={score_str}  elapsed={elapsed:.0f}s  run_id={run_id[:24]}...")

    return overall_score


async def main(smoke_only: bool = False) -> None:
    user_id  = get_user_id()
    engine   = AnalyserEngine()

    rishabh_resume = load_resume(RISHABH_RESUME_ID)
    gaurica_resume = load_resume(GAURICA_RESUME_ID)

    java_jobs = load_jobs(*JAVA_JOB_RANGE)
    ml_jobs   = load_jobs(*ML_JOB_RANGE)

    batches = [
        (java_jobs, rishabh_resume, "RISHABH RAGHAV"),
        (ml_jobs,   gaurica_resume, "GAURICA PURI"),
    ]

    total = sum(len(j) for j, _, _ in batches)
    print(f"Model: {MODEL}  |  Total jobs: {total} ({len(java_jobs)} Java / {len(ml_jobs)} ML)")
    print(f"Failure log: {FAILURE_LOG}")

    # --- SMOKE TEST: job 1 ---
    smoke_job_id, smoke_title, smoke_jd = java_jobs[0]
    print(f"\n[SMOKE] job_id={smoke_job_id} '{smoke_title}' (RISHABH RAGHAV) ...")

    if already_scored(smoke_job_id):
        print(f"  already scored — skipping smoke, check DB for existing result")
    else:
        ats_raw    = await asyncio.to_thread(score_ats, rishabh_resume, smoke_jd)
        ats_result = ats_to_dict(ats_raw) if ats_raw else {}
        ats_score  = ats_result.get("score", 100)
        sim        = await asyncio.to_thread(resume_jd_similarity, rishabh_resume, smoke_jd)

        t_smoke = time.time()
        smoke_result = await engine.run(
            resume_text=rishabh_resume,
            job_description=smoke_jd,
            llm_model=MODEL,
            skills_taxonomy=_taxonomy,
            api_key="",
            api_base_url="",
            ats_score=ats_score,
            semantic_similarity=sim,
        )
        smoke_elapsed = time.time() - t_smoke

        # Augment result with pipeline inputs so they're queryable in DB
        smoke_result["ats_preflight"] = ats_result
        smoke_result["semantic_similarity"] = sim

        print_smoke_report(smoke_job_id, smoke_title, smoke_result, ats_score, sim, smoke_elapsed)

        # Save smoke run
        run_id = f"analysis_{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S-%f')}"
        run_db.save_run(
            run_id=run_id,
            model=MODEL,
            resume_text=rishabh_resume,
            job_desc=smoke_jd,
            pipeline_result=smoke_result,
            ats_result=ats_result,
            candidate_name="RISHABH RAGHAV",
            job_title=smoke_title,
            user_id=user_id,
        )
        stamp_job_id(run_id, smoke_job_id)

    if smoke_only:
        return

    # --- FULL BATCH ---
    done = 0
    skipped = 0
    for jobs, resume, candidate_name in batches:
        for job_id, title, description in jobs:
            if job_id == smoke_job_id:
                continue  # already handled above

            done += 1
            print(f"[{done}/{total-1}] job_id={job_id} '{title}' ({candidate_name}) ...", flush=True)

            if already_scored(job_id):
                skipped += 1
                print(f"  → skipped (already scored)")
                continue

            try:
                await score_job(engine, resume, description, title, candidate_name, job_id, user_id)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                log_parse_failure(job_id, title, "ENGINE", str(e))

    print(f"\nDone. {done - skipped} scored, {skipped} skipped (already had run).")
    print("Next: python scripts/kappa.py export --jobs 1-12 --out data/kappa_java.csv")
    print("      python scripts/kappa.py export --jobs 13-31 --out data/kappa_ml.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run job 1 only")
    args = parser.parse_args()
    asyncio.run(main(smoke_only=args.smoke))
