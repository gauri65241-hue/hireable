"""
LLM-as-judge evaluation metrics — custom alternatives to DeepEval GEval.
Uses the existing call_llm_async infrastructure; no external eval library needed.

Four targeted meta-judges, each catching a specific failure mode:
  1. Skills false-negatives  — "missing" skill that's actually in the resume
  2. Reasoning-score misalignment — score contradicts the judge's own reasoning
  3. Experience false-UNMETs   — responsibility marked UNMET that candidate clearly meets
  4. Overall coherence          — all 5 scores collectively implausible for this pair

Usage:
    python scripts/eval_llm_judge.py [--model mistral-nemo:latest] [--limit 20] [--resume]

Results saved to data/eval_llm_judge_results.jsonl.
Re-run with --resume to skip completed IDs.
"""
import argparse
import asyncio
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from analysers.base import call_llm_async, parse_and_validate

RESULTS_FILE   = os.path.join(BASE_DIR, "data", "eval_results.jsonl")
EVAL_FILE      = os.path.join(BASE_DIR, "data", "eval_set.json")
LLM_JUDGE_FILE = os.path.join(BASE_DIR, "data", "eval_llm_judge_results.jsonl")

VERDICT_SCORES = {"CORRECT": 1.0, "QUESTIONABLE": 0.5, "INCORRECT": 0.0}

# ── Shared prompt infrastructure ─────────────────────────────────────────────

_META_SYSTEM = """You are a strict quality auditor reviewing an AI resume scoring system.
You will be shown the system's input and output for a single judge.
Your job: determine whether the judge's output is correct, questionable, or incorrect.

RULES:
- Output ONLY valid JSON. No markdown, no explanation outside the JSON.
- verdict MUST be exactly one of: "CORRECT", "QUESTIONABLE", "INCORRECT"
- confidence MUST be an integer 0-100 representing your certainty in the verdict.
- reasoning MUST be one sentence explaining the verdict.

Output format:
{"verdict": "CORRECT|QUESTIONABLE|INCORRECT", "confidence": 0, "reasoning": "one sentence"}"""


async def _call_meta_judge(user_prompt: str, model: str) -> dict:
    result = await call_llm_async(_META_SYSTEM, user_prompt, model, temperature=0.0)
    if not result["success"]:
        return {"verdict": "ERROR", "confidence": 0, "reasoning": result.get("error", "LLM call failed")}
    try:
        parsed = parse_and_validate(result["data"])
        verdict = parsed.get("verdict", "ERROR")
        if verdict not in ("CORRECT", "QUESTIONABLE", "INCORRECT"):
            verdict = "ERROR"
        return {
            "verdict": verdict,
            "confidence": int(parsed.get("confidence", 0)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return {"verdict": "ERROR", "confidence": 0, "reasoning": str(e)[:200]}


# ── Meta-judge 1: Skills false negatives ─────────────────────────────────────

async def meta_skills_false_negatives(
    resume_text: str,
    job_description: str,
    skill_analysis: list[dict],
    model: str,
) -> dict:
    """
    For each skill the judge marked ABSENT (is_present=0), checks whether
    that skill or a clear equivalent is actually present in the resume.
    Returns per-skill verdicts and an aggregate false-negative rate.
    """
    absent = [s for s in skill_analysis if s.get("is_present") == 0]
    if not absent:
        return {"skipped": True, "reason": "no absent skills to check", "false_negative_rate": 0.0}

    # Limit to 5 to control cost
    sample = absent[:5]
    skill_lines = "\n".join(
        f'- "{s.get("skill", s.get("skill_requirement", "?"))}"'
        for s in sample
    )

    prompt = f"""The skills judge marked these skills as ABSENT from the candidate's resume.
Review the resume and determine whether each skill is genuinely absent or whether
the judge made a false-negative error (skill present but not recognised).

ABSENT SKILLS:
{skill_lines}

RESUME (first 2000 chars):
{resume_text[:2000]}

JOB DESCRIPTION (first 800 chars):
{job_description[:800]}

For the overall verdict:
- CORRECT: all or nearly all marked-absent skills are genuinely absent
- QUESTIONABLE: 1-2 skills may have been missed
- INCORRECT: 3+ skills are actually present in the resume

Output JSON: {{"verdict": "...", "confidence": 0, "reasoning": "..."}}"""

    result = await _call_meta_judge(prompt, model)
    result["absent_skills_checked"] = len(sample)
    result["absent_skills_total"] = len(absent)
    return result


# ── Meta-judge 2: Reasoning-score misalignment ───────────────────────────────

async def meta_reasoning_score_alignment(
    judge_name: str,
    score: float,
    reasoning: str,
    model: str,
) -> dict:
    """
    Checks whether the judge's reasoning text is consistent with its numeric score.
    E.g., reasoning says "candidate has strong Python experience" but score=5%.
    """
    if not reasoning or reasoning.strip() in ("N/A", ""):
        return {"skipped": True, "reason": "no reasoning text"}

    prompt = f"""A resume scoring judge produced the following output.
Check whether the reasoning is consistent with the score.

JUDGE: {judge_name}
SCORE: {round(score, 1)}% (0% = completely absent, 100% = fully meets requirements)
REASONING: {reasoning[:600]}

Verdict:
- CORRECT: reasoning clearly justifies the score (both point the same direction)
- QUESTIONABLE: minor tension between reasoning tone and score magnitude
- INCORRECT: reasoning contradicts the score (e.g. positive reasoning + very low score, or vice versa)

Output JSON: {{"verdict": "...", "confidence": 0, "reasoning": "..."}}"""

    return await _call_meta_judge(prompt, model)


# ── Meta-judge 3: Experience false-UNMETs ────────────────────────────────────

async def meta_experience_false_unmets(
    resume_text: str,
    responsibility_analysis: list[dict],
    model: str,
) -> dict:
    """
    For responsibilities marked UNMET (experience_score=0.0), checks whether
    the candidate's resume actually demonstrates that responsibility.
    """
    unmet = [r for r in responsibility_analysis if r.get("experience_score", 1) == 0.0]
    if not unmet:
        return {"skipped": True, "reason": "no UNMET responsibilities", "false_unmet_rate": 0.0}

    sample = unmet[:3]
    resp_lines = "\n".join(
        f'- "{r.get("responsibility", "?")}"  (judge said: {r.get("reasoning", "no reasoning")[:100]})'
        for r in sample
    )

    prompt = f"""An experience judge marked these job responsibilities as UNMET by the candidate.
Review the resume and determine whether the judge is correct.

UNMET RESPONSIBILITIES:
{resp_lines}

RESUME (first 2000 chars):
{resume_text[:2000]}

Verdict:
- CORRECT: candidate genuinely lacks evidence for these responsibilities
- QUESTIONABLE: candidate shows partial evidence the judge may have overlooked
- INCORRECT: candidate clearly demonstrates one or more of these responsibilities

Output JSON: {{"verdict": "...", "confidence": 0, "reasoning": "..."}}"""

    result = await _call_meta_judge(prompt, model)
    result["unmet_checked"] = len(sample)
    result["unmet_total"] = len(unmet)
    return result


# ── Meta-judge 4: Overall coherence ──────────────────────────────────────────

async def meta_overall_coherence(
    resume_text: str,
    job_description: str,
    judge_scores: dict,
    overall_pct: float,
    human_tier: str,
    model: str,
) -> dict:
    """
    Reviews all 5 scores together and asks: does this holistic picture make sense
    for this resume-JD pair? Catches cases where individual judges are each
    plausible but the combination produces an implausible result.
    """
    score_lines = "\n".join(f"  {k}: {v}%" for k, v in judge_scores.items())

    prompt = f"""You are reviewing an AI system that scores how well a resume matches a job description.
The system produced these 5 dimension scores and an overall score.

SCORES:
{score_lines}
  Overall: {overall_pct}%

GROUND TRUTH TIER: {human_tier} (Strong=strong match, Moderate=partial match, Weak=poor match)

RESUME (first 1500 chars):
{resume_text[:1500]}

JOB DESCRIPTION (first 600 chars):
{job_description[:600]}

Does the overall scoring picture make sense for this resume-JD pair?
- CORRECT: scores are plausible and consistent with the ground truth tier
- QUESTIONABLE: scores are directionally right but magnitude seems off
- INCORRECT: scores are clearly wrong or inconsistent with the ground truth tier

Output JSON: {{"verdict": "...", "confidence": 0, "reasoning": "..."}}"""

    return await _call_meta_judge(prompt, model)


# ── Run all meta-judges for one eval item ────────────────────────────────────

async def run_meta_judges(result: dict, eval_item: dict, model: str) -> dict:
    pipeline = result.get("pipeline_data", {})

    skills_data = (pipeline.get("skills", {})
                   .get("analyser_result", {})
                   .get("data", {}))
    exp_data    = (pipeline.get("experience", {})
                   .get("analyser_result", {})
                   .get("data", {}))

    resume = eval_item["resume_text"]
    jd     = eval_item["job_description"]

    tasks = {
        "skills_false_negatives": meta_skills_false_negatives(
            resume, jd,
            skills_data.get("skill_analysis", []),
            model,
        ),
        "skills_reasoning": meta_reasoning_score_alignment(
            "skills",
            result["judge_scores"].get("skills", 0),
            skills_data.get("overall_reasoning", ""),
            model,
        ),
        "experience_false_unmets": meta_experience_false_unmets(
            resume,
            exp_data.get("responsibility_analysis", []),
            model,
        ),
        "experience_reasoning": meta_reasoning_score_alignment(
            "experience",
            result["judge_scores"].get("experience", 0),
            exp_data.get("overall_reasoning", ""),
            model,
        ),
        "overall_coherence": meta_overall_coherence(
            resume, jd,
            result["judge_scores"],
            result["overall_pct"],
            result["human_tier"],
            model,
        ),
    }

    outputs = await asyncio.gather(*tasks.values(), return_exceptions=True)
    meta_results = {}
    for key, output in zip(tasks.keys(), outputs):
        if isinstance(output, Exception):
            meta_results[key] = {"verdict": "ERROR", "reasoning": str(output)[:100]}
        else:
            meta_results[key] = output

    return {
        "id":         result["id"],
        "human_tier": result["human_tier"],
        "overall_pct": result["overall_pct"],
        "meta":       meta_results,
    }


# ── Aggregate report ──────────────────────────────────────────────────────────

def print_report(llm_results: list[dict]):
    print(f"\n{'='*65}")
    print("  LLM-AS-JUDGE META-EVALUATION REPORT")
    print(f"{'='*65}")
    print(f"  Items evaluated: {len(llm_results)}\n")

    meta_keys = [
        "skills_false_negatives",
        "skills_reasoning",
        "experience_false_unmets",
        "experience_reasoning",
        "overall_coherence",
    ]

    for key in meta_keys:
        verdicts = []
        for r in llm_results:
            v = r.get("meta", {}).get(key, {})
            if v.get("skipped") or v.get("verdict") == "ERROR":
                continue
            verdicts.append(v["verdict"])

        if not verdicts:
            print(f"  {key}: no data")
            continue

        correct      = verdicts.count("CORRECT")
        questionable = verdicts.count("QUESTIONABLE")
        incorrect    = verdicts.count("INCORRECT")
        total        = len(verdicts)
        correct_pct  = round(correct / total * 100, 1)

        status = "✓" if correct_pct >= 80 else ("⚠" if correct_pct >= 60 else "✗")
        print(f"  {status} {key}")
        print(f"    CORRECT={correct} QUESTIONABLE={questionable} INCORRECT={incorrect}  "
              f"({correct_pct}% correct)")

        # Show worst failures
        failures = [
            r for r in llm_results
            if r.get("meta", {}).get(key, {}).get("verdict") == "INCORRECT"
        ]
        for f in failures[:3]:
            reasoning = f["meta"][key].get("reasoning", "")[:80]
            print(f"    → {f['id']} ({f['human_tier']}, {f['overall_pct']}%): {reasoning}")
        print()

    print(f"{'='*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_completed_ids() -> set[str]:
    done = set()
    if not os.path.exists(LLM_JUDGE_FILE):
        return done
    with open(LLM_JUDGE_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="mistral-nemo:latest")
    ap.add_argument("--limit",  type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_items = {item["id"]: item for item in json.load(f)}

    if args.report_only:
        llm_results = []
        if os.path.exists(LLM_JUDGE_FILE):
            with open(LLM_JUDGE_FILE, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        llm_results.append(json.loads(line))
        print_report(llm_results)
        return

    # Load base eval results (need pipeline_data)
    base_results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("success") and r.get("pipeline_data"):
                        base_results.append(r)

    if not base_results:
        print("ERROR: No results with pipeline_data found in eval_results.jsonl.")
        print("Re-run run_eval.py — it now saves pipeline_data in each result.")
        sys.exit(1)

    completed = load_completed_ids() if args.resume else set()
    todo = [r for r in base_results if r["id"] not in completed]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Model   : {args.model}")
    print(f"Running : {len(todo)} items ({len(completed)} already done)")

    all_llm_results = []
    for i, result in enumerate(todo, 1):
        eval_item = eval_items.get(result["id"])
        if not eval_item:
            print(f"[{i:3d}] {result['id']} — not in eval_set.json, skipping")
            continue

        print(f"[{i:3d}/{len(todo)}] {result['id']} ({result['human_tier']}) ...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            llm_result = await run_meta_judges(result, eval_item, args.model)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        with open(LLM_JUDGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(llm_result) + "\n")
        all_llm_results.append(llm_result)

        # Quick inline summary
        coherence = llm_result["meta"].get("overall_coherence", {}).get("verdict", "?")
        sk_fn     = llm_result["meta"].get("skills_false_negatives", {}).get("verdict", "?")
        print(f"coherence={coherence} skills_fn={sk_fn} ({round(time.monotonic()-t0, 1)}s)")

    # Merge with previous if resuming
    if args.resume and completed:
        prev = []
        with open(LLM_JUDGE_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["id"] not in {x["id"] for x in all_llm_results}:
                        prev.append(r)
        all_llm_results = prev + all_llm_results

    print_report(all_llm_results)
    print(f"Results saved to: {LLM_JUDGE_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
