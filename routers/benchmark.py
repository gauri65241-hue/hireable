"""
Model benchmark — fires all OPENROUTER_MODEL_N entries in parallel,
measures latency + JSON validity for judge scoring and ATS deep audit tasks,
picks a winner per task type, saves to data/model_benchmark.json.
"""
import asyncio
import json
import logging
import math
import os
import time

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_FILE = os.path.join(BASE_DIR, "data", "model_benchmark.json")

# ── Fixed test fixtures ───────────────────────────────────────────────────────

_JUDGE_SYSTEM = "You are a resume scoring judge. Return ONLY valid JSON, no markdown fences, no prose outside the JSON object."
_JUDGE_USER = """\
Score the resume below for skills match against the job description.

RESUME:
Senior Python Engineer | 5 years experience
Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes
Led microservices migration cutting p99 latency by 40%.

JOB:
Senior Backend Engineer. Required: Python, FastAPI, PostgreSQL, Docker. Nice to have: Kubernetes, Redis.

Return exactly this JSON (nothing else):
{"score": <integer 0-100>, "confidence": <integer 0-100>, "reasoning": "<one sentence>", "matched_skills": ["skill1"], "missing_skills": []}"""

_AUDIT_SYSTEM = "You are a cynical ATS auditor. Return ONLY valid JSON, no markdown fences, no prose outside the JSON object."
_AUDIT_USER = """\
Audit the resume below. Return exactly this JSON (nothing else):
{"disqualifiers": [{"rank": 1, "title": "<flaw>", "explanation": "<detail>"}, {"rank": 2, "title": "<flaw>", "explanation": "<detail>"}], "scorecard": {"ats_parsability": {"score": 7.0, "verdict": "PASS", "note": "<finding>"}, "metric_caliber": {"score": 4.5, "verdict": "REVIEW", "note": "<finding>"}, "overall_verdict": "REVIEW"}}

RESUME:
John Smith | john@email.com | +1-555-0100
Senior Engineer @ Acme Corp 2020-2024.
Responsible for managing team. Worked on various projects. Helped with deployments.
Skills: Python, Java, SQL, Communication, Team player"""

_TASKS = [
    ("judge", _JUDGE_SYSTEM, _JUDGE_USER),
    ("audit", _AUDIT_SYSTEM, _AUDIT_USER),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pool() -> list[dict]:
    pool = []
    i = 1
    while True:
        model = os.getenv(f"OPENROUTER_MODEL_{i}")
        if not model:
            break
        key = os.getenv(f"OPENROUTER_KEY_{i}") or os.getenv("OPENROUTER_API_KEY", "")
        pool.append({"model": model, "key": key})
        i += 1
    return pool


def _parse_json(raw: str) -> bool:
    text = raw.strip()
    # Strip markdown fences if model adds them despite instructions
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip().lstrip("json").strip()
            try:
                json.loads(p)
                return True
            except Exception:
                pass
    try:
        json.loads(text)
        return True
    except Exception:
        return False


async def _bench_one(model: str, key: str, task: str, system: str, user: str, timeout: int = 60) -> dict:
    from analysers.base import call_llm_async
    start = time.time()
    try:
        result = await asyncio.wait_for(
            call_llm_async(system, user, model=model, api_key=key,
                           api_base_url="https://openrouter.ai/api/v1", temperature=0.0),
            timeout=timeout,
        )
        latency = round(time.time() - start, 2)
        if not result.get("success"):
            return {"model": model, "task": task, "ok": False,
                    "json_valid": False, "latency_s": latency,
                    "error": result.get("error", "LLM error")}
        json_valid = _parse_json(result.get("data", ""))
        return {"model": model, "task": task, "ok": True,
                "json_valid": json_valid, "latency_s": latency}
    except asyncio.TimeoutError:
        return {"model": model, "task": task, "ok": False,
                "json_valid": False, "latency_s": timeout, "error": "timeout"}
    except Exception as e:
        return {"model": model, "task": task, "ok": False,
                "json_valid": False, "latency_s": round(time.time() - start, 2),
                "error": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/benchmark-models")
async def run_benchmark():
    pool = _load_pool()
    if not pool:
        return {"success": False, "error": "No models in pool — check OPENROUTER_MODEL_N env vars."}

    logger.info("Benchmark starting: %d models × %d tasks = %d calls (all parallel)",
                len(pool), len(_TASKS), len(pool) * len(_TASKS))

    calls = [
        _bench_one(entry["model"], entry["key"], task, sys_p, user_p)
        for entry in pool
        for task, sys_p, user_p in _TASKS
    ]
    raw = await asyncio.gather(*calls)

    # Aggregate per model
    by_model: dict[str, dict] = {}
    for r in raw:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"model": m, "judge": None, "audit": None}
        by_model[m][r["task"]] = r

    summary = []
    for m, data in by_model.items():
        j = data["judge"] or {}
        a = data["audit"] or {}
        both_valid = j.get("json_valid", False) and a.get("json_valid", False)
        avg_latency = round(
            ((j.get("latency_s") or 999) + (a.get("latency_s") or 999)) / 2, 2
        )
        summary.append({
            "model": m,
            "judge_ok": j.get("ok", False),
            "judge_json_valid": j.get("json_valid", False),
            "judge_latency_s": j.get("latency_s", 999),
            "judge_error": j.get("error"),
            "audit_ok": a.get("ok", False),
            "audit_json_valid": a.get("json_valid", False),
            "audit_latency_s": a.get("latency_s", 999),
            "audit_error": a.get("error"),
            "both_json_valid": both_valid,
            "avg_latency_s": avg_latency,
        })

    # Sort: both valid first, then fastest
    summary.sort(key=lambda x: (not x["both_json_valid"], x["avg_latency_s"]))

    best_judge = next((m for m in summary if m["judge_json_valid"]), None)
    best_audit = next((m for m in summary if m["audit_json_valid"]), None)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_tested": len(pool),
        "results": summary,
        "recommended": {
            "judge": best_judge["model"] if best_judge else None,
            "audit": best_audit["model"] if best_audit else None,
        },
    }

    with open(BENCHMARK_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Benchmark done. Best judge=%s  best audit=%s",
                output["recommended"]["judge"], output["recommended"]["audit"])
    return output


@router.get("/benchmark-results")
async def get_benchmark_results():
    if not os.path.exists(BENCHMARK_FILE):
        return {"success": False, "error": "No benchmark run yet — POST /benchmark-models first."}
    with open(BENCHMARK_FILE) as f:
        return json.load(f)


# ── Real-data benchmark ───────────────────────────────────────────────────────

from pydantic import BaseModel

class RealBenchRequest(BaseModel):
    judge_resume_id: str
    job_ids: list[str]          # up to 3
    audit_resume_id: str

REAL_BENCHMARK_FILE = os.path.join(BASE_DIR, "data", "model_benchmark_real.json")


async def _bench_real_judge(model: str, key: str, resume_text: str,
                             job_desc: str, job_label: str, timeout: int = 180,
                             api_base_url: str = "https://openrouter.ai/api/v1") -> dict:
    from analysers.base import call_llm_async
    from analysers.skills import prompts as skills_prompts
    system, user, _ = skills_prompts.build_prompt(resume_text, job_desc)
    start = time.time()
    try:
        result = await asyncio.wait_for(
            call_llm_async(system, user, model=model, api_key=key,
                           api_base_url=api_base_url, temperature=0.0),
            timeout=timeout,
        )
        latency = round(time.time() - start, 2)
        if not result.get("success"):
            return {"model": model, "job": job_label, "task": "judge",
                    "ok": False, "json_valid": False, "score": None,
                    "latency_s": latency, "error": result.get("error")}
        raw = result.get("data", "")
        json_valid = _parse_json(raw)
        score = None
        if json_valid:
            try:
                from analysers.skills.schema import SkillsOutput
                from analysers.base import parse_and_validate
                parsed = parse_and_validate(raw, SkillsOutput)
                if parsed and parsed.get("success") is not False:
                    from analysers.skills.analyser import score as calc_score
                    scored = calc_score(parsed)
                    score = scored.get("score")
            except Exception:
                pass
        return {"model": model, "job": job_label, "task": "judge",
                "ok": True, "json_valid": json_valid, "score": score, "latency_s": latency}
    except asyncio.TimeoutError:
        return {"model": model, "job": job_label, "task": "judge",
                "ok": False, "json_valid": False, "score": None,
                "latency_s": timeout, "error": "timeout"}
    except Exception as e:
        return {"model": model, "job": job_label, "task": "judge",
                "ok": False, "json_valid": False, "score": None,
                "latency_s": round(time.time() - start, 2), "error": str(e)}


def _build_audit_prompts(resume_text: str) -> tuple[str, str]:
    import routers.tools as tools_mod
    system_prompt = tools_mod._ATS_DEEP_AUDIT_SYSTEM
    user_prompt = f"""Perform a full audit on the resume below using all four dimensions from your instructions.

RESUME:
{resume_text[:5000]}

Return ONLY valid JSON matching the schema in your instructions. Begin with {{ and end with }}."""
    return system_prompt, user_prompt


async def _bench_real_audit(model: str, key: str, resume_text: str, timeout: int = 180,
                             api_base_url: str = "https://openrouter.ai/api/v1") -> dict:
    from analysers.base import call_llm_async
    system_prompt, user_prompt = _build_audit_prompts(resume_text)
    start = time.time()
    try:
        result = await asyncio.wait_for(
            call_llm_async(system_prompt, user_prompt, model=model, api_key=key,
                           api_base_url=api_base_url, temperature=0.0),
            timeout=timeout,
        )
        latency = round(time.time() - start, 2)
        if not result.get("success"):
            return {"model": model, "task": "audit", "ok": False,
                    "json_valid": False, "latency_s": latency, "error": result.get("error")}
        json_valid = _parse_json(result.get("data", ""))
        return {"model": model, "task": "audit", "ok": True,
                "json_valid": json_valid, "latency_s": latency}
    except asyncio.TimeoutError:
        return {"model": model, "task": "audit", "ok": False,
                "json_valid": False, "latency_s": timeout, "error": "timeout"}
    except Exception as e:
        return {"model": model, "task": "audit", "ok": False,
                "json_valid": False, "latency_s": round(time.time() - start, 2), "error": str(e)}


@router.post("/benchmark-real")
async def run_real_benchmark(req: RealBenchRequest):
    from routers.deps import run_db
    pool = _load_pool()
    if not pool:
        return {"success": False, "error": "No models in pool."}

    # Filter to synthetic-benchmark winners if available — no point running
    # the heavy real prompt against models that already failed the lightweight test.
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE) as f:
            prev = json.load(f)
        passing = {r["model"] for r in prev.get("results", []) if r.get("judge_json_valid") or r.get("audit_json_valid")}
        if passing:
            pool = [e for e in pool if e["model"] in passing]
            logger.info("Filtered pool to %d synthetic-passing models: %s", len(pool), passing)

    # Fetch resume + job text from DB
    all_resumes = await asyncio.to_thread(run_db.get_all_resumes)
    all_jobs    = await asyncio.to_thread(run_db.get_all_jobs)

    judge_resume = next((r for r in all_resumes if str(r["resume_id"]) == str(req.judge_resume_id)), None)
    audit_resume = next((r for r in all_resumes if str(r["resume_id"]) == str(req.audit_resume_id)), None)

    if not judge_resume:
        return {"success": False, "error": f"Judge resume {req.judge_resume_id} not found"}
    if not audit_resume:
        return {"success": False, "error": f"Audit resume {req.audit_resume_id} not found"}

    selected_jobs = [j for j in all_jobs if str(j["job_id"]) in [str(x) for x in req.job_ids]][:3]
    if not selected_jobs:
        return {"success": False, "error": "No valid jobs found for given job_ids"}

    logger.info("Real benchmark: %d models, %d jobs, judge=%s, audit=%s",
                len(pool), len(selected_jobs), judge_resume.get("name"), audit_resume.get("name"))

    calls = []
    for entry in pool:
        for job in selected_jobs:
            skills_req = job.get("skills_required") or ""
            desc = job.get("description") or ""
            jd = (f"Required Skills: {skills_req}\n\n" + desc) if skills_req else desc
            label = f"{job.get('title','?')} @ {job.get('company','?')}"
            calls.append(_bench_real_judge(entry["model"], entry["key"],
                                           judge_resume["text"], jd, label))
        calls.append(_bench_real_audit(entry["model"], entry["key"], audit_resume["text"]))

    raw = await asyncio.gather(*calls)

    # Aggregate per model
    by_model: dict[str, dict] = {}
    for r in raw:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"model": m, "jobs": [], "audit": None}
        if r["task"] == "judge":
            by_model[m]["jobs"].append(r)
        else:
            by_model[m]["audit"] = r

    summary = []
    for m, data in by_model.items():
        jobs_data = data["jobs"]
        audit = data["audit"] or {}
        judge_valid = all(j.get("json_valid") for j in jobs_data) if jobs_data else False
        judge_avg_latency = round(sum(j.get("latency_s", 999) for j in jobs_data) / max(len(jobs_data), 1), 2)
        scores = [j["score"] for j in jobs_data if j.get("score") is not None]
        summary.append({
            "model": m,
            "judge_all_valid": judge_valid,
            "judge_avg_latency_s": judge_avg_latency,
            "job_results": [{"job": j["job"], "score": j.get("score"), "json_valid": j.get("json_valid"),
                             "latency_s": j.get("latency_s"), "error": j.get("error")} for j in jobs_data],
            "scores": scores,
            "score_avg": round(sum(scores) / len(scores), 1) if scores else None,
            "audit_json_valid": audit.get("json_valid", False),
            "audit_latency_s": audit.get("latency_s", 999),
            "audit_error": audit.get("error"),
        })

    # Composite rank: 60% score quality + 40% speed (normalised within pool)
    valid_judge = [x for x in summary if x["judge_all_valid"] and (x["score_avg"] or 0) > 0]
    if valid_judge:
        max_lat = max(x["judge_avg_latency_s"] for x in valid_judge) or 1
        for x in valid_judge:
            x["composite"] = (x["score_avg"] or 0) * 0.6 - (x["judge_avg_latency_s"] / max_lat) * 0.4
    summary.sort(key=lambda x: (
        not x["judge_all_valid"],
        -(x.get("composite") or 0),
    ))

    best_judge = next((m for m in summary if m["judge_all_valid"] and (m["score_avg"] or 0) > 0), None)
    best_audit = next((m for m in summary if m["audit_json_valid"]), None)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "real",
        "judge_resume": judge_resume.get("name"),
        "audit_resume": audit_resume.get("name"),
        "jobs_tested": [f"{j.get('title')} @ {j.get('company')}" for j in selected_jobs],
        "models_tested": len(pool),
        "results": summary,
        "recommended": {
            "judge": best_judge["model"] if best_judge else None,
            "audit": best_audit["model"] if best_audit else None,
        },
    }

    with open(REAL_BENCHMARK_FILE, "w") as f:
        json.dump(output, f, indent=2)

    return output


@router.get("/benchmark-real-results")
async def get_real_benchmark_results():
    if not os.path.exists(REAL_BENCHMARK_FILE):
        return {"success": False, "error": "No real benchmark run yet."}
    with open(REAL_BENCHMARK_FILE) as f:
        return json.load(f)


# ── Ollama benchmark ──────────────────────────────────────────────────────────

OLLAMA_BENCHMARK_FILE      = os.path.join(BASE_DIR, "data", "model_benchmark_ollama.json")
OLLAMA_REAL_BENCHMARK_FILE = os.path.join(BASE_DIR, "data", "model_benchmark_ollama_real.json")

_SKIP_MODEL_KEYWORDS = ("embed", "vision", "clip", "whisper")


async def _list_ollama_models() -> list[str]:
    """Runs `ollama list` and returns model names, excluding embed/vision models."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        models = []
        for line in stdout.decode().splitlines()[1:]:  # skip header
            name = line.split()[0].strip() if line.strip() else ""
            if name and not any(kw in name.lower() for kw in _SKIP_MODEL_KEYWORDS):
                models.append(name)
        return models
    except Exception as e:
        logger.warning("_list_ollama_models failed: %s", e)
        return []


async def _bench_one_ollama(model: str, task: str, system: str, user: str,
                             timeout: int = 120) -> dict:
    from analysers.base import call_llm_async
    start = time.time()
    try:
        result = await asyncio.wait_for(
            call_llm_async(system, user, model=model, temperature=0.0),
            timeout=timeout,
        )
        latency = round(time.time() - start, 2)
        if not result.get("success"):
            return {"model": model, "task": task, "ok": False,
                    "json_valid": False, "latency_s": latency,
                    "error": result.get("error", "LLM error")}
        json_valid = _parse_json(result.get("data", ""))
        return {"model": model, "task": task, "ok": True,
                "json_valid": json_valid, "latency_s": latency}
    except asyncio.TimeoutError:
        return {"model": model, "task": task, "ok": False,
                "json_valid": False, "latency_s": timeout, "error": "timeout"}
    except Exception as e:
        return {"model": model, "task": task, "ok": False,
                "json_valid": False, "latency_s": round(time.time() - start, 2),
                "error": str(e)}


@router.post("/benchmark-ollama")
async def run_ollama_benchmark():
    """Synthetic benchmark — runs all installed Ollama models sequentially (single GPU)."""
    models = await _list_ollama_models()
    if not models:
        return {"success": False, "error": "No Ollama models found. Is Ollama running?"}

    logger.info("Ollama synthetic benchmark: %d models × %d tasks (sequential)", len(models), len(_TASKS))

    results_raw = []
    for model in models:
        for task, sys_p, user_p in _TASKS:
            r = await _bench_one_ollama(model, task, sys_p, user_p, timeout=120)
            results_raw.append(r)
            logger.info("  %s / %s → ok=%s json=%s latency=%.1fs",
                        model, task, r["ok"], r["json_valid"], r["latency_s"])

    by_model: dict[str, dict] = {}
    for r in results_raw:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"model": m, "judge": None, "audit": None}
        by_model[m][r["task"]] = r

    summary = []
    for m, data in by_model.items():
        j = data["judge"] or {}
        a = data["audit"] or {}
        both_valid = j.get("json_valid", False) and a.get("json_valid", False)
        avg_latency = round(
            ((j.get("latency_s") or 999) + (a.get("latency_s") or 999)) / 2, 2
        )
        summary.append({
            "model": m,
            "judge_ok": j.get("ok", False),
            "judge_json_valid": j.get("json_valid", False),
            "judge_latency_s": j.get("latency_s", 999),
            "judge_error": j.get("error"),
            "audit_ok": a.get("ok", False),
            "audit_json_valid": a.get("json_valid", False),
            "audit_latency_s": a.get("latency_s", 999),
            "audit_error": a.get("error"),
            "both_json_valid": both_valid,
            "avg_latency_s": avg_latency,
        })

    summary.sort(key=lambda x: (not x["both_json_valid"], x["avg_latency_s"]))

    best_judge = next((m for m in summary if m["judge_json_valid"]), None)
    best_audit = next((m for m in summary if m["audit_json_valid"]), None)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": "ollama",
        "models_tested": len(models),
        "results": summary,
        "recommended": {
            "judge": best_judge["model"] if best_judge else None,
            "audit": best_audit["model"] if best_audit else None,
        },
    }

    with open(OLLAMA_BENCHMARK_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Ollama synthetic done. best_judge=%s  best_audit=%s",
                output["recommended"]["judge"], output["recommended"]["audit"])
    return output


@router.get("/benchmark-ollama-results")
async def get_ollama_benchmark_results():
    if not os.path.exists(OLLAMA_BENCHMARK_FILE):
        return {"success": False, "error": "No Ollama benchmark run yet — POST /benchmark-ollama first."}
    with open(OLLAMA_BENCHMARK_FILE) as f:
        return json.load(f)


async def _bench_ollama_judge(model: str, resume_text: str,
                               job_desc: str, job_label: str, timeout: int = 300) -> dict:
    return await _bench_real_judge(model, "", resume_text, job_desc, job_label,
                                   timeout=timeout, api_base_url="")


async def _bench_ollama_audit(model: str, resume_text: str, timeout: int = 300) -> dict:
    return await _bench_real_audit(model, "", resume_text, timeout=timeout, api_base_url="")


@router.post("/benchmark-ollama-real")
async def run_ollama_real_benchmark(req: RealBenchRequest):
    """Real-data benchmark on installed Ollama models (sequential, single GPU)."""
    from routers.deps import run_db

    models = await _list_ollama_models()
    if not models:
        return {"success": False, "error": "No Ollama models found."}

    # Filter to synthetic-passing models if available
    if os.path.exists(OLLAMA_BENCHMARK_FILE):
        with open(OLLAMA_BENCHMARK_FILE) as f:
            prev = json.load(f)
        passing = {r["model"] for r in prev.get("results", [])
                   if r.get("judge_json_valid") or r.get("audit_json_valid")}
        if passing:
            models = [m for m in models if m in passing]
            logger.info("Filtered Ollama pool to %d synthetic-passing models: %s", len(models), passing)

    all_resumes = await asyncio.to_thread(run_db.get_all_resumes)
    all_jobs    = await asyncio.to_thread(run_db.get_all_jobs)

    judge_resume = next((r for r in all_resumes if str(r["resume_id"]) == str(req.judge_resume_id)), None)
    audit_resume = next((r for r in all_resumes if str(r["resume_id"]) == str(req.audit_resume_id)), None)

    if not judge_resume:
        return {"success": False, "error": f"Judge resume {req.judge_resume_id} not found"}
    if not audit_resume:
        return {"success": False, "error": f"Audit resume {req.audit_resume_id} not found"}

    selected_jobs = [j for j in all_jobs if str(j["job_id"]) in [str(x) for x in req.job_ids]][:3]
    if not selected_jobs:
        return {"success": False, "error": "No valid jobs found for given job_ids"}

    logger.info("Ollama real benchmark: %d models, %d jobs (sequential)", len(models), len(selected_jobs))

    results_raw = []
    for model in models:
        for job in selected_jobs:
            skills_req = job.get("skills_required") or ""
            desc = job.get("description") or ""
            jd = (f"Required Skills: {skills_req}\n\n" + desc) if skills_req else desc
            label = f"{job.get('title','?')} @ {job.get('company','?')}"
            r = await _bench_ollama_judge(model, judge_resume["text"], jd, label)
            results_raw.append(r)
            logger.info("  judge %s / %s → ok=%s json=%s latency=%.1fs",
                        model, label, r["ok"], r["json_valid"], r["latency_s"])
        r = await _bench_ollama_audit(model, audit_resume["text"])
        results_raw.append(r)
        logger.info("  audit %s → ok=%s json=%s latency=%.1fs",
                    model, r["ok"], r["json_valid"], r["latency_s"])

    by_model: dict[str, dict] = {}
    for r in results_raw:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"model": m, "jobs": [], "audit": None}
        if r["task"] == "judge":
            by_model[m]["jobs"].append(r)
        else:
            by_model[m]["audit"] = r

    summary = []
    for m, data in by_model.items():
        jobs_data = data["jobs"]
        audit = data["audit"] or {}
        judge_valid = all(j.get("json_valid") for j in jobs_data) if jobs_data else False
        judge_avg_latency = round(sum(j.get("latency_s", 999) for j in jobs_data) / max(len(jobs_data), 1), 2)
        scores = [j["score"] for j in jobs_data if j.get("score") is not None]
        summary.append({
            "model": m,
            "judge_all_valid": judge_valid,
            "judge_avg_latency_s": judge_avg_latency,
            "job_results": [{"job": j["job"], "score": j.get("score"), "json_valid": j.get("json_valid"),
                             "latency_s": j.get("latency_s"), "error": j.get("error")} for j in jobs_data],
            "scores": scores,
            "score_avg": round(sum(scores) / len(scores), 1) if scores else None,
            "audit_json_valid": audit.get("json_valid", False),
            "audit_latency_s": audit.get("latency_s", 999),
            "audit_error": audit.get("error"),
        })

    # Composite rank: 60% score quality + 40% speed (normalised within pool)
    valid_judge = [x for x in summary if x["judge_all_valid"] and (x["score_avg"] or 0) > 0]
    if valid_judge:
        max_lat = max(x["judge_avg_latency_s"] for x in valid_judge) or 1
        for x in valid_judge:
            x["composite"] = (x["score_avg"] or 0) * 0.6 - (x["judge_avg_latency_s"] / max_lat) * 0.4
    summary.sort(key=lambda x: (
        not x["judge_all_valid"],
        -(x.get("composite") or 0),
    ))

    best_judge = next((m for m in summary if m["judge_all_valid"] and (m["score_avg"] or 0) > 0), None)
    best_audit = next((m for m in summary if m["audit_json_valid"]), None)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": "ollama",
        "mode": "real",
        "judge_resume": judge_resume.get("name"),
        "audit_resume": audit_resume.get("name"),
        "jobs_tested": [f"{j.get('title')} @ {j.get('company')}" for j in selected_jobs],
        "models_tested": len(models),
        "results": summary,
        "recommended": {
            "judge": best_judge["model"] if best_judge else None,
            "audit": best_audit["model"] if best_audit else None,
        },
    }

    with open(OLLAMA_REAL_BENCHMARK_FILE, "w") as f:
        json.dump(output, f, indent=2)

    return output


@router.get("/benchmark-ollama-real-results")
async def get_ollama_real_benchmark_results():
    if not os.path.exists(OLLAMA_REAL_BENCHMARK_FILE):
        return {"success": False, "error": "No Ollama real benchmark run yet."}
    with open(OLLAMA_REAL_BENCHMARK_FILE) as f:
        return json.load(f)


# ── Full benchmark (all 5 judges + audit) ─────────────────────────────────────

OLLAMA_FULL_BENCHMARK_FILE    = os.path.join(BASE_DIR, "data", "model_benchmark_ollama_full.json")
OLLAMA_FULL_PROGRESS_FILE     = os.path.join(BASE_DIR, "data", "model_benchmark_ollama_full_progress.json")
JUDGE_NAMES = ["skills", "experience", "seniority", "relevancy", "impact"]


def _build_judge_prompt(judge_name: str, resume_text: str, jd_text: str) -> tuple[str, str]:
    if judge_name == "skills":
        from analysers.skills.prompts import build_prompt
        r = build_prompt(resume_text, jd_text)
    elif judge_name == "experience":
        from analysers.experience.prompts import build_prompt
        r = build_prompt(resume_text, jd_text)
    elif judge_name == "seniority":
        from analysers.seniority.prompts import build_prompt
        r = build_prompt(resume_text, jd_text)
    elif judge_name == "relevancy":
        from analysers.relevancy.prompts import build_prompt
        r = build_prompt(resume_text, jd_text)
    elif judge_name == "impact":
        from analysers.impact.prompts import build_prompt
        r = build_prompt(resume_text)  # no JD
    else:
        raise ValueError(f"Unknown judge: {judge_name}")
    return r[0], r[1]


def _get_schema_and_score(judge_name: str):
    if judge_name == "skills":
        from analysers.skills.schema import SkillsOutput
        from analysers.skills.analyser import score
        return SkillsOutput, score
    elif judge_name == "experience":
        from analysers.experience.schema import ExperienceOutput
        from analysers.experience.analyser import score
        return ExperienceOutput, score
    elif judge_name == "seniority":
        from analysers.seniority.schema import SeniorityOutput
        from analysers.seniority.analyser import score
        return SeniorityOutput, score
    elif judge_name == "relevancy":
        from analysers.relevancy.schema import RelevancyOutput
        from analysers.relevancy.analyser import score
        return RelevancyOutput, score
    elif judge_name == "impact":
        from analysers.impact.schema import ImpactOutput
        from analysers.impact.analyser import score
        return ImpactOutput, score
    raise ValueError(f"Unknown judge: {judge_name}")


async def _full_judge_call(model: str, judge_name: str, jd_label: str,
                           system: str, user: str, schema_class, score_fn,
                           timeout: int = 300) -> dict:
    from analysers.base import call_llm_async, parse_and_validate
    entry = {
        "model": model, "judge": judge_name, "job": jd_label,
        "parse_success": False, "confidence": 0, "attempts": 0,
        "latency_s": 0.0, "score": None, "truncated": False,
        "first_attempt_success": False, "error": None,
    }
    start = time.time()
    for attempt in range(2):
        entry["attempts"] = attempt + 1
        try:
            res = await asyncio.wait_for(
                call_llm_async(system, user, model=model, temperature=0.0),
                timeout=timeout,
            )
            if not res.get("success"):
                entry["error"] = res.get("error", "LLM error")
                break
            raw = res.get("data", "")
            stripped = raw.strip()
            entry["truncated"] = not stripped.rstrip().endswith("}")
            try:
                if schema_class:
                    parsed = parse_and_validate(raw, schema_class)
                else:
                    parsed = parse_and_validate(raw)
                entry["parse_success"] = True
                entry["confidence"] = parsed.get("confidence", 0)
                entry["first_attempt_success"] = (attempt == 0)
                if score_fn:
                    try:
                        s = score_fn(parsed)
                        entry["score"] = round(s.get("score", 0) * 100, 1)
                    except Exception:
                        pass
                break
            except Exception as e:
                entry["error"] = f"parse_failed: {str(e)[:120]}"
        except asyncio.TimeoutError:
            entry["error"] = "timeout"
            break
        except Exception as e:
            entry["error"] = str(e)[:120]
            break
    entry["latency_s"] = round(time.time() - start, 2)
    return entry


def _judge_stats(results: list[dict]) -> dict:
    if not results:
        return None
    n = len(results)
    passed = [r for r in results if r["parse_success"]]
    pass_rate = round(len(passed) / n * 100, 1)
    avg_conf = round(sum(r["confidence"] for r in passed) / max(len(passed), 1), 1)
    avg_lat = round(sum(r["latency_s"] for r in results) / n, 2)
    first_att = round(sum(1 for r in results if r["first_attempt_success"]) / n * 100, 1)
    trunc_rate = round(sum(1 for r in results if r["truncated"]) / n * 100, 1)
    scores = [r["score"] for r in results if r["score"] is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    variance = 0.0
    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        variance = round(math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)), 1)
    return {
        "pass_rate": pass_rate,
        "avg_confidence": avg_conf,
        "avg_latency_s": avg_lat,
        "first_attempt_rate": first_att,
        "truncation_rate": trunc_rate,
        "avg_score": avg_score,
        "score_variance": variance,
    }


@router.post("/benchmark-ollama-full")
async def run_ollama_full_benchmark(req: RealBenchRequest):
    """Full benchmark: all 5 judges × all models × all JDs, sequential on single GPU."""
    from routers.deps import run_db

    # Reasoning models that generate multi-thousand-token think blocks — useless as judges
    _SKIP_MODELS = {"qwen3.5:latest", "qwen3.5"}

    models = await _list_ollama_models()
    if not models:
        return {"success": False, "error": "No Ollama models found."}
    models = [m for m in models if m not in _SKIP_MODELS]

    if os.path.exists(OLLAMA_BENCHMARK_FILE):
        with open(OLLAMA_BENCHMARK_FILE) as f:
            prev = json.load(f)
        passing = {r["model"] for r in prev.get("results", [])
                   if r.get("judge_json_valid") or r.get("audit_json_valid")}
        if passing:
            models = [m for m in models if m in passing]

    all_resumes = await asyncio.to_thread(run_db.get_all_resumes)
    all_jobs    = await asyncio.to_thread(run_db.get_all_jobs)

    judge_resume = next((r for r in all_resumes if str(r["resume_id"]) == str(req.judge_resume_id)), None)
    audit_resume = next((r for r in all_resumes if str(r["resume_id"]) == str(req.audit_resume_id)), None)
    if not judge_resume:
        return {"success": False, "error": f"Judge resume {req.judge_resume_id} not found"}
    if not audit_resume:
        return {"success": False, "error": f"Audit resume {req.audit_resume_id} not found"}

    selected_jobs = [j for j in all_jobs if str(j["job_id"]) in [str(x) for x in req.job_ids]][:3]
    if not selected_jobs:
        return {"success": False, "error": "No valid jobs found"}

    logger.info("Full benchmark: %d models × 5 judges × %d JDs (sequential)", len(models), len(selected_jobs))

    total_steps = len(models) * (len(selected_jobs) * len(JUDGE_NAMES) + 1)
    completed_steps = 0

    def _write_progress(current_model: str, current_judge: str, current_job: str) -> None:
        try:
            with open(OLLAMA_FULL_PROGRESS_FILE, "w") as _pf:
                json.dump({
                    "running": True,
                    "completed": completed_steps,
                    "total": total_steps,
                    "pct": round(completed_steps / total_steps * 100, 1) if total_steps else 0,
                    "current_model": current_model,
                    "current_judge": current_judge,
                    "current_job": current_job,
                    "models_total": len(models),
                    "model_index": models.index(current_model) + 1 if current_model in models else 0,
                }, _pf)
        except Exception:
            pass

    all_raw: list[dict] = []
    for model in models:
        for job in selected_jobs:
            skills_req = job.get("skills_required") or ""
            desc = job.get("description") or ""
            jd = (f"Required Skills: {skills_req}\n\n" + desc) if skills_req else desc
            label = f"{job.get('title','?')} @ {job.get('company','?')}"
            for judge_name in JUDGE_NAMES:
                _write_progress(model, judge_name, label)
                try:
                    system, user = _build_judge_prompt(judge_name, judge_resume["text"], jd)
                    schema_class, score_fn = _get_schema_and_score(judge_name)
                    r = await _full_judge_call(model, judge_name, label, system, user, schema_class, score_fn)
                except Exception as e:
                    r = {"model": model, "judge": judge_name, "job": label,
                         "parse_success": False, "confidence": 0, "attempts": 1,
                         "latency_s": 0.0, "score": None, "truncated": False,
                         "first_attempt_success": False, "error": str(e)[:120]}
                all_raw.append(r)
                completed_steps += 1
                logger.info("  %s/%s/%s → parse=%s conf=%d lat=%.1fs trunc=%s",
                            model, judge_name, label[:25],
                            r["parse_success"], r["confidence"], r["latency_s"], r["truncated"])

        # Audit
        _write_progress(model, "audit", "ats-deep-audit")
        try:
            sys_p, usr_p = _build_audit_prompts(audit_resume["text"])
            r = await _full_judge_call(model, "audit", "audit", sys_p, usr_p, None, None)
        except Exception as e:
            r = {"model": model, "judge": "audit", "job": "audit",
                 "parse_success": False, "confidence": 0, "attempts": 1,
                 "latency_s": 0.0, "score": None, "truncated": False,
                 "first_attempt_success": False, "error": str(e)[:120]}
        all_raw.append(r)
        completed_steps += 1
        logger.info("  %s/audit → parse=%s lat=%.1fs trunc=%s",
                    model, r["parse_success"], r["latency_s"], r["truncated"])

    # Aggregate per model
    model_buckets: dict[str, dict] = {}
    for r in all_raw:
        m = r["model"]
        if m not in model_buckets:
            model_buckets[m] = {j: [] for j in JUDGE_NAMES}
            model_buckets[m]["audit"] = []
        key = r["judge"] if r["judge"] in JUDGE_NAMES else "audit"
        model_buckets[m][key].append(r)

    summaries = []
    for m, buckets in model_buckets.items():
        per_judge = {j: _judge_stats(buckets[j]) for j in JUDGE_NAMES}
        audit_list = buckets["audit"]
        audit_stats = None
        if audit_list:
            ar = audit_list[0]
            audit_stats = {
                "parse_success": ar["parse_success"],
                "latency_s": ar["latency_s"],
                "truncated": ar["truncated"],
                "error": ar.get("error"),
            }

        valid_judges = [s for s in per_judge.values() if s]
        avg_pass    = sum(s["pass_rate"] for s in valid_judges) / len(valid_judges) if valid_judges else 0
        avg_conf    = sum(s["avg_confidence"] for s in valid_judges) / len(valid_judges) if valid_judges else 0
        avg_first   = sum(s["first_attempt_rate"] for s in valid_judges) / len(valid_judges) if valid_judges else 0
        avg_lat     = sum(s["avg_latency_s"] for s in valid_judges) / len(valid_judges) if valid_judges else 999
        speed_score = max(0.0, 100.0 - (avg_lat / 300.0 * 100.0))

        # skills+relevancy are the gates that cause 0% overall — weight them 2×
        _JUDGE_WEIGHTS = {"skills": 2.0, "relevancy": 2.0, "experience": 1.0, "seniority": 1.0, "impact": 1.0}
        _wtotal, _wsum = 0.0, 0.0
        for jname, stats in per_judge.items():
            if stats and stats.get("avg_score") is not None:
                w = _JUDGE_WEIGHTS.get(jname, 1.0)
                _wsum += stats["avg_score"] * w
                _wtotal += w
        avg_score = (_wsum / _wtotal) if _wtotal else 0.0

        # 40% score quality + 25% confidence + 20% speed + 10% pass rate + 5% first attempt
        composite = round(avg_score * 0.40 + avg_conf * 0.25 + speed_score * 0.20 + avg_pass * 0.10 + avg_first * 0.05, 1)

        summaries.append({
            "model": m,
            "per_judge": per_judge,
            "audit": audit_stats,
            "composite": composite,
            "avg_judge_score": round(avg_score, 1),
            "avg_pass_rate": round(avg_pass, 1),
            "avg_confidence": round(avg_conf, 1),
            "avg_first_attempt_rate": round(avg_first, 1),
            "avg_latency_s": round(avg_lat, 2),
            "avg_speed_score": round(speed_score, 1),
        })

    summaries.sort(key=lambda x: -x["composite"])
    best = summaries[0]["model"] if summaries else None

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": "ollama",
        "mode": "full",
        "judge_resume": judge_resume.get("name"),
        "audit_resume": audit_resume.get("name"),
        "jobs_tested": [f"{j.get('title')} @ {j.get('company')}" for j in selected_jobs],
        "models_tested": len(models),
        "results": summaries,
        "recommended": {"judge": best, "audit": best},
        "raw_calls": all_raw,
    }

    with open(OLLAMA_FULL_BENCHMARK_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Clear progress file
    try:
        with open(OLLAMA_FULL_PROGRESS_FILE, "w") as f:
            json.dump({"running": False, "completed": total_steps, "total": total_steps, "pct": 100.0}, f)
    except Exception:
        pass

    logger.info("Full benchmark done. Best overall: %s (composite=%.1f)", best,
                summaries[0]["composite"] if summaries else 0)
    return output


@router.get("/benchmark-ollama-full-results")
async def get_ollama_full_benchmark_results():
    if not os.path.exists(OLLAMA_FULL_BENCHMARK_FILE):
        return {"success": False, "error": "No full benchmark run yet — POST /benchmark-ollama-full first."}
    with open(OLLAMA_FULL_BENCHMARK_FILE) as f:
        return json.load(f)


@router.get("/benchmark-ollama-full-progress")
async def get_ollama_full_progress():
    if not os.path.exists(OLLAMA_FULL_PROGRESS_FILE):
        return {"running": False, "completed": 0, "total": 0, "pct": 0.0}
    with open(OLLAMA_FULL_PROGRESS_FILE) as f:
        return json.load(f)
