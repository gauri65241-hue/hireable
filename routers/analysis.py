import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
from pydantic import BaseModel

from auth_utils import get_current_user

from analysers.semantic import resume_jd_similarity, flag_suspect_scores

from routers.deps import (
    BASE_DIR,
    judge_engine, _analyze_sem, _analyser_cfg,
    run_db, score_ats, ats_to_dict,
    sanitize_user_content,
    calculate_overall_match,
    calculate_skills_score, calculate_experience_score,
    calculate_seniority_score, calculate_relevancy_score, calculate_impact_score,
    compute_learned_weights, append_feedback,
    INTERVIEW_JUDGES, JOB_JUDGES,
    execute_llm_call, get_api_config,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str
    use_coach: bool = True
    llm_models: list[str] = ["cogito:14b"]
    api_key: str = ""
    api_base_url: str = ""
    candidate_name: str = ""
    job_title: str = ""

class BatchSummaryRequest(BaseModel):
    batch_data: str
    model: str = "meta-llama/llama-3.1-8b-instruct:free"

class CandidateSummaryRequest(BaseModel):
    resume_name: str
    job_title: str
    score: float
    metrics: dict
    missing_skills: list
    model: str

class SummarizeResultRequest(BaseModel):
    result_id: str
    model: str = "cogito:14b"

class RecordOutcomeRequest(BaseModel):
    result_id: str
    predicted_score: float
    got_job: Optional[bool] = None
    got_interview: Optional[bool] = None
    interview_stage: Optional[str] = None    # phone_screen | technical | final | offer
    rejection_reason: Optional[str] = None  # skills_gap | seniority_mismatch | culture_fit | compensation | other

class RunEvalsRequest(BaseModel):
    model: str = ""
    api_key: str = ""
    api_base_url: str = ""


# ── /analyze ─────────────────────────────────────────────────────────────────

def _truncate_jd(text: str, limit: int = 10_000) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_space = truncated.rfind(' ')
    return truncated[:last_space] if last_space > limit // 2 else truncated


@router.post("/analyze")
@limiter.limit("5/minute")
async def analyze_resume_unified(request: Request, body: AnalyzeRequest,
                                  current_user: dict = Depends(get_current_user)):
    body = body.model_copy(update={"job_description": _truncate_jd(body.job_description)})
    try:
        sanitize_user_content(body.resume_text, "resume")
        sanitize_user_content(body.job_description, "job_description")
    except ValueError as inj_err:
        raise HTTPException(status_code=422, detail=str(inj_err))

    try:
        await asyncio.wait_for(_analyze_sem.acquire(), timeout=_analyser_cfg.sem_acquire_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server at capacity, try again shortly")
    try:
        job_description = body.job_description
        api_cfg = get_api_config(body.api_key, body.api_base_url, body.llm_models[0])
        model = api_cfg["llm_model"]
        timestamp = datetime.utcnow().isoformat()
        logger.info("analyze start model=%s api=%s resume=%dch jd=%dch",
                    model, "openrouter" if api_cfg["api_key"] else "ollama",
                    len(body.resume_text), len(job_description))

        ats_result = score_ats(body.resume_text, job_description)
        ats_dict = ats_to_dict(ats_result)
        logger.info("ATS %d/100 (%s) issues=%d", ats_result.score, ats_result.grade, len(ats_result.issues))

        semantic_similarity = await asyncio.to_thread(
            resume_jd_similarity, body.resume_text, job_description
        )
        logger.info("semantic_similarity=%.3f", semantic_similarity or 0)

        skills_taxonomy = {}
        try:
            taxonomy_path = os.path.join(BASE_DIR, "data", "skills_taxonomy.json")
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                skills_taxonomy = json.load(f)
        except Exception as e:
            logger.warning("skills taxonomy load failed: %s", e)

        pipeline_result = await judge_engine.run(
            resume_text=body.resume_text,
            job_description=job_description,
            llm_model=model,
            skills_taxonomy=skills_taxonomy,
            api_key=api_cfg["api_key"],
            api_base_url=api_cfg["api_base_url"],
            fast_model=api_cfg["fast_model"],
            ats_score=ats_result.score,
            semantic_similarity=semantic_similarity,
        )

        if not pipeline_result:
            raise HTTPException(status_code=500, detail="Judge pipeline failed to produce results")

        pipeline_result["ats_preflight"] = ats_dict

        learned = compute_learned_weights()
        if learned["source"] == "learned":
            s  = pipeline_result.get("skills", {}).get("scores", {})
            e  = pipeline_result.get("experience", {}).get("scores", {})
            sr = pipeline_result.get("seniority", {}).get("scores", {})
            rl = pipeline_result.get("relevancy", {}).get("scores", {})
            im = pipeline_result.get("impact", {}).get("scores", {})
            skills_ok = pipeline_result.get("skills", {}).get("analyser_result", {}).get("success", False)
            pipeline_result["overall"] = calculate_overall_match(s, e, sr, rl, im, learned["weights"], skills_gate_enabled=skills_ok)
            pipeline_result["weights_source"] = "learned"
            logger.info("learned weights applied: %s", learned["weights"])
        else:
            pipeline_result["weights_source"] = "default"

        overall          = pipeline_result.get("overall", {})
        skills_scores    = pipeline_result.get("skills", {}).get("scores", {})
        experience_scores = pipeline_result.get("experience", {}).get("scores", {})
        seniority_scores = pipeline_result.get("seniority", {}).get("scores", {})
        relevancy_scores = pipeline_result.get("relevancy", {}).get("scores", {})
        impact_scores    = pipeline_result.get("impact", {}).get("scores", {})
        coach_result     = pipeline_result.get("coach", {})

        skills_pct    = round(skills_scores.get("score", 0) * 100, 1)
        experience_pct = round(experience_scores.get("score", 0) * 100, 1)
        seniority_pct = round(seniority_scores.get("score", 0) * 100, 1)
        relevancy_pct = round(relevancy_scores.get("score", 0) * 100, 1)
        impact_pct    = round(impact_scores.get("score", 0) * 100, 1)
        overall_pct   = round(overall.get("overall_score", 0) * 100, 1)

        logger.info("scores overall=%.1f%% skills=%.1f%% exp=%.1f%%", overall_pct, skills_pct, experience_pct)

        missing_skills           = skills_scores.get("missing_skills", [])
        unmet_responsibilities   = experience_scores.get("unmet_responsibilities", [])
        absent_seniority_signals = seniority_scores.get("absent_signals", [])
        irrelevant_statements    = relevancy_scores.get("irrelevant_statements", [])
        unquantified_statements  = impact_scores.get("unquantified_statements", [])

        coach_recommendations = None
        if coach_result.get("success") and body.use_coach:
            coach_recommendations = coach_result.get("data", {}).get("improvement_analysis", [])

        result_id = f"analysis_{timestamp.replace(':', '-').replace('.', '-')}"
        try:
            await asyncio.to_thread(
                run_db.save_run,
                result_id, model, body.resume_text, body.job_description,
                pipeline_result, ats_dict, body.candidate_name, body.job_title,
                current_user["user_id"],
            )
        except Exception as db_err:
            logger.warning("SQLite save failed (non-fatal): %s", db_err)

        judge_score_map = {
            "skills":     skills_scores.get("score"),
            "relevancy":  relevancy_scores.get("score"),
            "experience": experience_scores.get("score"),
            "impact":     impact_scores.get("score"),
        }
        suspect_scores = flag_suspect_scores(judge_score_map, semantic_similarity)

        return {
            "success": True,
            "match_score": overall_pct,
            "metrics": {
                "skills_coverage":       skills_pct,
                "experience_depth":      experience_pct,
                "seniority_calibration": seniority_pct,
                "statement_relevancy":   relevancy_pct,
                "impact_quantification": impact_pct,
            },
            "analysis": {
                "missing_critical_skills":   missing_skills,
                "unmet_responsibilities":     unmet_responsibilities,
                "absent_seniority_signals":   absent_seniority_signals,
                "irrelevant_statements":      irrelevant_statements,
                "unquantified_statements":    unquantified_statements,
                "coach_recommendations":      coach_recommendations,
            },
            "pipeline_data": pipeline_result,
            "ats_preflight": ats_dict,
            "semantic_similarity": semantic_similarity,
            "suspect_scores": suspect_scores,
            "metadata": {
                "model_used":       model,
                "timestamp":        timestamp,
                "result_id":        result_id,
                "pipeline_version": "5-stage-v2.0",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _analyze_sem.release()


# ── /analyze/stream ───────────────────────────────────────────────────────────

@router.post("/analyze/stream")
@limiter.limit("5/minute")
async def analyze_stream(request: Request, body: AnalyzeRequest,
                         current_user: dict = Depends(get_current_user)):
    import asyncio as _asyncio

    body = body.model_copy(update={"job_description": _truncate_jd(body.job_description)})

    async def generate():
        def _event(obj: dict) -> str:
            payload = json.dumps(obj)
            pad = max(0, 1536 - len(payload))
            return f"data: {payload}\n: {'x' * pad}\n\n"

        try:
            sanitize_user_content(body.resume_text, "resume")
            sanitize_user_content(body.job_description, "job_description")
        except ValueError as inj_err:
            yield _event({"event": "error", "detail": str(inj_err), "code": 422})
            return

        try:
            await asyncio.wait_for(_analyze_sem.acquire(), timeout=_analyser_cfg.sem_acquire_timeout)
        except asyncio.TimeoutError:
            yield _event({"event": "error", "detail": "Server at capacity, try again shortly", "code": 503})
            return
        try:
            engine    = judge_engine
            api_cfg   = get_api_config(body.api_key, body.api_base_url, body.llm_models[0])
            model     = api_cfg["llm_model"]
            timestamp = datetime.utcnow().isoformat()
            logger.info("stream analyze start model=%s api=%s",
                        model, "openrouter" if api_cfg["api_key"] else "ollama")

            import time as _time
            _pipeline_start = _time.monotonic()

            from analysers.engine import ATS_SKIP_THRESHOLD, SIMILARITY_MISMATCH_THRESHOLD, _FAST_JUDGES
            from analysers.config import cfg as _cfg

            ats_result = score_ats(body.resume_text, body.job_description)
            ats_dict   = ats_to_dict(ats_result)
            yield _event({"event": "ats", "data": ats_dict, "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})

            # Fix 3a: ATS early exit on stream path
            if ats_result.score < ATS_SKIP_THRESHOLD:
                logger.info("stream: ATS early exit score=%.0f", ats_result.score)
                yield _event({
                    "event": "early_exit",
                    "reason": "ats_formatting",
                    "detail": (
                        f"ATS score {ats_result.score:.0f}/100 is below threshold. "
                        "Fix resume formatting before running LLM analysis."
                    ),
                    "elapsed_s": round(_time.monotonic() - _pipeline_start, 1),
                })
                return

            skills_taxonomy = {}
            try:
                with open(os.path.join(BASE_DIR, "data", "skills_taxonomy.json"), "r", encoding="utf-8") as f:
                    skills_taxonomy = json.load(f)
            except Exception:
                pass

            kw = dict(api_key=api_cfg["api_key"], api_base_url=api_cfg["api_base_url"])

            # Fix 3b + Fix 4: compute similarity before dispatching judges so we can
            # skip domain-mismatch judges, and route light judges to fast model
            semantic_similarity = await asyncio.to_thread(
                resume_jd_similarity, body.resume_text, body.job_description
            )
            yield _event({"event": "semantic_similarity", "value": semantic_similarity,
                          "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})

            domain_mismatch = (semantic_similarity is not None
                               and semantic_similarity < SIMILARITY_MISMATCH_THRESHOLD)
            skip_in_stream: set[str] = {"seniority", "relevancy"} if domain_mismatch else set()
            if domain_mismatch:
                logger.info("stream: domain mismatch similarity=%.3f — skipping %s", semantic_similarity, skip_in_stream)

            def _stream_model(judge: str) -> str:
                fast = api_cfg["fast_model"]
                return fast if (fast and judge in _FAST_JUDGES) else model

            JUDGE_ORDER = [j for j in ["skills", "experience", "seniority", "relevancy", "impact"]
                           if j not in skip_in_stream]
            judge_coros = {
                "skills":     engine.run_skills_judge(body.resume_text, body.job_description, _stream_model("skills"), skills_taxonomy, **kw),
                "experience": engine.run_experience_judge(body.resume_text, body.job_description, _stream_model("experience"), **kw),
                "seniority":  engine.run_seniority_judge(body.resume_text, body.job_description, _stream_model("seniority"), **kw),
                "relevancy":  engine.run_relevancy_judge(body.resume_text, body.job_description, _stream_model("relevancy"), **kw),
                "impact":     engine.run_impact_judge(body.resume_text, _stream_model("impact"), **kw),
            }
            # Remove skipped judges from coros so they never run
            for j in skip_in_stream:
                judge_coros.pop(j, None)
            task_to_name = {}
            judge_start_times = {}
            pending = set()
            for name in JUDGE_ORDER:
                t = _asyncio.create_task(judge_coros[name])
                task_to_name[t] = name
                judge_start_times[name] = _time.monotonic()
                pending.add(t)
                yield _event({"event": "judge_start", "judge": name, "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})
            await _asyncio.sleep(0)

            SCORE_FNS = {
                "skills":     calculate_skills_score,
                "experience": calculate_experience_score,
                "seniority":  calculate_seniority_score,
                "relevancy":  calculate_relevancy_score,
                "impact":     calculate_impact_score,
            }
            judge_results = {}
            all_scores    = {}

            while pending:
                done, pending = await _asyncio.wait(pending, return_when=_asyncio.FIRST_COMPLETED, timeout=10.0)
                if not done:
                    yield _event({"event": "tick", "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})
                    await _asyncio.sleep(0)
                    continue
                for task in done:
                    name = task_to_name[task]
                    judge_elapsed = round(_time.monotonic() - judge_start_times[name], 1)
                    try:
                        result = task.result()
                    except Exception as e:
                        result = {"success": False, "data": {}, "error": str(e)}
                    judge_results[name] = result
                    scores = SCORE_FNS[name](result.get("data", {}))
                    all_scores[name] = scores
                    yield _event({
                        "event":      "judge",
                        "judge":      name,
                        "score":      round(scores.get("score", 0) * 100, 1),
                        "success":    result.get("success", False),
                        "elapsed_s":  judge_elapsed,
                        "pipeline_elapsed_s": round(_time.monotonic() - _pipeline_start, 1),
                    })
                    await _asyncio.sleep(0)

            # semantic_similarity already computed before judges started (for adaptive routing)
            logger.info("stream semantic_similarity=%.3f", semantic_similarity or 0)

            yield _event({"event": "coach_start", "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})
            await _asyncio.sleep(0)
            s   = all_scores.get("skills", {})
            e   = all_scores.get("experience", {})
            sr  = all_scores.get("seniority", {})
            rel = all_scores.get("relevancy", {})
            imp = all_scores.get("impact", {})

            coach_task = _asyncio.create_task(engine.run_coach_judge(
                resume_text=body.resume_text,
                job_description=body.job_description,
                missing_skills=s.get("missing_skills", []),
                unmet_responsibilities=e.get("unmet_responsibilities", []),
                absent_seniority_signals=sr.get("absent_signals", []),
                irrelevant_statements=rel.get("irrelevant_statements", []),
                unquantified_statements=imp.get("unquantified_statements", []),
                llm_model=model, **kw,
            ))
            while not coach_task.done():
                try:
                    await _asyncio.wait_for(_asyncio.shield(coach_task), timeout=10.0)
                except _asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    await _asyncio.sleep(0)
            coach_result = coach_task.result()

            skills_succeeded = judge_results.get("skills", {}).get("success", False)
            learned = compute_learned_weights()
            overall = (
                calculate_overall_match(s, e, sr, rel, imp, learned["weights"], skills_gate_enabled=skills_succeeded)
                if learned["source"] == "learned"
                else calculate_overall_match(s, e, sr, rel, imp, skills_gate_enabled=skills_succeeded)
            )

            pipeline_result = {
                "overall":    overall,
                "skills":     {"analyser_result": judge_results.get("skills", {}),     "scores": s},
                "experience": {"analyser_result": judge_results.get("experience", {}), "scores": e},
                "seniority":  {"analyser_result": judge_results.get("seniority", {}),  "scores": sr},
                "relevancy":  {"analyser_result": judge_results.get("relevancy", {}),  "scores": rel},
                "impact":     {"analyser_result": judge_results.get("impact", {}),     "scores": imp},
                "coach":      coach_result,
                "ats_preflight": ats_dict,
            }

            result_id = f"analysis_{timestamp.replace(':', '-').replace('.', '-')}"
            try:
                await asyncio.to_thread(
                    run_db.save_run,
                    result_id, model, body.resume_text, body.job_description,
                    pipeline_result, ats_dict, body.candidate_name, body.job_title,
                    current_user["user_id"],
                )
            except Exception as db_err:
                logger.warning("SQLite save failed (non-fatal): %s", db_err)

            coach_recs = None
            if coach_result.get("success") and body.use_coach:
                coach_recs = coach_result.get("data", {}).get("improvement_analysis", [])

            stream_judge_scores = {
                "skills":     s.get("score"),
                "relevancy":  rel.get("score"),
                "experience": e.get("score"),
                "impact":     imp.get("score"),
            }
            suspect_scores = flag_suspect_scores(stream_judge_scores, semantic_similarity)

            yield _event({
                "event":       "complete",
                "match_score": round(overall.get("overall_score", 0) * 100, 1),
                "total_elapsed_s": round(_time.monotonic() - _pipeline_start, 1),
                "metrics": {
                    "skills_coverage":       round(s.get("score", 0) * 100, 1),
                    "experience_depth":      round(e.get("score", 0) * 100, 1),
                    "seniority_calibration": round(sr.get("score", 0) * 100, 1),
                    "statement_relevancy":   round(rel.get("score", 0) * 100, 1),
                    "impact_quantification": round(imp.get("score", 0) * 100, 1),
                },
                "analysis": {
                    "missing_critical_skills":   s.get("missing_skills", []),
                    "unmet_responsibilities":     e.get("unmet_responsibilities", []),
                    "absent_seniority_signals":   sr.get("absent_signals", []),
                    "irrelevant_statements":      rel.get("irrelevant_statements", []),
                    "unquantified_statements":    imp.get("unquantified_statements", []),
                    "coach_recommendations":      coach_recs,
                },
                "pipeline_data": pipeline_result,
                "ats_preflight": ats_dict,
                "semantic_similarity": semantic_similarity,
                "suspect_scores": suspect_scores,
                "metadata": {
                    "model_used":       model,
                    "timestamp":        timestamp,
                    "result_id":        result_id,
                    "pipeline_version": "5-stage-v2.0",
                },
            })

        except Exception as e:
            yield _event({"event": "error", "detail": str(e)})
        finally:
            _analyze_sem.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /list_results ─────────────────────────────────────────────────────────────

@router.get("/list_results")
@limiter.limit("60/minute")
async def list_results(request: Request, limit: int = 50, offset: int = 0,
                       current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    resume_lookup: dict[str, tuple[str, str]] = {}
    job_lookup: dict[str, str] = {}
    try:
        for r in await asyncio.to_thread(run_db.get_resumes, uid):
            prefix = (r.get("text") or "")[:200].strip()
            if prefix:
                resume_lookup[prefix] = (r.get("name", ""), r.get("title", ""))
    except Exception:
        pass
    try:
        for j in await asyncio.to_thread(run_db.get_jobs, uid):
            prefix = (j.get("description") or "")[:200].strip()
            if prefix:
                label = j.get("title", "")
                company = j.get("company", "")
                if company:
                    label = f"{label} – {company}"
                job_lookup[prefix] = label
    except Exception:
        pass

    def match_resume(text: str) -> str:
        if not text:
            return ""
        needle = text[:200].strip()
        for prefix, (name, title) in resume_lookup.items():
            if needle[:len(prefix)] == prefix or prefix[:len(needle)] == needle:
                return name or title or ""
        import re as _re
        _loc_re = _re.compile(r',\s*[A-Z]|\d{5}|@|http|www\.|linkedin\.com', _re.IGNORECASE)
        for line in text.splitlines():
            line = line.strip()
            if 2 <= len(line.split()) <= 5 and not _loc_re.search(line) and line[0].isupper():
                return line
        return ""

    def match_job(desc: str) -> str:
        if not desc:
            return ""
        needle = desc[:200].strip()
        for prefix, label in job_lookup.items():
            if needle[:len(prefix)] == prefix or prefix[:len(needle)] == needle:
                return label
        for line in desc.splitlines():
            line = line.strip()
            if line and not line.startswith("(window.") and not line.startswith("{"):
                return line[:80]
        return ""

    try:
        db_rows = await asyncio.to_thread(run_db.list_runs, uid, limit, offset)
        for row in db_rows:
            if not row.get("resume_name"):
                row["resume_name"] = match_resume(row.get("resume_snippet", ""))
            if not row.get("job_title"):
                row["job_title"] = match_job(row.get("job_snippet", ""))
        return db_rows
    except Exception as db_err:
        logger.warning("SQLite list_runs failed, falling back to JSON files: %s", db_err)

    pipeline_dir = os.path.join(BASE_DIR, "results", "pipeline")
    results = []
    try:
        files = sorted(
            [f for f in os.listdir(pipeline_dir) if f.endswith(".json")],
            reverse=True
        )
        for fname in files:
            fpath = os.path.join(pipeline_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pr = data.get("pipeline_result", {})
                overall = pr.get("overall", {})
                component_scores = overall.get("component_scores", {})
                coach = pr.get("coach", {})
                coach_recs = []
                if coach.get("success") and coach.get("data"):
                    coach_recs = coach["data"].get("improvement_analysis", [])
                skills_data    = pr.get("skills", {}).get("analyser_result", {}).get("data", {})
                exp_data       = pr.get("experience", {}).get("analyser_result", {}).get("data", {})
                seniority_data = pr.get("seniority", {}).get("analyser_result", {}).get("data", {})
                relevancy_data = pr.get("relevancy", {}).get("analyser_result", {}).get("data", {})
                impact_data    = pr.get("impact", {}).get("analyser_result", {}).get("data", {})
                resume_text    = data.get("resume_text", "")
                job_desc       = data.get("job_description", "")
                results.append({
                    "result_id":    data.get("result_id", fname.replace(".json", "")),
                    "timestamp":    data.get("timestamp", ""),
                    "model":        data.get("model", ""),
                    "resume_name":  match_resume(resume_text),
                    "job_title":    match_job(job_desc),
                    "resume_snippet": resume_text[:120],
                    "job_snippet":  job_desc[:120],
                    "match_score":  round(overall.get("overall_score", 0) * 100, 1),
                    "metrics": {
                        "skills_coverage":       round(component_scores.get("skills", 0) * 100, 1),
                        "experience_depth":      round(component_scores.get("experience", 0) * 100, 1),
                        "seniority_calibration": round(component_scores.get("seniority", 0) * 100, 1),
                        "statement_relevancy":   round(component_scores.get("relevancy", 0) * 100, 1),
                        "impact_quantification": round(component_scores.get("impact", 0) * 100, 1),
                    },
                    "analysis": {
                        "missing_critical_skills":  skills_data.get("missing_critical_skills", []),
                        "unmet_responsibilities":    exp_data.get("unmet_responsibilities", []),
                        "absent_seniority_signals":  seniority_data.get("absent_signals", []),
                        "irrelevant_statements":     relevancy_data.get("irrelevant_statements", []),
                        "unquantified_statements":   impact_data.get("unquantified_statements", []),
                        "coach_recommendations":     coach_recs,
                    },
                    "pipeline_data": {
                        "skills":     pr.get("skills", {}),
                        "experience": pr.get("experience", {}),
                        "seniority":  pr.get("seniority", {}),
                        "relevancy":  pr.get("relevancy", {}),
                        "impact":     pr.get("impact", {}),
                    }
                })
            except Exception as e:
                logger.warning("Could not parse result file %s: %s", fname, e)
    except FileNotFoundError:
        pass
    return results


# ── Summarise endpoints ───────────────────────────────────────────────────────

@router.post("/summarize_batch")
@limiter.limit("10/minute")
async def summarize_batch(request: Request, body: BatchSummaryRequest):
    prompt = f"""You are an executive HR Director. Review this batch of AI resume-to-job match scores:
{body.batch_data}

Write a 3-4 sentence executive summary. Identify the top candidate/job pairing, note any major trends, and be decisive.

IMPORTANT: Output plain text only. No HTML, no markdown, no formatting. Just plain text."""
    result = await execute_llm_call(prompt, body.model)
    try:
        summary_text = result["data"]
        if "<!DOCTYPE" in summary_text or "<html" in summary_text:
            import re as _re
            summary_text = _re.sub(r'<[^>]+>', '', summary_text)
            summary_text = ' '.join(summary_text.split())
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize_candidate")
@limiter.limit("10/minute")
async def summarize_candidate(request: Request, body: CandidateSummaryRequest):
    prompt = f"""Act as an expert Technical Recruiter. Write a punchy, 2-3 sentence summary of this specific candidate's fit for the role.

Candidate: {body.resume_name}
Role: {body.job_title}
Overall Match Score: {body.score}%
Skills Match: {body.metrics.get('skills_coverage', 'N/A')}%
Experience Match: {body.metrics.get('experience_depth', 'N/A')}%
Critical Missing Skills: {', '.join(body.missing_skills) if body.missing_skills else 'None'}

Highlight their strongest selling point and clearly state their biggest gap. Be objective, professional, and concise. Output plain text only."""
    result = await execute_llm_call(prompt, body.model)
    try:
        summary_text = result["data"]
        if "<!DOCTYPE" in summary_text or "<html" in summary_text:
            import re as _re
            summary_text = _re.sub(r'<[^>]+>', '', summary_text)
            summary_text = ' '.join(summary_text.split())
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize_result")
@limiter.limit("10/minute")
async def summarize_result(request: Request, req: SummarizeResultRequest):
    run = await asyncio.to_thread(run_db.get_run, req.result_id)
    if not run:
        raise HTTPException(status_code=404, detail="Result not found")
    pr = run.get("pipeline_result", {})
    overall = pr.get("overall", {})
    score = round((overall.get("overall_score") or 0) * 100, 1)
    coach = pr.get("coach", {})
    coach_recs = []
    if coach.get("success") and coach.get("data"):
        coach_recs = coach["data"].get("improvement_analysis", [])[:3]
    system = "You write concise candidate match summaries for recruiters. Two sentences max."
    prompt = f"""Summarize this resume analysis in 2 sentences:
Overall match: {score}%
Top coach notes: {'; '.join(r.get('recommendation','') for r in coach_recs if r.get('recommendation'))}
Resume snippet: {(run.get('resume_text') or '')[:400]}
Job snippet: {(run.get('job_description') or '')[:300]}"""
    try:
        from specialized_judges import call_ollama_async
        result = await call_ollama_async(system, prompt, req.model, temperature=0.3)
        if result.get("success"):
            return {"summary": result["data"].strip()}
    except Exception:
        pass
    return {"summary": f"Overall match score: {score}%. Review individual judge scores for details."}


# ── Outcome tracking ──────────────────────────────────────────────────────────

@router.post("/record_outcome")
@limiter.limit("30/minute")
async def record_outcome(request: Request, req: RecordOutcomeRequest):
    now = datetime.utcnow().isoformat()
    await asyncio.to_thread(
        run_db.save_outcome,
        req.result_id, req.predicted_score, req.got_job, req.got_interview,
        req.interview_stage, req.rejection_reason,
    )

    run_record = await asyncio.to_thread(run_db.get_run, req.result_id)
    if run_record:
        pr = run_record.get("pipeline_result", {})
    else:
        pipeline_file = os.path.join(BASE_DIR, "results", "pipeline", f"{req.result_id}.json")
        pr = {}
        if os.path.isfile(pipeline_file):
            with open(pipeline_file, "r", encoding="utf-8") as pf:
                pr = json.load(pf).get("pipeline_result", {})

    if pr:
        judge_scores = {
            "skills":     pr.get("skills", {}).get("scores", {}).get("score"),
            "experience": pr.get("experience", {}).get("scores", {}).get("score"),
            "seniority":  pr.get("seniority", {}).get("scores", {}).get("score"),
            "relevancy":  pr.get("relevancy", {}).get("scores", {}).get("score"),
            "impact":     pr.get("impact", {}).get("scores", {}).get("score"),
            "overall":    pr.get("overall", {}).get("overall_score"),
        }
        feedback_entries = []
        if req.got_interview is not None:
            for j in INTERVIEW_JUDGES:
                score = judge_scores.get(j)
                if score is None:
                    continue
                entry = {
                    "judge": j, "signal": "got_interview",
                    "predicted_score": score, "outcome": req.got_interview,
                    "correct": (score >= 0.5) == req.got_interview,
                    "timestamp": now,
                }
                if req.interview_stage:
                    entry["interview_stage"] = req.interview_stage
                if req.rejection_reason:
                    entry["rejection_reason"] = req.rejection_reason
                feedback_entries.append(entry)
        if req.got_job is not None:
            for j in JOB_JUDGES:
                score = judge_scores.get(j)
                if score is None:
                    continue
                entry = {
                    "judge": j, "signal": "got_job",
                    "predicted_score": score, "outcome": req.got_job,
                    "correct": (score >= 0.6) == req.got_job,
                    "timestamp": now,
                }
                if req.interview_stage:
                    entry["interview_stage"] = req.interview_stage
                if req.rejection_reason:
                    entry["rejection_reason"] = req.rejection_reason
                feedback_entries.append(entry)
        if feedback_entries:
            append_feedback(feedback_entries)

    return {"success": True}


@router.get("/outcomes")
@limiter.limit("60/minute")
async def get_outcomes(request: Request):
    outcomes = await asyncio.to_thread(run_db.get_outcomes_all)

    if not outcomes:
        return {"outcomes": [], "accuracy": None, "interview_accuracy": None, "total": 0}

    job_o = [o for o in outcomes if o.get("got_job") is not None]
    int_o = [o for o in outcomes if o.get("got_interview") is not None]
    correct_job = sum(1 for o in job_o if (o["predicted_score"] >= 60) == o["got_job"])
    correct_int = sum(1 for o in int_o if (o["predicted_score"] >= 50) == o["got_interview"])

    return {
        "outcomes":           outcomes,
        "accuracy":           round(correct_job / len(job_o) * 100, 1) if job_o else None,
        "interview_accuracy": round(correct_int / len(int_o) * 100, 1) if int_o else None,
        "total":              len(outcomes),
        "total_job":          len(job_o),
        "total_interview":    len(int_o),
        "correct":            correct_job,
        "correct_interview":  correct_int,
    }


# ── Results management ────────────────────────────────────────────────────────

@router.get("/results/{result_id}")
@limiter.limit("60/minute")
async def get_result(request: Request, result_id: str, current_user: dict = Depends(get_current_user)):
    record = await asyncio.to_thread(run_db.get_run, result_id, current_user["user_id"])
    if not record:
        raise HTTPException(status_code=404, detail=f"Result {result_id!r} not found")
    pr = record.get("pipeline_result", {})
    return {
        "result_id":    result_id,
        "pipeline_data": pr,
        "ats_preflight": pr.get("ats_preflight", {}),
    }


@router.delete("/results/{result_id}")
@limiter.limit("30/minute")
async def delete_result(request: Request, result_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    def _delete(conn):
        conn.execute("DELETE FROM runs WHERE id=? AND user_id=?", (result_id, uid))
    await asyncio.to_thread(run_db._get_pool().execute_tx, _delete)
    return {"success": True}


@router.post("/results/{result_id}/evals")
@limiter.limit("10/minute")
async def run_result_evals(request: Request, result_id: str, req: RunEvalsRequest):
    record = await asyncio.to_thread(run_db.get_run, result_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Result {result_id!r} not found")

    resume_text     = record.get("resume_text", "")
    job_description = record.get("job_description", "")
    pipeline_result = record.get("pipeline_result", {})

    if not resume_text or not pipeline_result:
        raise HTTPException(status_code=422, detail="Stored result has no resume text or pipeline data")

    model = req.model or (pipeline_result.get("metadata", {}).get("model_used") or _analyser_cfg.default_model)

    try:
        eval_results = await judge_engine.run_evals(
            pipeline_result=pipeline_result,
            resume_text=resume_text,
            job_description=job_description,
            llm_model=model,
            api_key=req.api_key,
            api_base_url=req.api_base_url,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result_id": result_id, "model": model, "evals": eval_results}


# ── Calibration / weights ─────────────────────────────────────────────────────

@router.get("/judge_calibration")
@limiter.limit("60/minute")
async def judge_calibration(request: Request):
    from analysers.weights import FEEDBACK_FILE
    feedback = []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feedback.append(json.loads(line))
    except FileNotFoundError:
        pass

    if not feedback:
        return {"calibration": {}, "total_feedback": 0}

    by_judge = defaultdict(list)
    for fb in feedback:
        by_judge[fb["judge"]].append(fb)

    calibration = {}
    for judge, entries in by_judge.items():
        correct = sum(1 for e in entries if e["correct"])
        pos = [e for e in entries if e["outcome"]]
        neg = [e for e in entries if not e["outcome"]]
        calibration[judge] = {
            "total":        len(entries),
            "correct":      correct,
            "accuracy":     round(correct / len(entries) * 100, 1),
            "signal":       entries[-1].get("signal", ""),
            "avg_score_pos": round(sum(e["predicted_score"] for e in pos) / len(pos), 3) if pos else None,
            "avg_score_neg": round(sum(e["predicted_score"] for e in neg) / len(neg), 3) if neg else None,
        }

    return {"calibration": calibration, "total_feedback": len(feedback)}


@router.get("/judge_weights")
@limiter.limit("60/minute")
async def judge_weights(request: Request):
    return compute_learned_weights()
