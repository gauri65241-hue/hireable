import asyncio
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ratelimit import limiter
from concurrency import run_blocking
from pydantic import BaseModel, Field

from auth_utils import get_current_user

from analysers.base import prepare_user_text
from analysers.engine import PIPELINE_VERSION, pipeline_status
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
    execute_llm_call, get_api_config, make_cache_key,
)
from db import get_cached_run_id, set_cached_run

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
    force_rerun: bool = False

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

RESULT_ID_PATTERN = r"^analysis_[0-9A-Za-z_\-]{1,64}$"


class SummarizeResultRequest(BaseModel):
    result_id: str = Field(pattern=RESULT_ID_PATTERN)
    model: str = "cogito:14b"

class RecordOutcomeRequest(BaseModel):
    result_id: str = Field(pattern=RESULT_ID_PATTERN)
    # Ignored: the score is read from the stored run. (A client-supplied score let anyone
    # write arbitrary "predictions" into the accuracy stats that feed learned weights.)
    predicted_score: Optional[float] = None
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
        # Normalise + neutralise instruction-like text (flag, don't reject, don't leak rules).
        clean_resume, resume_flags = prepare_user_text(body.resume_text, "resume")
        clean_jd, jd_flags = prepare_user_text(body.job_description, "job_description")
    except ValueError as inj_err:
        raise HTTPException(status_code=422, detail=str(inj_err))
    security_flags = resume_flags + jd_flags
    body = body.model_copy(update={"resume_text": clean_resume, "job_description": clean_jd})

    api_cfg = get_api_config(body.api_key, body.api_base_url, body.llm_models[0])   # validates api_base_url
    model = api_cfg["llm_model"]
    learned = await run_blocking(compute_learned_weights, timeout=10.0)
    weights = learned["weights"] if learned["source"] == "learned" else None
    cache_key = make_cache_key(current_user["user_id"], body.resume_text, body.job_description,
                               model, weights)
    if not body.force_rerun:
        cached_id = await asyncio.to_thread(get_cached_run_id, cache_key)
        if cached_id:
            run = await asyncio.to_thread(run_db.get_run, cached_id, current_user["user_id"])
            if run:
                pr = run.get("pipeline_result", {})
                ov = pr.get("overall", {})
                cs = ov.get("component_scores", {})
                s  = pr.get("skills", {}).get("scores", {})
                e  = pr.get("experience", {}).get("scores", {})
                sr = pr.get("seniority", {}).get("scores", {})
                rel= pr.get("relevancy", {}).get("scores", {})
                imp= pr.get("impact", {}).get("scores", {})
                coach = pr.get("coach", {})
                coach_recs = coach.get("data", {}).get("improvement_analysis", []) if coach.get("success") and body.use_coach else None
                return {
                    "success": True,
                    "match_score": round(ov.get("overall_score", 0) * 100, 1),
                    "metrics": {
                        "skills_coverage":       round((cs.get("skills")     or 0) * 100, 1),
                        "experience_depth":      round((cs.get("experience") or 0) * 100, 1),
                        "seniority_calibration": round((cs.get("seniority")  or 0) * 100, 1),
                        "statement_relevancy":   round((cs.get("relevancy")  or 0) * 100, 1),
                        "impact_quantification": round((cs.get("impact")     or 0) * 100, 1),
                    },
                    "analysis": {
                        "missing_critical_skills":  s.get("missing_skills", []),
                        "unmet_responsibilities":   e.get("unmet_responsibilities", []),
                        "absent_seniority_signals": sr.get("absent_signals", []),
                        "irrelevant_statements":    rel.get("irrelevant_statements", []),
                        "unquantified_statements":  imp.get("unquantified_statements", []),
                        "coach_recommendations":    coach_recs,
                    },
                    "pipeline_data": pr,
                    "ats_preflight": pr.get("ats_preflight", {}),
                    "semantic_similarity": None,
                    "suspect_scores": [],
                    "metadata": {
                        "model_used":       run.get("model", ""),
                        "timestamp":        run.get("created_at", ""),
                        "result_id":        cached_id,
                        "pipeline_version": ov.get("pipeline_version") or PIPELINE_VERSION,
                        "from_cache":       True,
                    },
                }

    try:
        await asyncio.wait_for(_analyze_sem.acquire(), timeout=_analyser_cfg.sem_acquire_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server at capacity, try again shortly",
                            headers={"Retry-After": "30"})
    try:
        job_description = body.job_description
        timestamp = datetime.utcnow().isoformat()
        logger.info("analyze start model=%s api=%s resume=%dch jd=%dch",
                    model, "openrouter" if api_cfg["api_key"] else "ollama",
                    len(body.resume_text), len(job_description))

        ats_result = await run_blocking(score_ats, body.resume_text, job_description, timeout=20.0)
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
            weights=weights,
            user_id=current_user["user_id"],
        )

        if not pipeline_result:
            raise HTTPException(status_code=500, detail="Judge pipeline failed to produce results")

        pipeline_result["ats_preflight"] = ats_dict
        pipeline_result["weights_source"] = "learned" if weights else "default"

        run_status = pipeline_status(pipeline_result.get("overall", {}))
        if run_status == "not_scored":
            raise HTTPException(status_code=422, detail={
                "error": "not_scored",
                "message": pipeline_result["overall"].get("early_exit_reason", "Resume could not be scored."),
                "ats_preflight": ats_dict,
            })
        if run_status == "insufficient":
            raise HTTPException(status_code=503, headers={"Retry-After": "30"}, detail={
                "error": "insufficient_judges",
                "message": "Too few evaluation judges completed to produce a trustworthy score. "
                           "No score was recorded; please retry.",
                "judge_statuses": pipeline_result["overall"].get("judge_statuses", {}),
            })

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
            if run_status == "complete":
                await asyncio.to_thread(set_cached_run, cache_key, result_id, current_user["user_id"])
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
            "status": run_status,
            "degraded": run_status == "degraded",
            "failed_judges": [k for k, v in overall.get("judge_statuses", {}).items()
                              if v.get("status") in ("failed", "timeout")],
            "security_flags": security_flags + [f for f in overall.get("security_flags", [])
                                                if f not in security_flags],
            "metadata": {
                "model_used":       model,
                "timestamp":        timestamp,
                "result_id":        result_id,
                "pipeline_version": PIPELINE_VERSION,
            }
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        _analyze_sem.release()


# ── /analyze/stream ───────────────────────────────────────────────────────────

@router.post("/analyze/stream")
@limiter.limit("5/minute")
async def analyze_stream(request: Request, body: AnalyzeRequest,
                         current_user: dict = Depends(get_current_user)):
    import time as _time

    body = body.model_copy(update={"job_description": _truncate_jd(body.job_description)})

    async def generate():
        def _event(obj: dict) -> str:
            payload = json.dumps(obj)
            pad = max(0, 1536 - len(payload))
            return f"data: {payload}\n: {'x' * pad}\n\n"

        try:
            clean_resume, resume_flags = prepare_user_text(body.resume_text, "resume")
            clean_jd, jd_flags = prepare_user_text(body.job_description, "job_description")
        except ValueError as inj_err:
            yield _event({"event": "error", "detail": str(inj_err), "code": 422})
            return
        security_flags = resume_flags + jd_flags
        run_body = body.model_copy(update={"resume_text": clean_resume, "job_description": clean_jd})

        try:
            api_cfg = get_api_config(run_body.api_key, run_body.api_base_url, run_body.llm_models[0])
        except HTTPException as http_err:
            yield _event({"event": "error", "detail": str(http_err.detail), "code": http_err.status_code})
            return
        model = api_cfg["llm_model"]
        learned = await run_blocking(compute_learned_weights, timeout=10.0)
        weights = learned["weights"] if learned["source"] == "learned" else None
        cache_key = make_cache_key(current_user["user_id"], run_body.resume_text,
                                   run_body.job_description, model, weights)

        # Cache check — same (user, resume, jd, model, weights, pipeline version) and a COMPLETE run
        if not body.force_rerun:
            cached_id = await asyncio.to_thread(get_cached_run_id, cache_key)
            if cached_id:
                run = await asyncio.to_thread(run_db.get_run, cached_id, current_user["user_id"])
                if run:
                    pr  = run.get("pipeline_result", {})
                    ats = pr.get("ats_preflight", {})
                    ov  = pr.get("overall", {})
                    cs  = ov.get("component_scores", {})
                    yield _event({"event": "cache_hit", "result_id": cached_id})
                    yield _event({"event": "ats", "data": ats, "elapsed_s": 0})
                    yield _event({"event": "semantic_similarity", "value": None, "elapsed_s": 0})
                    for jname in ["skills", "experience", "seniority", "relevancy", "impact"]:
                        jdata   = pr.get(jname, {})
                        jscores = jdata.get("scores", {})
                        yield _event({"event": "judge_start", "judge": jname, "elapsed_s": 0})
                        yield _event({
                            "event":    "judge",
                            "judge":    jname,
                            "score":    round(jscores.get("score", 0) * 100, 1),
                            "success":  jdata.get("analyser_result", {}).get("success", False),
                            "elapsed_s": 0,
                            "pipeline_elapsed_s": 0,
                        })
                    yield _event({"event": "coach_start", "elapsed_s": 0})
                    coach = pr.get("coach", {})
                    coach_recs = None
                    if coach.get("success") and body.use_coach:
                        coach_recs = coach.get("data", {}).get("improvement_analysis", [])
                    s   = pr.get("skills", {}).get("scores", {})
                    e   = pr.get("experience", {}).get("scores", {})
                    sr  = pr.get("seniority", {}).get("scores", {})
                    rel = pr.get("relevancy", {}).get("scores", {})
                    imp = pr.get("impact", {}).get("scores", {})
                    yield _event({
                        "event":       "complete",
                        "match_score": round(ov.get("overall_score", 0) * 100, 1),
                        "total_elapsed_s": 0,
                        "metrics": {
                            "skills_coverage":       round((cs.get("skills")     or 0) * 100, 1),
                            "experience_depth":      round((cs.get("experience") or 0) * 100, 1),
                            "seniority_calibration": round((cs.get("seniority")  or 0) * 100, 1),
                            "statement_relevancy":   round((cs.get("relevancy")  or 0) * 100, 1),
                            "impact_quantification": round((cs.get("impact")     or 0) * 100, 1),
                        },
                        "analysis": {
                            "missing_critical_skills":  s.get("missing_skills", []),
                            "unmet_responsibilities":   e.get("unmet_responsibilities", []),
                            "absent_seniority_signals": sr.get("absent_signals", []),
                            "irrelevant_statements":    rel.get("irrelevant_statements", []),
                            "unquantified_statements":  imp.get("unquantified_statements", []),
                            "coach_recommendations":    coach_recs,
                        },
                        "pipeline_data": pr,
                        "ats_preflight": ats,
                        "semantic_similarity": None,
                        "suspect_scores": [],
                        "metadata": {
                            "model_used":       run.get("model", ""),
                            "timestamp":        run.get("created_at", ""),
                            "result_id":        cached_id,
                            "pipeline_version": ov.get("pipeline_version") or PIPELINE_VERSION,
                            "from_cache":       True,
                        },
                    })
                    return

        try:
            await asyncio.wait_for(_analyze_sem.acquire(), timeout=_analyser_cfg.sem_acquire_timeout)
        except asyncio.TimeoutError:
            yield _event({"event": "error", "detail": "Server at capacity, try again shortly",
                          "code": 503, "retry_after_s": 30})
            return

        engine_task = None
        try:
            timestamp = datetime.utcnow().isoformat()
            logger.info("stream analyze start model=%s api=%s",
                        model, "openrouter" if api_cfg["api_key"] else "ollama")
            _pipeline_start = _time.monotonic()

            ats_result = await run_blocking(score_ats, run_body.resume_text, run_body.job_description, timeout=20.0)
            ats_dict   = ats_to_dict(ats_result)
            yield _event({"event": "ats", "data": ats_dict, "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})

            skills_taxonomy = {}
            try:
                with open(os.path.join(BASE_DIR, "data", "skills_taxonomy.json"), "r", encoding="utf-8") as f:
                    skills_taxonomy = json.load(f)
            except Exception:
                pass

            semantic_similarity = await asyncio.to_thread(
                resume_jd_similarity, run_body.resume_text, run_body.job_description
            )
            yield _event({"event": "semantic_similarity", "value": semantic_similarity,
                          "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})

            # ONE orchestration path: the engine (timeouts, quorum, weight renormalisation,
            # arbitration, gates) runs the judges and reports progress through a queue. The
            # previous stream handler re-implemented the pipeline without any of those
            # protections, so a failed judge counted as a real zero here.
            queue: asyncio.Queue = asyncio.Queue()
            engine_task = asyncio.create_task(judge_engine.run(
                resume_text=run_body.resume_text,
                job_description=run_body.job_description,
                llm_model=model,
                skills_taxonomy=skills_taxonomy,
                api_key=api_cfg["api_key"],
                api_base_url=api_cfg["api_base_url"],
                fast_model=api_cfg["fast_model"],
                ats_score=ats_result.score,
                semantic_similarity=semantic_similarity,
                weights=weights,
                user_id=current_user["user_id"],
                progress=queue.put_nowait,
            ))
            engine_task.add_done_callback(lambda _t: queue.put_nowait(None))

            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    yield _event({"event": "tick", "elapsed_s": round(_time.monotonic() - _pipeline_start, 1)})
                    continue
                if ev is None:
                    break
                yield _event(ev)

            try:
                pipeline_result = engine_task.result()
            except Exception:
                logger.exception("stream pipeline failed")
                yield _event({"event": "error", "detail": "Analysis failed. Please try again."})
                return

            pipeline_result["ats_preflight"] = ats_dict
            pipeline_result["weights_source"] = "learned" if weights else "default"
            overall = pipeline_result.get("overall", {})
            run_status = pipeline_status(overall)

            if run_status == "not_scored":
                yield _event({
                    "event": "early_exit", "reason": overall.get("early_exit", "unreadable_text"),
                    "detail": overall.get("early_exit_reason", "Resume could not be scored."),
                    "elapsed_s": round(_time.monotonic() - _pipeline_start, 1),
                })
                return
            if run_status == "insufficient":
                yield _event({
                    "event": "error", "code": 503, "retry_after_s": 30,
                    "detail": "Too few evaluation judges completed to produce a trustworthy score. "
                              "No score was recorded; please retry.",
                    "judge_statuses": overall.get("judge_statuses", {}),
                })
                return

            s   = pipeline_result.get("skills", {}).get("scores", {})
            e   = pipeline_result.get("experience", {}).get("scores", {})
            sr  = pipeline_result.get("seniority", {}).get("scores", {})
            rel = pipeline_result.get("relevancy", {}).get("scores", {})
            imp = pipeline_result.get("impact", {}).get("scores", {})
            coach_result = pipeline_result.get("coach", {})

            result_id = f"analysis_{timestamp.replace(':', '-').replace('.', '-')}"
            try:
                await asyncio.to_thread(
                    run_db.save_run,
                    result_id, model, run_body.resume_text, run_body.job_description,
                    pipeline_result, ats_dict, body.candidate_name, body.job_title,
                    current_user["user_id"],
                )
                if run_status == "complete":
                    await asyncio.to_thread(set_cached_run, cache_key, result_id, current_user["user_id"])
            except Exception as db_err:
                logger.warning("SQLite save failed (non-fatal): %s", db_err)

            coach_recs = None
            if coach_result.get("success") and body.use_coach:
                coach_recs = coach_result.get("data", {}).get("improvement_analysis", [])

            suspect_scores = flag_suspect_scores({
                "skills": s.get("score"), "relevancy": rel.get("score"),
                "experience": e.get("score"), "impact": imp.get("score"),
            }, semantic_similarity)

            yield _event({
                "event":       "complete",
                "match_score": round(overall.get("overall_score", 0) * 100, 1),
                "total_elapsed_s": round(_time.monotonic() - _pipeline_start, 1),
                "status":      run_status,
                "degraded":    run_status == "degraded",
                "failed_judges": [k for k, v in overall.get("judge_statuses", {}).items()
                                  if v.get("status") in ("failed", "timeout")],
                "security_flags": security_flags + [f for f in overall.get("security_flags", [])
                                                    if f not in security_flags],
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
                    "pipeline_version": PIPELINE_VERSION,
                },
            })

        except Exception:
            logger.exception("stream analyze failed")
            yield _event({"event": "error", "detail": "Analysis failed. Please try again."})
        finally:
            # Client disconnect (GeneratorExit/CancelledError) lands here too: stop the
            # pipeline so it does not keep spending LLM time for nobody. With native-async
            # LLM calls, cancellation really aborts the in-flight requests.
            if engine_task is not None and not engine_task.done():
                engine_task.cancel()
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
async def summarize_batch(request: Request, body: BatchSummaryRequest,
                          current_user: dict = Depends(get_current_user)):
    prompt = f"""You are an executive HR Director. Review this batch of AI resume-to-job match scores:
{body.batch_data}

Write a 3-4 sentence executive summary. Identify the top candidate/job pairing, note any major trends, and be decisive.

IMPORTANT: Output plain text only. No HTML, no markdown, no formatting. Just plain text."""
    from analysers.base import call_llm_async
    cfg = get_api_config()
    try:
        result = await asyncio.wait_for(
            call_llm_async("", prompt, cfg["llm_model"], temperature=0.1,
                           api_key=cfg["api_key"], api_base_url=cfg["api_base_url"]),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        return {"summary": "Summary unavailable — LLM timed out."}
    if not result.get("success"):
        return {"summary": "Summary unavailable — LLM call failed."}
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
async def summarize_candidate(request: Request, body: CandidateSummaryRequest,
                              current_user: dict = Depends(get_current_user)):
    prompt = f"""Act as an expert Technical Recruiter. Write a punchy, 2-3 sentence summary of this specific candidate's fit for the role.

Candidate: {body.resume_name}
Role: {body.job_title}
Overall Match Score: {body.score}%
Skills Match: {body.metrics.get('skills_coverage', 'N/A')}%
Experience Match: {body.metrics.get('experience_depth', 'N/A')}%
Critical Missing Skills: {', '.join(body.missing_skills) if body.missing_skills else 'None'}

Highlight their strongest selling point and clearly state their biggest gap. Be objective, professional, and concise. Output plain text only."""
    from analysers.base import call_llm_async
    cfg = get_api_config()
    try:
        result = await asyncio.wait_for(
            call_llm_async("", prompt, cfg["llm_model"], temperature=0.1,
                           api_key=cfg["api_key"], api_base_url=cfg["api_base_url"]),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        return {"summary": "Summary unavailable — LLM timed out."}
    if not result.get("success"):
        return {"summary": "Summary unavailable — LLM call failed."}
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
async def summarize_result(request: Request, req: SummarizeResultRequest,
                           current_user: dict = Depends(get_current_user)):
    run = await asyncio.to_thread(run_db.get_run, req.result_id, current_user["user_id"])
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
        from analysers.base import call_llm_async
        cfg = get_api_config()
        result = await asyncio.wait_for(
            call_llm_async(system, prompt, cfg["llm_model"], temperature=0.3,
                           api_key=cfg["api_key"], api_base_url=cfg["api_base_url"]),
            timeout=20.0,
        )
        if result.get("success"):
            return {"summary": result["data"].strip()}
    except Exception:
        pass
    return {"summary": f"Overall match score: {score}%. Review individual judge scores for details."}


# ── Outcome tracking ──────────────────────────────────────────────────────────

@router.post("/record_outcome")
@limiter.limit("30/minute")
async def record_outcome(request: Request, req: RecordOutcomeRequest,
                         current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    # Ownership: only the run's owner may record an outcome for it (this also makes the
    # lookup user-scoped; the old unscoped get_run(id) matched nothing for real users, so
    # the learning loop silently never received feedback).
    run_record = await asyncio.to_thread(run_db.get_run, req.result_id, uid)
    if not run_record:
        raise HTTPException(status_code=404, detail="Result not found")
    pr = run_record.get("pipeline_result", {})
    overall = pr.get("overall", {})
    stored_pct = round((overall.get("overall_score") or 0) * 100, 1)
    now = datetime.utcnow().isoformat()

    created = await asyncio.to_thread(
        run_db.save_outcome,
        req.result_id, stored_pct, req.got_job, req.got_interview,
        req.interview_stage, req.rejection_reason, uid,
    )

    # Only a NEW outcome on a COMPLETE run trains the judges, and only judges that actually
    # produced a score. Re-submits and degraded runs would otherwise inflate or poison the
    # weights/calibration that feed every future prompt.
    statuses = overall.get("judge_statuses", {})
    run_ok = overall.get("status", "complete") == "complete"
    if created and run_ok and pr:
        def _score(name: str):
            if statuses.get(name, {}).get("status", "success") != "success":
                return None
            if name == "overall":
                return overall.get("overall_score")
            return pr.get(name, {}).get("scores", {}).get("score")

        feedback_entries = []
        def _entries(judges, signal, outcome, threshold):
            for j in judges:
                score = _score(j)
                if score is None:
                    continue
                entry = {
                    "judge": j, "signal": signal, "predicted_score": score, "outcome": outcome,
                    "correct": (score >= threshold) == outcome, "timestamp": now,
                    "result_id": req.result_id, "user_id": uid,
                }
                if req.interview_stage:
                    entry["interview_stage"] = req.interview_stage
                if req.rejection_reason:
                    entry["rejection_reason"] = req.rejection_reason
                feedback_entries.append(entry)

        if req.got_interview is not None:
            _entries(INTERVIEW_JUDGES, "got_interview", req.got_interview, 0.5)
        if req.got_job is not None:
            _entries(JOB_JUDGES, "got_job", req.got_job, 0.6)
        if feedback_entries:
            await run_blocking(append_feedback, feedback_entries, timeout=10.0)

    return {"success": True, "recorded": bool(created)}


@router.get("/outcomes")
@limiter.limit("60/minute")
async def get_outcomes(request: Request, current_user: dict = Depends(get_current_user)):
    outcomes = await asyncio.to_thread(run_db.get_outcomes_all, current_user["user_id"])

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
async def run_result_evals(request: Request, result_id: str, req: RunEvalsRequest,
                           current_user: dict = Depends(get_current_user)):
    record = await asyncio.to_thread(run_db.get_run, result_id, current_user["user_id"])
    if not record:
        raise HTTPException(status_code=404, detail=f"Result {result_id!r} not found")

    resume_text     = record.get("resume_text", "")
    job_description = record.get("job_description", "")
    pipeline_result = record.get("pipeline_result", {})

    if not resume_text or not pipeline_result:
        raise HTTPException(status_code=422, detail="Stored result has no resume text or pipeline data")

    model = req.model or (pipeline_result.get("metadata", {}).get("model_used") or _analyser_cfg.default_model)
    ev_cfg = get_api_config(req.api_key, req.api_base_url, model)      # validates api_base_url

    try:
        eval_results = await judge_engine.run_evals(
            pipeline_result=pipeline_result,
            resume_text=resume_text,
            job_description=job_description,
            llm_model=model,
            api_key=ev_cfg["api_key"],
            api_base_url=ev_cfg["api_base_url"],
        )
    except Exception:
        logger.exception("run_evals failed")
        raise HTTPException(status_code=500, detail="Evaluation failed. Please try again.")

    return {"result_id": result_id, "model": model, "evals": eval_results}


# ── Calibration / weights ─────────────────────────────────────────────────────

@router.get("/judge_calibration")
@limiter.limit("60/minute")
async def judge_calibration(request: Request, current_user: dict = Depends(get_current_user)):
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
async def judge_weights(request: Request, current_user: dict = Depends(get_current_user)):
    return await run_blocking(compute_learned_weights, timeout=10.0)
