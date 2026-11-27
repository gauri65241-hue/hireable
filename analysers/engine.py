"""
AnalyserEngine — orchestrates the 5+1 analyser pipeline.
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger("AnalyserEngine")

from analysers.config import cfg

# Bump on any change to prompts, scoring, gates or weights: it is part of the result-cache
# key, so stale results are never served across a behaviour change.
PIPELINE_VERSION = "2.1.0"

_SKILLS_GATE_THRESHOLD = cfg.skills_gate_threshold
DEFAULT_WEIGHTS = cfg.default_weights

# Per-judge wall-clock timeout. With a local 14B model and up to 3 retry
# attempts, each attempt takes ~30s → 90s per judge is tight when Ollama
# queues requests. 300s gives headroom for serial processing + retries.
# Override with ANALYSER_JUDGE_TIMEOUT env var.
JUDGE_TIMEOUT_S = float(os.getenv("ANALYSER_JUDGE_TIMEOUT", "600"))
# Minimum judges that must succeed to produce a meaningful overall score.
JUDGE_QUORUM = 3

# Adaptive routing thresholds (Fix 3)
# ATS score below this → resume formatting is broken, skip all LLM judges
ATS_SKIP_THRESHOLD = 40.0
# SBERT similarity below this → complete domain mismatch, skip seniority + relevancy
SIMILARITY_MISMATCH_THRESHOLD = 0.15

# Judges that can use a smaller fast model (Fix 4)
# Seniority: binary signal detection. Relevancy: sentence classification. Impact: metric pattern matching.
_FAST_JUDGES = {"seniority", "relevancy", "impact"}

# Cross-judge contradiction pairs. Each entry defines a condition on the score map
# and which judge to re-run with an explicit arbitration context injected into its prompt.
_CONTRADICTION_PAIRS = [
    {
        "name": "skills_high_experience_low",
        "condition": lambda s: s.get("skills", 0) >= 0.75 and s.get("experience", 0) <= 0.25,
        "rerun": "experience",
        "msg": (
            "ARBITRATION CONTEXT: The skills and experience judges strongly disagree on this "
            "candidate (skills 75%+, experience 25% or below). Re-evaluate experience strictly "
            "against the resume text: for each responsibility either quote exact supporting text "
            "or mark it unmet. Do not infer experience that is not written. If your first "
            "judgement was correct, keep it."
        ),
    },
    {
        "name": "experience_high_skills_low",
        "condition": lambda s: s.get("experience", 0) >= 0.70 and s.get("skills", 0) <= 0.25,
        "rerun": "skills",
        "msg": (
            "ARBITRATION CONTEXT: The experience and skills judges strongly disagree on this "
            "candidate (experience 70%+, skills 25% or below). Re-evaluate skills strictly: for "
            "each required skill either quote exact resume text or mark it MISSING. Do not infer a "
            "skill from a job title or from general experience depth. Keep your original "
            "judgement wherever you cannot quote evidence."
        ),
    },
    {
        "name": "relevancy_high_seniority_low",
        "condition": lambda s: s.get("relevancy", 0) >= 0.80 and s.get("seniority", 0) <= 0.15,
        "rerun": "seniority",
        "msg": (
            "ARBITRATION CONTEXT: The relevancy and seniority judges strongly disagree on this "
            "candidate (relevancy 80%+, seniority 15% or below). Re-check the seniority signals "
            "(scale, ownership, complexity, leadership) against the resume text: demonstrate each "
            "with an exact quote or mark it absent. Do not assume signals that are not written. "
            "If your first judgement was correct, keep it."
        ),
    },
]


def calculate_overall(s: dict, e: dict, sr: dict, rel: dict, imp: dict,
                      weights: dict | None = None, skills_gate_enabled: bool = True) -> dict:
    w = weights or DEFAULT_WEIGHTS
    sk = s["score"]
    overall = (
        sk       * w["skills"]
        + e["score"]   * w["experience"]
        + sr["score"]  * w["seniority"]
        + rel["score"] * w["relevancy"]
        + imp["score"] * w["impact"]
    )
    gate = False
    if skills_gate_enabled:
        if sk < _SKILLS_GATE_THRESHOLD:
            cap = sk * 0.4
        elif cfg.skills_gate_ramp_upper and sk < cfg.skills_gate_ramp_upper:
            # Optional continuous ramp (off by default). The hard gate alone is a cliff:
            # skills 0.39 caps overall at 0.156 while skills 0.41 is uncapped (~0.8 if the
            # other four dimensions are strong). The ramp rises linearly from the gate's cap
            # at the threshold to 1.0 at `skills_gate_ramp_upper`.
            lo = _SKILLS_GATE_THRESHOLD * 0.4
            frac = (sk - _SKILLS_GATE_THRESHOLD) / (cfg.skills_gate_ramp_upper - _SKILLS_GATE_THRESHOLD)
            cap = lo + frac * (1.0 - lo)
        else:
            cap = None
        if cap is not None and overall > cap:
            overall = cap
            gate = True
    # NOTE: the former "low_fit_cap" (0.179) was unreachable whenever the gate was on
    # (sk < 0.25 already implies a gate cap <= 0.10) and operated on placeholder zeros when
    # the gate was off, so it was removed. The key is kept for response compatibility.
    return {
        "overall_score": round(overall, 3),
        "weights_used": w,
        "skills_gate_applied": gate,
        "low_fit_cap_applied": False,
        "component_scores": {
            "skills": sk, "experience": e["score"], "seniority": sr["score"],
            "relevancy": rel["score"], "impact": imp["score"],
        },
    }


def _safe(result, name: str) -> dict:
    if isinstance(result, BaseException):
        logger.error("Analyser %s raised unhandled exception: %s", name, result)
        return {"success": False, "error": str(result), "data": {}}
    return result


async def _run_judge_timed(name: str, coro, score_fn, on_done=None) -> tuple[str, dict, dict, str | None, int]:
    """
    Wraps a judge coroutine with a hard timeout and structured error envelope.
    Returns (status, raw_result, score_dict, error_msg, elapsed_ms).
    status: "success" | "failed" | "timeout"

    Callers must check status before trusting score_dict — a failed/timed-out
    judge returns empty scores, NOT meaningful zeros.
    """
    res = await _run_judge_timed_inner(name, coro, score_fn)
    if on_done is not None:
        try:
            on_done(name, *res)
        except Exception:                                  # a broken listener must never fail a judge
            logger.exception("on_done callback failed for %s", name)
    return res


async def _run_judge_timed_inner(name: str, coro, score_fn) -> tuple[str, dict, dict, str | None, int]:
    t0 = time.monotonic()
    try:
        raw = await asyncio.wait_for(coro, timeout=JUDGE_TIMEOUT_S)
        latency = round((time.monotonic() - t0) * 1000)
        if not raw.get("success"):
            logger.error("Judge %s returned success=False in %.0fms: %s",
                         name, latency, raw.get("error", ""))
            return "failed", raw, {}, raw.get("error", "judge returned success=False"), latency
        scores = score_fn(raw.get("data", {}))
        if scores.get("scorable") is False:
            # Parsed fine but contained nothing to score. That is missing evidence, not
            # evidence of a zero, so it must not enter the weighted sum.
            logger.error("Judge %s produced no scorable items in %.0fms", name, latency)
            return "failed", raw, {}, "judge output contained no scorable items", latency
        logger.info("Judge %s OK %.0fms score=%.1f%%",
                    name, latency, scores.get("score", 0) * 100)
        return "success", raw, scores, None, latency
    except asyncio.TimeoutError:
        latency = round((time.monotonic() - t0) * 1000)
        msg = f"timed out after {JUDGE_TIMEOUT_S:.0f}s"
        logger.error("Judge %s %s (%.0fms)", name, msg, latency)
        return "timeout", {}, {}, msg, latency
    except Exception as exc:
        latency = round((time.monotonic() - t0) * 1000)
        logger.exception("Judge %s raised unhandled exception in %.0fms", name, latency)
        return "failed", {}, {}, str(exc), latency


def _gap_or_none(status: str, scores: dict, key: str) -> list | None:
    """
    Returns the gap list from scores if the judge succeeded, else None.
    None signals to the Coach that the data is UNAVAILABLE — not that there
    are no gaps. The Coach prompt must render these differently.
    """
    if status != "success":
        return None
    return scores.get(key, [])


async def _skipped_judge(name: str, reason: str) -> tuple[str, dict, dict, str, int]:
    """Drop-in replacement for _run_judge_timed when a judge is adaptively skipped."""
    logger.info("Judge %s skipped: %s", name, reason)
    return "skipped", {}, {}, reason, 0


def _text_unreadable(text: str) -> bool:
    """True when the extracted text is too broken for any judge to read: nearly empty,
    mostly non-letters, or letter-spaced garbage ("J a n e  D o e") from a bad PDF
    extraction. Deliberately script-agnostic (no English heading/verb assumptions)."""
    t = text.strip()
    if len(t) < 200:
        return True
    letters = sum(1 for ch in t if ch.isalpha())
    if letters / len(t) < 0.45:
        return True
    tokens = t.split()
    return bool(tokens) and sum(1 for tok in tokens if len(tok) == 1) / len(tokens) > 0.45


def _ats_early_exit(ats_score: float) -> dict:
    reason = (
        f"ATS score {ats_score:.0f}/100 and the extracted text is unreadable "
        "(too short, mostly non-letters, or letter-spaced). Fix the resume file / extraction "
        "before scoring. No score was produced."
    )
    skipped = {"success": False, "skipped": True}
    js = {j: {"status": "skipped", "error": reason}
          for j in ["skills", "experience", "seniority", "relevancy", "impact"]}
    return {
        "overall": {
            # None, not 0.0: "could not be scored" must never look like "scored zero".
            "overall_score": None,
            "status": "not_scored",
            "weights_used": DEFAULT_WEIGHTS,
            "skills_gate_applied": False,
            "low_fit_cap_applied": False,
            "component_scores": {},
            "judge_statuses": js,
            "quorum_met": False,
            "early_exit": "unreadable_text",
            "early_exit_reason": reason,
        },
        "skills":     {"analyser_result": skipped, "scores": {}, "extracted_chunks": {}},
        "experience": {"analyser_result": skipped, "scores": {}},
        "seniority":  {"analyser_result": skipped, "scores": {}},
        "relevancy":  {"analyser_result": skipped, "scores": {}},
        "impact":     {"analyser_result": skipped, "scores": {}},
        "coach":      {**skipped, "data": {}},
    }


def pipeline_status(overall: dict) -> str:
    """complete | degraded | insufficient | not_scored  (single source of truth)."""
    if overall.get("status") == "not_scored" or overall.get("overall_score") is None:
        return "not_scored"
    if not overall.get("quorum_met", False):
        return "insufficient"
    statuses = overall.get("judge_statuses", {})
    if any(v.get("status") in ("failed", "timeout") for v in statuses.values()):
        return "degraded"
    return "complete"


def _renormalized_weights_for_successes(base_weights: dict, judge_statuses: dict) -> dict:
    """
    Failed/timed-out judges are unavailable evidence, not evidence of zero fit.
    Drop their weights and renormalize over successful judges so one timeout
    cannot turn an otherwise strong candidate into a false negative.
    """
    active = {
        name: weight
        for name, weight in base_weights.items()
        if judge_statuses.get(name, {}).get("status") == "success"
    }
    total = sum(active.values())
    if total <= 0:
        return base_weights
    return {
        name: round((active.get(name, 0.0) / total), 6)
        for name in base_weights
    }


def _detect_contradictions(scores: dict, statuses: dict) -> list[dict]:
    """Return contradiction entries whose conditions are met by succeeded judges only."""
    active_scores = {
        j: scores[j].get("score", 0)
        for j in scores
        if statuses.get(j, {}).get("status") == "success"
    }
    return [c for c in _CONTRADICTION_PAIRS if c["condition"](active_scores)]


async def _run_arbitration(
    contradictions: list[dict],
    engine,
    resume_text: str,
    job_description: str,
    skills_taxonomy,
    llm_model: str,
    kw_base: dict,
    scores: dict,
    results: dict,
    statuses: dict,
    score_fns: dict | None = None,
) -> None:
    """Re-run flagged judges in sequence; updates scores/results/statuses in-place.
    `score_fns` MUST be the same scorers used in stage 1 (e.g. the frequency-weighted
    skills scorer) so first-pass and arbitrated scores are comparable."""
    from analysers.skills.analyser     import score as skills_score
    from analysers.experience.analyser import score as experience_score
    from analysers.seniority.analyser  import score as seniority_score

    score_fns = {
        "skills": skills_score,
        "experience": experience_score,
        "seniority": seniority_score,
        **(score_fns or {}),
    }

    seen = set()
    for c in contradictions:
        judge = c["rerun"]
        if judge in seen:
            continue
        seen.add(judge)

        ctx = c["msg"]
        old_score = scores.get(judge, {}).get("score", 0)
        logger.info("Arbitration: re-running %s (trigger=%s, old_score=%.1f%%)", judge, c["name"], old_score * 100)

        if judge == "skills":
            coro = engine.skills.run(resume_text, job_description, skills_taxonomy=skills_taxonomy,
                                     llm_model=llm_model, arbitration_context=ctx, **kw_base)
        elif judge == "experience":
            coro = engine.experience.run(resume_text, job_description, llm_model=llm_model,
                                          arbitration_context=ctx, **kw_base)
        elif judge == "seniority":
            coro = engine.seniority.run(resume_text, job_description, llm_model=llm_model,
                                         arbitration_context=ctx, **kw_base)
        else:
            continue

        new_status, new_raw, new_scores, new_err, new_ms = await _run_judge_timed(judge, coro, score_fns[judge])
        if new_status == "success":
            new_score = new_scores.get("score", 0)
            logger.info("Arbitration %s: score %.1f%% → %.1f%%", judge, old_score * 100, new_score * 100)
            results[judge]  = new_raw
            scores[judge]   = new_scores
            statuses[judge] = {"status": new_status, "error": new_err, "elapsed_ms": new_ms,
                               "arbitrated": True, "contradiction_trigger": c["name"],
                               "score_before_arbitration": old_score,
                               "score_after_arbitration": new_score}
        else:
            logger.warning("Arbitration re-run %s failed (%s) — keeping original result", judge, new_err)


class AnalyserEngine:
    def __init__(self, confidence_threshold: int = 75, max_retries: int = 3):
        from analysers.skills.analyser     import SkillsAnalyser
        from analysers.experience.analyser import ExperienceAnalyser
        from analysers.seniority.analyser  import SeniorityAnalyser
        from analysers.relevancy.analyser  import RelevancyAnalyser
        from analysers.impact.analyser     import ImpactAnalyser
        from analysers.coach.analyser      import CoachAnalyser
        cfg = {"confidence_threshold": confidence_threshold, "max_retries": max_retries}
        self.skills     = SkillsAnalyser(**cfg)
        self.experience = ExperienceAnalyser(**cfg)
        self.seniority  = SeniorityAnalyser(**cfg)
        self.relevancy  = RelevancyAnalyser(**cfg)
        self.impact     = ImpactAnalyser(**cfg)
        self.coach      = CoachAnalyser(max_retries=max_retries)

    async def run(
        self,
        resume_text: str,
        job_description: str,
        llm_model: str = "deepseek-r1:14b",
        skills_taxonomy: dict | None = None,
        weights: dict | None = None,
        api_key: str = "",
        api_base_url: str = "",
        fast_model: str = "",
        ats_score: float | None = None,
        semantic_similarity: float | None = None,
        user_id: str | None = None,
        progress=None,
    ) -> dict:
        """`progress`, if given, is a plain callable(dict) invoked with UI events
        (judge_start / judge / arbitration / coach_start). It must not block."""
        def _emit(ev: dict) -> None:
            if progress is not None:
                try:
                    progress(ev)
                except Exception:
                    logger.exception("progress callback failed")
        _t_start = time.monotonic()

        logger.info("=" * 60)
        logger.info("AnalyserEngine.run  model=%s fast=%s", llm_model, fast_model or "none")
        logger.info("=" * 60)

        # Defence in depth: whatever the caller did, judges only ever see normalised text
        # with instruction-like spans neutralised (idempotent if already prepared).
        from analysers.base import prepare_user_text
        resume_text, _rflags = prepare_user_text(resume_text, "resume")
        job_description, _jflags = prepare_user_text(job_description, "job_description")
        security_flags = _rflags + _jflags

        # Fix 3a: skip all LLM judges only when the extracted text is genuinely unreadable.
        # (Previously ANY resume with a heuristic ATS score < 40 got overall 0.0 and no LLM
        # run - including well-formed resumes whose headings are not English: a clean Spanish
        # resume scored exactly 40.0.) A low-but-readable ATS score is now just a warning.
        ats_warning = None
        if ats_score is not None and ats_score < ATS_SKIP_THRESHOLD:
            if _text_unreadable(resume_text):
                logger.info("Unreadable text + ATS %.0f < %.0f - not scoring", ats_score, ATS_SKIP_THRESHOLD)
                out = _ats_early_exit(ats_score)
                out["overall"]["security_flags"] = security_flags
                return out
            ats_warning = (f"ATS formatting score {ats_score:.0f}/100 is low; text is readable so "
                           "scoring proceeded. ATS heuristics are English-centric.")

        from analysers.skills.analyser     import score as _skills_score_base
        from analysers.experience.analyser import score as experience_score
        from analysers.seniority.analyser  import score as seniority_score
        from analysers.relevancy.analyser  import score as relevancy_score
        from analysers.impact.analyser     import score as impact_score

        # Fetch market skill frequencies once — used to enrich missing_skills with demand signal
        try:
            import db as run_db
            _freq, _n_jobs = await asyncio.to_thread(run_db.get_skill_demand, user_id)
        except Exception:
            _freq, _n_jobs = {}, 0
        skills_score = lambda data: _skills_score_base(
            data, skill_frequencies=_freq or None, total_jobs=_n_jobs or None)

        # Fix 3b: Domain mismatch — skip seniority + relevancy when SBERT sim is near-zero
        skip_judges: set[str] = set()
        if semantic_similarity is not None and semantic_similarity < SIMILARITY_MISMATCH_THRESHOLD:
            skip_judges = {"seniority", "relevancy"}
            logger.info(
                "Domain mismatch: similarity=%.3f < %.3f — skipping %s",
                semantic_similarity, SIMILARITY_MISMATCH_THRESHOLD, skip_judges,
            )

        # Fix 4: Per-judge model routing — light judges use fast_model when set
        def _model(judge: str) -> str:
            if fast_model and judge in _FAST_JUDGES:
                return fast_model
            return llm_model

        kw_base = dict(api_key=api_key, api_base_url=api_base_url)

        skip_reason_sr = f"skipped: similarity={semantic_similarity:.3f} < {SIMILARITY_MISMATCH_THRESHOLD}" if "seniority" in skip_judges else ""
        skip_reason_rl = f"skipped: similarity={semantic_similarity:.3f} < {SIMILARITY_MISMATCH_THRESHOLD}" if "relevancy" in skip_judges else ""

        def _judge_done(name, status, _raw, scores, _err, ms):
            _emit({"event": "judge", "judge": name,
                   "score": round(scores.get("score", 0) * 100, 1) if status == "success" else 0,
                   "success": status == "success", "status": status,
                   "elapsed_s": round(ms / 1000, 1),
                   "pipeline_elapsed_s": round(time.monotonic() - _t_start, 1)})

        for _j in ("skills", "experience", "seniority", "relevancy", "impact"):
            if _j not in skip_judges:
                _emit({"event": "judge_start", "judge": _j,
                       "elapsed_s": round(time.monotonic() - _t_start, 1)})

        # Stage 1: 5 independent judges, each with a hard per-judge timeout.
        # _run_judge_timed never raises — returns structured (status, raw, scores, err).
        # Skipped judges use _skipped_judge() which returns immediately with status="skipped".
        (sk_status, skills_res,    s_sc,  sk_err, sk_ms), \
        (ex_status, exp_res,       e_sc,  ex_err, ex_ms), \
        (sr_status, seniority_res, sr_sc, sr_err, sr_ms), \
        (rl_status, relevancy_res, rl_sc, rl_err, rl_ms), \
        (im_status, impact_res,    im_sc, im_err, im_ms)  = await asyncio.gather(
            _run_judge_timed("skills",     self.skills.run(resume_text, job_description, skills_taxonomy=skills_taxonomy, llm_model=_model("skills"), **kw_base), skills_score, _judge_done),
            _run_judge_timed("experience", self.experience.run(resume_text, job_description, llm_model=_model("experience"), **kw_base), experience_score, _judge_done),
            _skipped_judge("seniority", skip_reason_sr) if "seniority" in skip_judges else
            _run_judge_timed("seniority",  self.seniority.run(resume_text, job_description, llm_model=_model("seniority"), **kw_base), seniority_score, _judge_done),
            _skipped_judge("relevancy", skip_reason_rl) if "relevancy" in skip_judges else
            _run_judge_timed("relevancy",  self.relevancy.run(resume_text, job_description, llm_model=_model("relevancy"), **kw_base), relevancy_score, _judge_done),
            _run_judge_timed("impact",     self.impact.run(resume_text, job_description, llm_model=_model("impact"), **kw_base), impact_score, _judge_done),
        )

        judge_statuses = {
            "skills":     {"status": sk_status, "error": sk_err, "elapsed_ms": sk_ms},
            "experience": {"status": ex_status, "error": ex_err, "elapsed_ms": ex_ms},
            "seniority":  {"status": sr_status, "error": sr_err, "elapsed_ms": sr_ms},
            "relevancy":  {"status": rl_status, "error": rl_err, "elapsed_ms": rl_ms},
            "impact":     {"status": im_status, "error": im_err, "elapsed_ms": im_ms},
        }
        score_map = {
            "skills": s_sc, "experience": e_sc, "seniority": sr_sc,
            "relevancy": rl_sc, "impact": im_sc,
        }
        result_map = {
            "skills": skills_res, "experience": exp_res, "seniority": seniority_res,
            "relevancy": relevancy_res, "impact": impact_res,
        }

        # Stage 1b: Cross-judge arbitration — detect contradictions, re-run flagged judges
        contradictions = _detect_contradictions(score_map, judge_statuses)
        if contradictions:
            logger.info("Arbitration triggered: %s", [c["name"] for c in contradictions])
            _emit({"event": "tick", "stage": "arbitration",
                   "elapsed_s": round(time.monotonic() - _t_start, 1)})
            await _run_arbitration(
                contradictions, self, resume_text, job_description, skills_taxonomy,
                llm_model, kw_base, score_map, result_map, judge_statuses,
                score_fns={"skills": skills_score},
            )
            # Unpack potentially updated values
            s_sc  = score_map["skills"];   skills_res    = result_map["skills"]
            e_sc  = score_map["experience"]; exp_res      = result_map["experience"]
            sr_sc = score_map["seniority"]; seniority_res = result_map["seniority"]
            rl_sc = score_map["relevancy"]; relevancy_res = result_map["relevancy"]
            im_sc = score_map["impact"];   impact_res    = result_map["impact"]

        # Refresh individual status vars from judge_statuses so Stage 2 + Stage 3
        # see post-arbitration statuses, not the original Stage 1 values.
        sk_status = judge_statuses["skills"]["status"]
        ex_status = judge_statuses["experience"]["status"]
        sr_status = judge_statuses["seniority"]["status"]
        rl_status = judge_statuses["relevancy"]["status"]
        im_status = judge_statuses["impact"]["status"]

        succeeded = sum(1 for v in judge_statuses.values() if v["status"] == "success")
        # Skipped judges are intentional — exclude from quorum denominator
        active = sum(1 for v in judge_statuses.values() if v["status"] != "skipped")
        if succeeded < min(JUDGE_QUORUM, active):
            failed_names = [k for k, v in judge_statuses.items() if v["status"] == "failed"]
            skipped_names = [k for k, v in judge_statuses.items() if v["status"] == "skipped"]
            logger.error("Quorum not met: %d/%d active judges succeeded. Failed=%s Skipped=%s",
                         succeeded, active, failed_names, skipped_names)

        # Stage 2: Use zero-scores for failed/skipped judges so the overall formula still runs.
        def _scores_or_zero(status, scores, zero_fn):
            return scores if status == "success" else zero_fn({})

        s_sc  = _scores_or_zero(sk_status, s_sc,  skills_score)
        e_sc  = _scores_or_zero(ex_status, e_sc,  experience_score)
        sr_sc = _scores_or_zero(sr_status, sr_sc, seniority_score)
        rl_sc = _scores_or_zero(rl_status, rl_sc, relevancy_score)
        im_sc = _scores_or_zero(im_status, im_sc, impact_score)

        # Stage 3: Coach receives None for any dimension whose judge failed or was skipped.
        # None ≠ empty list — the Coach prompt treats None as "data unavailable".
        quorum_ok = succeeded >= min(JUDGE_QUORUM, active)
        if quorum_ok:
            _emit({"event": "coach_start", "elapsed_s": round(time.monotonic() - _t_start, 1)})
            coach_res = await self.coach.run(
                resume_text=resume_text,
                job_description=job_description,
                missing_skills=            _gap_or_none(sk_status, s_sc,  "missing_skills"),
                unmet_responsibilities=    _gap_or_none(ex_status, e_sc,  "unmet_responsibilities"),
                absent_seniority_signals=  _gap_or_none(sr_status, sr_sc, "absent_signals"),
                irrelevant_statements=     _gap_or_none(rl_status, rl_sc, "irrelevant_statements"),
                unquantified_statements=   _gap_or_none(im_status, im_sc, "unquantified_statements"),
                llm_model=llm_model,
                **kw_base,
            )
        else:
            # No score will be shown, so do not spend a 20-40s LLM call on advice.
            coach_res = {"success": False, "skipped": True, "data": {},
                         "error": "coach skipped: judge quorum not met"}

        # Stage 4: Overall score — gate only applies when skills judge actually succeeded.
        # Skipped judges are excluded from weights the same way failed judges are.
        base_weights = weights or DEFAULT_WEIGHTS
        effective_weights = _renormalized_weights_for_successes(base_weights, judge_statuses)
        overall = calculate_overall(s_sc, e_sc, sr_sc, rl_sc, im_sc, effective_weights,
                                    skills_gate_enabled=(sk_status == "success"))
        overall["base_weights"] = base_weights
        overall["weights_adjusted_for_failed_judges"] = effective_weights != base_weights
        overall["judge_statuses"] = judge_statuses
        overall["quorum_met"] = quorum_ok
        overall["security_flags"] = security_flags
        if ats_warning:
            overall["ats_warning"] = ats_warning
        overall["pipeline_version"] = PIPELINE_VERSION
        overall["status"] = pipeline_status(overall)
        if skip_judges:
            overall["skipped_judges"] = sorted(skip_judges)
            overall["skip_reason"] = f"semantic_similarity={semantic_similarity:.3f} below threshold"
        logger.info("Overall: %.1f%%  gate=%s  quorum=%d/%d active  skipped=%s",
                    overall["overall_score"] * 100, overall["skills_gate_applied"],
                    succeeded, active, sorted(skip_judges) or "none")

        return {
            "overall": overall,
            "skills":     {"analyser_result": skills_res,    "scores": s_sc,  "extracted_chunks": skills_res.get("extracted_chunks", {})},
            "experience": {"analyser_result": exp_res,       "scores": e_sc},
            "seniority":  {"analyser_result": seniority_res, "scores": sr_sc},
            "relevancy":  {"analyser_result": relevancy_res, "scores": rl_sc},
            "impact":     {"analyser_result": impact_res,    "scores": im_sc},
            "coach":      coach_res,
        }

    async def run_skills_judge(self, resume_text: str, job_description: str, llm_model: str,
                               skills_taxonomy: dict | None = None, api_key: str = "", api_base_url: str = "") -> dict:
        from analysers.skills.analyser import score as _score
        status, raw, _, err, _ = await _run_judge_timed(
            "skills",
            self.skills.run(resume_text, job_description, llm_model=llm_model,
                            skills_taxonomy=skills_taxonomy, api_key=api_key, api_base_url=api_base_url),
            _score,
        )
        return raw if status == "success" else {"success": False, "error": err, "data": {}}

    async def run_experience_judge(self, resume_text: str, job_description: str, llm_model: str,
                                   api_key: str = "", api_base_url: str = "") -> dict:
        from analysers.experience.analyser import score as _score
        status, raw, _, err, _ = await _run_judge_timed(
            "experience",
            self.experience.run(resume_text, job_description, llm_model=llm_model,
                                api_key=api_key, api_base_url=api_base_url),
            _score,
        )
        return raw if status == "success" else {"success": False, "error": err, "data": {}}

    async def run_seniority_judge(self, resume_text: str, job_description: str, llm_model: str,
                                  api_key: str = "", api_base_url: str = "") -> dict:
        from analysers.seniority.analyser import score as _score
        status, raw, _, err, _ = await _run_judge_timed(
            "seniority",
            self.seniority.run(resume_text, job_description, llm_model=llm_model,
                               api_key=api_key, api_base_url=api_base_url),
            _score,
        )
        return raw if status == "success" else {"success": False, "error": err, "data": {}}

    async def run_relevancy_judge(self, resume_text: str, job_description: str, llm_model: str,
                                  api_key: str = "", api_base_url: str = "") -> dict:
        from analysers.relevancy.analyser import score as _score
        status, raw, _, err, _ = await _run_judge_timed(
            "relevancy",
            self.relevancy.run(resume_text, job_description, llm_model=llm_model,
                               api_key=api_key, api_base_url=api_base_url),
            _score,
        )
        return raw if status == "success" else {"success": False, "error": err, "data": {}}

    async def run_impact_judge(self, resume_text: str, llm_model: str,
                               job_description: str = "",
                               api_key: str = "", api_base_url: str = "") -> dict:
        from analysers.impact.analyser import score as _score
        status, raw, _, err, _ = await _run_judge_timed(
            "impact",
            self.impact.run(resume_text, job_description, llm_model=llm_model,
                            api_key=api_key, api_base_url=api_base_url),
            _score,
        )
        return raw if status == "success" else {"success": False, "error": err, "data": {}}

    async def run_coach_judge(self, resume_text: str, job_description: str,
                              missing_skills: list, unmet_responsibilities: list,
                              absent_seniority_signals: list, irrelevant_statements: list,
                              unquantified_statements: list,
                              llm_model: str = "deepseek-r1:14b",
                              api_key: str = "", api_base_url: str = "") -> dict:
        return await self.coach.run(
            resume_text=resume_text, job_description=job_description,
            missing_skills=missing_skills, unmet_responsibilities=unmet_responsibilities,
            absent_seniority_signals=absent_seniority_signals,
            irrelevant_statements=irrelevant_statements,
            unquantified_statements=unquantified_statements,
            llm_model=llm_model, api_key=api_key, api_base_url=api_base_url,
        )

    async def run_evals(
        self,
        pipeline_result: dict,
        resume_text: str,
        job_description: str,
        llm_model: str = "deepseek-r1:14b",
        api_key: str = "",
        api_base_url: str = "",
    ) -> dict:
        """Run all 6 eval judges against an existing pipeline result. Returns eval scores per analyser."""
        from analysers.skills.eval    import run_eval as skills_eval
        from analysers.experience.eval import run_eval as experience_eval
        from analysers.seniority.eval  import run_eval as seniority_eval
        from analysers.relevancy.eval  import run_eval as relevancy_eval
        from analysers.impact.eval     import run_eval as impact_eval
        from analysers.coach.eval      import run_eval as coach_eval

        kw = dict(resume_text=resume_text, job_description=job_description,
                  llm_model=llm_model, api_key=api_key, api_base_url=api_base_url)

        raw = await asyncio.gather(
            skills_eval(pipeline_result.get("skills", {}).get("analyser_result", {}).get("data", {}), **kw),
            experience_eval(pipeline_result.get("experience", {}).get("analyser_result", {}).get("data", {}), **kw),
            seniority_eval(pipeline_result.get("seniority", {}).get("analyser_result", {}).get("data", {}), **kw),
            relevancy_eval(pipeline_result.get("relevancy", {}).get("analyser_result", {}).get("data", {}), **kw),
            impact_eval(pipeline_result.get("impact", {}).get("analyser_result", {}).get("data", {}), **kw),
            coach_eval(pipeline_result.get("coach", {}).get("data", {}), **kw),
            return_exceptions=True,
        )

        names = ["skills", "experience", "seniority", "relevancy", "impact", "coach"]
        return {name: (_safe(r, name) if isinstance(r, BaseException) else r)
                for name, r in zip(names, raw)}
