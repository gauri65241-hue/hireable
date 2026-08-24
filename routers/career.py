import json
import logging
import os
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from routers.deps import (
    BASE_DIR, run_db, score_ats, ats_to_dict,
    get_free_model, execute_llm_call,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class InterviewPrepRequest(BaseModel):
    result_id: str
    model: str = ""
    force_regen: bool = False

class EditResumeRequest(BaseModel):
    result_id: str
    model: str = ""
    force_regen: bool = False


@router.post("/interview_prep")
async def interview_prep(req: InterviewPrepRequest):
    data = run_db.get_run(req.result_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Result {req.result_id} not found")

    pr             = data.get("pipeline_result", {})
    resume_text    = data.get("resume_text", "")
    job_description= data.get("job_description", "")

    skills_data    = pr.get("skills", {}).get("analyser_result", {}).get("data", {})
    exp_data       = pr.get("experience", {}).get("analyser_result", {}).get("data", {})
    seniority_data = pr.get("seniority", {}).get("analyser_result", {}).get("data", {})
    impact_data    = pr.get("impact", {}).get("analyser_result", {}).get("data", {})

    missing_skills   = skills_data.get("missing_critical_skills", [])
    present_skills   = skills_data.get("present_skills", [])
    unmet_resp       = [r.get("responsibility","") for r in exp_data.get("responsibility_analysis",[]) if r.get("experience_score",1) < 0.5]
    absent_signals   = [s.get("signal","") for s in seniority_data.get("seniority_analysis",[]) if s.get("is_demonstrated") == 0]
    unquantified     = [s.get("resume_statement","") for s in impact_data.get("impact_analysis",[]) if s.get("is_quantified") == 0]

    DOMAIN_GAP_KEYWORDS = {
        "pentest":      ["offensive security mindset", "vulnerability assessment", "security tooling (Burp Suite / OWASP)"],
        "security":     ["threat modeling", "CVE/exploit research", "security review process"],
        "devops":       ["CI/CD pipeline ownership", "infrastructure-as-code", "on-call incident response"],
        "frontend":     ["UI component architecture", "browser performance profiling", "accessibility standards"],
        "backend":      ["database schema design", "API rate limiting and auth", "distributed systems"],
        "data engineer":["pipeline orchestration (Airflow/Prefect)", "data warehouse modeling", "dbt/SQL at scale"],
        "mlops":        ["model serving infrastructure", "drift monitoring", "A/B experiment pipelines"],
    }
    inferred_domain_gaps = []
    jd_lower = job_description.lower()
    for keyword, gaps in DOMAIN_GAP_KEYWORDS.items():
        if keyword in jd_lower:
            inferred_domain_gaps.extend(gaps)

    model = req.model or get_free_model()

    system_prompt = """You are a senior technical interviewer preparing a candidate for a specific job interview.
Given gap analysis from 4 AI judges plus inferred domain gaps, generate the realistic questions this candidate WILL face.

RULES:
- Prioritise domain-specific gaps and missing skills over generic quantification questions
- If domain gaps exist, generate at least 3 questions targeting that domain directly
- Model answers must be honest — use only what the candidate actually has; bridge gaps with transferable experience
- Do NOT fabricate experience the candidate doesn't have in model answers
- Do NOT generate more than 2 "how did you measure X" questions — these are low-value filler
- Mark difficulty: easy | medium | hard
- Mark type: technical | behavioral | situational

Output ONLY valid JSON, no markdown, no text outside JSON:
{
  "questions": [
    {
      "gap": "<the specific gap this question targets>",
      "question": "<realistic interview question>",
      "model_answer": "<honest answer bridging candidate's real experience to the gap, with [ADD METRIC] placeholders only where a real number would fit>",
      "difficulty": "easy | medium | hard",
      "type": "technical | behavioral | situational"
    }
  ]
}"""

    user_prompt = f"""JOB DESCRIPTION:
{job_description[:800]}

CANDIDATE RESUME:
{resume_text[:1200]}

WHAT THE CANDIDATE HAS (present skills): {json.dumps(present_skills[:15])}

GAPS — generate questions for these in priority order:
1. Inferred Domain Gaps (highest priority — job domain skills not on resume): {json.dumps(inferred_domain_gaps)}
2. Missing Skills (from skills judge): {json.dumps(missing_skills)}
3. Unmet Responsibilities: {json.dumps(unmet_resp)}
4. Absent Seniority Signals: {json.dumps(absent_signals[:4])}
5. Unquantified Statements (max 2 questions from this list): {json.dumps(unquantified[:4])}

Generate 8-12 focused questions weighted toward domain gaps. Output the JSON."""

    result = await execute_llm_call(system_prompt, user_prompt, model)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=f"LLM failed: {result.get('error','unknown')}")

    raw = result.get("data", "")
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            raw_questions = parsed.get("questions", [])
            seen_gaps = set()
            questions = []
            for q in raw_questions:
                gap_key = q.get("gap", "").lower().strip()
                if gap_key not in seen_gaps:
                    seen_gaps.add(gap_key)
                    questions.append(q)
            return {"questions": questions, "result_id": req.result_id, "cached": False}
        except json.JSONDecodeError:
            pass
    raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON")


@router.post("/edit_resume")
async def edit_resume(req: EditResumeRequest):
    data = run_db.get_run(req.result_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Result {req.result_id} not found")

    if not req.force_regen and data.get("tailored_text"):
        return {
            "edited_text":    data["tailored_text"],
            "result_id":      req.result_id,
            "cached":         True,
            "ats_validation": data.get("ats_validation"),
        }

    resume_text = data.get("resume_text", "")
    pr = data.get("pipeline_result", {})
    coach = pr.get("coach", {})
    coach_recs = []
    if coach.get("success") and coach.get("data"):
        coach_recs = coach["data"].get("improvement_analysis", [])

    PLACEHOLDER_RE = re.compile(r'\[.{1,30}?\]|\b[A-Z]%|\b[A-Z]\s*%')
    REWRITE_TYPES  = {"bullet_rewrite", "deletion"}
    actionable = [
        r for r in coach_recs
        if r.get("improvement_type") in REWRITE_TYPES
        and r.get("is_actionable") == 1
        and r.get("original_content", "").strip()
        and not PLACEHOLDER_RE.search(r.get("suggested_improvement", ""))
    ]

    model = req.model or get_free_model()

    if actionable:
        def fmt_rec(i, r):
            orig = r['original_content'].strip()
            sugg = r.get('suggested_improvement', '').strip()
            if r.get('improvement_type') == 'deletion':
                return (f"{i+1}. DELETE — remove this sentence/bullet entirely from the resume "
                        f"(do not reword it, do not keep any part of it):\n   >>> {orig}")
            return (f"{i+1}. FIND this exact text in the resume:\n"
                    f"   >>> {orig}\n"
                    f"   REPLACE with:\n"
                    f"   >>> {sugg}")
        recs_text = "\n".join(fmt_rec(i, r) for i, r in enumerate(actionable))
        system_prompt = (
            "You are a precise resume editor. Follow each numbered instruction exactly. "
            "DELETE instructions: remove that sentence completely — do not reword, do not keep any part of it. "
            "FIND/REPLACE instructions: swap only that text, leave everything else unchanged. "
            "Do not add bullets, sections, or metrics that are not in the original. "
            "Return ONLY the complete edited resume. No commentary, no markdown fences."
        )
        user_prompt = f"""ORIGINAL RESUME:
{resume_text}

FIND AND REPLACE INSTRUCTIONS:
{recs_text}

Return the complete resume with only those replacements applied."""
    else:
        job_snippet  = data.get("job_description", "")[:600]
        gap_warnings = [r.get("gap_addressed","") for r in coach_recs if r.get("improvement_type") == "gap_warning"]
        system_prompt = (
            "You are a professional resume editor. Reframe the existing resume bullets to highlight transferable "
            "skills most relevant to the target job. Do NOT add new bullets, new sections, or invent metrics. "
            "Only rephrase language in bullets that already exist. Keep all sections, dates, and company names intact. "
            "Return ONLY the complete edited resume. No commentary, no markdown fences."
        )
        user_prompt = f"""TARGET JOB:
{job_snippet}

KNOWN GAPS (do NOT attempt to cover these with new bullets — leave them as honest gaps):
{chr(10).join(f'- {g}' for g in gap_warnings) or '- See job description for domain gaps'}

ORIGINAL RESUME (reframe existing bullets only — do not add or remove any):
{resume_text}

Return the complete resume with existing bullets reworded to better highlight relevance to the target job."""

    result = await execute_llm_call(system_prompt, user_prompt, model)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=f"LLM edit failed: {result.get('error','unknown')}")

    edited_text = result.get("data", "").strip()
    if not edited_text:
        raise HTTPException(status_code=500, detail="LLM returned empty resume")

    original_resume = data.get("resume_text", "")
    job_desc_text   = data.get("job_description", "")
    ats_before = ats_to_dict(score_ats(original_resume, job_desc_text))
    ats_after  = ats_to_dict(score_ats(edited_text, job_desc_text))
    delta  = round(ats_after["score"] - ats_before["score"], 1)
    fixed  = [c["label"] for c in ats_after["criteria"]
              if c["passed"] and not next((b for b in ats_before["criteria"] if b["id"] == c["id"]), {}).get("passed")]
    regressed = [c["label"] for c in ats_after["criteria"]
                 if not c["passed"] and next((b for b in ats_before["criteria"] if b["id"] == c["id"]), {}).get("passed")]
    ats_validation = {
        "before":  {"score": ats_before["score"], "grade": ats_before["grade"]},
        "after":   {"score": ats_after["score"],  "grade": ats_after["grade"]},
        "delta":   delta,
        "fixed":   fixed,
        "regressed": regressed,
        "verdict": "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged",
    }
    try:
        run_db.update_run_tailored(req.result_id, edited_text, ats_validation)
    except Exception as db_err:
        logger.warning("SQLite update_run_tailored failed (non-fatal): %s", db_err)

    return {
        "edited_text":    edited_text,
        "result_id":      req.result_id,
        "cached":         False,
        "ats_validation": ats_validation,
    }
