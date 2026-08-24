import asyncio
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth_utils import get_current_user
from routers.deps import run_db, strip_html

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class AddResumeRequest(BaseModel):
    name: str
    title: str
    skills: str
    text: str
    experience_years: str = ""

class AddJobRequest(BaseModel):
    title: str
    company: str
    experience_required: str
    skills_required: str
    description: str
    source_url: str = ""

class ToggleAppliedRequest(BaseModel):
    applied: bool

class FeedbackRequest(BaseModel):
    resume_id: str
    job_id: str
    got_interview: bool
    is_golden: bool = False

class MatchJobRequest(BaseModel):
    resume_id: str
    job_description: str


# ── TF-IDF / Jaccard helpers ──────────────────────────────────────────────────

def _tfidf_score(resume_text: str, job_text: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf = vec.fit_transform([resume_text, job_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


def _skills_jaccard(resume_skills: str, job_skills: str) -> tuple[float, list[str]]:
    def tokenise(text: str) -> set[str]:
        tokens = re.split(r"[,\s/|]+", text.lower())
        return {t.strip() for t in tokens if len(t.strip()) > 2}
    resume_set = tokenise(resume_skills)
    job_set    = tokenise(job_skills)
    if not job_set:
        return 0.0, []
    intersection = resume_set & job_set
    union = resume_set | job_set
    missing = sorted(job_set - resume_set)[:10]
    return len(intersection) / len(union), missing


def _experience_stub(resume_years: str, job_years: str) -> float:
    import re as _re
    def parse_years(s: str) -> float:
        nums = _re.findall(r"\d+\.?\d*", str(s))
        return float(nums[0]) if nums else 0.0
    r = parse_years(resume_years)
    j = parse_years(job_years)
    if j == 0:
        return 1.0
    if r >= j:
        return 1.0
    if r >= j * 0.6:
        return 0.5
    return 0.0


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/resumes")
async def get_resumes(current_user: dict = Depends(get_current_user)):
    return await asyncio.to_thread(run_db.get_resumes, current_user["user_id"])


@router.get("/jobs")
async def get_jobs(current_user: dict = Depends(get_current_user)):
    return await asyncio.to_thread(run_db.get_jobs, current_user["user_id"])


@router.post("/resumes")
async def add_resume(req: AddResumeRequest, current_user: dict = Depends(get_current_user)):
    new_id = await asyncio.to_thread(
        run_db.add_resume, req.name, req.title, req.experience_years, req.skills, req.text,
        current_user["user_id"]
    )
    return {"success": True, "id": new_id}


@router.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str, current_user: dict = Depends(get_current_user)):
    found = await asyncio.to_thread(run_db.delete_resume, resume_id, current_user["user_id"])
    if not found:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return {"success": True}


@router.post("/jobs")
async def add_job(req: AddJobRequest, current_user: dict = Depends(get_current_user)):
    clean_desc = strip_html(req.description)
    new_id = await asyncio.to_thread(
        run_db.add_job, req.title, req.company, req.experience_required,
        req.skills_required, clean_desc, req.source_url, current_user["user_id"]
    )
    return {"success": True, "id": new_id}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: int, current_user: dict = Depends(get_current_user)):
    found = await asyncio.to_thread(run_db.delete_job, job_id, current_user["user_id"])
    if not found:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"success": True}


@router.patch("/jobs/{job_id}/applied")
async def toggle_applied(job_id: str, req: ToggleAppliedRequest,
                         current_user: dict = Depends(get_current_user)):
    found = await asyncio.to_thread(
        run_db.toggle_job_applied, int(job_id), req.applied, current_user["user_id"]
    )
    if not found:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"success": True}


@router.post("/feedback")
async def add_feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    await asyncio.to_thread(
        run_db.add_label,
        req.resume_id, req.job_id,
        1 if req.got_interview else 0,
        1 if req.is_golden else 0,
    )
    return {"success": True}


@router.post("/match_job")
async def match_job(req: MatchJobRequest, current_user: dict = Depends(get_current_user)):
    resumes = await asyncio.to_thread(run_db.get_resumes, current_user["user_id"])
    resume = next((r for r in resumes if str(r["resume_id"]) == str(req.resume_id)), None)
    if not resume:
        raise HTTPException(status_code=404, detail=f"resume_id {req.resume_id} not found")

    tfidf = _tfidf_score(resume["text"], req.job_description)
    skills, missing = _skills_jaccard(resume.get("skills", ""), req.job_description)
    exp = _experience_stub(resume.get("experience_years", "0"), req.job_description)
    final = round(0.4 * tfidf + 0.4 * skills + 0.2 * exp, 4)

    return {
        "resume_id":        req.resume_id,
        "name":             resume["name"],
        "final_score":      final,
        "tfidf_score":      round(tfidf, 4),
        "skills_score":     round(skills, 4),
        "experience_score": round(exp, 4),
        "missing_skills":   missing,
    }


@router.post("/match_job_by_id")
async def match_job_by_id(resume_id: str, job_id: str,
                          current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    resumes, jobs = await asyncio.gather(
        asyncio.to_thread(run_db.get_resumes, uid),
        asyncio.to_thread(run_db.get_jobs, uid),
    )
    resume = next((r for r in resumes if str(r["resume_id"]) == str(resume_id)), None)
    job    = next((j for j in jobs    if str(j["job_id"])    == str(job_id)),    None)
    if not resume:
        raise HTTPException(status_code=404, detail=f"resume_id {resume_id} not found")
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id {job_id} not found")

    job_text = f"{job['title']} {job['skills_required']} {job['description']}"
    tfidf = _tfidf_score(resume["text"], job_text)
    skills, missing = _skills_jaccard(resume.get("skills", ""), job.get("skills_required", ""))
    exp = _experience_stub(resume.get("experience_years", "0"), job.get("experience_required", "0"))
    final = round(0.4 * tfidf + 0.4 * skills + 0.2 * exp, 4)

    return {
        "resume_id":        resume_id,
        "job_id":           job_id,
        "name":             resume["name"],
        "job_title":        job["title"],
        "final_score":      final,
        "tfidf_score":      round(tfidf, 4),
        "skills_score":     round(skills, 4),
        "experience_score": round(exp, 4),
        "missing_skills":   missing,
    }
