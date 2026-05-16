import os
import re
import json
import csv
import logging
from collections import defaultdict
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def strip_html(text: str) -> str:
    """Remove HTML/JS/CSS from job description text. Falls back to skills_required hint."""
    if not text:
        return text
    # Try JSON-LD schema.org description extraction first (LinkedIn, job boards)
    match = re.search(r'"description":\s*"((?:[^\\"]|\\.)*)"', text, re.DOTALL)
    if match:
        try:
            extracted = json.loads('"' + match.group(1) + '"').strip()
            if len(extracted) > 100:
                return extracted
        except Exception:
            pass
    # Strip script/style blocks
    text = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = (text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"'))
    # Drop lines that are JS/CSS artifacts
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 6:
            continue
        if re.search(r'(function\s*\(|window\.|NREUM|gtm\.js|\.push\s*\(|var\s+[a-zA-Z_$]|document\.|__cf_|dataLayer)', line):
            continue
        if re.match(r'^[{}\[\];:,\s\d.%#]+$', line):
            continue
        clean_lines.append(line)
    result = ' '.join(clean_lines).strip()
    return result if len(result) > 50 else text  # if stripping destroyed it, return original

# 1. Load Environment Variables First!
from dotenv import load_dotenv
load_dotenv()

# Import the new Specialized Judges System
from specialized_judges import (
    SpecializedJudgeEngine,
    calculate_overall_match,
    JUDGE_DEFAULT_WEIGHTS,
)

# Import data loaders from your CLI script
from linkedin_cli_v1 import load_resumes, load_jobs

from ats_checker import score_ats, ats_to_dict
import db as run_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUMES_CSV = os.path.join(BASE_DIR, "data", "resumes.csv")
JOBS_CSV = os.path.join(BASE_DIR, "data", "jobs.csv")
LABELS_CSV = os.path.join(BASE_DIR, "data", "labels.csv")

# --- Global State ---
resumes_data = {}
jobs_data = {}

def get_llm_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

# --- Pydantic Models ---
class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str
    use_coach: bool = True
    llm_models: list[str] = ["deepseek-r1:14b"]
    api_key: str = ""
    api_base_url: str = ""
    candidate_name: str = ""
    job_title: str = ""

class BatchSummaryRequest(BaseModel):
    batch_data: str
    model: str = "meta-llama/llama-3.1-8b-instruct:free"

class FeedbackRequest(BaseModel):
    resume_id: int
    job_id: int
    got_interview: bool
    is_golden: bool = False

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

# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global resumes_data, jobs_data
    print("🚀 Starting hireable API...")
    if not get_llm_api_key():
        print("⚠️ WARNING: No LLM API Key detected in .env file.")
    try:
        resumes_data = load_resumes(RESUMES_CSV)
        jobs_data = load_jobs(JOBS_CSV)
        print(f"✅ Loaded {len(resumes_data)} resumes and {len(jobs_data)} jobs.")
    except Exception as e:
        print(f"⚠️ Error loading CSV data: {e}")
    run_db.init_db()
    print("✅ SQLite run history DB initialised.")
    yield

app = FastAPI(title="hireable API", lifespan=lifespan)

# --- Initialize Judge Engine ---
judge_engine = SpecializedJudgeEngine()

OLLAMA_BASE = "http://127.0.0.1:11434"

def get_free_model() -> str:
    """Pick an available Ollama model not currently running inference."""
    import httpx
    try:
        busy = set()
        ps = httpx.get(f"{OLLAMA_BASE}/api/ps", timeout=3).json()
        for m in ps.get("models", []):
            busy.add(m["name"].split(":")[0])

        EMBEDDING_MODELS = {"nomic-embed-text", "nomic-embed", "mxbai-embed-large", "all-minilm", "bge-m3"}

        tags = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3).json()
        for m in tags.get("models", []):
            name = m["name"]
            base = name.split(":")[0]
            if base in EMBEDDING_MODELS:
                continue
            if base not in busy:
                print(f"🟢 Selected free model: {name}")
                return name

        # All models busy — return first available anyway
        models = tags.get("models", [])
        if models:
            fallback = models[0]["name"]
            print(f"🟡 All models busy, using: {fallback}")
            return fallback
    except Exception as e:
        print(f"⚠️ get_free_model failed: {e}")
    return "deepseek-r1:14b"

# --- LLM Caller (Updated for local Ollama) ---
async def execute_llm_call(system_or_prompt: str, prompt_or_model: str = None, model: str = None) -> dict:
    """Legacy function for compatibility - now uses local Ollama.
    Supports two call signatures:
      execute_llm_call(prompt, model)           -- original 2-arg form
      execute_llm_call(system, prompt, model)   -- 3-arg form used in /analyze
    """
    # Detect which calling convention is being used
    if model is not None:
        # 3-arg form: (system, user_prompt, model)
        system_prompt = system_or_prompt
        prompt = prompt_or_model
    else:
        # 2-arg form: (prompt, model)
        system_prompt = ""
        prompt = system_or_prompt
        model = prompt_or_model

    try:
        from specialized_judges import call_ollama
        print(f"🔍 Attempting LLM call with model: {model}")
        result = call_ollama(system_prompt, prompt, model, temperature=0.1)
        
        # If the selected model fails, fallback to deepseek-r1:14b
        if not result["success"]:
            fallback = get_free_model()
            if fallback != model:
                print(f"⚠️ Model {model} failed, falling back to {fallback}")
                result = call_ollama("", prompt, fallback, temperature=0.1)
        
        return result
    except Exception as e:
        print(f"❌ execute_llm_call failed: {str(e)}")
        return {"success": False, "error": str(e)}

# --- HEALTH ENDPOINT ---
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    llm_status = "disconnected"
    llm_detail = "Ollama not responding"
    
    # Test local Ollama connection — prefer a model not busy with benchmarking
    free = get_free_model()
    test_models = [free] + (["deepseek-r1:14b"] if free != "deepseek-r1:14b" else [])

    for model in test_models:
        try:
            test_result = await execute_llm_call("Respond with: OK", model)
            if test_result["success"]:
                llm_status = "connected"
                llm_detail = f"LLM service responding - Model: {model}"
                break
            else:
                llm_status = "error"
                llm_detail = f"LLM error: {test_result.get('error', 'Unknown error')[:50]}"
        except Exception as e:
            llm_status = "error"
            llm_detail = f"Connection failed: {str(e)[:50]}"
            continue  # Try next model
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "llm": llm_status,
            "llm_detail": llm_detail
        }
    }

# --- MATCH JOB: CLI-style TF-IDF + skills Jaccard scoring (no LLM) ---
# Week 3: REST conversion of linkedin_cli_v1.py composite score logic

class MatchJobRequest(BaseModel):
    resume_id: int
    job_description: str


def _tfidf_score(resume_text: str, job_text: str) -> float:
    """TF-IDF cosine similarity between resume and job description."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf = vec.fit_transform([resume_text, job_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


def _skills_jaccard(resume_skills: str, job_skills: str) -> tuple[float, list[str]]:
    """Jaccard similarity on skill tokens; returns score + missing skills list."""
    def tokenise(text: str) -> set[str]:
        import re
        tokens = re.split(r"[,\s/|]+", text.lower())
        return {t.strip() for t in tokens if len(t.strip()) > 2}

    resume_set = tokenise(resume_skills)
    job_set = tokenise(job_skills)
    if not job_set:
        return 0.0, []
    intersection = resume_set & job_set
    union = resume_set | job_set
    missing = sorted(job_set - resume_set)[:10]
    return len(intersection) / len(union), missing


def _experience_stub(resume_years: str, job_years: str) -> float:
    """Simple experience match stub: 1.0 meets, 0.5 partial, 0.0 unmet."""
    import re
    def parse_years(s: str) -> float:
        nums = re.findall(r"\d+\.?\d*", str(s))
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


@app.post("/match_job")
async def match_job(req: MatchJobRequest):
    """CLI-style composite score: TF-IDF 40% + skills Jaccard 40% + exp stub 20%."""
    from linkedin_cli_v1 import load_resumes
    base = os.path.dirname(os.path.abspath(__file__))
    resumes = load_resumes(os.path.join(base, "data", "resumes.csv"))
    resume = next((r for r in resumes if int(r["resume_id"]) == req.resume_id), None)
    if not resume:
        raise HTTPException(status_code=404, detail=f"resume_id {req.resume_id} not found")

    tfidf = _tfidf_score(resume["text"], req.job_description)
    skills, missing = _skills_jaccard(resume.get("skills", ""), req.job_description)
    exp = _experience_stub(resume.get("experience_years", "0"), req.job_description)
    final = round(0.4 * tfidf + 0.4 * skills + 0.2 * exp, 4)

    return {
        "resume_id": req.resume_id,
        "name": resume["name"],
        "final_score": final,
        "tfidf_score": round(tfidf, 4),
        "skills_score": round(skills, 4),
        "experience_score": round(exp, 4),
        "missing_skills": missing,
    }


@app.post("/match_job_by_id")
async def match_job_by_id(resume_id: int, job_id: int):
    """Match resume against a job from data/jobs.csv by ID."""
    from linkedin_cli_v1 import load_resumes, load_jobs
    base = os.path.dirname(os.path.abspath(__file__))
    resumes = load_resumes(os.path.join(base, "data", "resumes.csv"))
    jobs = load_jobs(os.path.join(base, "data", "jobs.csv"))
    resume = next((r for r in resumes if int(r["resume_id"]) == resume_id), None)
    job = next((j for j in jobs if int(j["job_id"]) == job_id), None)
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
        "resume_id": resume_id,
        "job_id": job_id,
        "name": resume["name"],
        "job_title": job["title"],
        "final_score": final,
        "tfidf_score": round(tfidf, 4),
        "skills_score": round(skills, 4),
        "experience_score": round(exp, 4),
        "missing_skills": missing,
    }


# --- FRONTEND ROUTE ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    with open(dashboard_path, encoding="utf-8") as f:
        return f.read()

# --- DATABASE ROUTES (The missing links!) ---
@app.get("/resumes")
async def get_resumes():
    global resumes_data
    resumes_data = load_resumes(RESUMES_CSV) # Refresh from disk
    return resumes_data

@app.get("/jobs")
async def get_jobs():
    global jobs_data
    jobs_data = load_jobs(JOBS_CSV) # Refresh from disk
    return jobs_data

@app.post("/resumes")
async def add_resume(req: AddResumeRequest):
    global resumes_data
    resumes_data = load_resumes(RESUMES_CSV)  # refresh before ID calculation
    # Parse numeric part from both "resume_001" and plain "1" formats
    def parse_num(rid):
        s = str(rid).lower().replace("resume_", "").strip()
        return int(s) if s.isdigit() else 0
    existing_nums = [parse_num(r.get('resume_id', 0)) for r in resumes_data]
    next_num = max(existing_nums) + 1 if existing_nums else 1
    new_id = f"resume_{next_num:03d}"
    with open(RESUMES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([new_id, req.name, req.title, req.experience_years, req.skills, req.text])
    resumes_data = load_resumes(RESUMES_CSV)
    return {"success": True, "id": new_id}

@app.post("/jobs")
async def add_job(req: AddJobRequest):
    global jobs_data
    existing_ids = [int(j.get('job_id', 0)) for j in jobs_data if str(j.get('job_id', '')).isdigit()]
    new_id = max(existing_ids) + 1 if existing_ids else 1
    clean_desc = strip_html(req.description)
    with open(JOBS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([new_id, req.title, req.company, req.experience_required, req.skills_required, clean_desc])
    jobs_data = load_jobs(JOBS_CSV)
    return {"success": True, "id": new_id}

@app.post("/feedback")
async def add_feedback(req: FeedbackRequest):
    fit_label = 1 if req.got_interview else 0
    golden_flag = 1 if req.is_golden else 0
    file_exists = os.path.isfile(LABELS_CSV)
    with open(LABELS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["resume_id", "job_id", "fit_label", "golden"])
        writer.writerow([req.resume_id, req.job_id, fit_label, golden_flag])
    return {"success": True}

# --- EVALUATION ROUTES ---
@app.post("/analyze")
async def analyze_resume_unified(request: AnalyzeRequest):
    try:
        job_description = request.job_description  # chunker.py handles extraction per-judge

        print(f"🚀 Starting 5-stage analysis with model: {request.llm_models[0]}")
        print(f"📝 Resume length: {len(request.resume_text)} chars")
        print(f"📝 Job length: {len(job_description)} chars")

        model = request.llm_models[0]
        timestamp = datetime.utcnow().isoformat()

        # --- ATS pre-flight (instant, no LLM) ---
        ats_result = score_ats(request.resume_text, job_description)
        ats_dict = ats_to_dict(ats_result)
        print(f"🗂  ATS pre-flight: {ats_result.score}/100 ({ats_result.grade}) — {len(ats_result.issues)} issues")

        # Load skills taxonomy
        skills_taxonomy = {}
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            SKILLS_TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "skills_taxonomy.json")
            with open(SKILLS_TAXONOMY_PATH, 'r', encoding='utf-8') as f:
                skills_taxonomy = json.load(f)
            print(f"✅ Loaded skills taxonomy with {len(skills_taxonomy)} domains")
        except Exception as e:
            print(f"Warning: Could not load skills taxonomy: {e}")
        
        # Use new 5-stage judge pipeline
        from specialized_judges import SpecializedJudgeEngine
        
        print("🔧 Initializing 5-stage judge engine...")
        engine = SpecializedJudgeEngine(confidence_threshold=75, max_retries=3)
        
        # Run full pipeline
        pipeline_result = await engine.run_all_judges(
            resume_text=request.resume_text,
            job_description=job_description,
            llm_model=model,
            skills_taxonomy=skills_taxonomy,
            api_key=request.api_key,
            api_base_url=request.api_base_url,
        )
        
        if not pipeline_result:
            raise HTTPException(status_code=500, detail="Judge pipeline failed to produce results")

        # Embed ATS into pipeline_result so it's persisted in pipeline_json
        pipeline_result["ats_preflight"] = ats_dict

        # Recompute overall with learned weights if enough feedback exists
        learned = compute_learned_weights()
        if learned["source"] == "learned":
            s = pipeline_result.get("skills", {}).get("scores", {})
            e = pipeline_result.get("experience", {}).get("scores", {})
            sr = pipeline_result.get("seniority", {}).get("scores", {})
            rl = pipeline_result.get("relevancy", {}).get("scores", {})
            im = pipeline_result.get("impact", {}).get("scores", {})
            pipeline_result["overall"] = calculate_overall_match(s, e, sr, rl, im, learned["weights"])
            pipeline_result["weights_source"] = "learned"
            print(f"⚖️  Using learned weights: {learned['weights']}")
        else:
            pipeline_result["weights_source"] = "default"

        print("📊 Processing pipeline results...")

        # Extract component scores
        overall = pipeline_result.get("overall", {})
        skills_scores = pipeline_result.get("skills", {}).get("scores", {})
        experience_scores = pipeline_result.get("experience", {}).get("scores", {})
        seniority_scores = pipeline_result.get("seniority", {}).get("scores", {})
        relevancy_scores = pipeline_result.get("relevancy", {}).get("scores", {})
        impact_scores = pipeline_result.get("impact", {}).get("scores", {})
        coach_result = pipeline_result.get("coach", {})
        
        # Convert to percentage format for frontend
        skills_pct = round(skills_scores.get("score", 0) * 100, 1)
        experience_pct = round(experience_scores.get("score", 0) * 100, 1)
        seniority_pct = round(seniority_scores.get("score", 0) * 100, 1)
        relevancy_pct = round(relevancy_scores.get("score", 0) * 100, 1)
        impact_pct = round(impact_scores.get("score", 0) * 100, 1)
        overall_pct = round(overall.get("overall_score", 0) * 100, 1)
        
        print(f"📊 Final scores - Overall: {overall_pct}%, Skills: {skills_pct}%, Experience: {experience_pct}%")
        
        # Extract gap lists for frontend
        missing_skills = skills_scores.get("missing_skills", [])
        unmet_responsibilities = experience_scores.get("unmet_responsibilities", [])
        absent_seniority_signals = seniority_scores.get("absent_signals", [])
        irrelevant_statements = relevancy_scores.get("irrelevant_statements", [])
        unquantified_statements = impact_scores.get("unquantified_statements", [])
        
        # Extract coach recommendations
        coach_recommendations = None
        if coach_result.get("success") and request.use_coach:
            coach_data = coach_result.get("data", {})
            coach_recommendations = coach_data.get("improvement_analysis", [])
        
        final_evaluation = {
            "overall_score": overall.get("overall_score", 0),
            "skills_score": skills_scores.get("score", 0),
            "experience_score": experience_scores.get("score", 0),
            "seniority_score": seniority_scores.get("score", 0),
            "relevancy_score": relevancy_scores.get("score", 0),
            "impact_score": impact_scores.get("score", 0),
            "pipeline_data": pipeline_result  # Full pipeline data for debugging
        }
        # Save to SQLite for persistent history
        result_id = f"analysis_{timestamp.replace(':', '-').replace('.', '-')}"
        try:
            run_db.save_run(
                run_id=result_id,
                model=model,
                resume_text=request.resume_text,
                job_desc=request.job_description,
                pipeline_result=pipeline_result,
                ats_result=ats_dict,
                candidate_name=request.candidate_name,
                job_title=request.job_title,
            )
        except Exception as db_err:
            logger.warning(f"SQLite save failed (non-fatal): {db_err}")

        # Save results to JSON files (legacy)
        pipeline_save = {
            "timestamp": timestamp,
            "model": model,
            "resume_text": request.resume_text,
            "job_description": request.job_description,
            "pipeline_result": pipeline_result,
            "result_id": result_id
        }
        with open(os.path.join(BASE_DIR, "results", "pipeline", f"{result_id}.json"), "w", encoding="utf-8") as f:
            json.dump(pipeline_save, f, indent=2, ensure_ascii=False)
        
        # Save individual judge results for compatibility
        skills_judge_result = pipeline_result.get("skills", {}).get("judge_result", {})
        experience_judge_result = pipeline_result.get("experience", {}).get("judge_result", {})
        seniority_judge_result = pipeline_result.get("seniority", {}).get("judge_result", {})
        relevancy_judge_result = pipeline_result.get("relevancy", {}).get("judge_result", {})
        impact_judge_result = pipeline_result.get("impact", {}).get("judge_result", {})
        coach_judge_result = pipeline_result.get("coach", {})
        
        # Save individual results (legacy compatibility)
        for judge_name, judge_data in [
            ("skills_judge", skills_judge_result),
            ("experience_judge", experience_judge_result), 
            ("seniority_judge", seniority_judge_result),
            ("relevancy_judge", relevancy_judge_result),
            ("impact_judge", impact_judge_result),
            ("coach_judge", coach_judge_result)
        ]:
            if judge_data:
                save_data = {
                    "timestamp": timestamp,
                    "model": model,
                    "resume_text": request.resume_text,
                    "job_description": request.job_description,
                    "raw_response": judge_data.get("raw_response", ""),
                    "parsed_data": judge_data.get("data", {}),
                    "result_id": result_id
                }
                with open(os.path.join(BASE_DIR, "results", judge_name, f"{result_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)

        # Build response structure with all 5 components
        return {
            "success": True,
            "match_score": overall_pct,
            "metrics": {
                "skills_coverage": skills_pct,
                "experience_depth": experience_pct,
                "seniority_calibration": seniority_pct,
                "statement_relevancy": relevancy_pct,
                "impact_quantification": impact_pct
            },
            "analysis": {
                "missing_critical_skills": missing_skills,
                "unmet_responsibilities": unmet_responsibilities,
                "absent_seniority_signals": absent_seniority_signals,
                "irrelevant_statements": irrelevant_statements,
                "unquantified_statements": unquantified_statements,
                "coach_recommendations": coach_recommendations
            },
            "pipeline_data": pipeline_result,
            "ats_preflight": ats_dict,
            "metadata": {
                "model_used": model,
                "timestamp": timestamp,
                "result_id": result_id,
                "pipeline_version": "5-stage-v2.0"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_results")
async def list_results(limit: int = 50, offset: int = 0):
    """Return saved pipeline results from SQLite, falling back to JSON files for legacy runs."""

    # Build CSV lookups once — used for both DB enrichment and JSON fallback
    resume_lookup: dict[str, tuple[str, str]] = {}
    job_lookup: dict[str, str] = {}
    try:
        for r in load_resumes(RESUMES_CSV):
            prefix = (r.get("text") or "")[:200].strip()
            if prefix:
                resume_lookup[prefix] = (r.get("name", ""), r.get("title", ""))
    except Exception:
        pass
    try:
        for j in load_jobs(JOBS_CSV):
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
        # Last resort: find first short line that looks like a name (not a location/email/url)
        _loc_re = re.compile(r',\s*[A-Z]|\d{5}|@|http|www\.|linkedin\.com', re.IGNORECASE)
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

    # Primary: SQLite
    try:
        db_rows = run_db.list_runs(limit=limit, offset=offset)
        if db_rows:
            # Enrich rows where candidate_name/job_title were not stored (pre-fix runs)
            for row in db_rows:
                if not row.get("resume_name"):
                    row["resume_name"] = match_resume(row.get("resume_snippet", ""))
                if not row.get("job_title"):
                    row["job_title"] = match_job(row.get("job_snippet", ""))
            return db_rows
    except Exception as db_err:
        logger.warning(f"SQLite list_runs failed, falling back to JSON files: {db_err}")

    # Fallback: scan JSON files (pre-SQLite runs)
    pipeline_dir = os.path.join(BASE_DIR, "results", "pipeline")
    results = []
    try:
        files = sorted(
            [f for f in os.listdir(pipeline_dir) if f.endswith(".json")],
            reverse=True  # newest first
        )
        for fname in files:
            fpath = os.path.join(pipeline_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pr = data.get("pipeline_result", {})
                overall = pr.get("overall", {})
                component_scores = overall.get("component_scores", {})
                # Extract coach recommendations
                coach = pr.get("coach", {})
                coach_recs = []
                if coach.get("success") and coach.get("data"):
                    coach_recs = coach["data"].get("improvement_analysis", [])
                # Extract gaps
                skills_data = pr.get("skills", {}).get("judge_result", {}).get("data", {})
                exp_data = pr.get("experience", {}).get("judge_result", {}).get("data", {})
                seniority_data = pr.get("seniority", {}).get("judge_result", {}).get("data", {})
                relevancy_data = pr.get("relevancy", {}).get("judge_result", {}).get("data", {})
                impact_data = pr.get("impact", {}).get("judge_result", {}).get("data", {})

                # Extract candidate name and job title from CSV lookup
                resume_text = data.get("resume_text", "")
                job_desc = data.get("job_description", "")
                resume_name = match_resume(resume_text)
                job_title = match_job(job_desc)

                results.append({
                    "result_id": data.get("result_id", fname.replace(".json", "")),
                    "timestamp": data.get("timestamp", ""),
                    "model": data.get("model", ""),
                    "resume_name": resume_name,
                    "job_title": job_title,
                    "resume_snippet": resume_text[:120],
                    "job_snippet": job_desc[:120],
                    "match_score": round(overall.get("overall_score", 0) * 100, 1),
                    "metrics": {
                        "skills_coverage": round(component_scores.get("skills", 0) * 100, 1),
                        "experience_depth": round(component_scores.get("experience", 0) * 100, 1),
                        "seniority_calibration": round(component_scores.get("seniority", 0) * 100, 1),
                        "statement_relevancy": round(component_scores.get("relevancy", 0) * 100, 1),
                        "impact_quantification": round(component_scores.get("impact", 0) * 100, 1),
                    },
                    "analysis": {
                        "missing_critical_skills": skills_data.get("missing_critical_skills", []),
                        "unmet_responsibilities": exp_data.get("unmet_responsibilities", []),
                        "absent_seniority_signals": seniority_data.get("absent_signals", []),
                        "irrelevant_statements": relevancy_data.get("irrelevant_statements", []),
                        "unquantified_statements": impact_data.get("unquantified_statements", []),
                        "coach_recommendations": coach_recs,
                    },
                    "pipeline_data": {
                        "skills": pr.get("skills", {}),
                        "experience": pr.get("experience", {}),
                        "seniority": pr.get("seniority", {}),
                        "relevancy": pr.get("relevancy", {}),
                        "impact": pr.get("impact", {}),
                    }
                })
            except Exception as e:
                logger.warning(f"Could not parse result file {fname}: {e}")
    except FileNotFoundError:
        pass
    return results

class SummarizeResultRequest(BaseModel):
    result_id: str
    model: str = "deepseek-r1:14b"

@app.post("/summarize_result")
async def summarize_result(request: SummarizeResultRequest):
    """Generate an AI summary for a single result combining all judge outputs."""
    timestamp = datetime.utcnow().isoformat()
    result_id = request.result_id
    model = request.model

    # Load the full pipeline data
    pipeline_dir = os.path.join(BASE_DIR, "results", "pipeline")
    pipeline_file = os.path.join(pipeline_dir, f"{result_id}.json")
    if not os.path.exists(pipeline_file):
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

    with open(pipeline_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pr = data.get("pipeline_result", {})
    overall = pr.get("overall", {})
    component_scores = overall.get("component_scores", {})

    resume_text = data.get("resume_text", "")
    job_description = data.get("job_description", "")
    skills_data = pr.get("skills", {}).get("judge_result", {}).get("data", {})
    exp_data = pr.get("experience", {}).get("judge_result", {}).get("data", {})
    seniority_data = pr.get("seniority", {}).get("judge_result", {}).get("data", {})
    impact_data = pr.get("impact", {}).get("judge_result", {}).get("data", {})
    coach = pr.get("coach", {})
    coach_recs = []
    if coach.get("success") and coach.get("data"):
        coach_recs = coach["data"].get("improvement_analysis", [])[:3]

    prompt = f"""You are a senior technical recruiter. Write a concise 3-4 sentence summary of this candidate's fit for the role.

CANDIDATE RESUME (excerpt):
{resume_text[:600]}

JOB DESCRIPTION (excerpt):
{job_description[:400]}

JUDGE SCORES:
- Overall Match: {round(overall.get('overall_score', 0) * 100, 1)}%
- Skills Coverage: {round(component_scores.get('skills', 0) * 100, 1)}%
- Experience Depth: {round(component_scores.get('experience', 0) * 100, 1)}%
- Seniority Calibration: {round(component_scores.get('seniority', 0) * 100, 1)}%
- Statement Relevancy: {round(component_scores.get('relevancy', 0) * 100, 1)}%
- Impact Quantification: {round(component_scores.get('impact', 0) * 100, 1)}%

CRITICAL GAPS IDENTIFIED:
- Missing Skills: {', '.join(skills_data.get('missing_critical_skills', [])) or 'None'}
- Unmet Responsibilities: {', '.join(exp_data.get('unmet_responsibilities', [])) or 'None'}
- Missing Seniority Signals: {', '.join(seniority_data.get('absent_signals', [])) or 'None'}
- Unquantified Statements: {len(impact_data.get('unquantified_statements', []))} items lack metrics

TOP COACH RECOMMENDATIONS:
{chr(10).join([f"- {r.get('suggested_improvement', '')}" for r in coach_recs]) or 'None generated'}

Write a clear, actionable summary: state the candidate's strongest selling point, their biggest gap, and one specific thing they should fix. Output plain text only, no markdown, no bullet points."""

    result = await execute_llm_call(prompt, model)
    summary_text = result.get("data", "Summary generation failed.")

    # Clean any HTML/markdown
    import re as _re
    summary_text = _re.sub(r'<[^>]+>', '', summary_text).strip()

    # Save to summary folder
    summary_id = f"analysis_{timestamp.replace(':', '-').replace('.', '-')}"
    summary_data = {
        "timestamp": timestamp,
        "model": model,
        "result_id": result_id,
        "summary": summary_text,
        "summary_id": summary_id
    }
    summary_path = os.path.join(BASE_DIR, "results", "summary", f"{summary_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {"summary": summary_text}

@app.post("/summarize_batch")
async def summarize_batch(request: BatchSummaryRequest):
    timestamp = datetime.utcnow().isoformat()
    summary_id = f"summary_{timestamp.replace(':', '-').replace('.', '-')}"
    
    prompt = f"""You are an executive HR Director. Review this batch of AI resume-to-job match scores:
{request.batch_data}

Write a 3-4 sentence executive summary. Identify the top candidate/job pairing, note any major trends, and be decisive. 

IMPORTANT: Output plain text only. No HTML, no markdown, no formatting. Just plain text."""
    result = await execute_llm_call(prompt, request.model)
    
    try:
        summary_text = result["data"]
        # Clean up any HTML/markdown that might slip through
        if "<!DOCTYPE" in summary_text or "<html" in summary_text:
            # Extract text from HTML if needed
            import re
            # Remove HTML tags
            summary_text = re.sub(r'<[^>]+>', '', summary_text)
            # Clean up extra whitespace
            summary_text = ' '.join(summary_text.split())
        
        # Save summary to file
        summary_data = {
            "timestamp": timestamp,
            "model": request.model,
            "batch_data": request.batch_data,
            "summary": summary_text,
            "summary_id": summary_id
        }
        with open(os.path.join(BASE_DIR, "results", "summary", f"{summary_id}.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Individual candidate summary endpoint
class CandidateSummaryRequest(BaseModel):
    resume_name: str
    job_title: str
    score: float
    metrics: dict
    missing_skills: list
    model: str

@app.post("/summarize_candidate")
async def summarize_candidate(request: CandidateSummaryRequest):
    timestamp = datetime.utcnow().isoformat()
    summary_id = f"candidate_summary_{timestamp.replace(':', '-').replace('.', '-')}"
    
    prompt = f"""
    Act as an expert Technical Recruiter. Write a punchy, 2-3 sentence summary of this specific candidate's fit for the role.
    
    Candidate: {request.resume_name}
    Role: {request.job_title}
    Overall Match Score: {request.score}%
    Skills Match: {request.metrics.get('skills_coverage', 'N/A')}%
    Experience Match: {request.metrics.get('experience_depth', 'N/A')}%
    Critical Missing Skills: {', '.join(request.missing_skills) if request.missing_skills else 'None'}
    
    Highlight their strongest selling point and clearly state their biggest gap. Be objective, professional, and concise. Output plain text only.
    """
    
    result = await execute_llm_call(prompt, request.model)
    
    try:
        summary_text = result["data"]
        # Clean up any HTML/markdown that might slip through
        if "<!DOCTYPE" in summary_text or "<html" in summary_text:
            import re
            summary_text = re.sub(r'<[^>]+>', '', summary_text)
            summary_text = ' '.join(summary_text.split())
        
        # Save summary to file
        summary_data = {
            "timestamp": timestamp,
            "model": request.model,
            "resume_name": request.resume_name,
            "job_title": request.job_title,
            "score": request.score,
            "metrics": request.metrics,
            "missing_skills": request.missing_skills,
            "summary": summary_text,
            "summary_id": summary_id
        }
        with open(os.path.join(BASE_DIR, "results", "summary", f"{summary_id}.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

OUTCOMES_FILE = os.path.join(BASE_DIR, "results", "outcomes.jsonl")
JUDGE_FEEDBACK_FILE = os.path.join(BASE_DIR, "results", "judge_feedback.jsonl")

# Which judges each outcome signal trains
INTERVIEW_JUDGES = ["skills", "experience", "seniority", "relevancy"]
JOB_JUDGES = ["impact", "overall"]
MIN_SAMPLES_FOR_LEARNING = 3


def compute_learned_weights(feedback_file: str = None) -> dict:
    """Derive judge weights from outcome feedback. Falls back to defaults for judges with <3 samples."""
    feedback_file = feedback_file or JUDGE_FEEDBACK_FILE
    entries = []
    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except FileNotFoundError:
        return {"weights": JUDGE_DEFAULT_WEIGHTS, "source": "default", "judge_stats": {}}

    component_judges = list(JUDGE_DEFAULT_WEIGHTS.keys())  # skills, experience, seniority, relevancy, impact
    by_judge = defaultdict(list)
    for e in entries:
        if e.get("judge") in component_judges:
            by_judge[e["judge"]].append(e)

    raw, judge_stats = {}, {}
    for judge in component_judges:
        jentries = by_judge[judge]
        if len(jentries) >= MIN_SAMPLES_FOR_LEARNING:
            accuracy = sum(1 for e in jentries if e["correct"]) / len(jentries)
            raw[judge] = max(0.1, accuracy)  # floor so no judge is zeroed out
            judge_stats[judge] = {"accuracy": round(accuracy * 100, 1), "n": len(jentries), "source": "learned"}
        else:
            raw[judge] = JUDGE_DEFAULT_WEIGHTS[judge]
            judge_stats[judge] = {"n": len(jentries), "source": "default", "accuracy": None}

    total = sum(raw.values())
    normalized = {j: round(w / total, 4) for j, w in raw.items()}
    # Fix rounding drift — add remainder to highest-weight judge
    diff = round(1.0 - sum(normalized.values()), 4)
    if diff:
        normalized[max(normalized, key=normalized.get)] = round(
            normalized[max(normalized, key=normalized.get)] + diff, 4
        )

    any_learned = any(s.get("source") == "learned" for s in judge_stats.values())
    return {"weights": normalized, "source": "learned" if any_learned else "default", "judge_stats": judge_stats}


class RecordOutcomeRequest(BaseModel):
    result_id: str
    predicted_score: float
    got_job: Optional[bool] = None
    got_interview: Optional[bool] = None

@app.post("/record_outcome")
async def record_outcome(req: RecordOutcomeRequest):
    entry = {
        "result_id": req.result_id,
        "predicted_score": req.predicted_score,
        "got_job": req.got_job,
        "got_interview": req.got_interview,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(OUTCOMES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # Write per-judge feedback so each LLM can calibrate on real outcomes
    pipeline_file = os.path.join(BASE_DIR, "results", "pipeline", f"{req.result_id}.json")
    if os.path.isfile(pipeline_file):
        with open(pipeline_file, "r", encoding="utf-8") as pf:
            pipeline_data = json.load(pf)
        pr = pipeline_data.get("pipeline_result", {})
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
                feedback_entries.append({
                    "judge": j, "signal": "got_interview",
                    "predicted_score": score, "outcome": req.got_interview,
                    "correct": (score >= 0.5) == req.got_interview,
                    "timestamp": entry["timestamp"],
                })
        if req.got_job is not None:
            for j in JOB_JUDGES:
                score = judge_scores.get(j)
                if score is None:
                    continue
                feedback_entries.append({
                    "judge": j, "signal": "got_job",
                    "predicted_score": score, "outcome": req.got_job,
                    "correct": (score >= 0.6) == req.got_job,
                    "timestamp": entry["timestamp"],
                })
        if feedback_entries:
            with open(JUDGE_FEEDBACK_FILE, "a", encoding="utf-8") as f:
                for fb in feedback_entries:
                    f.write(json.dumps(fb) + "\n")

    return {"success": True}

@app.get("/outcomes")
async def get_outcomes():
    """Return all recorded outcomes and prediction accuracy stats."""
    outcomes = []
    try:
        with open(OUTCOMES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    outcomes.append(json.loads(line))
    except FileNotFoundError:
        pass

    if not outcomes:
        return {"outcomes": [], "accuracy": None, "interview_accuracy": None, "total": 0}

    job_o = [o for o in outcomes if o.get("got_job") is not None]
    int_o = [o for o in outcomes if o.get("got_interview") is not None]
    correct_job = sum(1 for o in job_o if (o["predicted_score"] >= 60) == o["got_job"])
    correct_int = sum(1 for o in int_o if (o["predicted_score"] >= 50) == o["got_interview"])

    return {
        "outcomes": outcomes,
        "accuracy": round(correct_job / len(job_o) * 100, 1) if job_o else None,
        "interview_accuracy": round(correct_int / len(int_o) * 100, 1) if int_o else None,
        "total": len(outcomes),
        "total_job": len(job_o),
        "total_interview": len(int_o),
        "correct": correct_job,
        "correct_interview": correct_int,
    }

@app.get("/judge_calibration")
async def judge_calibration():
    """Per-judge accuracy stats from real outcome feedback."""
    feedback = []
    try:
        with open(JUDGE_FEEDBACK_FILE, "r", encoding="utf-8") as f:
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
            "total": len(entries),
            "correct": correct,
            "accuracy": round(correct / len(entries) * 100, 1),
            "signal": entries[-1].get("signal", ""),
            "avg_score_pos": round(sum(e["predicted_score"] for e in pos) / len(pos), 3) if pos else None,
            "avg_score_neg": round(sum(e["predicted_score"] for e in neg) / len(neg), 3) if neg else None,
        }

    return {"calibration": calibration, "total_feedback": len(feedback)}


@app.get("/judge_weights")
async def judge_weights():
    """Return current active judge weights — learned from feedback or defaults."""
    return compute_learned_weights()


class InterviewPrepRequest(BaseModel):
    result_id: str
    model: str = ""
    force_regen: bool = False

@app.post("/interview_prep")
async def interview_prep(req: InterviewPrepRequest):
    """Generate interview questions + model answers. Caches result to results/interview_prep/."""
    prep_dir = os.path.join(BASE_DIR, "results", "interview_prep")
    os.makedirs(prep_dir, exist_ok=True)
    cache_file = os.path.join(prep_dir, f"{req.result_id}.json")

    # Return cached version unless force_regen requested
    if os.path.exists(cache_file) and not req.force_regen:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return {"questions": cached.get("questions", []), "result_id": req.result_id, "cached": True}

    pipeline_dir = os.path.join(BASE_DIR, "results", "pipeline")
    pipeline_file = os.path.join(pipeline_dir, f"{req.result_id}.json")
    if not os.path.exists(pipeline_file):
        raise HTTPException(status_code=404, detail=f"Result {req.result_id} not found")

    with open(pipeline_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pr = data.get("pipeline_result", {})
    resume_text = data.get("resume_text", "")
    job_description = data.get("job_description", "")

    skills_data   = pr.get("skills", {}).get("judge_result", {}).get("data", {})
    exp_data      = pr.get("experience", {}).get("judge_result", {}).get("data", {})
    seniority_data= pr.get("seniority", {}).get("judge_result", {}).get("data", {})
    impact_data   = pr.get("impact", {}).get("judge_result", {}).get("data", {})

    missing_skills      = skills_data.get("missing_critical_skills", [])
    present_skills      = skills_data.get("present_skills", [])
    unmet_resp          = [r.get("responsibility","") for r in exp_data.get("responsibility_analysis",[]) if r.get("experience_score",1) < 0.5]
    absent_signals      = [s.get("signal","") for s in seniority_data.get("seniority_analysis",[]) if s.get("is_demonstrated") == 0]
    unquantified        = [s.get("resume_statement","") for s in impact_data.get("impact_analysis",[]) if s.get("is_quantified") == 0]

    # Detect domain gaps from job title that skills judge may have missed
    # (e.g. a pentesting role needing security domain knowledge)
    DOMAIN_GAP_KEYWORDS = {
        "pentest": ["offensive security mindset", "vulnerability assessment", "security tooling (Burp Suite / OWASP)"],
        "security": ["threat modeling", "CVE/exploit research", "security review process"],
        "devops": ["CI/CD pipeline ownership", "infrastructure-as-code", "on-call incident response"],
        "frontend": ["UI component architecture", "browser performance profiling", "accessibility standards"],
        "backend": ["database schema design", "API rate limiting and auth", "distributed systems"],
        "data engineer": ["pipeline orchestration (Airflow/Prefect)", "data warehouse modeling", "dbt/SQL at scale"],
        "mlops": ["model serving infrastructure", "drift monitoring", "A/B experiment pipelines"],
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
    # Parse JSON from response
    import re as _re
    json_match = _re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            raw_questions = parsed.get("questions", [])
            # Deduplicate — max 1 question per unique gap
            seen_gaps = set()
            questions = []
            for q in raw_questions:
                gap_key = q.get("gap", "").lower().strip()
                if gap_key not in seen_gaps:
                    seen_gaps.add(gap_key)
                    questions.append(q)
            # Cache to disk
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"questions": questions, "result_id": req.result_id}, f, indent=2, ensure_ascii=False)
            return {"questions": questions, "result_id": req.result_id, "cached": False}
        except json.JSONDecodeError:
            pass
    raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON")


class EditResumeRequest(BaseModel):
    result_id: str
    model: str = ""
    force_regen: bool = False

@app.post("/edit_resume")
async def edit_resume(req: EditResumeRequest):
    """Apply coach recommendations to resume text. Caches result to results/tailored_resumes/."""
    tailored_dir = os.path.join(BASE_DIR, "results", "tailored_resumes")
    os.makedirs(tailored_dir, exist_ok=True)
    cache_file = os.path.join(tailored_dir, f"{req.result_id}.txt")

    # Return cached version unless force_regen requested
    if os.path.exists(cache_file) and not req.force_regen:
        with open(cache_file, "r", encoding="utf-8") as f:
            return {"edited_text": f.read(), "result_id": req.result_id, "cached": True}

    pipeline_file = os.path.join(BASE_DIR, "results", "pipeline", f"{req.result_id}.json")
    if not os.path.exists(pipeline_file):
        raise HTTPException(status_code=404, detail=f"Result {req.result_id} not found")

    with open(pipeline_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    resume_text = data.get("resume_text", "")

    pr = data.get("pipeline_result", {})
    coach = pr.get("coach", {})
    coach_recs = []
    if coach.get("success") and coach.get("data"):
        coach_recs = coach["data"].get("improvement_analysis", [])

    # Only apply rewrites that modify existing content (bullet_rewrite, deletion).
    # Skip strategic_addition and gap_warning — those either fabricate new bullets
    # or are informational flags, not edits to apply.
    # Reject rewrites containing placeholder metrics: [X%], bare X%/Y%/Z%, or bracket-wrapped values.
    import re as _re
    PLACEHOLDER_RE = _re.compile(r'\[.{1,30}?\]|\b[A-Z]%|\b[A-Z]\s*%')
    REWRITE_TYPES = {"bullet_rewrite", "deletion"}
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
        # No bullet_rewrite items — do a direct JD-guided reframe.
        # Reword existing bullets to surface transferable skills for this specific role.
        # Never add new bullets or invent metrics.
        job_snippet = data.get("job_description", "")[:600]
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

    # Cache to disk
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(edited_text)

    # Validation loop: ATS score before vs after (instant — no LLM)
    original_resume = data.get("resume_text", "")
    job_desc_text = data.get("job_description", "")
    ats_before = ats_to_dict(score_ats(original_resume, job_desc_text))
    ats_after  = ats_to_dict(score_ats(edited_text, job_desc_text))
    delta = round(ats_after["score"] - ats_before["score"], 1)
    fixed = [c["label"] for c in ats_after["criteria"]
             if c["passed"] and not next((b for b in ats_before["criteria"] if b["id"] == c["id"]), {}).get("passed")]
    regressed = [c["label"] for c in ats_after["criteria"]
                 if not c["passed"] and next((b for b in ats_before["criteria"] if b["id"] == c["id"]), {}).get("passed")]
    ats_validation = {
        "before": {"score": ats_before["score"], "grade": ats_before["grade"]},
        "after":  {"score": ats_after["score"],  "grade": ats_after["grade"]},
        "delta": delta,
        "fixed": fixed,
        "regressed": regressed,
        "verdict": "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged",
    }
    try:
        run_db.update_run_tailored(req.result_id, edited_text, ats_validation)
    except Exception as db_err:
        logger.warning(f"SQLite update_run_tailored failed (non-fatal): {db_err}")

    return {
        "edited_text": edited_text,
        "result_id": req.result_id,
        "cached": False,
        "ats_validation": ats_validation,
    }


# --- PDF RESUME EXTRACTION ---
@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract plain text from a PDF resume and parse name/title/skills via LLM."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    from io import BytesIO
    contents = await file.read()
    full_text = ""
    extraction_method = "pypdf"

    # Stage 1: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(contents))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}")

    # Confidence check: flag sparse/spaced-char PDFs
    def _is_sparse(text: str) -> bool:
        tokens = text.split()
        if not tokens:
            return True
        single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
        return len(text.strip()) < 100 or single_char_ratio > 0.4

    # Fix spaced-character encoding artifact
    if full_text and not _is_sparse(full_text):
        tokens = full_text.split()
        if tokens and sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.4:
            full_text = re.sub(r'(?<=[A-Za-z0-9.,@]) (?=[A-Za-z0-9.,@])', '', full_text)

    # Stage 2: pdfminer fallback (better layout analysis for multi-column PDFs)
    if _is_sparse(full_text):
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            full_text = pdfminer_extract(BytesIO(contents)) or ""
            extraction_method = "pdfminer"
            logger.info("pypdf sparse — fell back to pdfminer")
        except ImportError:
            logger.warning("pdfminer not installed — skipping pdfminer fallback")
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {e}")

    # Stage 3: OCR via pytesseract (image-based / scanned PDFs)
    if _is_sparse(full_text):
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(contents, dpi=200)
            ocr_pages = [pytesseract.image_to_string(img) for img in images]
            full_text = "\n\n".join(p.strip() for p in ocr_pages if p.strip())
            extraction_method = "ocr"
            logger.info(f"Fell back to OCR — extracted {len(full_text)} chars from {len(images)} pages")
        except ImportError:
            logger.warning("pytesseract/pdf2image not installed — OCR fallback unavailable")
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from PDF — it may be image-based or encrypted. "
                   "Install pytesseract + pdf2image for OCR support, or paste resume text directly."
        )

    confidence = "high"
    tokens = full_text.split()
    if tokens:
        single_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
        chars_per_word = len(full_text) / len(tokens)
        if single_ratio > 0.2 or chars_per_word < 3:
            confidence = "low"
        elif single_ratio > 0.1 or chars_per_word < 4:
            confidence = "medium"

    # LLM-parse structured fields from the first ~1500 chars (name/title/skills are always near the top)
    parsed = {"name": "", "title": "", "skills": ""}
    try:
        from specialized_judges import call_ollama
        model = get_free_model()
        system = "You extract structured data from resume text. Return ONLY valid JSON, no markdown, no explanation."
        prompt = f"""Extract these fields from the resume below:
- name: the candidate's full name (exactly as written)
- title: their primary/most recent job title (one title only, not a list)
- skills: comma-separated list of up to 12 core technical skills (tools, languages, frameworks only — no soft skills)

Resume (top section):
{full_text[:1500]}

Return exactly: {{"name": "...", "title": "...", "skills": "..."}}"""
        result = call_ollama(system, prompt, model, temperature=0.0)
        if result.get("success"):
            raw = result.get("data", "")
            # Strip markdown fences if present
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(raw)
            parsed["name"] = str(data.get("name", "")).strip()
            parsed["title"] = str(data.get("title", "")).strip()
            parsed["skills"] = str(data.get("skills", "")).strip()
    except Exception:
        pass  # structured parsing is best-effort; raw text is always returned

    return {
        "text": full_text,
        "name": parsed["name"],
        "title": parsed["title"],
        "skills": parsed["skills"],
        "extraction_method": extraction_method,
        "confidence": confidence,
    }


# --- JOB URL FETCH ---
class FetchJobURLRequest(BaseModel):
    url: str

@app.post("/fetch-job-url")
async def fetch_job_url(req: FetchJobURLRequest):
    """Fetch a job posting URL and extract the job description text."""
    import httpx
    from bs4 import BeautifulSoup

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Job page returned HTTP {resp.status_code}")
        html = resp.text
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out — job site took too long to respond")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch URL: {str(e)}")

    # Quick check: if the static fetch returned almost nothing, the page is JS-rendered.
    # Fall back to Playwright to actually execute the JavaScript and get the real DOM.
    static_text_len = len(strip_html(html))
    if static_text_len < 500:
        print(f"  Static fetch too thin ({static_text_len} chars) — trying Playwright for JS-rendered page")
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    user_agent=headers["User-Agent"],
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                # Wait for any job-content container to appear
                for selector in ["[class*='job']", "[class*='description']", "main", "article"]:
                    try:
                        await page.wait_for_selector(selector, timeout=4000)
                        break
                    except Exception:
                        continue
                html = await page.content()
                await browser.close()
            print(f"  Playwright fetch succeeded: {len(html)} chars of HTML")
        except Exception as pw_err:
            print(f"  Playwright fallback failed: {pw_err} — proceeding with static HTML")

    # Step 1: best-effort HTML → plain text
    # Try JSON-LD structured data first (Greenhouse, Lever, Workable embed this)
    raw_text = strip_html(html)

    # BeautifulSoup fallback — also always run to get cleaner text than regex alone
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()
    content = None
    for selector in [
        "[class*='job-description']", "[class*='jobDescription']",
        "[class*='job_description']", "[id*='job-description']",
        "[class*='description']", "main", "article", "[role='main']",
    ]:
        content = soup.select_one(selector)
        if content and len(content.get_text(strip=True)) > 200:
            break
    target = content or soup.find("body") or soup
    bs_text = "\n".join(
        line for line in target.get_text(separator="\n", strip=True).splitlines()
        if line.strip()
    )

    # Use whichever extraction is longer (more content)
    raw_text = bs_text if len(bs_text) > len(raw_text) else raw_text

    if len(raw_text) < 100:
        raise HTTPException(
            status_code=422,
            detail="Could not extract job description — the page may require login or JavaScript rendering (LinkedIn requires auth)"
        )

    raw_text = raw_text[:5000]  # cap before LLM call

    # Step 2: LLM parses structured fields + cleans the JD text
    # Works even on noisy LinkedIn HTML — the model can find signal in the noise
    structured = {"title": "", "company": "", "experience_required": "", "skills_required": "", "job_description": raw_text}
    try:
        from specialized_judges import call_ollama
        model = get_free_model()
        system = "You extract structured job posting data. Return ONLY valid JSON, no markdown, no explanation."
        prompt = f"""From the job posting text below, extract these fields:
- title: job title only (e.g. "Applied AI Engineer")
- company: company name only (e.g. "Talently")
- experience_required: years/level of experience (e.g. "1-4 years" or "Mid-Senior")
- skills_required: comma-separated list of up to 12 core technical skills only
- job_description: the clean job description text — remove all navigation, "Sign in", "Skip to main content", similar jobs sections, and other page noise. Keep only the actual job posting content.

Job posting text:
{raw_text}

Return exactly: {{"title": "...", "company": "...", "experience_required": "...", "skills_required": "...", "job_description": "..."}}"""

        result = call_ollama(system, prompt, model, temperature=0.0)
        if result.get("success"):
            raw = re.sub(r"```[a-z]*\n?", "", result.get("data", "")).strip()
            # Extract JSON from response (model may add reasoning before JSON for deepseek-r1)
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                structured["title"]               = str(data.get("title", "")).strip()
                structured["company"]             = str(data.get("company", "")).strip()
                structured["experience_required"] = str(data.get("experience_required", "")).strip()
                structured["skills_required"]     = str(data.get("skills_required", "")).strip()
                clean_jd = str(data.get("job_description", "")).strip()
                if len(clean_jd) > 100:
                    structured["job_description"] = clean_jd
    except Exception:
        pass  # structured parsing best-effort; raw text always returned

    return {
        "job_description":     structured["job_description"],
        "title":               structured["title"],
        "company":             structured["company"],
        "experience_required": structured["experience_required"],
        "skills_required":     structured["skills_required"],
        "source_url":          url,
    }


# --- ATS CHECKER (standalone, no LLM) ---
class AtsCheckRequest(BaseModel):
    resume_text: str
    job_description: str = ""

@app.post("/ats-check")
async def ats_check(req: AtsCheckRequest):
    """Instant ATS scan — no LLM. Returns score, grade, 9-criterion breakdown."""
    result = score_ats(req.resume_text, req.job_description)
    return ats_to_dict(result)


# --- SKILL MATCHER ---
INCOME_PATHS_FILE = os.path.join(BASE_DIR, "data", "income_paths.json")

def _load_income_paths() -> list[dict]:
    with open(INCOME_PATHS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_resume_skill_tokens(resume_text: str) -> set[str]:
    """Lowercase token set from resume text — simple but fast."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-\.+#]{1,30}", resume_text.lower())
    # Expand common multi-word phrases that appear as single tokens in resume
    text_lower = resume_text.lower()
    extras = set()
    ALIASES = {
        "machine learning": "machine_learning", "deep learning": "deep_learning",
        "natural language": "nlp", "large language": "llm", "langchain": "langchain",
        "fast api": "fastapi", "scikit": "scikit-learn", "sklearn": "scikit-learn",
        "pytorch": "pytorch", "tensorflow": "tensorflow", "docker": "docker",
        "kubernetes": "kubernetes", "aws": "aws", "azure": "azure", "gcp": "gcp",
        "sql": "sql", "postgresql": "sql", "postgres": "sql", "mysql": "sql",
        "react": "react", "next.js": "nextjs", "nextjs": "nextjs",
        "typescript": "typescript", "javascript": "javascript",
        "prompt engineering": "prompt_engineering", "fine.tuning": "fine_tuning",
        "rag": "rag", "vector": "vector_databases", "chromadb": "vector_databases",
        "pinecone": "vector_databases", "mlops": "mlops", "airflow": "airflow",
        "dbt": "dbt", "spark": "spark", "kafka": "kafka",
    }
    for phrase, skill in ALIASES.items():
        if phrase in text_lower:
            extras.add(skill)
    return set(tokens) | extras

class SkillMatchRequest(BaseModel):
    resume_text: str

@app.post("/skill-match")
async def skill_match(req: SkillMatchRequest):
    """Match resume skills against all income paths. Returns top 10 ranked by fit %."""
    paths = _load_income_paths()
    resume_tokens = _extract_resume_skill_tokens(req.resume_text)

    matches = []
    for path in paths:
        required = [s.lower().replace("-", "_") for s in path.get("required_skills", [])]
        nice = [s.lower().replace("-", "_") for s in path.get("nice_to_have", [])]
        all_skills = required + nice

        matched_req  = [s for s in required if any(s in tok or tok in s for tok in resume_tokens)]
        matched_nice = [s for s in nice if any(s in tok or tok in s for tok in resume_tokens)]
        missing_req  = [s for s in required if s not in matched_req]

        if not required:
            pct = 0.0
        else:
            # 70% weight on required, 30% on nice-to-have
            req_score  = len(matched_req) / len(required)
            nice_score = len(matched_nice) / len(nice) if nice else 0.0
            pct = round((req_score * 0.7 + nice_score * 0.3) * 100, 1)

        matches.append({
            "id": path["id"],
            "title": path["title"],
            "category": path["category"],
            "description": path["description"],
            "income_range": path["income_range"]["display"],
            "time_to_first_income": path["time_to_first_income"],
            "market_demand": path["market_demand"],
            "match_pct": pct,
            "matched_skills": matched_req + matched_nice,
            "missing_required": missing_req,
            "starter_actions": path.get("starter_actions", []),
        })

    matches.sort(key=lambda x: x["match_pct"], reverse=True)
    return {"matches": matches[:10], "total_paths": len(paths)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)