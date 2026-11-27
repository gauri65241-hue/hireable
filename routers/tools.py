import json
import logging
import os
import re

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File

from ratelimit import limiter
from pydantic import BaseModel, field_validator
from typing import Literal

from auth_utils import get_current_user
from concurrency import run_blocking
from net_safety import UnsafeURLError, resolve_public, safe_get
from routers.deps import BASE_DIR, get_free_model, strip_html, score_ats, ats_to_dict

logger = logging.getLogger(__name__)
router = APIRouter()

INCOME_PATHS_FILE = os.path.join(BASE_DIR, "data", "income_paths.json")

_ATS_DEEP_AUDIT_SYSTEM = """You are operating as a cynical, highly selective Enterprise Talent Acquisition Director and an expert in ATS (Applicant Tracking System) parsing heuristics. Your task is to perform an uncompromising, comprehensive audit on the provided resume to determine its true readiness for a highly competitive job market.

Evaluate the resume across the four operational dimensions outlined below. Do not provide generic encouragement; look for hidden disqualifiers, formatting risks, execution gaps, and soft narrative phrasing that corporate evaluators reject.

---

### 1. ATS Parsability & Structural Integrity
Analyze how standard programmatic parsers (e.g., Taleo, Workday, Greenhouse) ingest this document layout:
- Tables, Columns, and Visual Clutter: Does the resume use multi-column layouts, sidebars, text boxes, or embedded graphic metrics (like skill progress bars)? Explain how these elements cause text-merging and data truncation errors during programmatic ingestion.
- Section Hierarchy and Font Semantics: Are the headers standard (e.g., "Professional Experience", "Education") or overly creative? Will an automated parser correctly map chronologies, job titles, and company names, or will it misinterpret dates and throw parsing exceptions?

### 2. Metric Caliber & Impact-to-Activity Ratio
Evaluate the strength of the bullet points under professional history:
- Passive Duty vs. Quantified Impact: Identify bullets that read like a passive list of day-to-day responsibilities (e.g., "Responsible for managing a team" or "Assisted with code deployments") instead of active accomplishments.
- The Metric Equation: Audit the presence of hard, verifiable metrics. Does the candidate prove scale, volume, velocity, or revenue impact? For every major project, did they define the baseline, the action taken, and the quantified outcome (e.g., "Reduced latency by 40% resulting in $50k infrastructure savings")? Mark every bullet that lacks measurable data.

### 3. Role Targeting & Signal-to-Noise Ratio
Assess the clarity and positioning of the candidate's professional profile:
- Skill Indexing and Buzzword Stuffing: Is the skills section an unstructured dump of general keywords, or is it categorized logically by technical domain or proficiency? Are there highly generic phrases present (e.g., "Team player," "Results-driven leader," "Excellent communication skills") that offer zero signal to a hiring manager?
- Scope and Seniority Alignment: Does the narrative arc matching their titles align with their intended career trajectory? Is there a mismatch where a candidate seeking a senior role describes purely junior tasks, or vice versa? Identify any irrelevant legacy experience that dilutes their core value proposition.

### 4. Semantic Red Flags & "The Human Scan" Bias
Simulate a human hiring manager performing an initial 6-second manual review of the document:
- Typos and Formatting Flashes: Are there inconsistent date formats (e.g., mixing "Jan 2023" with "05/2024"), irregular bullet styling, or trailing whitespace issues that signal a lack of attention to detail?
- Framing and Tone Fixes: Are bullet points leading with weak, passive verbs (e.g., "Helped," "Worked on," "Participated in") instead of strong, high-agency action verbs (e.g., "Architected," "Spearheaded," "Optimized")?"""


# ── Pydantic models ───────────────────────────────────────────────────────────

class AtsCheckRequest(BaseModel):
    resume_text: str
    job_description: str = ""

class AtsDeepAuditRequest(BaseModel):
    resume_text: str
    job_description: str = ""

class _Disqualifier(BaseModel):
    rank: int
    title: str
    explanation: str

class _Rewrite(BaseModel):
    original: str
    why_fails: str
    refactored: str

_VERDICT_MAP = {
    "pass": "PASS", "review": "REVIEW", "fail": "FAIL",
}

class _ScorecardDim(BaseModel):
    score: float
    verdict: Literal["PASS", "REVIEW", "FAIL"] = "REVIEW"
    note: str
    improvement: str

    @field_validator("verdict", mode="before")
    @classmethod
    def normalise_verdict(cls, v):
        if not isinstance(v, str): return "REVIEW"
        return _VERDICT_MAP.get(v.strip().lower(), "REVIEW")

class _Scorecard(BaseModel):
    ats_parsability: _ScorecardDim
    metric_caliber: _ScorecardDim
    role_targeting: _ScorecardDim
    semantic_red_flags: _ScorecardDim
    overall_verdict: Literal["PASS", "REVIEW", "FAIL"] = "REVIEW"

    @field_validator("overall_verdict", mode="before")
    @classmethod
    def normalise_overall(cls, v):
        if not isinstance(v, str): return "REVIEW"
        return _VERDICT_MAP.get(v.strip().lower(), "REVIEW")

class AtsDeepAuditResult(BaseModel):
    disqualifiers: list[_Disqualifier]
    rewrites: list[_Rewrite]
    scorecard: _Scorecard

class FetchJobURLRequest(BaseModel):
    url: str

class ParseJDTextRequest(BaseModel):
    text: str

class SkillMatchRequest(BaseModel):
    resume_text: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_income_paths() -> list[dict]:
    with open(INCOME_PATHS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_resume_skill_tokens(resume_text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-\.+#]{1,30}", resume_text.lower()))
    text_lower = resume_text.lower()
    extras = set()
    # Phrase → canonical skill name. Checked as substring of lowercased resume.
    ALIASES = {
        "machine learning": "machine_learning", "deep learning": "deep_learning",
        "natural language": "nlp", "large language": "llm", "langchain": "langchain",
        "fast api": "fastapi", "scikit": "scikit-learn", "sklearn": "scikit-learn",
        "pytorch": "pytorch", "tensorflow": "tensorflow", "docker": "docker",
        "kubernetes": "kubernetes", "aws": "aws", "azure": "azure", "gcp": "gcp",
        "sql": "sql", "postgresql": "sql", "postgres": "sql", "mysql": "sql",
        "spring boot": "spring_boot", "spring-boot": "spring_boot",
        "rest api": "rest_api", "rest-api": "rest_api",
        "node.js": "node", "node js": "node",
        "react": "react", "next.js": "nextjs", "nextjs": "nextjs",
        "typescript": "typescript", "javascript": "javascript",
        "prompt engineering": "prompt_engineering", "fine-tuning": "fine_tuning",
        "rag": "rag", "vector": "vector_databases", "chromadb": "vector_databases",
        "pinecone": "vector_databases", "mlops": "mlops", "airflow": "airflow",
        "dbt": "dbt", "spark": "spark", "kafka": "kafka",
        "ci/cd": "ci_cd", "ci cd": "ci_cd", "cicd": "ci_cd",
        "system design": "system_design", "data pipeline": "data_pipelines",
        "data engineering": "data_engineering", "cloud platform": "cloud_platforms",
        "junit": "junit", "mockito": "mockito", "agile": "agile", "scrum": "scrum",
        "hibernate": "hibernate", "microservice": "microservices",
    }
    for phrase, skill in ALIASES.items():
        if phrase in text_lower:
            extras.add(skill)
    return tokens | extras


def _skill_matched(skill: str, resume_tokens: set[str]) -> bool:
    """Exact match first, then compound-part match for multi-word skills."""
    if skill in resume_tokens:
        return True
    # For compound skills (spring_boot, rest_api, etc.), require ALL significant
    # parts (≥4 chars) to appear as individual resume tokens.
    # This lets "spring boot" on a resume match skill "spring_boot" without
    # the false-positive `tok in skill` substring direction.
    parts = [p for p in re.split(r'[_\-]', skill) if len(p) >= 4]
    if parts and all(p in resume_tokens for p in parts):
        return True
    return False


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/client-config")
@limiter.limit("60/minute")
async def client_config(request: Request):
    """Expose non-secret UI defaults so the frontend can pre-fill provider/model/key."""
    return {
        "api_key": "",
        "provider": "ollama",
        "model": "mistral-nemo:latest",
    }


@router.post("/ats-check")
@limiter.limit("20/minute")
async def ats_check(request: Request, req: AtsCheckRequest,
                    current_user: dict = Depends(get_current_user)):
    result = score_ats(req.resume_text, req.job_description)
    return ats_to_dict(result)


@router.post("/ats-deep-audit")
@limiter.limit("10/minute")
async def ats_deep_audit(request: Request, req: AtsDeepAuditRequest,
                         current_user: dict = Depends(get_current_user)):
    from analysers.base import call_llm_async, parse_and_validate
    from routers.deps import get_llm_api_key, get_free_model

    resume_text = req.resume_text.strip()
    if not resume_text:
        return {"success": False, "error": "Resume text required."}

    jd_section = f"\n\nJOB DESCRIPTION:\n{req.job_description.strip()[:3000]}" if req.job_description.strip() else ""

    system_prompt = _ATS_DEEP_AUDIT_SYSTEM

    user_prompt = f"""Perform a full audit on the resume below using all four dimensions from your instructions.{jd_section}

RESUME:
{resume_text[:5000]}

---

### Expected Deliverables — return ONLY valid JSON, no prose, no markdown fences, no text outside the JSON object:

{{
  "disqualifiers": [
    {{"rank": 1, "title": "<top structural or content flaw that causes instant ATS filter-out or 6-second drop>", "explanation": "<specific, direct explanation referencing actual resume content — no generics>"}},
    {{"rank": 2, "title": "<second disqualifier>", "explanation": "<explanation>"}},
    {{"rank": 3, "title": "<third disqualifier>", "explanation": "<explanation>"}}
  ],
  "rewrites": [
    {{"original": "<exact weakest bullet or phrase copied verbatim from the resume>", "why_fails": "<specific lack of impact, metrics, or agency>", "refactored": "<rewrite maximising technical depth, active verbs, and quantified outcomes>"}},
    {{"original": "<second weak bullet verbatim>", "why_fails": "<why it fails>", "refactored": "<high-impact refactor>"}},
    {{"original": "<third weak bullet verbatim>", "why_fails": "<why it fails>", "refactored": "<high-impact refactor>"}}
  ],
  "scorecard": {{
    "ats_parsability":    {{"score": 0.0, "verdict": "PASS", "note": "<one-line finding on structure, headers, column layout, date parsing>", "improvement": "<one specific action to fix this dimension>"}},
    "metric_caliber":     {{"score": 0.0, "verdict": "REVIEW", "note": "<one-line finding on quantification density and impact-to-activity ratio>", "improvement": "<one specific action to fix this dimension>"}},
    "role_targeting":     {{"score": 0.0, "verdict": "FAIL", "note": "<one-line finding on skill categorisation, seniority alignment, noise>", "improvement": "<one specific action to fix this dimension>"}},
    "semantic_red_flags": {{"score": 0.0, "verdict": "PASS", "note": "<one-line finding on passive verbs, formatting inconsistencies, tone>", "improvement": "<one specific action to fix this dimension>"}},
    "overall_verdict": "REVIEW"
  }}
}}

Scoring rules:
- score: 0.0–10.0, one decimal place
- verdict per dimension: PASS if score ≥ 7.0 | REVIEW if 4.0–6.9 | FAIL if < 4.0
- overall_verdict: PASS only if all 4 dimensions ≥ 7.0 | FAIL if any dimension < 4.0 | REVIEW otherwise
- ats_parsability covers: standard section headers, single-column layout, consistent date formats, parseable contact block
- metric_caliber covers: % of experience bullets with hard numbers, baseline→action→outcome structure
- role_targeting covers: skills categorised by domain, no generic soft-skill filler, seniority narrative matches target level{' and alignment with provided JD' if req.job_description.strip() else ''}
- semantic_red_flags covers: passive verb leads ("Helped", "Worked on", "Responsible for"), inconsistent formatting, attention-to-detail signals
- improvement: one concrete, actionable fix for this specific dimension — not generic advice. For PASS dimensions, state what to do to maintain or push the score higher."""

    try:
        api_key = get_llm_api_key()
        model = "anthropic/claude-3.5-haiku" if api_key else "qwen2.5-coder:7b"
        result = await call_llm_async(
            system_prompt, user_prompt, model=model,
            temperature=0.0, api_key=api_key or ""
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "LLM unavailable")}
        parsed = parse_and_validate(result["data"], AtsDeepAuditResult)
        return parsed
    except Exception as e:
        logger.error("ats-deep-audit failed: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/skill-match")
@limiter.limit("20/minute")
async def skill_match(request: Request, req: SkillMatchRequest,
                      current_user: dict = Depends(get_current_user)):
    paths = _load_income_paths()
    resume_tokens = _extract_resume_skill_tokens(req.resume_text)

    matches = []
    for path in paths:
        required = [s.lower().replace("-", "_") for s in path.get("required_skills", [])]
        nice     = [s.lower().replace("-", "_") for s in path.get("nice_to_have", [])]

        matched_req  = [s for s in required if _skill_matched(s, resume_tokens)]
        matched_nice = [s for s in nice     if _skill_matched(s, resume_tokens)]
        missing_req  = [s for s in required if s not in matched_req]

        if not required:
            pct = 0.0
        else:
            req_score  = len(matched_req) / len(required)
            nice_score = len(matched_nice) / len(nice) if nice else 0.0
            pct = round((req_score * 0.7 + nice_score * 0.3) * 100, 1)

        matches.append({
            "id":                  path["id"],
            "title":               path["title"],
            "category":            path["category"],
            "description":         path["description"],
            "income_range":        path["income_range"]["display"],
            "time_to_first_income":path["time_to_first_income"],
            "market_demand":       path["market_demand"],
            "match_pct":           pct,
            "matched_skills":      matched_req + matched_nice,
            "missing_required":    missing_req,
            "starter_actions":     path.get("starter_actions", []),
        })

    matches.sort(key=lambda x: x["match_pct"], reverse=True)
    return {"matches": matches[:10], "total_paths": len(paths)}


def _is_sparse_text(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return True
    single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    return len(text.strip()) < 100 or single_char_ratio > 0.4


def _extract_pdf_text_blocking(contents: bytes, max_pages: int) -> tuple[str, str, int]:
    """CPU-bound PDF text extraction. Runs in the blocking pool, never on the event loop.
    Returns (text, method, page_count). Raises ValueError if the PDF exceeds max_pages."""
    from io import BytesIO
    full_text = ""
    extraction_method = "pypdf"
    page_count = 0

    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(contents))
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise ValueError(f"PDF has {page_count} pages; the limit is {max_pages}.")
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(p.strip() for p in pages if p.strip())
    except ValueError:
        raise
    except Exception as e:
        logger.warning("pypdf extraction failed: %s", e)

    if full_text and not _is_sparse_text(full_text):
        tokens = full_text.split()
        if tokens and sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.4:
            full_text = re.sub(r'(?<=[A-Za-z0-9.,@]) (?=[A-Za-z0-9.,@])', '', full_text)

    if _is_sparse_text(full_text):
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            full_text = pdfminer_extract(BytesIO(contents), maxpages=max_pages) or ""
            extraction_method = "pdfminer"
            logger.info("pypdf sparse - fell back to pdfminer")
        except ImportError:
            logger.warning("pdfminer not installed - skipping pdfminer fallback")
        except Exception as e:
            logger.warning("pdfminer extraction failed: %s", e)

    if _is_sparse_text(full_text):
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(contents, dpi=200, last_page=max_pages)
            ocr_pages = [pytesseract.image_to_string(img) for img in images]
            full_text = "\n\n".join(p.strip() for p in ocr_pages if p.strip())
            extraction_method = "ocr"
            logger.info("Fell back to OCR - extracted %d chars from %d pages", len(full_text), len(images))
        except ImportError:
            logger.warning("pytesseract/pdf2image not installed - OCR fallback unavailable")
        except Exception as e:
            logger.warning("OCR extraction failed: %s", e)

    return full_text, extraction_method, page_count


MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(5 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "30"))


async def _read_capped(file: UploadFile, limit: int) -> bytes:
    """Read an upload without ever holding more than limit+1 bytes."""
    buf = bytearray()
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status_code=413,
                                detail=f"File too large; limit is {limit // (1024 * 1024)} MB.")
    return bytes(buf)


@router.post("/extract-pdf")
@limiter.limit("30/minute")
async def extract_pdf(request: Request, file: UploadFile = File(...),
                      current_user: dict = Depends(get_current_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await _read_capped(file, MAX_PDF_BYTES)
    if not contents.lstrip()[:5] == b"%PDF-":
        raise HTTPException(status_code=400, detail="File is not a valid PDF")
    try:
        full_text, extraction_method, page_count = await run_blocking(
            _extract_pdf_text_blocking, contents, MAX_PDF_PAGES, timeout=45.0
        )
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=422, detail="PDF took too long to process.")

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

    clean_text = full_text

    _SECTION_STOP = (r'(?:EXPERIENCE|EDUCATION|PROJECTS?|WORK\s+HISTORY|CERTIFICATIONS?'
                     r'|PUBLICATIONS?|AWARDS?|LANGUAGES?|INTERESTS?|SUMMARY|OBJECTIVE'
                     r'|PROFILE|CONTACT|REFERENCES?)')

    def _detect_format(text: str) -> str:
        first_500 = text[:500]
        if (re.search(r'[A-Z]{3,}[A-Z][a-z/|]', first_500) or
                re.search(r'SKILLS(?:TECHNICAL|CORE|PROFESSIONAL)', text, re.IGNORECASE)):
            return 'format_a'
        name_pat    = re.compile(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b')
        first_third = text[:max(1, len(text) // 3)]
        last_quarter= text[max(0, int(len(text) * 0.75)):]
        if not name_pat.search(first_third) and name_pat.search(last_quarter):
            return 'format_b'
        lines = text.splitlines()
        if lines and sum(1 for ln in lines if not ln.strip()) / len(lines) > 0.35:
            return 'format_c'
        return 'generic'

    def _extract_skills_section(text: str, fmt: str) -> str:
        if fmt == 'format_a':
            pattern = (r'(?:SKILLS?(?:\s*TECHNICAL\s+SKILLS?)?|TECHNICAL\s+SKILLS?|CORE\s+SKILLS?)'
                       r'\s*[\n:]?(.*?)(?=\n' + _SECTION_STOP + r'|\Z)')
        elif fmt == 'format_c':
            pattern = (r'(?:TECHNICAL\s+SKILLS?|CORE\s+SKILLS?|SKILLS?)\s*[\n:]'
                       r'(.*?)(?=\n\n' + _SECTION_STOP + r'|\Z)')
        else:
            pattern = (r'(?:TECHNICAL\s+SKILLS?|CORE\s+SKILLS?|SKILLS?)\s*[\n:]'
                       r'(.*?)(?=\n[A-Z]{4,}[\s\n]|\Z)')
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip()[:1500] if m else ""

    fmt = _detect_format(clean_text)

    if fmt == 'format_a':
        header = clean_text[:300]
        header = re.sub(r'([A-Z]{3,})([A-Z][a-z])', r'\1\n\2', header)
        header = re.sub(r'([A-Z]{3,})([|/])', r'\1\n\2', header)
        clean_text = header + clean_text[300:]

    skills_section = _extract_skills_section(clean_text, fmt)
    skills_hint = (f"\n\nSKILLS SECTION (extracted directly — use this for the skills field):\n{skills_section}"
                   if skills_section else "")

    # ── Regex fallback extraction (fast, no LLM) ────────────────────────────────
    def _regex_extract(text: str) -> dict:
        result = {"name": "", "title": ""}
        # Name: first ALL-CAPS sequence of 2+ words in top 1000 chars
        m = re.search(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b', text[:1000])
        if m:
            candidate = m.group(1).strip()
            title_kw = r'\b(ENGINEER|ANALYST|SCIENTIST|MANAGER|DEVELOPER|ARCHITECT|INTERN|LEAD|DIRECTOR|CONSULTANT|SPECIALIST|OFFICER|DESIGNER|RESEARCHER|ASSOCIATE|MLOPS|ML)\b'
            kw_match = re.search(title_kw, candidate, re.IGNORECASE)
            result["name"] = candidate[:kw_match.start()].strip().rstrip('|/-,') if kw_match else candidate
        # Title: ALL-CAPS line with pipes or slash after name
        m2 = re.search(r'\b([A-Z][A-Z /|&]+(?:\|\s*[A-Z][A-Z /|&]+)+)\b', text[:1500])
        if m2:
            result["title"] = m2.group(1).strip()
        return result

    regex_parsed = _regex_extract(clean_text)
    parsed = {"name": regex_parsed["name"], "title": regex_parsed["title"], "skills": ""}

    # Use skills section if regex found it
    if skills_section:
        # Extract individual skills (strip category headers like "AI/ML & GenAI:")
        raw_skills = re.sub(r'^[A-Z][^:]{0,40}:\s*', '', skills_section, flags=re.MULTILINE)
        skill_tokens = [s.strip() for s in re.split(r'[,\n]+', raw_skills) if s.strip() and len(s.strip()) > 1]
        parsed["skills"] = ", ".join(skill_tokens[:25])

    # ── LLM refinement (overwrites regex if successful) ────────────────────────
    # Skip LLM if regex already got both name and title — avoid blocking Ollama during analysis
    _regex_has_enough = bool(parsed.get("name") and parsed.get("title"))
    try:
        from specialized_judges import call_ollama_async
        from analysers.base import _extract_outermost_json_object, _strip_markdown_fences
        model = get_free_model()
        if _regex_has_enough:
            raise ValueError("regex sufficient — skip LLM")
        system = "You extract structured data from resume text. Return ONLY valid JSON, no markdown, no explanation."
        prompt = f"""Extract structured fields from this resume. Resumes come in many formats — some have clear sections, some are dense text with no line breaks due to PDF extraction artifacts.

Rules:
- name: the person's full name. It may appear at the top OR at the bottom of the resume (some templates put contact info at the end). Look in both the first 500 and last 500 characters. Names are usually 2-3 words, often ALL-CAPS. In PDF-extracted text, the name may run directly into a job title (e.g. "JANE DOESENIOR ENGINEER" — name is "JANE DOE"). Extract only the name, not the title, company, or location.
- title: the professional headline or tagline from the resume header — the line that appears immediately after the person's name (e.g. "AI/MLOPS ENGINEER | RAG & LLM EVALUATION ARCHITECT | MACHINE LEARNING ENGINEER"). This is usually on the same line or directly below the name, before the contact info or summary. Return it exactly as written, including pipe separators if present. If no header tagline exists, fall back to the most recent job title from the EXPERIENCE section.
- skills: comma-separated string of up to 25 specific technical skills. Look for a section headed "SKILLS", "TECHNICAL SKILLS", or "CORE SKILLS" anywhere in the resume. Extract individual skill items only — NOT category headers. CRITICAL: only include skills explicitly listed in the resume text. Do NOT infer, guess, or add skills not written in the document. Return a plain comma-separated string, no brackets, no quotes. Exclude soft skills.

Resume (start):
{clean_text[:3000]}

Resume (end):
{clean_text[-1500:]}{skills_hint}

Return exactly: {{"name": "...", "title": "...", "skills": "..."}}"""
        result = await call_ollama_async(system, prompt, model, temperature=0.0)
        if result.get("success"):
            raw = result.get("data", "")
            raw = _strip_markdown_fences(raw)
            raw = _extract_outermost_json_object(raw) or raw
            data = json.loads(raw)
            name = str(data.get("name", "")).strip()
            title_kw = r'\b(ENGINEER|ANALYST|SCIENTIST|MANAGER|DEVELOPER|ARCHITECT|INTERN|LEAD|DIRECTOR|CONSULTANT|SPECIALIST|OFFICER|DESIGNER|RESEARCHER|ASSOCIATE)\b'
            name_match = re.search(title_kw, name, re.IGNORECASE)
            if name_match:
                name = name[:name_match.start()].strip().rstrip('|/-,')
            if name:
                parsed["name"] = name
            if data.get("title", "").strip():
                parsed["title"] = str(data["title"]).strip()
            if data.get("skills", "").strip():
                parsed["skills"] = str(data["skills"]).strip()
    except Exception:
        pass


    return {
        "text":              full_text,
        "pages":             page_count,
        "name":              parsed["name"],
        "title":             parsed["title"],
        "skills":            parsed["skills"],
        "experience_years":  "",
        "extraction_method": extraction_method,
        "confidence":        confidence,
    }


@router.post("/fetch-job-url")
@limiter.limit("30/minute")
async def fetch_job_url(request: Request, req: FetchJobURLRequest,
                        current_user: dict = Depends(get_current_user)):
    from bs4 import BeautifulSoup

    url = req.url.strip()
    # SSRF guard: scheme/port/host checks + every resolved address must be public.
    try:
        await resolve_public(url)
    except UnsafeURLError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _BROWSER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # ── Tier 1: static fetch ──────────────────────────────────────────────────
    html = ""
    try:
        # Pinned-IP fetch, manual redirect re-validation, 2 MB body cap.
        resp = await safe_get(url, headers=_BROWSER_HEADERS, timeout=15)
        if resp.status_code == 200:
            html = resp.text
        else:
            logger.warning("Static fetch HTTP %d for %s", resp.status_code, url)
    except UnsafeURLError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("Static fetch failed: %s", e)

    # ── Tier 2: JSON-LD structured data (works for most major job boards) ────
    def _extract_jsonld(html_text: str) -> dict:
        """Parse JobPosting schema.org JSON-LD embedded in page HTML."""
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
                # Handle @graph arrays and single objects
                entries = data if isinstance(data, list) else [data]
                for entry in entries:
                    if isinstance(entry, dict):
                        graph = entry.get("@graph", [])
                        entries_to_check = graph if graph else [entry]
                        for item in entries_to_check:
                            if isinstance(item, dict) and "JobPosting" in str(item.get("@type", "")):
                                org = item.get("hiringOrganization", {})
                                company = (org.get("name", "") if isinstance(org, dict) else str(org)).strip()
                                desc_html = item.get("description", "")
                                desc_soup = BeautifulSoup(desc_html, "html.parser")
                                desc_text = desc_soup.get_text(separator="\n", strip=True)
                                exp = item.get("experienceRequirements", "") or item.get("qualifications", "")
                                if isinstance(exp, dict):
                                    exp = exp.get("description", "") or exp.get("value", "")
                                skills = item.get("skills", "") or item.get("competencies", "")
                                if isinstance(skills, list):
                                    skills = ", ".join(str(s) for s in skills)
                                return {
                                    "title":   item.get("title", ""),
                                    "company": company,
                                    "experience_required": str(exp).strip(),
                                    "skills_required": str(skills).strip(),
                                    "job_description": desc_text,
                                }
            except Exception:
                continue
        return {}

    jsonld = _extract_jsonld(html) if html else {}
    if jsonld.get("job_description") and len(jsonld["job_description"]) > 200:
        logger.info("JSON-LD extraction succeeded for %s (%d chars)", url, len(jsonld["job_description"]))
        raw_text = jsonld["job_description"][:15000]
        structured = {
            "title":               jsonld.get("title", ""),
            "company":             jsonld.get("company", ""),
            "experience_required": jsonld.get("experience_required", ""),
            "skills_required":     jsonld.get("skills_required", ""),
            "job_description":     raw_text,
        }
        # If JSON-LD gave us title+company, skip LLM entirely
        if structured["title"] and structured["company"]:
            return {**structured, "source_url": url, "extraction_method": "json-ld"}
        # Otherwise fall through to LLM for metadata only
        html = ""  # don't re-scrape, use what we have
    else:
        jsonld = {}

    # ── Tier 3: Playwright for JS-rendered / SPA sites ────────────────────────
    intercepted_json: dict = {}
    static_text_len = len(strip_html(html)) if html else 0

    if static_text_len < 500:
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=_BROWSER_HEADERS["User-Agent"],
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page = await context.new_page()

                # The browser follows redirects and loads sub-resources on its own, so
                # every request it makes is checked against the same SSRF policy.
                _host_ok: dict[str, bool] = {}
                async def _guard(route):
                    req_url = route.request.url
                    if req_url.startswith(("data:", "blob:", "about:")):
                        await route.continue_()
                        return
                    host = req_url.split("/")[2] if "//" in req_url else req_url
                    if host not in _host_ok:
                        try:
                            await resolve_public(req_url)
                            _host_ok[host] = True
                        except UnsafeURLError:
                            _host_ok[host] = False
                    if _host_ok[host]:
                        await route.continue_()
                    else:
                        await route.abort("blockedbyclient")
                await context.route("**/*", _guard)

                # Intercept XHR/fetch responses that look like job API payloads
                api_responses: list[str] = []
                async def _handle_response(response):
                    try:
                        ct = response.headers.get("content-type", "")
                        if "json" in ct and response.status == 200:
                            rurl = response.url.lower()
                            if any(kw in rurl for kw in ["job", "position", "vacancy", "posting", "role", "opening"]):
                                body = await response.text()
                                if len(body) > 200:
                                    api_responses.append(body)
                    except Exception:
                        pass
                page.on("response", _handle_response)

                await page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Wait for job content to appear
                _JOB_SELECTORS = [
                    "[class*='job-description']", "[class*='jobDescription']",
                    "[class*='job_description']", "[class*='job-detail']",
                    "[class*='jobDetail']", "[class*='posting']",
                    "main article", "article", "main",
                ]
                for selector in _JOB_SELECTORS:
                    try:
                        await page.wait_for_selector(selector, timeout=4000)
                        break
                    except Exception:
                        continue
                else:
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass

                html = (await page.content())[:2_000_000]

                # Re-check JSON-LD after JS rendered the page
                if not jsonld:
                    jsonld = _extract_jsonld(html)
                    if jsonld.get("job_description") and len(jsonld["job_description"]) > 200:
                        logger.info("JSON-LD found after Playwright render for %s", url)

                await browser.close()

                # Try to parse intercepted API JSON for job content
                for api_body in api_responses:
                    try:
                        api_data = json.loads(api_body)
                        # Flatten nested structures to find description-like fields
                        def _find_desc(obj, depth=0):
                            if depth > 5:
                                return ""
                            if isinstance(obj, dict):
                                for key in ["description", "jobDescription", "job_description", "content", "body", "details"]:
                                    val = obj.get(key, "")
                                    if isinstance(val, str) and len(val) > 300:
                                        return val
                                for v in obj.values():
                                    r = _find_desc(v, depth+1)
                                    if r:
                                        return r
                            elif isinstance(obj, list):
                                for item in obj[:5]:
                                    r = _find_desc(item, depth+1)
                                    if r:
                                        return r
                            return ""
                        desc = _find_desc(api_data)
                        if desc:
                            # Clean HTML tags from API response description
                            desc_soup = BeautifulSoup(desc, "html.parser")
                            desc_clean = desc_soup.get_text(separator="\n", strip=True)
                            if len(desc_clean) > len(jsonld.get("job_description", "")):
                                intercepted_json["job_description"] = desc_clean
                                # Try to find title/company
                                for key in ["title", "jobTitle", "job_title", "name"]:
                                    v = api_data.get(key) or _find_field(api_data, key)
                                    if v and isinstance(v, str) and len(v) < 100:
                                        intercepted_json["title"] = v.strip()
                                        break
                                for key in ["company", "companyName", "company_name", "employer", "organization"]:
                                    v = api_data.get(key) or _find_field(api_data, key)
                                    if v and isinstance(v, str) and len(v) < 100:
                                        intercepted_json["company"] = v.strip()
                                        break
                            break
                    except Exception:
                        continue

        except Exception as pw_err:
            logger.warning("Playwright failed: %s", pw_err)

    # ── Pick best content source ──────────────────────────────────────────────
    def _find_field(obj, key, depth=0):
        if depth > 4 or not isinstance(obj, dict):
            return None
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_field(v, key, depth+1)
            if r:
                return r
        return None

    best_jsonld = jsonld if (jsonld.get("job_description") and
                              len(jsonld["job_description"]) > len(intercepted_json.get("job_description", ""))) else {}
    best_api = intercepted_json if intercepted_json.get("job_description") else {}

    if best_jsonld.get("job_description"):
        raw_text = best_jsonld["job_description"][:15000]
        structured = {
            "title":               best_jsonld.get("title", ""),
            "company":             best_jsonld.get("company", ""),
            "experience_required": best_jsonld.get("experience_required", ""),
            "skills_required":     best_jsonld.get("skills_required", ""),
            "job_description":     raw_text,
        }
        extraction_method = "json-ld"
    elif best_api.get("job_description"):
        raw_text = best_api["job_description"][:15000]
        structured = {
            "title":               best_api.get("title", ""),
            "company":             best_api.get("company", ""),
            "experience_required": "",
            "skills_required":     "",
            "job_description":     raw_text,
        }
        extraction_method = "api-intercept"
    else:
        # DOM scrape fallback
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
        raw_text = strip_html(html)
        raw_text = bs_text if len(bs_text) > len(raw_text) else raw_text
        if len(raw_text) < 100:
            raise HTTPException(
                status_code=422,
                detail="Could not extract job description — the page may require login or JavaScript rendering. Try Quick Paste instead."
            )
        raw_text = raw_text[:15000]
        structured = {"title": "", "company": "", "experience_required": "", "skills_required": "", "job_description": raw_text}
        extraction_method = "dom-scrape"

    logger.info("fetch_job_url method=%s title=%r chars=%d", extraction_method, structured["title"], len(structured["job_description"]))

    # ── LLM: fill missing metadata fields only ───────────────────────────────
    needs_llm = not structured["title"] or not structured["company"] or not structured["skills_required"]
    if needs_llm:
        try:
            from specialized_judges import call_ollama_async
            model = get_free_model()
            system = "You extract structured job posting metadata. Return ONLY valid JSON, no markdown, no explanation."
            # Only send first 4000 chars for metadata extraction — faster
            snippet = structured["job_description"][:4000]
            prompt = f"""From this job posting extract:
- title: job title only (short, e.g. "Data Scientist")
- company: hiring company name only
- experience_required: years or level (e.g. "3-5 years", "Senior")
- skills_required: comma-separated list of up to 12 core technical skills

Job posting:
{snippet}

Return exactly: {{"title": "...", "company": "...", "experience_required": "...", "skills_required": "..."}}"""
            result = await call_ollama_async(system, prompt, model, temperature=0.0)
            if result.get("success"):
                raw = re.sub(r"```[a-z]*\n?", "", result.get("data", "")).strip()
                json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    if not structured["title"]:
                        structured["title"] = str(data.get("title", "")).strip()
                    if not structured["company"]:
                        structured["company"] = str(data.get("company", "")).strip()
                    if not structured["experience_required"]:
                        structured["experience_required"] = str(data.get("experience_required", "")).strip()
                    if not structured["skills_required"]:
                        structured["skills_required"] = str(data.get("skills_required", "")).strip()
        except Exception:
            pass

    return {
        "job_description":     structured["job_description"],
        "title":               structured["title"],
        "company":             structured["company"],
        "experience_required": structured["experience_required"],
        "skills_required":     structured["skills_required"],
        "source_url":          url,
        "extraction_method":   extraction_method,
    }


@router.post("/parse-jd-text")
@limiter.limit("30/minute")
async def parse_jd_text(request: Request, req: ParseJDTextRequest,
                        current_user: dict = Depends(get_current_user)):
    raw_text   = req.text.strip()[:15000]
    structured = {"title": "", "company": "", "experience_required": "", "skills_required": "", "job_description": raw_text}
    try:
        from specialized_judges import call_ollama_async
        model = get_free_model()
        system = "You extract structured job posting data. Return ONLY valid JSON, no markdown, no explanation."
        prompt = f"""From the job posting text below, extract these fields:
- title: job title only
- company: company name only
- experience_required: years/level of experience
- skills_required: comma-separated list of up to 12 core technical skills only
- job_description: the COMPLETE full job description text verbatim — do NOT summarize. Remove only page chrome.

Job posting text:
{raw_text}

Return exactly: {{"title": "...", "company": "...", "experience_required": "...", "skills_required": "...", "job_description": "..."}}"""

        result = await call_ollama_async(system, prompt, model, temperature=0.0)
        if result.get("success"):
            raw = re.sub(r"```[a-z]*\n?", "", result.get("data", "")).strip()
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
        pass

    return {
        "title":               structured["title"],
        "company":             structured["company"],
        "experience_required": structured["experience_required"],
        "skills_required":     structured["skills_required"],
        "job_description":     structured["job_description"],
    }
