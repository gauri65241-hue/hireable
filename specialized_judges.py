"""
Specialized LLM Judges for LinkedIn Agent
==========================================
Fixes & Upgrades applied:
  1. robust_parse_response: fixed backtick typo that silently broke JSON parsing
  2. Retry logic: parse/LLM failures on non-final attempts now continue instead of returning early
  3. Retry prompt: uses a single retry_note injected once
  4. 5-Stage Stage-1 Pipeline: Added Statement Relevancy Judge and Impact & Metrics Judge.
  5. Synthesizing Coach Judge: Combines 5 distinct gap lists to prevent duplicate advice and 
     provides copy-paste-ready rewritten bullet points.
"""

import json
import re
import asyncio
import os
import logging
from typing import Dict, Any, Optional, List

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from chunker import parse_jd, parse_resume, format_for_skills_judge, format_for_experience_judge, format_for_seniority_judge, format_for_relevancy_judge, format_for_impact_judge, format_for_coach_judge

# ─── Skills taxonomy (optional) ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "skills_taxonomy.json")
JUDGE_FEEDBACK_FILE = os.path.join(BASE_DIR, "results", "judge_feedback.jsonl")

try:
    with open(SKILLS_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        SKILLS_TAXONOMY = json.load(f)
    print(f"✅ Loaded skills taxonomy with {len(SKILLS_TAXONOMY)} domains")
except Exception as e:
    print(f"⚠️  Could not load skills taxonomy: {e}")
    SKILLS_TAXONOMY = {}

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LLM_Judge_Engine")


# ============================================================================
# SECTION 1 — PROMPTS
# ============================================================================

# ── 1a. Adversarial Skills Gap Judge ─────────────────────────────────────────
ADVERSARIAL_SKILLS_JUDGE_SYSTEM_PROMPT = """You are a systematic skills evaluator that follows exact algorithmic steps.

MANDATORY REASONING ALGORITHM:
STEP 1: Identify all required skills from the Job Description (technical, soft, domain-specific)
STEP 2: Use the provided SKILLS TAXONOMY to categorize each identified skill
STEP 3: For EACH required skill, search the Resume for specific evidence
STEP 4: Classify skill presence as PRESENT (1) or MISSING (0) — no other values
STEP 5: Provide a specific evidence quote from the Resume for each classification
STEP 6: Identify all missing critical skills and list them separately

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- Focus on MISSING skills first (adversarial approach).
- is_present MUST be exactly the integer 1 (found) or 0 (missing). No decimals.
- confidence MUST be an integer between 0-100 representing your certainty in the analysis.
- ONLY extract actual technical skills, tools, frameworks, programming languages, and domain knowledge as required_skills.
- DO NOT include education degrees (e.g. "Bachelor's degree", "Master's", "PhD"), years of experience, or soft skills like "communication" as required_skills items.

Output format:
{
    "required_skills": ["skill1", "skill2"],
    "missing_critical_skills": ["skill1"],
    "present_skills": ["skill2"],
    "taxonomy_categorization": {"skill1": "category"},
    "skill_analysis": [
        {
            "skill_requirement": "<exact text from JD>",
            "skill": "skill_name",
            "is_present": 1,
            "reasoning": "<why present or missing>",
            "resume_evidence": "<exact quote or 'Not found'>",
            "evidence_strength": "strong | medium | weak"
        }
    ],
    "overall_reasoning": "<final summary>",
    "confidence_calculation": "<reasoning for confidence level>",
    "confidence": 0
}"""

ADVERSARIAL_SKILLS_JUDGE_PROMPT = """Job Description:
{job_description_text}

Candidate Resume:
{resume_text}

SKILLS TAXONOMY:
{skills_taxonomy}

{retry_note}

Perform the skills coverage analysis using statement-by-statement evaluation and output the JSON.
Focus on identifying missing skills first.

IMPORTANT: Your entire response must be ONLY the JSON object above. No explanations, no markdown, no text before or after the JSON."""

# ── 1b. Experience Depth Judge ────────────────────────────────────────────────
EXPERIENCE_JUDGE_SYSTEM_PROMPT = """You are an expert hiring manager conducting a deep-dive relevancy evaluation.
Determine whether the candidate's experience depth matches the Job Description's scope.

MANDATORY REASONING ALGORITHM:
STEP 1: Identify the 3 most critical core responsibilities in the Job Description.
STEP 2: For EACH responsibility, assess whether the candidate performed work of similar scope, complexity, and scale.
STEP 3: Classify each responsibility as MET (1.0), PARTIAL (0.5), or UNMET (0.0).
STEP 4: Provide a brief reasoning and a direct resume quote for each classification.
STEP 5: Calculate your confidence based on evidence strength.

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- experience_score MUST be exactly 1.0, 0.5, or 0.0.
- confidence MUST be an integer between 0-100 representing your certainty in the analysis.

Output format:
{
    "responsibility_analysis": [
        {
            "responsibility": "<exact text from JD>",
            "experience_score": 1.0,
            "reasoning": "<why this score>",
            "resume_evidence": "<exact quote>",
            "evidence_strength": "strong | medium | weak"
        }
    ],
    "overall_reasoning": "<final summary>",
    "confidence": 0
}"""

EXPERIENCE_JUDGE_PROMPT = """Job Description:
{job_description_text}

Candidate Resume:
{resume_text}

{retry_note}

Perform the experience depth analysis using statement-by-statement evaluation and output the JSON.

IMPORTANT: Your entire response must be ONLY the JSON object above. No explanations, no markdown, no text before or after the JSON."""

# ── 1c. Seniority Calibration Judge ────────────────────────────────────
SENIORITY_JUDGE_SYSTEM_PROMPT = """You are a senior talent partner who specialises in detecting keyword-stuffers.
A keyword-stuffer lists required tools/skills but shows no evidence of working at senior scale.

Your job: evaluate whether the candidate demonstrates SENIORITY SIGNALS, not just skill keywords.

SENIORITY SIGNALS TO CHECK:
- Scale: did they work with large systems, high volumes, or production-grade infrastructure?
- Ownership: did they own outcomes end-to-end, or just contribute to someone else's work?
- Leadership: did they mentor, lead, or influence cross-functional teams?
- Complexity: did they solve ambiguous or novel problems, or follow defined playbooks?

MANDATORY REASONING ALGORITHM:
STEP 1: Extract every seniority signal required by the Job Description.
STEP 2: For EACH signal, search the Resume for concrete evidence.
STEP 3: Classify each signal as DEMONSTRATED (1) or ABSENT (0).
STEP 4: Provide reasoning and a direct resume quote for each classification.
STEP 5: Calculate your confidence.

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- is_demonstrated MUST be exactly 1 (present) or 0 (absent). No decimals.
- confidence MUST be an integer between 0-100 representing your certainty in the analysis.

Output format:
{
    "seniority_analysis": [
        {
            "signal": "<seniority signal from JD>",
            "signal_type": "scale | ownership | leadership | complexity",
            "is_demonstrated": 1,
            "reasoning": "<why present or absent>",
            "resume_evidence": "<exact quote or 'Not found'>"
        }
    ],
    "red_flags": ["<keyword-stuffing patterns spotted>"],
    "overall_reasoning": "<final seniority assessment>",
    "confidence": 0
}"""

SENIORITY_JUDGE_PROMPT = """Job Description:
{job_description_text}

Candidate Resume:
{resume_text}

{retry_note}

Evaluate the candidate's seniority signals using statement-by-statement analysis and output the JSON.

IMPORTANT: Your entire response must be ONLY the JSON object above. No explanations, no markdown, no text before or after the JSON."""

# ── 1d. Statement Relevancy Judge (NEW) ────────────────────────────────────
RELEVANCY_JUDGE_SYSTEM_PROMPT = """You are an expert technical recruiter conducting a granular, statement-by-statement relevancy evaluation.

MANDATORY REASONING ALGORITHM:
STEP 1: Extract every distinct experiential statement/bullet from the Candidate's Resume.
STEP 2: For EACH extracted statement, determine if it aligns with the core requirements or domain of the Job Description.
STEP 3: Classify each statement as RELEVANT (1) or IRRELEVANT (0). No decimals.
STEP 4: Provide reasoning mapping it to the JD (or explaining why it fails to map).

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- 'is_relevant' MUST be exactly 1 or 0.
- confidence MUST be an integer between 0-100 representing your certainty in the analysis.

Output format:
{
    "statement_analysis": [
        {
            "resume_statement": "<exact text of extracted bullet>",
            "mapped_jd_requirement": "<JD requirement addressed, or 'None'>",
            "is_relevant": 1,
            "reasoning": "<why relevant or irrelevant>"
        }
    ],
    "overall_reasoning": "<summary of resume noise vs signal>",
    "confidence": 0
}"""

RELEVANCY_JUDGE_PROMPT = """Job Description:
{job_description_text}

Candidate Resume:
{resume_text}

{retry_note}

Evaluate statement-by-statement relevancy and output the JSON.

IMPORTANT: Your entire response must be ONLY the JSON object above. No explanations, no markdown, no text before or after the JSON."""

# ── 1e. Impact & Metrics Judge (NEW) ───────────────────────────────────────
IMPACT_JUDGE_SYSTEM_PROMPT = """You are a strict data-driven hiring manager.
Your job is to ensure the candidate quantifies their impact, rather than just listing duties.

MANDATORY REASONING ALGORITHM:
STEP 1: Extract every experiential statement/bullet from the Candidate's Resume.
STEP 2: For EACH statement, check if it contains explicit, quantified metrics (e.g., percentages, revenue, latency improvements, dataset scale, user counts).
STEP 3: Classify each statement using this scale:
  - 1 = FULLY QUANTIFIED: contains explicit numbers, percentages, revenue, latency, scale, user counts
  - 0.5 = DIRECTIONAL: mentions measurable improvement/reduction/increase but no specific number (e.g. "improved accuracy", "reduced latency", "scaled the system")
  - 0 = DUTY-ONLY: lists a task or responsibility with no mention of outcome or impact at all
STEP 4: If not fully quantified, suggest a metric category they *could* use (e.g., "needs scale metric", "needs latency metric").

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- 'is_quantified' MUST be exactly 1, 0.5, or 0. No other values.
- confidence MUST be an integer between 0-100 representing your certainty in the analysis.

Output format:
{
    "impact_analysis": [
        {
            "resume_statement": "<exact text of extracted bullet>",
            "is_quantified": 1,
            "extracted_metrics": ["<metric 1>", "<metric 2>"] /* empty if 0 or 0.5 */,
            "missing_metric_suggestion": "<what metric they should add if 0 or 0.5>"
        }
    ],
    "overall_reasoning": "<summary of the candidate's focus on impact>",
    "confidence": 0
}"""

IMPACT_JUDGE_PROMPT = """Candidate Resume:
{resume_text}

{retry_note}

Evaluate the candidate's use of metrics and impact, and output the JSON.

IMPORTANT: Your entire response must be ONLY the JSON object above. No explanations, no markdown, no text before or after the JSON."""

# ── 1f. Synthesizing Coach Judge ──────────────────────────────────────────────
COACH_JUDGE_SYSTEM_PROMPT = """You are an elite Technical Career Coach and Resume Writer.
You are receiving the raw, structured output from 5 specialized AI evaluation agents that just analyzed the candidate's resume against the target Job Description.

Your goal is to synthesize these granular gaps into highly specific, copy-paste-ready bullet point rewrites.

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- is_actionable MUST be exactly 1 or 0.
- NEVER FABRICATE EXPERIENCE: You may only rewrite or reframe bullets that already exist in the candidate's resume. Do NOT add new bullets claiming skills or experience the candidate did not demonstrate. If a required skill is genuinely absent, flag it as a gap warning, not a fabricated bullet.
- NO INVENTED METRICS: Only use numbers that already appear in the candidate's existing resume. If no real number exists, rewrite without any metric — do NOT insert placeholders like [X%], [Insert metric], [N], or any bracket-wrapped value. A clean rewrite with no metric is better than a fake placeholder.
- GAP WARNINGS: For skills or responsibilities the candidate provably lacks, set improvement_type to "gap_warning" and is_actionable to 0. Explain the gap honestly — do not attempt to paper over it.
- REWRITES ONLY: improvement_type "bullet_rewrite" means reframing an existing bullet to better highlight transferable skills. It must reference original_content from the resume.

Output format:
{
    "improvement_analysis": [
        {
            "gap_addressed": "<the specific skills/responsibilities/metrics this fixes>",
            "improvement_type": "bullet_rewrite | deletion | gap_warning",
            "is_actionable": 1,
            "original_content": "<exact current resume text being rewritten, required for bullet_rewrite>",
            "suggested_improvement": "<rewritten bullet using only real experience, or gap warning message>",
            "reasoning": "<why this rewrite highlights transferable value, or why this is a genuine gap>",
            "priority": "high | medium | low"
        }
    ],
    "overall_reasoning": "<honest coaching summary: what genuinely transfers, what gaps exist, expected realistic fit>"
}"""

COACH_JUDGE_PROMPT = """=========================================
SOURCE MATERIAL
=========================================
JOB DESCRIPTION:
{job_description_text}

CANDIDATE'S CURRENT RESUME:
{resume_text}

=========================================
AGENT EVALUATION RESULTS (THE GAPS)
=========================================
1. MISSING SKILLS (Adversarial Skills Judge):
{missing_skills}

2. UNMET CORE RESPONSIBILITIES (Experience Depth Judge):
{unmet_responsibilities}

3. ABSENT SENIORITY SIGNALS (Seniority Calibration Judge):
{absent_seniority_signals}

4. IRRELEVANT STATEMENTS (Relevancy Judge):
{irrelevant_statements}

5. UNQUANTIFIED STATEMENTS (Impact & Metrics Judge):
{unquantified_statements}

=========================================
COACHING & SYNTHESIS DIRECTIVES
=========================================
1. COMBINE FIXES: If a bullet is unquantified AND lacks a seniority signal, produce ONE rewrite that fixes both — do not duplicate.
2. REWRITES ONLY: Each bullet_rewrite MUST reference real text from the candidate's resume in original_content. Rephrase to surface transferable skills more clearly. Never add capabilities the resume doesn't evidence.
3. HONEST GAPS: Skills in the missing list that do not appear anywhere in the resume → set as gap_warning with is_actionable=0. State clearly: "Candidate has no demonstrated experience in X. Do not add this to the resume."
4. PRUNING: For irrelevant statements, use improvement_type "deletion". Do not replace deleted content with fabricated bullets.
5. PRIORITIZATION:
   - HIGH: Rewrites that reframe real experience to match core responsibilities.
   - MEDIUM: Adding real metrics / elevating seniority signals using existing content.
   - LOW: Minor wording improvements.
   - gap_warning: Always flag genuine skill absences regardless of priority.

Analyze the 5 gap lists and output your final coaching plan using the JSON format.

IMPORTANT: Your entire response must be ONLY the JSON object above. No explanations, no markdown, no text before or after the JSON.
{retry_note}"""


# ============================================================================
# SECTION 2 — SCORING FUNCTIONS
# ============================================================================

def calculate_skills_score(judge_json: dict) -> dict:
    analysis = judge_json.get("skill_analysis", [])
    if not analysis:
        return {"score": 0.0, "present": 0, "missing": 0, "total": 0, "missing_skills": []}

    total = len(analysis)
    present = sum(1 for s in analysis if s.get("is_present") == 1)
    partial = sum(1 for s in analysis if s.get("is_present") == 0.5)
    missing_skills = [
        s.get("skill", s.get("skill_requirement", "unknown"))
        for s in analysis if s.get("is_present") == 0
    ]

    score = (present + 0.5 * partial) / total if total > 0 else 0.0

    return {
        "score": round(score, 3),
        "present": present,
        "partial": partial,
        "missing": total - present - partial,
        "total": total,
        "missing_skills": missing_skills,
    }

def calculate_experience_score(judge_json: dict) -> dict:
    analysis = judge_json.get("responsibility_analysis", [])
    if not analysis:
        return {"score": 0.0, "met": 0, "partial": 0, "unmet": 0, "total": 0, "unmet_responsibilities": []}

    total = len(analysis)
    met = sum(1 for r in analysis if r.get("experience_score") == 1.0)
    partial = sum(1 for r in analysis if r.get("experience_score") == 0.5)
    unmet_responsibilities = [
        r.get("responsibility", "unknown")
        for r in analysis if r.get("experience_score", 1.0) < 0.5
    ]

    score = (met + 0.5 * partial) / total if total > 0 else 0.0

    return {
        "score": round(score, 3),
        "met": met,
        "partial": partial,
        "unmet": total - met - partial,
        "total": total,
        "unmet_responsibilities": unmet_responsibilities,
    }

def calculate_seniority_score(judge_json: dict) -> dict:
    analysis = judge_json.get("seniority_analysis", [])
    if not analysis:
        return {"score": 0.0, "demonstrated": 0, "absent": 0, "total": 0, "absent_signals": [], "red_flags": []}

    total = len(analysis)
    demonstrated = sum(1 for s in analysis if s.get("is_demonstrated") == 1)
    absent_signals = [s.get("signal", "unknown") for s in analysis if s.get("is_demonstrated") == 0]
    score = demonstrated / total if total > 0 else 0.0

    return {
        "score": round(score, 3),
        "demonstrated": demonstrated,
        "absent": total - demonstrated,
        "total": total,
        "absent_signals": absent_signals,
        "red_flags": judge_json.get("red_flags", []),
    }

def calculate_relevancy_score(judge_json: dict) -> dict:
    analysis = judge_json.get("statement_analysis", [])
    if not analysis:
        return {"score": 0.0, "relevant": 0, "irrelevant": 0, "total": 0, "irrelevant_statements": []}

    total = len(analysis)
    relevant = sum(1 for s in analysis if s.get("is_relevant") == 1)
    irrelevant_statements = [s.get("resume_statement") for s in analysis if s.get("is_relevant") == 0]
    
    return {
        "score": round(relevant / total, 3) if total else 0.0,
        "relevant": relevant,
        "irrelevant": total - relevant,
        "total": total,
        "irrelevant_statements": irrelevant_statements
    }

def calculate_impact_score(judge_json: dict) -> dict:
    analysis = judge_json.get("impact_analysis", [])
    if not analysis:
        return {"score": 0.0, "quantified": 0, "directional": 0, "unquantified": 0, "total": 0, "unquantified_statements": []}

    total = len(analysis)
    quantified = sum(1 for s in analysis if s.get("is_quantified") == 1)
    directional = sum(1 for s in analysis if s.get("is_quantified") == 0.5)
    unquantified_statements = [
        {"statement": s.get("resume_statement"), "suggestion": s.get("missing_metric_suggestion")}
        for s in analysis if s.get("is_quantified") == 0
    ]

    # Full credit for quantified, half credit for directional impact statements
    score = (quantified + 0.5 * directional) / total if total else 0.0

    return {
        "score": round(score, 3),
        "quantified": quantified,
        "directional": directional,
        "unquantified": total - quantified - directional,
        "total": total,
        "unquantified_statements": unquantified_statements,
    }

JUDGE_DEFAULT_WEIGHTS = {"skills": 0.25, "experience": 0.25, "seniority": 0.20, "relevancy": 0.15, "impact": 0.15}

def calculate_overall_match(skills: dict, experience: dict, seniority: dict, relevancy: dict, impact: dict, weights: dict = None) -> dict:
    """Weighted overall match score using all 5 judges."""
    w = weights if weights is not None else JUDGE_DEFAULT_WEIGHTS
    overall = (
        skills["score"] * w["skills"]
        + experience["score"] * w["experience"]
        + seniority["score"] * w["seniority"]
        + relevancy["score"] * w["relevancy"]
        + impact["score"] * w["impact"]
    )
    return {
        "overall_score": round(overall, 3),
        "weights_used": w,
        "component_scores": {
            "skills": skills["score"],
            "experience": experience["score"],
            "seniority": seniority["score"],
            "relevancy": relevancy["score"],
            "impact": impact["score"]
        },
    }


# ============================================================================
# SECTION 3 — LLM INTEGRATION
# ============================================================================

def call_api_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    api_base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.1,
) -> dict:
    """Call any OpenAI-compatible API (OpenRouter, OpenAI, etc.)."""
    try:
        url = api_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        print(f"🌐 Calling API model {model} at {api_base_url}")
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"✅ API response: {len(content)} chars")
        return {"success": True, "data": content}
    except Exception as e:
        print(f"❌ API model call failed: {e}")
        return {"success": False, "error": str(e)}


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = "deepseek-r1:14b",
    temperature: float = 0.1,
    api_key: str = "",
    api_base_url: str = "",
) -> dict:
    if api_key:
        base = api_base_url or "https://openrouter.ai/api/v1"
        return call_api_model(system_prompt, user_prompt, model, api_key, base, temperature)
    try:
        total_chars = len(system_prompt) + len(user_prompt)
        # ~4 chars per token; add 2048 headroom for output
        num_ctx = min(131072, max(4096, (total_chars // 4) + 2048))
        print(f"🚀 Calling {model} | sys={len(system_prompt)}ch usr={len(user_prompt)}ch ctx={num_ctx}")
        llm = ChatOllama(model=model, temperature=temperature, num_ctx=num_ctx)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        print(f"✅ Response received: {len(response.content)} chars")
        return {"success": True, "data": response.content}
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return {"success": False, "error": str(e)}

def extract_confidence_from_text(text: str) -> int:
    patterns = [
        r"confidence[:\s]*(\d+)%?", r"confident[:\s]*(\d+)%?",
        r"(\d+)%\s*confidence", r"(\d+)/10", r"(\d+)\s*out\s*of\s*10",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                v = int(matches[0])
                if v == 0:
                    continue  # 0 is the template placeholder, not a real confidence
                return v * 10 if v <= 10 else min(100, v)
            except ValueError:
                continue

    positive = ["good", "excellent", "strong", "accurate", "clear", "comprehensive"]
    negative = ["poor", "bad", "unclear", "uncertain", "weak", "incomplete"]
    tl = text.lower()
    pos = sum(1 for w in positive if w in tl)
    neg = sum(1 for w in negative if w in tl)
    return 75 if pos > neg else (60 if neg > pos else 50)

def robust_parse_response(response: str, judge_type: str = "skills") -> dict:
    cleaned = response.strip()
    
    # More robust JSON extraction from markdown blocks
    if "```json" in cleaned:
        try:
            # Extract content between ```json and ```
            json_start = cleaned.find("```json") + 7
            json_end = cleaned.find("```", json_start)
            if json_end > json_start:
                cleaned = cleaned[json_start:json_end].strip()
            else:
                # Fallback: take everything after ```json
                cleaned = cleaned.split("```json")[1].strip()
        except:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        try:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()
            else:
                cleaned = cleaned.replace("```", "").strip()
        except:
            cleaned = cleaned.replace("```", "").strip()
    
    # Try to find JSON object in the response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            result["source"] = "json_extracted"
            result["success"] = True
            return result
        except json.JSONDecodeError:
            pass

    try:
        result = json.loads(cleaned)
        result["source"] = "json"
        result["success"] = True
        return result
    except json.JSONDecodeError:
        pass

    print(f"🔧 JSON parse failed for {judge_type} judge, falling back to text extraction")
    summary = (cleaned[:200] + "...") if len(cleaned) > 200 else cleaned

    base = {
        "success": True,
        "source": "text_extraction",
        "overall_reasoning": summary,
        "confidence": extract_confidence_from_text(response),
    }

    if judge_type == "skills":
        return {**base, "required_skills": [], "taxonomy_categorization": {}, "skill_analysis": [], "missing_critical_skills": [], "present_skills": [], "confidence_calculation": "JSON parsing failed"}
    elif judge_type == "experience":
        return {**base, "responsibility_analysis": []}
    elif judge_type == "seniority":
        return {**base, "seniority_analysis": [], "red_flags": []}
    elif judge_type == "relevancy":
        return {**base, "statement_analysis": []}
    elif judge_type == "impact":
        return {**base, "impact_analysis": []}
    else:  # coach
        return {**base, "improvement_analysis": []}


# ============================================================================
# SECTION 4 — SEMANTIC TAXONOMY CHUNKING
# ============================================================================

def extract_semantic_chunks(job_description: str, resume_text: str, skills_taxonomy: dict) -> dict:
    if not skills_taxonomy: return {}

    combined = f"{job_description} {resume_text}".lower()
    chunk_defs = {
        "technical_skills": {"keywords": ["python", "java", "javascript", "sql", "aws", "docker", "kubernetes", "react", "node", "api", "database", "cloud", "devops", "git", "linux", "machine learning", "ai", "data science", "algorithm", "programming"], "priority": 1},
        "business_skills": {"keywords": ["strategy", "management", "leadership", "project", "product", "marketing", "sales", "revenue", "growth", "business", "customer", "stakeholder", "negotiation", "planning", "analysis", "reporting", "communication"], "priority": 2},
        "domain_expertise": {"keywords": ["healthcare", "finance", "legal", "education", "retail", "manufacturing", "logistics", "supply chain", "ecommerce", "fintech", "biotech", "pharmaceutical", "insurance", "real estate", "energy"], "priority": 3},
        "process_methodology": {"keywords": ["agile", "scrum", "kanban", "waterfall", "lean", "six sigma", "itil", "devops", "cicd", "testing", "quality assurance", "compliance", "risk management", "governance", "documentation"], "priority": 4},
        "soft_skills": {"keywords": ["teamwork", "collaboration", "communication", "problem solving", "critical thinking", "adaptability", "creativity", "innovation", "mentoring", "leadership", "time management", "analytical", "strategic"], "priority": 5},
    }

    scored = {}
    for name, cfg in chunk_defs.items():
        hits = [kw for kw in cfg["keywords"] if kw in combined]
        score = sum(combined.count(kw) for kw in hits)
        if score > 0:
            scored[name] = {"score": score, "priority": cfg["priority"]}

    top3 = sorted(scored, key=lambda n: (scored[n]["score"], -scored[n]["priority"]), reverse=True)[:3]

    extractors = {
        "technical_skills": _extract_technical,
        "business_skills": _extract_business,
        "domain_expertise": _extract_domain,
        "process_methodology": _extract_process,
        "soft_skills": _extract_soft,
    }
    return {name: extractors[name](skills_taxonomy, job_description, resume_text) for name in top3}

def _extract_technical(taxonomy, job_text, resume_text):
    combined = f"{job_text} {resume_text}".lower()
    out = {"tools": {}, "concepts": {}}
    for domain_data in taxonomy.values():
        for cat, items in domain_data.get("tools", {}).items():
            matches = [i for i in items if i.lower() in combined]
            if matches: out["tools"].setdefault(cat, []).extend(matches)
        for cat, items in domain_data.get("concepts", {}).items():
            matches = [i for i in items if i.lower() in combined]
            if matches: out["concepts"].setdefault(cat, []).extend(matches)
    return out

def _extract_business(taxonomy, job_text, resume_text):
    combined = f"{job_text} {resume_text}".lower()
    out = {"tools": {}, "concepts": {}, "soft_skills": []}
    biz_domains = {"sales_and_marketing", "executive_and_management", "hr_and_people_ops"}
    for name, data in taxonomy.items():
        if name not in biz_domains: continue
        for cat, items in data.get("tools", {}).items():
            matches = [i for i in items if i.lower() in combined]
            if matches: out["tools"].setdefault(cat, []).extend(matches)
        for cat, items in data.get("concepts", {}).items():
            matches = [i for i in items if i.lower() in combined]
            if matches: out["concepts"].setdefault(cat, []).extend(matches)
        out["soft_skills"].extend(s for s in data.get("soft_skills", []) if s.lower() in combined)
    return out

def _extract_domain(taxonomy, job_text, resume_text):
    combined = f"{job_text} {resume_text}".lower()
    domain_names = {"healthcare_and_medical", "finance_and_accounting", "legal_and_compliance", "supply_chain_and_logistics", "culinary_and_kitchen"}
    out = {}
    for name, data in taxonomy.items():
        if name not in domain_names: continue
        titles = [t for t in data.get("titles", []) if t.lower() in combined]
        concepts = {c: [i for i in items if i.lower() in combined] for c, items in data.get("concepts", {}).items() if any(i.lower() in combined for i in items)}
        tools = {c: [i for i in items if i.lower() in combined] for c, items in data.get("tools", {}).items() if any(i.lower() in combined for i in items)}
        if titles or concepts or tools: out[name] = {"titles": titles, "concepts": concepts, "tools": tools}
    return out

def _extract_process(taxonomy, job_text, resume_text):
    combined = f"{job_text} {resume_text}".lower()
    out = {"concepts": {}}
    for data in taxonomy.values():
        for cat, items in data.get("concepts", {}).items():
            matches = [i for i in items if i.lower() in combined]
            if matches: out["concepts"].setdefault(cat, []).extend(matches)
    return out

def _extract_soft(taxonomy, job_text, resume_text):
    combined = f"{job_text} {resume_text}".lower()
    found = set()
    for data in taxonomy.values():
        found.update(s for s in data.get("soft_skills", []) if s.lower() in combined)
    return {"soft_skills": list(found)}


# ============================================================================
# SECTION 5 — PROMPT BUILDERS
# ============================================================================

def load_judge_calibration(judge_name: str) -> str:
    """Read last 5 feedback entries for a judge, return a calibration note for its system prompt."""
    entries = []
    try:
        with open(JUDGE_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    if e.get("judge") == judge_name:
                        entries.append(e)
    except FileNotFoundError:
        return ""

    recent = entries[-5:]
    if len(recent) < 2:
        return ""

    total = len(recent)
    correct = sum(1 for e in recent if e["correct"])
    pos = [e for e in recent if e["outcome"]]
    neg = [e for e in recent if not e["outcome"]]
    avg_pos = sum(e["predicted_score"] for e in pos) / len(pos) if pos else None
    avg_neg = sum(e["predicted_score"] for e in neg) / len(neg) if neg else None

    lines = [f"\nCALIBRATION NOTE (from {total} real-world outcomes):"]
    lines.append(f"- Your recent prediction accuracy: {correct/total:.0%} ({correct}/{total} correct)")

    if avg_pos is not None and avg_neg is not None:
        if avg_pos < 0.5:
            lines.append("- You UNDER-SCORE successful candidates. Be more generous when evidence is present but implicit.")
        elif avg_neg > 0.55:
            lines.append("- You OVER-SCORE unsuitable candidates. Apply stricter evidence standards before scoring high.")
        else:
            lines.append("- Scoring appears well-calibrated. Maintain current standards.")
    elif avg_pos is not None and avg_pos < 0.5:
        lines.append("- You UNDER-SCORE successful candidates. Be more generous when evidence is present but implicit.")

    return "\n".join(lines)


def _build_retry_note(attempt: int, prev_confidence: int, threshold: int) -> str:
    if attempt == 0: return ""
    return (f"IMPORTANT: Your previous response had confidence {prev_confidence}% "
            f"which is below the required threshold of {threshold}%. "
            "Be more specific in your evidence quotes and more decisive in your classifications.")

def create_skills_judge_prompt(resume_text: str, job_description: str, skills_taxonomy: dict = None, attempt: int = 0, prev_confidence: int = 0, confidence_threshold: int = 75) -> tuple:
    jd = parse_jd(job_description)
    resume = parse_resume(resume_text)
    jd_block, resume_block = format_for_skills_judge(jd, resume)

    # Taxonomy filtered to just the detected domain
    semantic = extract_semantic_chunks(job_description, resume_text, skills_taxonomy or {})
    taxonomy_str = json.dumps(semantic, indent=2) if semantic else json.dumps({"technical_skills": {"tools": {"languages": ["python", "sql"]}}}, indent=2)

    user = ADVERSARIAL_SKILLS_JUDGE_PROMPT.format(
        job_description_text=jd_block,
        resume_text=resume_block,
        skills_taxonomy=taxonomy_str,
        retry_note=_build_retry_note(attempt, prev_confidence, confidence_threshold),
    )
    return ADVERSARIAL_SKILLS_JUDGE_SYSTEM_PROMPT + load_judge_calibration("skills"), user, semantic


def create_experience_judge_prompt(resume_text: str, job_description: str, attempt: int = 0, prev_confidence: int = 0, confidence_threshold: int = 75) -> tuple:
    jd = parse_jd(job_description)
    resume = parse_resume(resume_text)
    jd_block, resume_block = format_for_experience_judge(jd, resume)

    user = EXPERIENCE_JUDGE_PROMPT.format(
        job_description_text=jd_block,
        resume_text=resume_block,
        retry_note=_build_retry_note(attempt, prev_confidence, confidence_threshold),
    )
    return EXPERIENCE_JUDGE_SYSTEM_PROMPT + load_judge_calibration("experience"), user


def create_seniority_judge_prompt(resume_text: str, job_description: str, attempt: int = 0, prev_confidence: int = 0, confidence_threshold: int = 75) -> tuple:
    jd = parse_jd(job_description)
    resume = parse_resume(resume_text)
    jd_block, resume_block = format_for_seniority_judge(jd, resume)

    user = SENIORITY_JUDGE_PROMPT.format(
        job_description_text=jd_block,
        resume_text=resume_block,
        retry_note=_build_retry_note(attempt, prev_confidence, confidence_threshold),
    )
    return SENIORITY_JUDGE_SYSTEM_PROMPT + load_judge_calibration("seniority"), user


def create_relevancy_judge_prompt(resume_text: str, job_description: str, attempt: int = 0, prev_confidence: int = 0, confidence_threshold: int = 75) -> tuple:
    jd = parse_jd(job_description)
    resume = parse_resume(resume_text)
    jd_block, resume_block = format_for_relevancy_judge(jd, resume)

    user = RELEVANCY_JUDGE_PROMPT.format(
        job_description_text=jd_block,
        resume_text=resume_block,
        retry_note=_build_retry_note(attempt, prev_confidence, confidence_threshold),
    )
    return RELEVANCY_JUDGE_SYSTEM_PROMPT + load_judge_calibration("relevancy"), user


def create_impact_judge_prompt(resume_text: str, attempt: int = 0, prev_confidence: int = 0, confidence_threshold: int = 75) -> tuple:
    resume = parse_resume(resume_text)
    resume_block = format_for_impact_judge(resume)

    user = IMPACT_JUDGE_PROMPT.format(
        resume_text=resume_block,
        retry_note=_build_retry_note(attempt, prev_confidence, confidence_threshold),
    )
    return IMPACT_JUDGE_SYSTEM_PROMPT + load_judge_calibration("impact"), user

def create_coach_judge_prompt(
    resume_text: str, job_description: str,
    missing_skills: List[str], unmet_responsibilities: List[str], absent_seniority_signals: List[str],
    irrelevant_statements: List[str], unquantified_statements: List[dict]
) -> tuple:
    jd = parse_jd(job_description)
    resume = parse_resume(resume_text)
    jd_block, resume_block = format_for_coach_judge(jd, resume)

    user = COACH_JUDGE_PROMPT.format(
        job_description_text=jd_block,
        resume_text=resume_block,
        missing_skills=json.dumps(missing_skills, indent=2),
        unmet_responsibilities=json.dumps(unmet_responsibilities, indent=2),
        absent_seniority_signals=json.dumps(absent_seniority_signals, indent=2),
        irrelevant_statements=json.dumps(irrelevant_statements, indent=2),
        unquantified_statements=json.dumps(unquantified_statements, indent=2),
        retry_note=""
    )
    return COACH_JUDGE_SYSTEM_PROMPT, user


# ============================================================================
# SECTION 6 — RETRY RUNNER
# ============================================================================

async def _run_with_retry(
    prompt_builder,
    builder_args: tuple, judge_type: str, llm_model: str,
    max_retries: int, confidence_threshold: int, extra_return_key: str = None,
    api_key: str = "", api_base_url: str = "",
) -> dict:
    prev_confidence = 0

    for attempt in range(max_retries):
        print(f"🔄 {judge_type} judge — attempt {attempt + 1}/{max_retries}")
        built = prompt_builder(*builder_args, attempt=attempt, prev_confidence=prev_confidence, confidence_threshold=confidence_threshold)
        
        if len(built) == 3:
            system_prompt, user_prompt, extra = built
        else:
            system_prompt, user_prompt = built
            extra = None

        try:
            result = call_ollama(system_prompt, user_prompt, llm_model, temperature=0.1, api_key=api_key, api_base_url=api_base_url)
            if not result["success"]:
                if attempt == max_retries - 1:
                    return {"success": False, "error": result["error"], "raw_response": "", "attempts": attempt + 1}
                continue

            llm_response = result["data"]
            parsed = robust_parse_response(llm_response, judge_type)

            if parsed.get("success"):
                confidence = parsed.get("confidence", 0)
                prev_confidence = confidence
                print(f"📊 Confidence: {confidence}% (threshold: {confidence_threshold}%)")

                if confidence >= confidence_threshold or attempt == max_retries - 1:
                    out = {
                        "success": True, "data": parsed, "raw_response": llm_response,
                        "confidence_met": confidence >= confidence_threshold, "attempts": attempt + 1,
                    }
                    if extra_return_key and extra is not None:
                        out[extra_return_key] = extra
                    return out

                logger.warning("Confidence below threshold — retrying", extra={"judge": judge_type, "attempt": attempt + 1, "confidence": confidence})
                continue

            if attempt == max_retries - 1:
                return {"success": False, "error": "Parsing failed after all retries", "raw_response": llm_response, "attempts": attempt + 1}
            continue

        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e), "raw_response": "", "attempts": attempt + 1}
            continue

    return {"success": False, "error": "Exhausted retries", "raw_response": "", "attempts": max_retries}


# ============================================================================
# SECTION 7 — SPECIALIZED JUDGE ENGINE
# ============================================================================

class SpecializedJudgeEngine:
    def __init__(self, confidence_threshold: int = 75, max_retries: int = 3):
        self.config = {"confidence_threshold": confidence_threshold, "max_retries": max_retries}

    async def run_skills_judge(self, resume_text: str, job_description: str, llm_model: str = "deepseek-r1:14b", skills_taxonomy: dict = None, api_key: str = "", api_base_url: str = "") -> dict:
        print(f"\n🎯 Skills Gap Judge | model={llm_model}")
        def builder(resume, jd, taxonomy, *, attempt, prev_confidence, confidence_threshold):
            return create_skills_judge_prompt(resume, jd, taxonomy, attempt=attempt, prev_confidence=prev_confidence, confidence_threshold=confidence_threshold)
        return await _run_with_retry(builder, (resume_text, job_description, skills_taxonomy), "skills", llm_model, self.config["max_retries"], self.config["confidence_threshold"], "extracted_chunks", api_key=api_key, api_base_url=api_base_url)

    async def run_experience_judge(self, resume_text: str, job_description: str, llm_model: str = "deepseek-r1:14b", api_key: str = "", api_base_url: str = "") -> dict:
        print(f"\n📋 Experience Depth Judge | model={llm_model}")
        def builder(resume, jd, *, attempt, prev_confidence, confidence_threshold):
            return create_experience_judge_prompt(resume, jd, attempt=attempt, prev_confidence=prev_confidence, confidence_threshold=confidence_threshold)
        return await _run_with_retry(builder, (resume_text, job_description), "experience", llm_model, self.config["max_retries"], self.config["confidence_threshold"], api_key=api_key, api_base_url=api_base_url)

    async def run_seniority_judge(self, resume_text: str, job_description: str, llm_model: str = "deepseek-r1:14b", api_key: str = "", api_base_url: str = "") -> dict:
        print(f"\n🏅 Seniority Calibration Judge | model={llm_model}")
        def builder(resume, jd, *, attempt, prev_confidence, confidence_threshold):
            return create_seniority_judge_prompt(resume, jd, attempt=attempt, prev_confidence=prev_confidence, confidence_threshold=confidence_threshold)
        return await _run_with_retry(builder, (resume_text, job_description), "seniority", llm_model, self.config["max_retries"], self.config["confidence_threshold"], api_key=api_key, api_base_url=api_base_url)

    async def run_relevancy_judge(self, resume_text: str, job_description: str, llm_model: str = "deepseek-r1:14b", api_key: str = "", api_base_url: str = "") -> dict:
        print(f"\n🔍 Statement Relevancy Judge | model={llm_model}")
        def builder(resume, jd, *, attempt, prev_confidence, confidence_threshold):
            return create_relevancy_judge_prompt(resume, jd, attempt=attempt, prev_confidence=prev_confidence, confidence_threshold=confidence_threshold)
        return await _run_with_retry(builder, (resume_text, job_description), "relevancy", llm_model, self.config["max_retries"], self.config["confidence_threshold"], api_key=api_key, api_base_url=api_base_url)

    async def run_impact_judge(self, resume_text: str, llm_model: str = "deepseek-r1:14b", api_key: str = "", api_base_url: str = "") -> dict:
        print(f"\n📈 Impact & Metrics Judge | model={llm_model}")
        def builder(resume, *, attempt, prev_confidence, confidence_threshold):
            return create_impact_judge_prompt(resume, attempt=attempt, prev_confidence=prev_confidence, confidence_threshold=confidence_threshold)
        return await _run_with_retry(builder, (resume_text,), "impact", llm_model, self.config["max_retries"], self.config["confidence_threshold"], api_key=api_key, api_base_url=api_base_url)

    async def run_coach_judge(
        self, resume_text: str, job_description: str,
        missing_skills: List[str], unmet_responsibilities: List[str], absent_seniority_signals: List[str],
        irrelevant_statements: List[str], unquantified_statements: List[dict],
        llm_model: str = "deepseek-r1:14b", api_key: str = "", api_base_url: str = "",
    ) -> dict:
        print(f"\n🎓 Coach Judge | model={llm_model}")
        system_prompt, user_prompt = create_coach_judge_prompt(
            resume_text, job_description, missing_skills, unmet_responsibilities,
            absent_seniority_signals, irrelevant_statements, unquantified_statements
        )
        try:
            result = call_ollama(system_prompt, user_prompt, llm_model, temperature=0.2, api_key=api_key, api_base_url=api_base_url)
            if not result["success"]: return {"success": False, "error": result["error"], "raw_response": ""}
            parsed = robust_parse_response(result["data"], "coach")
            return {"success": parsed.get("success", False), "data": parsed, "raw_response": result["data"]}
        except Exception as e:
            return {"success": False, "error": str(e), "raw_response": ""}

    async def run_all_judges(self, resume_text: str, job_description: str, llm_model: str = "deepseek-r1:14b", skills_taxonomy: dict = None, api_key: str = "", api_base_url: str = "") -> dict:
        print("\n" + "=" * 60)
        print("🚀 Running Full 5-Stage Judge Pipeline")
        print("=" * 60)

        # ── Stage 1: Independent concurrent judges ──────────────────────────────
        results = await asyncio.gather(
            self.run_skills_judge(resume_text, job_description, llm_model, skills_taxonomy, api_key=api_key, api_base_url=api_base_url),
            self.run_experience_judge(resume_text, job_description, llm_model, api_key=api_key, api_base_url=api_base_url),
            self.run_seniority_judge(resume_text, job_description, llm_model, api_key=api_key, api_base_url=api_base_url),
            self.run_relevancy_judge(resume_text, job_description, llm_model, api_key=api_key, api_base_url=api_base_url),
            self.run_impact_judge(resume_text, llm_model, api_key=api_key, api_base_url=api_base_url)
        )
        
        skills_res, exp_res, seniority_res, relevancy_res, impact_res = results

        # ── Stage 2: Python scoring ───────────────────────────────────────────
        s_scores = calculate_skills_score(skills_res.get("data", {}))
        e_scores = calculate_experience_score(exp_res.get("data", {}))
        sr_scores = calculate_seniority_score(seniority_res.get("data", {}))
        rel_scores = calculate_relevancy_score(relevancy_res.get("data", {}))
        imp_scores = calculate_impact_score(impact_res.get("data", {}))

        print(f"\n📊 Stage 2 Component Scores:")
        print(f"   Skills:     {s_scores['score']:.1%}")
        print(f"   Experience: {e_scores['score']:.1%}")
        print(f"   Seniority:  {sr_scores['score']:.1%}")
        print(f"   Relevancy:  {rel_scores['score']:.1%}")
        print(f"   Impact:     {imp_scores['score']:.1%}")

        # ── Stage 3: Coach with full gap context ──────────────────────────────
        coach_result = await self.run_coach_judge(
            resume_text=resume_text,
            job_description=job_description,
            missing_skills=s_scores["missing_skills"],
            unmet_responsibilities=e_scores["unmet_responsibilities"],
            absent_seniority_signals=sr_scores["absent_signals"],
            irrelevant_statements=rel_scores["irrelevant_statements"],
            unquantified_statements=imp_scores["unquantified_statements"],
            llm_model=llm_model, api_key=api_key, api_base_url=api_base_url,
        )

        # ── Stage 4: Overall match score ──────────────────────────────────────
        overall = calculate_overall_match(s_scores, e_scores, sr_scores, rel_scores, imp_scores)
        print(f"\n🏆 Overall Match Score: {overall['overall_score']:.1%}")

        return {
            "overall": overall,
            "skills": {"judge_result": skills_res, "scores": s_scores, "extracted_chunks": skills_res.get("extracted_chunks", {})},
            "experience": {"judge_result": exp_res, "scores": e_scores},
            "seniority": {"judge_result": seniority_res, "scores": sr_scores},
            "relevancy": {"judge_result": relevancy_res, "scores": rel_scores},
            "impact": {"judge_result": impact_res, "scores": imp_scores},
            "coach": coach_result,
        }


# ============================================================================
# SECTION 8 — MAIN / STRESS TEST
# ============================================================================

if __name__ == "__main__":
    mock_jd = """
    Role: Senior Machine Learning Engineer
    Core Responsibilities:
    - Architect and scale machine learning models to handle millions of daily predictions.
    - Mentor junior data scientists and lead cross-functional project planning.
    - Deploy and orchestrate ML pipelines using Kubernetes and AWS SageMaker.
    Required Skills: Python, PyTorch, SQL, Kubernetes, AWS, Docker.
    """

    mock_resume = """
    John Doe - Data Scientist
    Skills: Python, PyTorch, SQL, AWS, Docker, Machine Learning.

    Experience:
    Data Scientist | Tech Corp | 2022 - Present
    - Wrote Python scripts to clean data and train PyTorch models.
    - Used SQL to pull data for business reports.
    - Put models in AWS Docker containers for the web team.
    - Attended sprint planning meetings with the manager.
    - Designed 3D game assets for a side project.
    """

    mock_taxonomy = {
        "technical_skills": {
            "tools": {
                "languages": ["python", "sql", "java"],
                "infrastructure": ["aws", "docker", "kubernetes", "gcp"],
                "frameworks": ["pytorch", "tensorflow", "scikit-learn"],
            }
        },
        "business_skills": {"soft_skills": ["mentoring", "leadership", "project planning"]},
    }

    async def run_stress_test():
        print("🚀 Starting Edge-Case Stress Test...\n")
        engine = SpecializedJudgeEngine(confidence_threshold=75, max_retries=3)
        pipeline_result = await engine.run_all_judges(
            resume_text=mock_resume, job_description=mock_jd,
            llm_model="deepseek-r1:14b", skills_taxonomy=mock_taxonomy,
        )

        print("\n" + "=" * 60)
        print("📋 PIPELINE RESULTS SUMMARY")
        print("=" * 60)
        
        overall = pipeline_result["overall"]
        print(f"\n🏆 Overall Match: {overall['overall_score']:.1%}")
        for k, v in overall['component_scores'].items():
            print(f"   {k.capitalize():<12}: {v:.1%}")

        s = pipeline_result["skills"]["scores"]
        print(f"\n🔧 Skills: {s['present']} present, {s['missing']} missing | Missing: {s['missing_skills']}")

        e = pipeline_result["experience"]["scores"]
        print(f"📋 Experience: {e['met']} met, {e['unmet']} unmet | Unmet: {e['unmet_responsibilities']}")

        sr = pipeline_result["seniority"]["scores"]
        print(f"🏅 Seniority: {sr['demonstrated']} demonstrated, {sr['absent']} absent | Absent: {sr['absent_signals']}")

        rel = pipeline_result["relevancy"]["scores"]
        print(f"🔍 Relevancy: {rel['irrelevant']} irrelevant statements spotted.")

        imp = pipeline_result["impact"]["scores"]
        print(f"📈 Impact: {imp['unquantified']} unquantified statements spotted.")

        coach = pipeline_result["coach"]
        if coach.get("success"):
            improvements = coach["data"].get("improvement_analysis", [])
            print(f"\n🎓 Coach: {len(improvements)} improvements suggested.")
            for item in improvements[:3]:
                print(f"   [{item.get('priority', 'medium').upper()}] {item.get('gap_addressed')}")
                print(f"   → {item.get('suggested_improvement', '')[:120]}...")
        else:
            print(f"\n🎓 Coach judge failed: {coach.get('error')}")

    asyncio.run(run_stress_test())