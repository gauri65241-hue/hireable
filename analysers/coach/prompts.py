import json

from analysers.chunker_compat import parse_jd, parse_resume, format_for_coach_judge

SYSTEM = """You are an elite Technical Career Coach and Resume Writer.
You are receiving the raw, structured output from 5 specialized AI evaluation agents that just analyzed the candidate's resume against the target Job Description.

Your goal is to synthesize these granular gaps into highly specific, copy-paste-ready bullet point rewrites.

CRITICAL RULES:
- Output ONLY valid JSON. No markdown, no explanations, no text outside JSON.
- is_actionable MUST be exactly 1 or 0.
- NEVER FABRICATE EXPERIENCE: You may only rewrite or reframe bullets that already exist in the candidate's resume. Do NOT add new bullets claiming skills or experience the candidate did not demonstrate. If a required skill is genuinely absent, flag it as a gap warning, not a fabricated bullet.
- NO INVENTED METRICS: Only use numbers that already appear in the candidate's existing resume. If no real number exists, rewrite without any metric — do NOT insert placeholders like [X%], [Insert metric], [N], X%, Y%, Z%, or any single letter used as a stand-in for a number, whether bracketed or bare.
- GAP WARNINGS: For skills or responsibilities the candidate provably lacks, set improvement_type to "gap_warning" and is_actionable to 0. Explain the gap honestly.
- REWRITES ONLY: improvement_type "bullet_rewrite" must reference original_content from the resume.

Output format:
{
    "improvement_analysis": [
        {
            "gap_addressed": "<the specific skills/responsibilities/metrics this fixes>",
            "improvement_type": "bullet_rewrite | deletion | gap_warning",
            "is_actionable": 1,
            "original_content": "<exact current resume text being rewritten>",
            "suggested_improvement": "<rewritten bullet using only real experience, or gap warning message>",
            "reasoning": "<why this rewrite highlights transferable value, or why this is a genuine gap>",
            "priority": "high | medium | low"
        }
    ],
    "overall_reasoning": "<honest coaching summary: what genuinely transfers, what gaps exist, expected realistic fit>"
}"""

_USER_TEMPLATE = """Analyze the boundaries below and output your coaching synthesis.

<target_job_description>
{job_description_text}
</target_job_description>

<candidate_supplied_resume>
{resume_text}
</candidate_supplied_resume>

<agent_evaluation_gaps>
1. MISSING SKILLS (Skills Analyser):
{missing_skills}

2. UNMET CORE RESPONSIBILITIES (Experience Analyser):
{unmet_responsibilities}

3. ABSENT SENIORITY SIGNALS (Seniority Analyser):
{absent_seniority_signals}

4. IRRELEVANT STATEMENTS (Relevancy Analyser):
{irrelevant_statements}

5. UNQUANTIFIED STATEMENTS (Impact Analyser):
{unquantified_statements}
</agent_evaluation_gaps>

<synthesis_directives>
1. COMBINE FIXES: If a bullet is unquantified AND lacks a seniority signal, produce ONE rewrite that fixes both — do not duplicate.
2. REWRITES ONLY: Each bullet_rewrite MUST reference real text from the candidate's resume in original_content.
3. HONEST GAPS: Skills in the missing list that do not appear anywhere in the resume → set as gap_warning with is_actionable=0.
4. PRUNING: For irrelevant statements, use improvement_type "deletion". Do not replace deleted content with fabricated bullets.
5. PRIORITIZATION: HIGH = reframe real experience to match core responsibilities. MEDIUM = add real metrics / elevate seniority. LOW = minor wording. gap_warning = always flag.
</synthesis_directives>

Output JSON only. Begin with {{ and end with }}. No preamble, no markdown fences, no trailing text.
{retry_note}"""


def _fmt_gap(data: list | None, dimension: str) -> str:
    """
    Render a gap list for the Coach prompt.
    None means the judge that produced this data failed or timed out — the Coach
    must NOT treat this as 'no gaps'. Render it explicitly so the Coach knows
    to skip recommendations for that dimension rather than invent positive ones.
    """
    if data is None:
        return f"[DATA UNAVAILABLE — {dimension} judge failed or timed out. Do NOT infer gaps or strengths for this dimension.]"
    if not data:
        return "[]"
    return json.dumps(data, separators=(',', ':'))


def build_prompt(
    resume_text: str, job_description: str,
    missing_skills: list | None, unmet_responsibilities: list | None,
    absent_seniority_signals: list | None,
    irrelevant_statements: list | None, unquantified_statements: list | None,
    retry_note: str = "",
) -> tuple[str, str]:
    jd = parse_jd(job_description)
    resume = parse_resume(resume_text)
    jd_block, resume_block = format_for_coach_judge(jd, resume)

    user = _USER_TEMPLATE.format(
        job_description_text=jd_block,
        resume_text=resume_block,
        missing_skills=            _fmt_gap(missing_skills,             "Skills"),
        unmet_responsibilities=    _fmt_gap(unmet_responsibilities,     "Experience"),
        absent_seniority_signals=  _fmt_gap(absent_seniority_signals,   "Seniority"),
        irrelevant_statements=     _fmt_gap(irrelevant_statements,       "Relevancy"),
        unquantified_statements=   _fmt_gap(unquantified_statements,     "Impact"),
        retry_note=retry_note,
    )
    return SYSTEM, user
