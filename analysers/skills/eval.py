"""
Eval judge for the Skills Analyser.
Takes the analyser output + original resume/JD and grades the quality of the analysis.
Run on-demand (sampling / QA) — not on every production request.
"""

from pydantic import BaseModel, Field
from analysers.base import call_llm_async, parse_and_validate


class SkillsEvalOutput(BaseModel):
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fabrication_detected: bool = Field(default=False)
    missed_skills: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)


_SYSTEM = """You are a meta-evaluator assessing the quality of a Skills Analyser's output.
You will be given the analyser's verdict alongside the original resume and job description.

Your job:
1. Check every skill marked PRESENT — does the resume actually contain clear evidence?
2. Check every skill marked MISSING — is it genuinely absent, or did the analyser overlook it?
3. Detect fabricated evidence: did the analyser cite text that does not appear in the resume?
4. Identify missed required skills the analyser failed to extract from the JD.

Output ONLY valid JSON matching the schema. Start your response with `{` and end with `}`. No preamble, no markdown fences."""

_USER_TEMPLATE = """Audit the skills evaluation below.

<target_job_description>
{job_description}
</target_job_description>

<source_candidate_resume>
{resume_text}
</source_candidate_resume>

<runtime_skills_analyser_output>
{analyser_output}
</runtime_skills_analyser_output>

Execute the meta-evaluation and return the JSON object."""


async def run_eval(
    analyser_output: dict,
    resume_text: str,
    job_description: str,
    llm_model: str = "deepseek-r1:14b",
    api_key: str = "",
    api_base_url: str = "",
) -> dict:
    import json
    user = _USER_TEMPLATE.format(
        job_description=job_description,
        resume_text=resume_text,
        analyser_output=json.dumps(analyser_output, indent=2),
    )
    result = await call_llm_async(_SYSTEM, user, llm_model,
                                  temperature=0.0, api_key=api_key, api_base_url=api_base_url)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    try:
        parsed = parse_and_validate(result["data"], SkillsEvalOutput)
        return {"success": True, "eval": parsed}
    except ValueError as e:
        return {"success": False, "error": str(e)}
