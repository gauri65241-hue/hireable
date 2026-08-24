"""
Eval judge for the Experience Analyser.
"""

from pydantic import BaseModel, Field
from analysers.base import call_llm_async, parse_and_validate


class ExperienceEvalOutput(BaseModel):
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fabrication_detected: bool = Field(default=False)
    over_credited: list[str] = Field(default_factory=list)
    under_credited: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)


_SYSTEM = """You are a meta-evaluator assessing the quality of an Experience Analyser's output.
You will be given the analyser's verdict alongside the original resume and job description.

Your job:
1. For each responsibility scored MET (1.0) — does the resume actually show comparable scope/scale?
2. For each scored UNMET (0.0) — is it genuinely absent, or did the analyser miss evidence?
3. Detect fabricated evidence: did the analyser cite text not present in the resume?
4. Identify responsibilities the analyser should have extracted from the JD but missed.

Output ONLY valid JSON matching the schema. Start your response with `{` and end with `}`. No preamble, no markdown fences."""

_USER_TEMPLATE = """Audit the experience evaluation below.

<target_job_description>
{job_description}
</target_job_description>

<source_candidate_resume>
{resume_text}
</source_candidate_resume>

<runtime_experience_analyser_output>
{analyser_output}
</runtime_experience_analyser_output>

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
        parsed = parse_and_validate(result["data"], ExperienceEvalOutput)
        return {"success": True, "eval": parsed}
    except ValueError as e:
        return {"success": False, "error": str(e)}
