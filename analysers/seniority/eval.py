"""
Eval judge for the Seniority Analyser.
"""

from pydantic import BaseModel, Field
from analysers.base import call_llm_async, parse_and_validate


class SeniorityEvalOutput(BaseModel):
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fabrication_detected: bool = Field(default=False)
    false_red_flags: list[str] = Field(default_factory=list)
    missed_red_flags: list[str] = Field(default_factory=list)
    wrongly_demonstrated: list[str] = Field(default_factory=list)
    wrongly_absent: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)


_SYSTEM = """You are a meta-evaluator assessing the quality of a Seniority Analyser's output.

Your job:
1. For each signal marked DEMONSTRATED — does the resume actually prove it at senior scale?
2. For each signal marked ABSENT — is it genuinely missing, or was evidence overlooked?
3. Check red_flags — are they genuine keyword-stuffing patterns, or false accusations?
4. Detect fabricated evidence: did the analyser cite text not in the resume?

Output ONLY valid JSON matching the schema. Start your response with `{` and end with `}`. No preamble, no markdown fences."""

_USER_TEMPLATE = """Audit the seniority evaluation below.

<target_job_description>
{job_description}
</target_job_description>

<source_candidate_resume>
{resume_text}
</source_candidate_resume>

<runtime_seniority_analyser_output>
{analyser_output}
</runtime_seniority_analyser_output>

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
        parsed = parse_and_validate(result["data"], SeniorityEvalOutput)
        return {"success": True, "eval": parsed}
    except ValueError as e:
        return {"success": False, "error": str(e)}
