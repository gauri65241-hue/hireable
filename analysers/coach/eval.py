"""
Eval judge for the Coach Analyser.
Checks for fabricated content, hallucinated metrics, and honesty of gap warnings.
"""

from pydantic import BaseModel, Field
from analysers.base import call_llm_async, parse_and_validate


class CoachEvalOutput(BaseModel):
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fabrication_detected: bool = Field(default=False)
    invented_metrics: list[str] = Field(default_factory=list)
    papered_over_gaps: list[str] = Field(default_factory=list)
    honest_gap_warnings: int = Field(default=0, ge=0)
    reasoning: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)


_SYSTEM = """You are a meta-evaluator assessing the quality of a Coach Analyser's output.
The coach produces resume improvement suggestions based on gap analysis.

Your job:
1. Check every bullet_rewrite: does original_content actually appear (verbatim or near-verbatim) in the resume? Flag any rewrite where the original does not exist — this is fabrication.
2. Check for invented metrics: does any suggested_improvement contain numbers that do not appear in the original resume? Flag each one.
3. Check gap_warnings: are genuine skill absences correctly flagged as gap_warning with is_actionable=0? Flag any gap that was papered over with a fabricated bullet instead.
4. Compute an overall accuracy score: fraction of suggestions that are grounded in actual resume content.

Output ONLY valid JSON:
{
    "accuracy_score": 0.0,
    "fabrication_detected": false,
    "invented_metrics": ["<suggestion text containing invented numbers>"],
    "papered_over_gaps": ["<gap_addressed values where a missing skill was given a fabricated bullet>"],
    "honest_gap_warnings": 0,
    "reasoning": "<summary>",
    "confidence": 0
}"""

_USER_TEMPLATE = """Audit the execution context below.

<source_candidate_resume>
{resume_text}
</source_candidate_resume>

<target_job_description>
{job_description}
</target_job_description>

<runtime_coach_analyser_output>
{analyser_output}
</runtime_coach_analyser_output>

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
        resume_text=resume_text,
        job_description=job_description,
        analyser_output=json.dumps(analyser_output, indent=2),
    )
    result = await call_llm_async(_SYSTEM, user, llm_model,
                                  temperature=0.0, api_key=api_key, api_base_url=api_base_url)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    try:
        parsed = parse_and_validate(result["data"], CoachEvalOutput)
        return {"success": True, "eval": parsed}
    except ValueError as e:
        return {"success": False, "error": str(e)}
