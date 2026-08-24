"""
Eval judge for the Relevancy Analyser.
"""

from pydantic import BaseModel, Field
from analysers.base import call_llm_async, parse_and_validate


class RelevancyEvalOutput(BaseModel):
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    wrongly_irrelevant: list[str] = Field(default_factory=list)
    wrongly_relevant: list[str] = Field(default_factory=list)
    missed_statements: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)


_SYSTEM = """You are a meta-evaluator assessing the quality of a Relevancy Analyser's output.

Your job:
1. For statements marked IRRELEVANT — could any of them reasonably map to the JD?
2. For statements marked RELEVANT — do they genuinely connect to the JD's requirements?
3. Did the analyser miss any resume bullets entirely (not in statement_analysis)?
4. Is the overall noise-vs-signal judgement accurate?

Output ONLY valid JSON matching the schema. Start your response with `{` and end with `}`. No preamble, no markdown fences."""

_USER_TEMPLATE = """Audit the relevancy evaluation below.

<target_job_description>
{job_description}
</target_job_description>

<source_candidate_resume>
{resume_text}
</source_candidate_resume>

<runtime_relevancy_analyser_output>
{analyser_output}
</runtime_relevancy_analyser_output>

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
        parsed = parse_and_validate(result["data"], RelevancyEvalOutput)
        return {"success": True, "eval": parsed}
    except ValueError as e:
        return {"success": False, "error": str(e)}
