"""
Backward-compatibility shim.
All logic has moved to the analysers/ package.
Import from analysers directly in new code.
"""

from analysers import AnalyserEngine, calculate_overall, DEFAULT_WEIGHTS, sanitize_user_content
from analysers.base import call_llm_async, parse_and_validate, _build_retry_note
from analysers.skills.analyser     import score as calculate_skills_score
from analysers.experience.analyser import score as calculate_experience_score
from analysers.seniority.analyser  import score as calculate_seniority_score
from analysers.relevancy.analyser  import score as calculate_relevancy_score
from analysers.impact.analyser     import score as calculate_impact_score

# Old names
SpecializedJudgeEngine = AnalyserEngine
calculate_overall_match = calculate_overall
JUDGE_DEFAULT_WEIGHTS = DEFAULT_WEIGHTS
call_ollama_async = call_llm_async

__all__ = [
    "AnalyserEngine",
    "SpecializedJudgeEngine",
    "calculate_overall_match",
    "calculate_overall",
    "DEFAULT_WEIGHTS",
    "JUDGE_DEFAULT_WEIGHTS",
    "sanitize_user_content",
    "calculate_skills_score",
    "calculate_experience_score",
    "calculate_seniority_score",
    "calculate_relevancy_score",
    "calculate_impact_score",
    "call_llm_async",
    "parse_and_validate",
]
