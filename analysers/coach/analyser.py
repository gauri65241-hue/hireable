import logging

from analysers.base import call_llm_async, parse_and_validate
from analysers.coach.schema import CoachOutput
from analysers.coach import prompts

logger = logging.getLogger("AnalyserEngine")


class CoachAnalyser:
    name = "coach"

    def __init__(self, max_retries: int = 3):
        self._mr = max_retries

    async def run(
        self,
        resume_text: str, job_description: str,
        missing_skills: list, unmet_responsibilities: list, absent_seniority_signals: list,
        irrelevant_statements: list, unquantified_statements: list,
        llm_model: str = "deepseek-r1:14b", api_key: str = "", api_base_url: str = "",
    ) -> dict:
        retry_note = ""
        for attempt in range(self._mr):
            logger.info("coach attempt %d/%d", attempt + 1, self._mr)
            system_prompt, user_prompt = prompts.build_prompt(
                resume_text, job_description,
                missing_skills, unmet_responsibilities, absent_seniority_signals,
                irrelevant_statements, unquantified_statements,
                retry_note=retry_note,
            )
            try:
                result = await call_llm_async(system_prompt, user_prompt, llm_model,
                                              temperature=0.1, api_key=api_key, api_base_url=api_base_url)
                if not result["success"]:
                    if attempt == self._mr - 1:
                        return {"success": False, "error": result["error"], "raw_response": "", "attempts": attempt + 1}
                    continue
                try:
                    parsed = parse_and_validate(result["data"], CoachOutput)
                    return {"success": True, "data": parsed, "raw_response": result["data"], "attempts": attempt + 1}
                except ValueError as e:
                    logger.error("coach parse failed on attempt %d: %s", attempt + 1, e)
                    retry_note = f"\n[PREVIOUS ATTEMPT FAILED VALIDATION: {e}. Fix your property types and JSON structure before responding.]"
                    if attempt == self._mr - 1:
                        return {"success": False, "error": str(e), "raw_response": result["data"], "attempts": attempt + 1}
            except Exception as e:
                if attempt == self._mr - 1:
                    return {"success": False, "error": str(e), "raw_response": "", "attempts": attempt + 1}
        return {"success": False, "error": "Exhausted retries", "raw_response": "", "attempts": self._mr}
