from analysers.base import run_with_retry, load_calibration
from analysers.relevancy.schema import RelevancyOutput
from analysers.relevancy import prompts


def score(analyser_output: dict) -> dict:
    analysis = analyser_output.get("statement_analysis", [])
    if not analysis:
        return {"score": 0.0, "relevant": 0, "irrelevant": 0, "total": 0, "irrelevant_statements": []}
    total = len(analysis)
    relevant = sum(1 for s in analysis if s.get("is_relevant") == 1)
    irrelevant = [s.get("resume_statement") for s in analysis if s.get("is_relevant") == 0]
    return {
        "score": round(relevant / total, 3) if total else 0.0,
        "relevant": relevant, "irrelevant": total - relevant,
        "total": total, "irrelevant_statements": irrelevant,
    }


class RelevancyAnalyser:
    name = "relevancy"

    def __init__(self, confidence_threshold: int = 75, max_retries: int = 3):
        self._ct = confidence_threshold
        self._mr = max_retries

    async def run(self, resume_text: str, job_description: str,
                  llm_model: str = "deepseek-r1:14b",
                  api_key: str = "", api_base_url: str = "") -> dict:
        calibration = load_calibration(self.name)

        def _build(resume, jd, *, attempt, prev_confidence, confidence_threshold, **kwargs):
            return prompts.build_prompt(resume, jd,
                                        attempt=attempt, prev_confidence=prev_confidence,
                                        confidence_threshold=confidence_threshold,
                                        calibration_note=calibration)

        return await run_with_retry(
            _build, (resume_text, job_description),
            RelevancyOutput, llm_model, self._mr, self._ct,
            api_key=api_key, api_base_url=api_base_url, judge_name="relevancy",
        )
