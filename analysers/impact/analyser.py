from analysers.base import run_with_retry, load_calibration
from analysers.impact.schema import ImpactOutput
from analysers.impact import prompts


def score(analyser_output: dict) -> dict:
    analysis = analyser_output.get("impact_analysis", [])
    if not analysis:
        return {"scorable": False, "score": 0.0, "quantified": 0, "directional": 0, "unquantified": 0, "total": 0, "unquantified_statements": []}
    total = len(analysis)
    quantified = sum(1 for s in analysis if s.get("is_quantified") == 1)
    directional = sum(1 for s in analysis if s.get("is_quantified") == 0.5)
    unquantified = [
        {"statement": s.get("resume_statement"), "suggestion": s.get("missing_metric_suggestion")}
        for s in analysis if s.get("is_quantified") == 0
    ]
    return {
        "scorable": True,
        "score": round((quantified + 0.5 * directional) / total, 3) if total else 0.0,
        "quantified": quantified, "directional": directional,
        "unquantified": total - quantified - directional,
        "total": total, "unquantified_statements": unquantified,
    }


class ImpactAnalyser:
    name = "impact"

    def __init__(self, confidence_threshold: int = 75, max_retries: int = 3):
        self._ct = confidence_threshold
        self._mr = max_retries

    async def run(self, resume_text: str, job_description: str = "",
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
            ImpactOutput, llm_model, self._mr, self._ct,
            api_key=api_key, api_base_url=api_base_url, judge_name="impact",
        )
