from analysers.base import run_with_retry, load_calibration
from analysers.experience.schema import ExperienceOutput
from analysers.experience import prompts


def score(analyser_output: dict) -> dict:
    analysis = analyser_output.get("responsibility_analysis", [])
    if not analysis:
        return {"score": 0.0, "met": 0, "partial": 0, "unmet": 0, "total": 0,
                "unmet_responsibilities": []}
    total = len(analysis)
    met     = sum(1 for r in analysis if r.get("experience_score", 0) >= 0.75)
    partial = sum(1 for r in analysis if 0.25 <= r.get("experience_score", 0) < 0.75)
    unmet   = sum(1 for r in analysis if r.get("experience_score", 0) < 0.25)
    unmet_list = [r.get("responsibility", "") for r in analysis
                  if r.get("experience_score", 0) < 0.25]
    raw_score = (met + 0.5 * partial) / total if total else 0.0
    return {
        "score": round(raw_score, 3),
        "met": met, "partial": partial, "unmet": unmet,
        "total": total, "unmet_responsibilities": unmet_list,
    }


class ExperienceAnalyser:
    name = "experience"

    def __init__(self, confidence_threshold: int = 75, max_retries: int = 3):
        self._ct = confidence_threshold
        self._mr = max_retries

    async def run(self, resume_text: str, job_description: str,
                  llm_model: str = "deepseek-r1:14b",
                  api_key: str = "", api_base_url: str = "",
                  arbitration_context: str = "") -> dict:
        calibration = load_calibration(self.name)

        def _build(resume, jd, *, attempt, prev_confidence, confidence_threshold, **kwargs):
            return prompts.build_prompt(resume, jd,
                                        attempt=attempt, prev_confidence=prev_confidence,
                                        confidence_threshold=confidence_threshold,
                                        calibration_note=calibration, **kwargs)

        return await run_with_retry(
            _build, (resume_text, job_description),
            ExperienceOutput, llm_model, self._mr, self._ct,
            api_key=api_key, api_base_url=api_base_url, judge_name="experience",
            resume_text=resume_text, extra_system_context=arbitration_context,
        )
