from analysers.base import run_with_retry, load_calibration
from analysers.seniority.schema import SeniorityOutput
from analysers.seniority import prompts


def score(analyser_output: dict) -> dict:
    analysis = analyser_output.get("seniority_analysis", [])
    if not analysis:
        return {"score": 0.0, "demonstrated": 0, "absent": 0, "total": 0, "absent_signals": [], "red_flags": []}
    total = len(analysis)
    demonstrated = sum(1 for s in analysis if s.get("is_demonstrated") == 1)
    absent = [s.get("signal", "unknown") for s in analysis if s.get("is_demonstrated") == 0]
    return {
        "score": round(demonstrated / total, 3) if total else 0.0,
        "demonstrated": demonstrated, "absent": total - demonstrated,
        "total": total, "absent_signals": absent,
        "red_flags": analyser_output.get("red_flags", []),
    }


class SeniorityAnalyser:
    name = "seniority"

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
            SeniorityOutput, llm_model, self._mr, self._ct,
            api_key=api_key, api_base_url=api_base_url, judge_name="seniority",
            resume_text=resume_text, extra_system_context=arbitration_context,
        )
