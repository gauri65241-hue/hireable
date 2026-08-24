from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

_STRENGTH_MAP = {
    "strong": "strong", "high": "strong",
    "medium": "medium", "moderate": "medium", "partial": "medium",
    "weak": "weak", "low": "weak",
}


class _ResponsibilityItem(BaseModel):
    responsibility: str = ""
    experience_score: float = Field(default=0.0)
    reasoning: str = ""
    resume_evidence: str = ""
    evidence_strength: Literal["strong", "medium", "weak"] = "weak"

    @field_validator("experience_score", mode="before")
    @classmethod
    def coerce_score(cls, v):
        try:
            f = float(v)
            if f >= 0.75: return 1.0
            if f >= 0.25: return 0.5
            return 0.0
        except (TypeError, ValueError): return 0.0

    @field_validator("evidence_strength", mode="before")
    @classmethod
    def normalise_strength(cls, v):
        if not isinstance(v, str): return "weak"
        return _STRENGTH_MAP.get(v.strip().lower().replace("-", "_"), "weak")


class ExperienceOutput(BaseModel):
    responsibility_analysis: list[_ResponsibilityItem] = Field(default_factory=list)
    overall_reasoning: str = ""
    confidence: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def require_non_empty_analysis(self):
        if not self.responsibility_analysis:
            raise ValueError("responsibility_analysis must contain at least one item")
        return self
