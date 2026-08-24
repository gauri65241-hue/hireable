from typing import Literal
from pydantic import BaseModel, Field, field_validator

_STRENGTH_MAP = {
    "strong": "strong", "high": "strong",
    "medium": "medium", "moderate": "medium", "partial": "medium",
    "weak": "weak", "low": "weak",
}


class _SkillItem(BaseModel):
    skill: str = ""
    skill_requirement: str = ""
    is_present: int = Field(default=0)
    reasoning: str = ""
    resume_evidence: str = ""
    evidence_strength: Literal["strong", "medium", "weak"] = "weak"

    @field_validator("is_present", mode="before")
    @classmethod
    def coerce_is_present(cls, v):
        if isinstance(v, bool): return int(v)
        if isinstance(v, str): return 1 if v.lower() in ("1", "true", "yes") else 0
        try: return int(v)
        except (TypeError, ValueError): return 0

    @field_validator("evidence_strength", mode="before")
    @classmethod
    def normalise_strength(cls, v):
        if not isinstance(v, str): return "weak"
        return _STRENGTH_MAP.get(v.strip().lower().replace("-", "_"), "weak")


class SkillsOutput(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    missing_critical_skills: list[str] = Field(default_factory=list)
    present_skills: list[str] = Field(default_factory=list)
    taxonomy_categorization: dict = Field(default_factory=dict)
    skill_analysis: list[_SkillItem] = Field(default_factory=list)
    overall_reasoning: str = ""
    confidence_calculation: str = ""
    confidence: int = Field(default=0, ge=0, le=100)
