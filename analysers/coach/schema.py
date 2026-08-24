from typing import Literal
from pydantic import BaseModel, Field, field_validator

_IMPROVEMENT_TYPE_MAP = {
    "bullet_rewrite": "bullet_rewrite", "bulletrewrite": "bullet_rewrite", "rewrite": "bullet_rewrite",
    "deletion": "deletion", "delete": "deletion", "remove": "deletion",
    "gap_warning": "gap_warning", "gapwarning": "gap_warning", "warning": "gap_warning", "gap": "gap_warning",
}
_PRIORITY_MAP = {
    "high": "high", "high_priority": "high",
    "medium": "medium", "mid": "medium", "moderate": "medium",
    "low": "low", "low_priority": "low",
}


class _CoachItem(BaseModel):
    gap_addressed: str = ""
    improvement_type: Literal["bullet_rewrite", "deletion", "gap_warning"] = "gap_warning"
    is_actionable: int = Field(default=0, ge=0, le=1)
    original_content: str = ""
    suggested_improvement: str = ""
    reasoning: str = ""
    priority: Literal["high", "medium", "low"] = "medium"

    @field_validator("improvement_type", mode="before")
    @classmethod
    def normalise_improvement_type(cls, v):
        if not isinstance(v, str): return "gap_warning"
        return _IMPROVEMENT_TYPE_MAP.get(v.strip().lower().replace("-", "_"), "gap_warning")

    @field_validator("priority", mode="before")
    @classmethod
    def normalise_priority(cls, v):
        if not isinstance(v, str): return "medium"
        return _PRIORITY_MAP.get(v.strip().lower().replace("-", "_"), "medium")


class CoachOutput(BaseModel):
    improvement_analysis: list[_CoachItem] = Field(default_factory=list)
    overall_reasoning: str = ""
