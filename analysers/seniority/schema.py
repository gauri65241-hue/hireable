from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

_SIGNAL_TYPE_MAP = {
    "scale": "scale", "volume": "scale", "scope": "scale",
    "ownership": "ownership", "accountability": "ownership",
    "leadership": "leadership", "management": "leadership", "mentoring": "leadership",
    "complexity": "complexity", "complex": "complexity", "technical": "complexity",
    "technical_depth": "complexity",
}


class _SeniorityItem(BaseModel):
    signal: str = ""
    signal_type: Literal["scale", "ownership", "leadership", "complexity"] = "complexity"
    is_demonstrated: int = Field(default=0)
    reasoning: str = ""
    resume_evidence: str = ""

    @field_validator("is_demonstrated", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, bool): return int(v)
        if isinstance(v, str): return 1 if v.lower() in ("1", "true", "yes") else 0
        try: return int(v)
        except (TypeError, ValueError): return 0

    @field_validator("signal_type", mode="before")
    @classmethod
    def normalise_signal_type(cls, v):
        if not isinstance(v, str): return "complexity"
        return _SIGNAL_TYPE_MAP.get(v.strip().lower().replace("-", "_"), "complexity")


class SeniorityOutput(BaseModel):
    seniority_analysis: list[_SeniorityItem] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    overall_reasoning: str = ""
    confidence: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def require_non_empty_analysis(self):
        # An output with nothing to score must FAIL validation. Previously every field
        # defaulted, so `{}` or `{"foo": 1}` validated as a "successful" empty result,
        # scored 0.0 and counted at full weight.
        if not self.seniority_analysis:
            raise ValueError("seniority_analysis must contain at least one item")
        return self
