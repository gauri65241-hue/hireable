from pydantic import BaseModel, Field, field_validator, model_validator


class _ImpactItem(BaseModel):
    resume_statement: str = ""
    is_quantified: float = Field(default=0.0)
    extracted_metrics: list[str] = Field(default_factory=list)
    missing_metric_suggestion: str = ""

    @field_validator("is_quantified", mode="before")
    @classmethod
    def coerce_quantified(cls, v):
        try:
            f = float(v)
            if f >= 0.75: return 1.0
            if f >= 0.25: return 0.5
            return 0.0
        except (TypeError, ValueError): return 0.0


class ImpactOutput(BaseModel):
    impact_analysis: list[_ImpactItem] = Field(default_factory=list)
    overall_reasoning: str = ""
    confidence: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def require_non_empty_analysis(self):
        # An output with nothing to score must FAIL validation. Previously every field
        # defaulted, so `{}` or `{"foo": 1}` validated as a "successful" empty result,
        # scored 0.0 and counted at full weight.
        if not self.impact_analysis:
            raise ValueError("impact_analysis must contain at least one item")
        return self
