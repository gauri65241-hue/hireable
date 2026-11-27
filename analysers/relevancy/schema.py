from pydantic import BaseModel, Field, field_validator, model_validator


class _StatementItem(BaseModel):
    resume_statement: str = ""
    mapped_jd_requirement: str = ""
    is_relevant: int = Field(default=0)
    reasoning: str = ""

    @field_validator("is_relevant", mode="before")
    @classmethod
    def coerce_relevance(cls, v):
        if isinstance(v, bool): return int(v)
        try: return int(float(v))
        except (TypeError, ValueError): return 0


class RelevancyOutput(BaseModel):
    statement_analysis: list[_StatementItem] = Field(default_factory=list)
    overall_reasoning: str = ""
    confidence: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def require_non_empty_analysis(self):
        # An output with nothing to score must FAIL validation. Previously every field
        # defaulted, so `{}` or `{"foo": 1}` validated as a "successful" empty result,
        # scored 0.0 and counted at full weight.
        if not self.statement_analysis:
            raise ValueError("statement_analysis must contain at least one item")
        return self
