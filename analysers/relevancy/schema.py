from pydantic import BaseModel, Field, field_validator


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
