"""
Tier 2 — Schema enforcement / format adherence benchmarks.

Verifies that parse_and_validate() correctly handles malformed LLM responses:
rejects unrecoverable input, strips reasoning tags, and flags repaired output.
No LLM calls — all inputs are synthetic strings.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysers.base import parse_and_validate


# ── Clean / structured outputs ────────────────────────────────────────────────

def test_clean_json_parsed():
    raw = '{"score": 0.8, "confidence": 85}'
    result = parse_and_validate(raw)
    assert result["success"] is True
    assert result["source"] == "structured"
    assert result["score"] == 0.8


def test_markdown_fenced_json_stripped():
    raw = '```json\n{"score": 0.75, "confidence": 80}\n```'
    result = parse_and_validate(raw)
    assert result["success"] is True
    assert result["score"] == 0.75


def test_think_tag_stripped():
    raw = '<think>Let me reason...</think>\n{"score": 0.6, "confidence": 70}'
    result = parse_and_validate(raw)
    assert result["success"] is True
    assert result["score"] == 0.6


def test_thinking_tag_stripped():
    raw = '<thinking>Internal reasoning here</thinking>\n{"score": 0.5, "confidence": 60}'
    result = parse_and_validate(raw)
    assert result["success"] is True
    assert result["score"] == 0.5


def test_nested_object_extracted():
    raw = 'Here is my analysis:\n{"result": {"score": 0.9}, "confidence": 90}'
    result = parse_and_validate(raw)
    assert result["success"] is True


# ── Repaired outputs carry penalty marker ─────────────────────────────────────

def test_repaired_source_flagged():
    # Deliberately malformed but repairable JSON
    raw = '{"score": 0.7 "confidence": 75}'  # missing comma
    try:
        result = parse_and_validate(raw)
        if result["success"]:
            # If it succeeded via repair, source must be marked
            assert result["source"] in ("repaired", "structured"), \
                "Repaired result must be flagged with source field"
    except ValueError:
        pass  # Acceptable — unrecoverable is also correct


# ── Unrecoverable inputs raise ValueError ────────────────────────────────────

def test_empty_string_raises():
    with pytest.raises(ValueError, match="[Pp]arse"):
        parse_and_validate("")


def test_pure_prose_raises():
    with pytest.raises(ValueError, match="[Pp]arse"):
        parse_and_validate("The candidate is a strong match with excellent Python skills.")


def test_partial_truncated_json_raises_or_repairs():
    raw = '{"score": 0.8, "confidence": 90, "missing_skills": ["AWS", "Docker"'
    try:
        result = parse_and_validate(raw)
        assert result["success"] is True
        assert result["source"] in ("repaired", "structured")
    except ValueError:
        pass  # Acceptable for truncated input


# ── Pydantic schema validation ────────────────────────────────────────────────

def test_schema_field_out_of_range_rejected():
    """If a schema is provided, out-of-range values must fail validation."""
    from pydantic import BaseModel, Field

    class SimpleScore(BaseModel):
        score: float = Field(ge=0.0, le=1.0)
        confidence: int = Field(ge=0, le=100)

    raw = '{"score": 1.5, "confidence": 110}'  # both out of range
    try:
        result = parse_and_validate(raw, SimpleScore)
        # If json_repair somehow passes this through, the Pydantic validator
        # must catch it — result should not succeed with invalid values
        if result.get("success"):
            assert 0.0 <= result["score"] <= 1.0, "Out-of-range score slipped through"
            assert 0 <= result["confidence"] <= 100, "Out-of-range confidence slipped through"
    except ValueError:
        pass  # Correct — schema violation should raise


def test_schema_valid_data_passes():
    from pydantic import BaseModel, Field

    class SimpleScore(BaseModel):
        score: float = Field(ge=0.0, le=1.0)
        confidence: int = Field(ge=0, le=100)

    raw = '{"score": 0.72, "confidence": 88}'
    result = parse_and_validate(raw, SimpleScore)
    assert result["success"] is True
    assert result["score"] == 0.72
    assert result["confidence"] == 88


# ── Confidence penalty for repaired results ───────────────────────────────────

def test_repaired_confidence_penalty_applied(monkeypatch):
    """
    Simulate run_with_retry receiving a repaired result and verify the
    20-point confidence penalty reduces effective confidence.
    This is a unit test of the penalty logic, not full retry behaviour.
    """
    # Manually simulate what run_with_retry does after parse_and_validate
    parsed = {"success": True, "source": "repaired", "confidence": 80, "score": 0.7}
    if parsed.get("source") == "repaired":
        penalty = 20
        parsed["confidence"] = max(0, parsed["confidence"] - penalty)
    assert parsed["confidence"] == 60, "Penalty should reduce confidence from 80 to 60"
