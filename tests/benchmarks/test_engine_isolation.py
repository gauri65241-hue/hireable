"""
Tier 1 (unit) — Engine failure isolation and quorum logic.

Verifies _run_judge_timed, _gap_or_none, and the engine's response to partial
judge failure without making LLM calls (uses coroutine mocks).
"""
import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysers.engine import _run_judge_timed, _gap_or_none, JUDGE_TIMEOUT_S, JUDGE_QUORUM


# ── _run_judge_timed ──────────────────────────────────────────────────────────

def test_successful_judge_returns_success():
    async def _ok():
        return {"success": True, "data": {"score_raw": 0.8}, "confidence": 90}

    def _score(data):
        return {"score": data.get("score_raw", 0.0), "missing_skills": []}

    status, raw, scores, err, _ms = asyncio.run(_run_judge_timed("skills", _ok(), _score))
    assert status == "success"
    assert err is None
    assert scores["score"] == 0.8


def test_failed_judge_returns_failed():
    async def _fail():
        return {"success": False, "error": "model crashed", "data": {}}

    def _score(data):
        return {"score": 0.0}

    status, raw, scores, err, _ms = asyncio.run(_run_judge_timed("skills", _fail(), _score))
    assert status == "failed"
    assert "crashed" in err
    assert scores == {}


def test_exception_in_judge_returns_failed():
    async def _crash():
        raise RuntimeError("Unexpected GPU error")

    def _score(data):
        return {"score": 0.0}

    status, raw, scores, err, _ms = asyncio.run(_run_judge_timed("skills", _crash(), _score))
    assert status == "failed"
    assert "GPU error" in err
    assert scores == {}


def test_timeout_judge_returns_timeout():
    async def _slow():
        await asyncio.sleep(200)
        return {"success": True, "data": {}}

    def _score(data):
        return {"score": 0.0}

    import analysers.engine as eng
    original = eng.JUDGE_TIMEOUT_S
    eng.JUDGE_TIMEOUT_S = 0.05
    try:
        status, raw, scores, err, _ms = asyncio.run(_run_judge_timed("skills", _slow(), _score))
        assert status == "timeout"
        assert "timed out" in err
    finally:
        eng.JUDGE_TIMEOUT_S = original


# ── _gap_or_none ──────────────────────────────────────────────────────────────

def test_gap_or_none_returns_list_on_success():
    scores = {"missing_skills": ["Python", "AWS"], "score": 0.4}
    result = _gap_or_none("success", scores, "missing_skills")
    assert result == ["Python", "AWS"]


def test_gap_or_none_returns_none_on_failed():
    result = _gap_or_none("failed", {}, "missing_skills")
    assert result is None


def test_gap_or_none_returns_none_on_timeout():
    result = _gap_or_none("timeout", {}, "missing_skills")
    assert result is None


def test_gap_or_none_returns_empty_list_when_no_gaps():
    scores = {"missing_skills": [], "score": 0.9}
    result = _gap_or_none("success", scores, "missing_skills")
    assert result == []  # empty list != None; "no gaps" != "data unavailable"


# ── Coach prompt None rendering ───────────────────────────────────────────────

def test_coach_prompt_renders_none_as_unavailable():
    from analysers.coach.prompts import _fmt_gap
    rendered = _fmt_gap(None, "Skills")
    assert "UNAVAILABLE" in rendered
    assert "Do NOT infer" in rendered


def test_coach_prompt_renders_empty_list_correctly():
    from analysers.coach.prompts import _fmt_gap
    rendered = _fmt_gap([], "Skills")
    assert rendered == "[]"


def test_coach_prompt_renders_populated_list():
    from analysers.coach.prompts import _fmt_gap
    rendered = _fmt_gap(["Python", "AWS"], "Skills")
    assert "Python" in rendered
    assert "AWS" in rendered


# ── JUDGE_QUORUM constant sanity ──────────────────────────────────────────────

def test_quorum_less_than_total_judges():
    assert JUDGE_QUORUM < 5, "Quorum must be achievable with partial failures"
    assert JUDGE_QUORUM >= 3, "Quorum must require majority of judges"


def test_judge_timeout_well_below_httpx_timeout():
    assert JUDGE_TIMEOUT_S < 620.0, "Judge timeout must be less than httpx connection timeout (620s)"
    assert JUDGE_TIMEOUT_S >= 30.0, "Judge timeout must be enough for real LLM inference"
