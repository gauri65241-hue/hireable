"""Native-async LLM transport, retry loop and fallback — with a fake HTTP layer (no network)."""
import asyncio
import json

import httpx
import pytest

import analysers.base as base
from analysers.config import cfg
from analysers.skills.schema import SkillsOutput

GOOD = json.dumps({"required_skills": ["a"], "present_skills": ["a"], "confidence": 90,
                   "skill_analysis": [{"skill": "a", "is_present": 1, "resume_evidence": "python",
                                       "evidence_strength": "strong"}]})


def _ollama_ok(content=GOOD):
    return httpx.Response(200, json={"message": {"content": content}})


def _prompt_fn(*a, **k):
    return ("sys", "usr")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def post(monkeypatch):
    """Install a scripted fake for base._http_post; returns the list of recorded calls."""
    calls, script = [], {"fn": None}
    async def fake(url, payload, headers, timeout):
        calls.append({"url": url, "payload": payload, "headers": headers})
        return script["fn"](len(calls), url, payload)
    monkeypatch.setattr(base, "_http_post", fake)
    return calls, script


def test_ollama_uses_schema_format_then_degrades_to_json(post):
    calls, script = post
    script["fn"] = lambda n, url, p: httpx.Response(400, text="bad format") if n == 1 else _ollama_ok()
    r = _run(base.call_llm_async("s", "u", "m", json_schema={"type": "object"}))
    assert r == {"success": True, "data": GOOD}
    assert calls[0]["payload"]["format"] == {"type": "object"} and calls[1]["payload"]["format"] == "json"
    assert calls[0]["url"].endswith("/api/chat")


def test_api_429_honours_retry_after_but_caps_the_wait(post, monkeypatch):
    calls, script = post
    waits = []
    async def fake_sleep(s): waits.append(s)
    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    def fn(n, url, p):
        if n <= 2:
            return httpx.Response(429, headers={"Retry-After": "9999"}, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": GOOD}}]})
    script["fn"] = fn
    r = _run(base.call_llm_async("s", "u", "m", api_key="k", api_base_url="https://api.example/v1"))
    assert r["success"] and len(calls) == 3
    assert waits and max(waits) <= cfg.api_max_retry_wait           # was 20/40/80/160 s sleeps in a thread
    assert calls[0]["url"] == "https://api.example/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer k"


def test_api_response_format_degrades_schema_to_json_object_to_plain(post):
    calls, script = post
    def fn(n, url, p):
        if "response_format" in p:
            return httpx.Response(400, text="unsupported")
        return httpx.Response(200, json={"choices": [{"message": {"content": GOOD}}]})
    script["fn"] = fn
    r = _run(base.call_llm_async("s", "u", "m", api_key="k", json_schema={"type": "object"}))
    assert r["success"]
    kinds = [c["payload"].get("response_format", {}).get("type") for c in calls]
    assert kinds == ["json_schema", "json_object", None]


def test_api_gives_up_after_bounded_attempts(post, monkeypatch):
    calls, script = post
    async def no_sleep(s): pass
    monkeypatch.setattr(base.asyncio, "sleep", no_sleep)
    script["fn"] = lambda n, url, p: httpx.Response(503, text="down")
    r = _run(base.call_llm_async("s", "u", "m", api_key="k"))
    assert r["success"] is False and "503" in r["error"] and len(calls) == 4


def test_network_error_becomes_a_failed_result_not_an_exception(monkeypatch):
    async def boom(*a, **k): raise httpx.ConnectError("refused")
    monkeypatch.setattr(base, "_http_post", boom)
    r = _run(base.call_llm_async("s", "u", "m"))
    assert r["success"] is False and "ConnectError" in r["error"]


def test_timeout_cancels_inflight_request_and_semaphore_is_truthful(monkeypatch):
    """The reproduced defect: with to_thread, 6 timed-out calls put 4 requests in flight
    against a limit of 2 and left them running. Now cancellation aborts the request."""
    state = {"inflight": 0, "peak": 0}
    async def slow(url, payload, headers, timeout):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        try:
            await asyncio.sleep(30)
        finally:
            state["inflight"] -= 1
    monkeypatch.setattr(base, "_http_post", slow)
    async def main():
        res = await asyncio.gather(*[
            asyncio.wait_for(base.call_llm_async("s", "u", "m"), timeout=0.1) for _ in range(6)
        ], return_exceptions=True)
        await asyncio.sleep(0.05)
        return res
    res = _run(main())
    assert all(isinstance(r, asyncio.TimeoutError) for r in res)
    assert state["peak"] <= cfg.ollama_semaphore_size
    assert state["inflight"] == 0                      # nothing left running after cancellation


# ── retry loop / fallback ─────────────────────────────────────────────────────
def _retry(llm_model="primary:14b", **kw):
    return _run(base.run_with_retry(_prompt_fn, (), SkillsOutput, llm_model, kw.pop("max_retries", 2), 75, **kw))


def test_fallback_model_is_actually_used(post):
    calls, script = post
    def fn(n, url, p):
        return httpx.Response(500, text="oom") if p["model"] == "primary:14b" else _ollama_ok()
    script["fn"] = fn
    r = _retry()
    assert r["success"] and r["used_fallback"] and r["fallback_model"] == cfg.fallback_model
    models = [c["payload"]["model"] for c in calls]
    assert models.count("primary:14b") >= 2 and models[-1] == cfg.fallback_model


def test_all_models_failing_returns_failure_with_both_errors(post):
    calls, script = post
    script["fn"] = lambda n, url, p: httpx.Response(500, text="oom")
    r = _retry()
    assert r["success"] is False and "fallback_error" in r


def test_garbage_json_on_final_attempt_is_a_failure_not_a_zero_score(post):
    calls, script = post
    script["fn"] = lambda n, url, p: _ollama_ok('{"foo": 1}')
    r = _retry()
    assert r["success"] is False                      # used to be success=True -> score 0.0, full weight


def test_low_confidence_valid_answer_is_accepted_but_flagged(post):
    calls, script = post
    low = json.loads(GOOD); low["confidence"] = 40
    script["fn"] = lambda n, url, p: _ollama_ok(json.dumps(low))
    r = _retry(max_retries=2)
    assert r["success"] and r["confidence_met"] is False and r["attempts"] == 2


def test_retry_prompt_does_not_leak_threshold(post):
    seen = []
    def prompt_fn(*a, attempt=0, prev_confidence=0, confidence_threshold=0, failure_reason="",
                  specific_issues=None, **k):
        seen.append(base._build_retry_note(attempt, prev_confidence, confidence_threshold, failure_reason, specific_issues))
        return ("sys", "usr")
    calls, script = post
    low = json.loads(GOOD); low["confidence"] = 40
    script["fn"] = lambda n, url, p: _ollama_ok(json.dumps(low))
    _run(base.run_with_retry(prompt_fn, (), SkillsOutput, "primary:14b", 2, 75))
    assert seen[0] == "" and "75" not in seen[1] and "40" not in seen[1]


def test_input_rejection_is_not_retried_and_does_not_try_fallback(post):
    calls, script = post
    def bad_prompt(*a, **k): raise ValueError("Injection fragment in skills_formatter[1].")
    r = _run(base.run_with_retry(bad_prompt, (), SkillsOutput, "primary:14b", 3, 75))
    assert r["success"] is False and r["error_type"] == "input_rejected" and calls == []


def test_structured_schema_is_sent_to_the_backend(post):
    calls, script = post
    script["fn"] = lambda n, url, p: _ollama_ok()
    _retry()
    assert isinstance(calls[0]["payload"]["format"], dict)
    assert "skill_analysis" in calls[0]["payload"]["format"]["properties"]
