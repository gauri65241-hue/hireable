"""Regression tests for scoring / schema / weight defects found in the audit."""
import json
import threading

import pytest

from analysers import weights as W
from analysers.base import (
    _build_retry_note, _check_citation, _warnings_to_issues, parse_and_validate, post_validate,
)
from analysers.config import cfg
from analysers.engine import DEFAULT_WEIGHTS, calculate_overall
from analysers.impact.schema import ImpactOutput
from analysers.relevancy.schema import RelevancyOutput
from analysers.seniority.schema import SeniorityOutput
from analysers.skills.analyser import _skill_weight, score as skills_score
from analysers.skills.schema import SkillsOutput


# ── schemas: nothing-to-score must be a parse FAILURE ─────────────────────────
@pytest.mark.parametrize("schema", [SkillsOutput, SeniorityOutput, RelevancyOutput, ImpactOutput])
@pytest.mark.parametrize("payload", ["{}", '{"foo": 1}', '{"confidence": 90}'])
def test_empty_or_garbage_json_is_rejected(schema, payload):
    with pytest.raises(ValueError):
        parse_and_validate(payload, schema)


def test_truncated_skills_output_still_accepted():
    out = parse_and_validate(json.dumps({"required_skills": ["a", "b"], "present_skills": ["a"],
                                         "confidence": 80}), SkillsOutput)
    assert out["required_skills"] == ["a", "b"]


# ── skills scoring ────────────────────────────────────────────────────────────
def test_synthesized_present_skills_earn_only_partial_credit():
    data = {"required_skills": ["a", "b", "c", "d"], "present_skills": ["a", "b", "c", "d"],
            "skill_analysis": []}
    s = skills_score(data)
    assert s["scorable"] is True
    assert s["score"] <= 0.5            # was 1.0: four unverified claims counted as full matches


def test_unscorable_is_flagged_not_silently_zero():
    s = skills_score({"skill_analysis": [], "required_skills": []})
    assert s["scorable"] is False and s["score"] == 0.0


def test_frequency_weight_uses_job_count_not_sum_of_counts():
    freqs = {f"skill{i}": 50 for i in range(20)}          # 50 jobs, each needing all 20 skills
    assert _skill_weight("skill0", freqs, 50) == 2.0       # documented range 1.0-2.0
    assert _skill_weight("unseen", freqs, 50) == 1.0
    assert _skill_weight("skill0", freqs, 5) == 2.0        # clamped, never above 2.0


def test_total_jobs_fallback_is_most_demanded_skill():
    data = {"required_skills": ["hot", "cold"], "present_skills": [],
            "skill_analysis": [
                {"skill": "hot", "is_present": 1, "evidence_strength": "strong"},
                {"skill": "cold", "is_present": 0, "evidence_strength": "weak"}]}
    s = skills_score(data, skill_frequencies={"hot": 10, "cold": 1})
    assert s["frequency_weighted"] is True and s["score"] > 0.5      # 'hot' weighs 2.0, 'cold' 1.1


# ── post_validate / citations ─────────────────────────────────────────────────
RESUME = "Built REST APIs in JavaScript and Python. Led a team of 8 engineers on a Kubernetes migration."


def test_absent_skill_with_empty_evidence_is_not_flagged():
    data = {"score": 0.5, "confidence": 80,
            "skill_analysis": [{"skill": "Kafka", "is_present": 0, "resume_evidence": ""}]}
    assert "_warnings" not in post_validate(data, "skills", "")


def test_present_skill_without_evidence_is_flagged():
    data = {"score": 0.5, "confidence": 80,
            "skill_analysis": [{"skill": "Kafka", "is_present": 1, "resume_evidence": ""}]}
    assert any(w.startswith("claimed_present_no_evidence") for w in post_validate(data, "skills", "")["_warnings"])


def test_citation_matching_rules():
    assert _check_citation("Led a team of 8 engineers", RESUME)
    assert _check_citation("led a team of 8 engineers on a Kubernetes migration", RESUME)
    assert not _check_citation("java", RESUME)                       # not a substring of "JavaScript"
    assert not _check_citation("Not found", RESUME)
    assert not _check_citation("", RESUME)
    # order-blind word stuffing used to pass the 65% bag-of-words test
    assert not _check_citation("engineers team Kubernetes Python APIs led built", RESUME)
    # a lightly trimmed real quote still passes
    assert _check_citation("Built REST APIs in JavaScript and Python", RESUME)


def test_ungrounded_present_claim_is_downgraded():
    data = {"score": 0.9, "confidence": 80, "skill_analysis": [
        {"skill": "Rust", "is_present": 1, "resume_evidence": "Expert Rust systems programmer"}]}
    out = post_validate(data, "skills", RESUME)
    assert out["skill_analysis"][0]["is_present"] == 0.5 and out["skill_analysis"][0]["_citation_invalid"]


# ── retry guidance must not push scores in one direction or leak the threshold ─
def test_retry_note_does_not_disclose_threshold_or_confidence():
    note = _build_retry_note(1, prev_confidence=70, threshold=75)
    assert "75" not in note and "70" not in note and "confidence" not in note.lower()


def test_retry_guidance_is_two_sided():
    text = " ".join(_warnings_to_issues(["high_conf_low_score", "low_conf_high_score",
                                         "reasoning_positive_score_low"])).lower()
    assert "implicit" not in text and "generous" not in text
    assert "keep" in text and "quote" in text


# ── gate behaviour ────────────────────────────────────────────────────────────
def _overall(sk, others=1.0):
    mk = lambda v: {"score": v}
    return calculate_overall({"score": sk, "missing_skills": []}, mk(others), mk(others),
                             mk(others), mk(others), DEFAULT_WEIGHTS)["overall_score"]


def test_hard_gate_cliff_is_documented_behaviour_by_default():
    assert _overall(0.39) < 0.2 and _overall(0.41) > 0.7


def test_optional_ramp_removes_the_cliff():
    object.__setattr__(cfg, "skills_gate_ramp_upper", 0.7)
    try:
        lo, hi = _overall(0.39), _overall(0.41)
        assert abs(hi - lo) < 0.05                       # continuous across the threshold
        assert _overall(0.69) < _overall(0.71) + 1e-9
        assert _overall(0.75) == pytest.approx(0.35 * 0.75 + 0.65, abs=1e-3)   # ungated above ramp
    finally:
        object.__setattr__(cfg, "skills_gate_ramp_upper", 0.0)


# ── learned weights ───────────────────────────────────────────────────────────
def _feedback(path, judge, n_correct, n_wrong):
    with open(path, "w") as f:
        for i in range(n_correct + n_wrong):
            f.write(json.dumps({"judge": judge, "correct": i < n_correct, "outcome": True,
                                "predicted_score": 0.8, "timestamp": "2026-11-01T00:00:00+00:00"}) + "\n")


def test_three_lucky_outcomes_no_longer_hijack_the_weights(tmp_path):
    p = tmp_path / "fb.jsonl"
    _feedback(p, "skills", 5, 0)                            # perfect accuracy, but only n=5
    w = W.compute_learned_weights(str(p))["weights"]
    assert w["skills"] < 0.45                               # old scheme: ~0.64
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_coin_flip_accuracy_leaves_prior_unchanged(tmp_path):
    p = tmp_path / "fb.jsonl"
    _feedback(p, "skills", 10, 10)
    w = W.compute_learned_weights(str(p))["weights"]
    for k, v in DEFAULT_WEIGHTS.items():
        assert w[k] == pytest.approx(v, abs=0.01)


def test_corrupt_line_does_not_disable_learning(tmp_path):
    p = tmp_path / "fb.jsonl"
    _feedback(p, "skills", 30, 0)
    with open(p, "a") as f:
        f.write('{"judge": "skills", "corr\n')
    assert W.compute_learned_weights(str(p))["source"] == "learned"


def test_feedback_appends_are_not_interleaved(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "FEEDBACK_FILE", str(tmp_path / "fb.jsonl"))
    def worker(i):
        W.append_feedback([{"judge": "skills", "i": i, "pad": "x" * 2000} for _ in range(5)])
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    [t.start() for t in ts]; [t.join() for t in ts]
    lines = open(tmp_path / "fb.jsonl").read().splitlines()
    assert len(lines) == 200 and all(json.loads(l)["judge"] == "skills" for l in lines)
