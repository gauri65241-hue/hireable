"""Tests for eval.py faithfulness checks and scoring utilities."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval import (
    _content_words,
    coach_faithfulness,
    impact_factuality,
    relevancy_grounding,
    compute_metrics,
)


# ── _content_words ────────────────────────────────────────────────────────────

def test_content_words_removes_stopwords():
    words = _content_words("a the and python machine learning")
    assert "python" in words
    assert "machine" in words
    assert "a" not in words
    assert "the" not in words


def test_content_words_case_insensitive():
    words = _content_words("Python DOCKER LangChain")
    assert "python" in words
    assert "docker" in words
    assert "langchain" in words


def test_content_words_min_length():
    words = _content_words("ai ml nlp deep learning")
    # 'ai' and 'ml' are 2 chars — should be excluded
    assert "ai" not in words
    assert "ml" not in words
    assert "deep" in words
    assert "learning" in words


def test_content_words_empty():
    assert _content_words("") == set()


# ── coach_faithfulness ────────────────────────────────────────────────────────

def _make_result(resume_text="", rewrites=None, impact=None, relevancy=None):
    rewrites = rewrites or []
    return {
        "resume_text": resume_text,
        "pipeline_result": {
            "coach": {
                "data": {"improvement_analysis": rewrites}
            },
            "impact": {
                "judge_result": {"data": {"impact_analysis": impact or []}}
            },
            "relevancy": {
                "judge_result": {"data": {"statement_analysis": relevancy or []}}
            },
        }
    }


def test_coach_faithfulness_none_when_no_rewrites():
    result = _make_result(resume_text="python engineer at startup")
    assert coach_faithfulness(result) is None


def test_coach_faithfulness_grounded_rewrite():
    resume = "built python pipeline for processing customer data using pandas"
    rewrite = {
        "improvement_type": "bullet_rewrite",
        "original_content": "built python pipeline processing customer data",
    }
    result = _make_result(resume_text=resume, rewrites=[rewrite])
    cf = coach_faithfulness(result)
    assert cf is not None
    assert cf["score"] == 1.0
    assert cf["n_grounded"] == 1


def test_coach_faithfulness_ungrounded_rewrite():
    resume = "excel dashboards for finance team"
    rewrite = {
        "improvement_type": "bullet_rewrite",
        "original_content": "designed distributed kubernetes microservices architecture",
    }
    result = _make_result(resume_text=resume, rewrites=[rewrite])
    cf = coach_faithfulness(result)
    assert cf is not None
    assert cf["score"] == 0.0
    assert cf["n_ungrounded"] == 1


# ── impact_factuality ─────────────────────────────────────────────────────────

def test_impact_factuality_none_when_no_quantified():
    result = _make_result(impact=[{"is_quantified": 0, "extracted_metrics": []}])
    assert impact_factuality(result) is None


def test_impact_factuality_supported_metric():
    resume = "reduced latency by 40% across the ml serving pipeline"
    impact = [{
        "is_quantified": 1,
        "extracted_metrics": ["40%"],
        "resume_statement": "reduced latency by 40%",
    }]
    result = _make_result(resume_text=resume, impact=impact)
    imf = impact_factuality(result)
    assert imf is not None
    assert imf["score"] == 1.0


def test_impact_factuality_hallucinated_metric():
    resume = "improved model performance significantly"
    impact = [{
        "is_quantified": 1,
        "extracted_metrics": ["73%"],
        "resume_statement": "improved model performance by 73%",
    }]
    result = _make_result(resume_text=resume, impact=impact)
    imf = impact_factuality(result)
    assert imf is not None
    assert imf["score"] == 0.0
    assert imf["n_unsupported"] == 1


# ── relevancy_grounding ───────────────────────────────────────────────────────

def test_relevancy_grounding_none_when_no_relevant():
    result = _make_result(relevancy=[{"is_relevant": 0, "mapped_jd_requirement": "python"}])
    assert relevancy_grounding(result) is None


def test_relevancy_grounding_returns_score():
    relevancy = [{
        "is_relevant": 1,
        "resume_statement": "built machine learning pipelines using python and sklearn",
        "mapped_jd_requirement": "python machine learning sklearn experience required",
    }]
    result = _make_result(relevancy=relevancy)
    rg = relevancy_grounding(result)
    assert rg is not None
    assert 0.0 <= rg["score"] <= 1.0
    assert "method" in rg


def test_relevancy_grounding_low_similarity_unrelated():
    relevancy = [{
        "is_relevant": 1,
        "resume_statement": "managed excel spreadsheets for quarterly reports",
        "mapped_jd_requirement": "kubernetes docker microservices deployment",
    }]
    result = _make_result(relevancy=relevancy)
    rg = relevancy_grounding(result)
    assert rg is not None
    # unrelated pair should score low regardless of method
    assert rg["score"] < 0.5


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_compute_metrics_empty():
    assert compute_metrics([]) == {}


def test_compute_metrics_basic():
    matched = [
        {"score": 0.8, "fit_label": 1},
        {"score": 0.3, "fit_label": 0},
    ]
    m = compute_metrics(matched, threshold=0.60)
    assert m["n"] == 2
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["accuracy"] == 1.0


def test_compute_metrics_f1_zero_when_all_wrong():
    matched = [
        {"score": 0.8, "fit_label": 0},
        {"score": 0.3, "fit_label": 1},
    ]
    m = compute_metrics(matched, threshold=0.60)
    assert m["f1"] == 0.0
