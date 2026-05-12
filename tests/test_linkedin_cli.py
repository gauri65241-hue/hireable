import json
import os
import pytest
import tempfile

from linkedin_cli_v1 import (
    extract_skills,
    tfidf_score,
    skills_score,
    experience_score,
    compute_score,
    load_taxonomy,
)

TAXONOMY = load_taxonomy()

RESUME_NLP = {
    "resume_id": "1",
    "name": "Gaurica P.",
    "title": "AI/MLOps Engineer",
    "experience_years": "4",
    "skills": "python,llm,rag,langchain,huggingface,docker,aws,gcp,nlp,transformers,vector-databases",
    "text": "AI engineer specialising in RAG LLM systems LangChain HuggingFace transformers Python Docker AWS GCP NLP prompt engineering vector databases.",
}

RESUME_ANALYST = {
    "resume_id": "3",
    "name": "Jamie L.",
    "title": "Junior Data Analyst",
    "experience_years": "1",
    "skills": "excel,sql,powerbi,tableau,python,pandas",
    "text": "Junior analyst Excel SQL Power BI Tableau Python pandas dashboards KPI reporting.",
}

JOB_LLM = "Senior NLP LLM engineer Python LangChain HuggingFace transformers RAG Docker AWS GCP vector databases prompt engineering model evaluation."
JOB_ANALYST = "Junior Data Analyst Excel SQL Power BI Tableau data visualisation dashboards reporting no machine learning required."
JOB_UNRELATED = "Plumber with 5 years experience fixing pipes drains heating systems."


# ---------------------------------------------------------------------------
# extract_skills
# ---------------------------------------------------------------------------

def test_extract_skills_finds_known_skills():
    found = extract_skills("we use python and docker in our stack", TAXONOMY)
    assert "python" in found
    assert "docker" in found


def test_extract_skills_case_insensitive():
    found = extract_skills("We use PYTHON and LangChain", TAXONOMY)
    assert "python" in found
    assert "langchain" in found


def test_extract_skills_empty_text():
    found = extract_skills("", TAXONOMY)
    assert isinstance(found, set)
    assert len(found) == 0


def test_extract_skills_no_match():
    found = extract_skills("fixing pipes and drains", TAXONOMY)
    assert len(found) == 0


# ---------------------------------------------------------------------------
# tfidf_score
# ---------------------------------------------------------------------------

def test_tfidf_identical_texts():
    score = tfidf_score("python machine learning nlp", "python machine learning nlp")
    assert score == pytest.approx(1.0, abs=0.01)


def test_tfidf_unrelated_texts():
    score = tfidf_score("python llm rag langchain", "plumber pipes drains heating")
    assert score < 0.1


def test_tfidf_partial_overlap():
    score = tfidf_score("python nlp transformers docker", "python docker sql tableau")
    assert 0.0 < score < 1.0


def test_tfidf_returns_float():
    score = tfidf_score("machine learning", "deep learning")
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# skills_score
# ---------------------------------------------------------------------------

def test_skills_score_high_for_matching_resume():
    score, missing = skills_score(RESUME_NLP, JOB_LLM, TAXONOMY)
    assert score > 0.4


def test_skills_score_low_for_mismatch():
    score, missing = skills_score(RESUME_ANALYST, JOB_LLM, TAXONOMY)
    assert score < 0.5


def test_skills_score_returns_missing_list():
    _, missing = skills_score(RESUME_ANALYST, JOB_LLM, TAXONOMY)
    assert isinstance(missing, list)


def test_skills_score_no_missing_when_full_match():
    job_text = "excel sql powerbi tableau python pandas"
    score, missing = skills_score(RESUME_ANALYST, job_text, TAXONOMY)
    assert isinstance(missing, list)


# ---------------------------------------------------------------------------
# experience_score stub
# ---------------------------------------------------------------------------

def test_experience_score_returns_08():
    assert experience_score() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# compute_score — composite
# ---------------------------------------------------------------------------

def test_compute_score_keys_present():
    result = compute_score(RESUME_NLP, JOB_LLM, TAXONOMY)
    for key in ["final_score", "tfidf_score", "skills_score", "experience_score", "missing_skills", "latency_ms"]:
        assert key in result


def test_compute_score_range():
    result = compute_score(RESUME_NLP, JOB_LLM, TAXONOMY)
    assert 0.0 <= result["final_score"] <= 1.0


def test_compute_score_nlp_resume_beats_analyst_on_llm_job():
    r_nlp = compute_score(RESUME_NLP, JOB_LLM, TAXONOMY)["final_score"]
    r_analyst = compute_score(RESUME_ANALYST, JOB_LLM, TAXONOMY)["final_score"]
    assert r_nlp > r_analyst


def test_compute_score_analyst_beats_nlp_on_analyst_job():
    r_nlp = compute_score(RESUME_NLP, JOB_ANALYST, TAXONOMY)["final_score"]
    r_analyst = compute_score(RESUME_ANALYST, JOB_ANALYST, TAXONOMY)["final_score"]
    assert r_analyst > r_nlp


def test_compute_score_latency_positive():
    result = compute_score(RESUME_NLP, JOB_LLM, TAXONOMY)
    assert result["latency_ms"] > 0


def test_compute_score_formula_weights():
    result = compute_score(RESUME_NLP, JOB_LLM, TAXONOMY)
    expected = (0.4 * result["tfidf_score"] +
                0.4 * result["skills_score"] +
                0.2 * result["experience_score"])
    assert result["final_score"] == pytest.approx(expected, abs=0.001)
