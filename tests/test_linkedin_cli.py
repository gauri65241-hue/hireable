"""Tests for linkedin_cli_v1 data loading utilities."""

import json

from linkedin_cli_v1 import load_resumes, load_jobs, load_skills_taxonomy, validate_data


# ── load_resumes ──────────────────────────────────────────────────────────────

def test_load_resumes_returns_list(tmp_path):
    csv = tmp_path / "resumes.csv"
    csv.write_text("resume_id,name,title,experience_years,skills,text\n1,Alice,Engineer,3,python,python developer\n")
    result = load_resumes(str(csv))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["resume_id"] == "1"
    assert result[0]["name"] == "Alice"


def test_load_resumes_missing_file():
    result = load_resumes("/nonexistent/path.csv")
    assert result == []


def test_load_resumes_empty_csv(tmp_path):
    csv = tmp_path / "resumes.csv"
    csv.write_text("resume_id,name,title,experience_years,skills,text\n")
    result = load_resumes(str(csv))
    assert result == []


# ── load_jobs ─────────────────────────────────────────────────────────────────

def test_load_jobs_returns_list(tmp_path):
    csv = tmp_path / "jobs.csv"
    csv.write_text("job_id,title,company,experience_required,skills_required,description\nj1,ML Engineer,Acme,3,python,build ml systems\n")
    result = load_jobs(str(csv))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["job_id"] == "j1"
    assert result[0]["title"] == "ML Engineer"


def test_load_jobs_missing_file():
    result = load_jobs("/nonexistent/path.csv")
    assert result == []


# ── load_skills_taxonomy ──────────────────────────────────────────────────────

def test_load_skills_taxonomy_returns_dict(tmp_path):
    tax = tmp_path / "taxonomy.json"
    tax.write_text(json.dumps({"languages": ["python", "java"]}))
    result = load_skills_taxonomy(str(tax))
    assert isinstance(result, dict)
    assert "languages" in result


def test_load_skills_taxonomy_missing_file():
    result = load_skills_taxonomy("/nonexistent/taxonomy.json")
    assert result == {}


# ── validate_data ─────────────────────────────────────────────────────────────

def test_validate_data_passes_with_valid_data():
    resumes = [{"resume_id": "1", "name": "Alice", "text": "python engineer"}]
    jobs = [{"job_id": "j1", "title": "ML Engineer", "description": "build ml systems"}]
    assert validate_data(resumes, jobs) is True


def test_validate_data_fails_missing_resume_field():
    resumes = [{"resume_id": "1", "name": "Alice"}]  # missing text
    jobs = [{"job_id": "j1", "title": "ML Engineer", "description": "build ml systems"}]
    assert validate_data(resumes, jobs) is False


def test_validate_data_fails_missing_job_field():
    resumes = [{"resume_id": "1", "name": "Alice", "text": "python engineer"}]
    jobs = [{"job_id": "j1", "title": "ML Engineer"}]  # missing description
    assert validate_data(resumes, jobs) is False


def test_validate_data_empty_inputs():
    assert validate_data([], []) is True
