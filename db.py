"""
SQLite persistence for analysis run history.
Replaces scanning JSON files from results/pipeline/ for list_runs.
"""

import os
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "runs.db")


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id                  TEXT PRIMARY KEY,
                created_at          TEXT NOT NULL,
                model               TEXT,
                candidate_name      TEXT,
                job_title           TEXT,
                resume_text         TEXT,
                job_desc            TEXT,
                overall_score       REAL,
                skills_score        REAL,
                experience_score    REAL,
                seniority_score     REAL,
                relevancy_score     REAL,
                impact_score        REAL,
                ats_score           REAL,
                ats_grade           TEXT,
                pipeline_json       TEXT,
                tailored_text       TEXT,
                ats_validation_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC)")
        # Migrate existing DBs that predate candidate_name/job_title columns
        for col in ("candidate_name", "job_title"):
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
            except Exception:
                pass


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_run(run_id: str, model: str, resume_text: str, job_desc: str,
             pipeline_result: dict, ats_result: dict | None = None,
             candidate_name: str = "", job_title: str = ""):
    overall = pipeline_result.get("overall", {})
    cs = overall.get("component_scores", {})
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO runs
            (id, created_at, model, candidate_name, job_title,
             resume_text, job_desc,
             overall_score, skills_score, experience_score,
             seniority_score, relevancy_score, impact_score,
             ats_score, ats_grade, pipeline_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id,
            datetime.utcnow().isoformat(),
            model,
            candidate_name,
            job_title,
            resume_text[:5000],
            job_desc[:3000],
            overall.get("overall_score"),
            cs.get("skills"),
            cs.get("experience"),
            cs.get("seniority"),
            cs.get("relevancy"),
            cs.get("impact"),
            ats_result.get("score") if ats_result else None,
            ats_result.get("grade") if ats_result else None,
            json.dumps(pipeline_result),
        ))


def update_run_tailored(run_id: str, tailored_text: str, ats_validation: dict | None = None):
    with _conn() as conn:
        conn.execute(
            "UPDATE runs SET tailored_text=?, ats_validation_json=? WHERE id=?",
            (tailored_text, json.dumps(ats_validation) if ats_validation else None, run_id),
        )


def list_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT id, created_at, model,
                   candidate_name, job_title,
                   substr(resume_text, 1, 120) AS resume_snippet,
                   substr(job_desc, 1, 120)    AS job_snippet,
                   overall_score, skills_score, experience_score,
                   seniority_score, relevancy_score, impact_score,
                   ats_score, ats_grade, pipeline_json, ats_validation_json
            FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

    result = []
    for row in rows:
        pr = json.loads(row["pipeline_json"] or "{}")
        skills_data    = pr.get("skills",    {}).get("judge_result", {}).get("data", {})
        exp_data       = pr.get("experience",{}).get("judge_result", {}).get("data", {})
        seniority_data = pr.get("seniority", {}).get("judge_result", {}).get("data", {})
        relevancy_data = pr.get("relevancy", {}).get("judge_result", {}).get("data", {})
        impact_data    = pr.get("impact",    {}).get("judge_result", {}).get("data", {})
        coach          = pr.get("coach", {})
        coach_recs     = []
        if coach.get("success") and coach.get("data"):
            coach_recs = coach["data"].get("improvement_analysis", [])
        ats_val = json.loads(row["ats_validation_json"]) if row["ats_validation_json"] else None

        ats_preflight = pr.get("ats_preflight") or (
            {"score": row["ats_score"], "grade": row["ats_grade"], "issues": [], "criteria": []}
            if row["ats_score"] is not None else None
        )
        result.append({
            "result_id": row["id"],
            "timestamp": row["created_at"],
            "model": row["model"],
            "resume_name": row["candidate_name"] or "",
            "job_title": row["job_title"] or "",
            "resume_snippet": row["resume_snippet"],
            "job_snippet": row["job_snippet"],
            "match_score": round((row["overall_score"] or 0) * 100, 1),
            "ats_score": row["ats_score"],
            "ats_grade": row["ats_grade"],
            "ats_preflight": ats_preflight,
            "metrics": {
                "skills_coverage":       round((row["skills_score"]      or 0) * 100, 1),
                "experience_depth":      round((row["experience_score"]   or 0) * 100, 1),
                "seniority_calibration": round((row["seniority_score"]    or 0) * 100, 1),
                "statement_relevancy":   round((row["relevancy_score"]    or 0) * 100, 1),
                "impact_quantification": round((row["impact_score"]       or 0) * 100, 1),
            },
            "analysis": {
                "missing_critical_skills":  skills_data.get("missing_critical_skills", []),
                "unmet_responsibilities":   exp_data.get("unmet_responsibilities", []),
                "absent_seniority_signals": seniority_data.get("absent_signals", []),
                "irrelevant_statements":    relevancy_data.get("irrelevant_statements", []),
                "unquantified_statements":  impact_data.get("unquantified_statements", []),
                "coach_recommendations":    coach_recs,
            },
            "ats_validation": ats_val,
            "pipeline_data": pr,
        })
    return result


def get_run(run_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    pr = json.loads(row["pipeline_json"] or "{}")
    return {
        "result_id": row["id"],
        "created_at": row["created_at"],
        "model": row["model"],
        "resume_text": row["resume_text"],
        "job_description": row["job_desc"],
        "pipeline_result": pr,
        "tailored_text": row["tailored_text"],
        "ats_score": row["ats_score"],
        "ats_grade": row["ats_grade"],
        "ats_validation": json.loads(row["ats_validation_json"]) if row["ats_validation_json"] else None,
    }
