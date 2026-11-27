"""
SQLite persistence for analysis run history.
Production hardening:
  - WAL mode + synchronous=FULL for crash-safe durability (fsync on every commit).
  - Connection pool (queue.Queue) — avoids per-request connect/close overhead.
  - run_summaries denormalized table — list_runs never deserializes pipeline_json.
  - save_run writes both tables in a single explicit transaction.
  - DB_PATH: override with HIREABLE_DB_PATH env var for containerised deployments.
"""

import os
import json
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta

DB_PATH = os.getenv(
    "HIREABLE_DB_PATH",
    os.path.join(__import__("paths").STATE_DIR, "runs.db"),
)
_POOL_SIZE = int(os.getenv("HIREABLE_DB_POOL_SIZE", "4"))
_POOL_TIMEOUT = 30.0


class _SQLitePool:
    def __init__(self, path: str, size: int):
        self._path = path
        self._pool: queue.Queue = queue.Queue(maxsize=size)
        for _ in range(size):
            self._pool.put(self._make_conn())

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=_POOL_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")   # fsync on every commit — crash-safe
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def acquire(self) -> sqlite3.Connection:
        try:
            return self._pool.get(timeout=_POOL_TIMEOUT)
        except queue.Empty:
            raise TimeoutError("DB connection pool exhausted after 30s")

    def release(self, conn: sqlite3.Connection):
        self._pool.put(conn)

    def execute_tx(self, fn):
        """Acquire a connection, run fn(conn) inside BEGIN/COMMIT, return result."""
        conn = self.acquire()
        try:
            # IMMEDIATE takes the write lock up front. A deferred BEGIN that reads and then
            # writes can hit SQLITE_BUSY on lock upgrade *without* honouring the busy timeout.
            conn.execute("BEGIN IMMEDIATE")
            result = fn(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release(conn)


_pool: _SQLitePool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> _SQLitePool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                _pool = _SQLitePool(DB_PATH, _POOL_SIZE)
    return _pool


def init_db():
    def _create(conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id         TEXT PRIMARY KEY,
                email           TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                resume_id       TEXT PRIMARY KEY,
                user_id         TEXT,
                name            TEXT NOT NULL,
                title           TEXT,
                experience_years TEXT,
                skills          TEXT,
                text            TEXT,
                created_at      TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             TEXT,
                title               TEXT NOT NULL,
                company             TEXT,
                experience_required TEXT,
                skills_required     TEXT,
                description         TEXT,
                source_url          TEXT,
                applied             INTEGER DEFAULT 0,
                created_at          TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id   TEXT NOT NULL,
                job_id      TEXT NOT NULL,
                fit_label   INTEGER NOT NULL,
                golden      INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT,
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
                skills_gate_applied INTEGER DEFAULT 0,
                pipeline_json       TEXT,
                tailored_text       TEXT,
                ats_validation_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id         TEXT NOT NULL,
                predicted_score   REAL,
                got_job           INTEGER,
                got_interview     INTEGER,
                recorded_at       TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_result ON outcomes(result_id)")

        # Denormalized summary — list_runs reads this; never touches pipeline_json
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_summaries (
                run_id                      TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                missing_critical_skills     TEXT DEFAULT '[]',
                unmet_responsibilities      TEXT DEFAULT '[]',
                absent_seniority_signals    TEXT DEFAULT '[]',
                irrelevant_statements       TEXT DEFAULT '[]',
                unquantified_statements     TEXT DEFAULT '[]',
                coach_recommendations       TEXT DEFAULT '[]'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS score_cache (
                cache_key   TEXT PRIMARY KEY,
                run_id      TEXT NOT NULL,
                user_id     TEXT,
                created_at  TEXT NOT NULL
            )
        """)

        # Migrate pre-existing DBs that predate some columns
        for table, col, defn in [
            ("runs",     "candidate_name",    "TEXT"),
            ("runs",     "job_title",         "TEXT"),
            ("runs",     "skills_gate_applied","INTEGER DEFAULT 0"),
            ("runs",     "user_id",           "TEXT"),
            ("resumes",  "user_id",           "TEXT"),
            ("jobs",     "user_id",           "TEXT"),
            ("outcomes", "interview_stage",   "TEXT"),
            ("outcomes", "rejection_reason",  "TEXT"),
            ("outcomes", "user_id",           "TEXT"),
            ("runs",     "status",            "TEXT DEFAULT 'complete'"),
            ("runs",     "pipeline_version",  "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
            except Exception:
                pass

    _get_pool().execute_tx(_create)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(user_id: str, email: str, hashed_password: str) -> None:
    def _insert(conn):
        conn.execute(
            "INSERT INTO users (user_id, email, hashed_password, created_at) VALUES (?,?,?,?)",
            (user_id, email, hashed_password, datetime.utcnow().isoformat()),
        )
    _get_pool().execute_tx(_insert)


def get_user_by_email(email: str) -> dict | None:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    finally:
        pool.release(conn)
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    finally:
        pool.release(conn)
    return dict(row) if row else None


def _slim_pipeline(result: dict) -> dict:
    """
    Return a shallow-stripped copy of pipeline_result suitable for storage.
    Removes raw_response fields (full LLM output including think blocks) from
    every judge section — these are 20–50KB each and never rendered to users.
    The parsed data dict, scores, and metadata are preserved intact.
    Reduces pipeline_json column size by ~80% for reasoning models.
    """
    slim = {}
    for key, val in result.items():
        if isinstance(val, dict):
            slim[key] = {k: v for k, v in val.items() if k != "raw_response"}
        else:
            slim[key] = val
    return slim


# ── Resumes ───────────────────────────────────────────────────────────────────

def get_resumes(user_id: str) -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        rows = conn.execute(
            "SELECT * FROM resumes WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    finally:
        pool.release(conn)
    return [dict(r) for r in rows]


def get_all_resumes() -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        rows = conn.execute(
            "SELECT * FROM resumes ORDER BY created_at DESC"
        ).fetchall()
    finally:
        pool.release(conn)
    return [dict(r) for r in rows]


def add_resume(name: str, title: str, experience_years: str, skills: str, text: str,
               user_id: str = "") -> str:
    def _insert(conn):
        new_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO resumes (resume_id, user_id, name, title, experience_years, skills, text, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (new_id, user_id, name, title, experience_years, skills, text, datetime.utcnow().isoformat()),
        )
        return new_id
    return _get_pool().execute_tx(_insert)


def delete_resume(resume_id: str, user_id: str = "") -> bool:
    def _delete(conn):
        cur = conn.execute(
            "DELETE FROM resumes WHERE resume_id=? AND user_id=?", (resume_id, user_id)
        )
        return cur.rowcount > 0
    return _get_pool().execute_tx(_delete)


# ── Jobs ──────────────────────────────────────────────────────────────────────

def get_jobs(user_id: str) -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    finally:
        pool.release(conn)
    return [{**dict(r), "applied": bool(r["applied"])} for r in rows]


def get_skill_demand(user_id: str | None = None) -> tuple[dict[str, int], int]:
    """Return ({skill: number of jobs requiring it}, number_of_jobs).

    Scoped to `user_id` when given. (It used to read EVERY user's jobs, so one account
    could change the skill weights - and therefore the scores - of all others.) Returns
    ({}, n) when there are fewer than cfg.min_jobs_for_frequency postings: a handful of
    jobs says nothing about market demand."""
    from collections import Counter
    from analysers.config import cfg
    pool = _get_pool()
    conn = pool.acquire()
    try:
        sql = "SELECT skills_required FROM jobs WHERE skills_required IS NOT NULL AND skills_required != ''"
        params: tuple = ()
        if user_id is not None:
            sql += " AND user_id = ?"
            params = (str(user_id),)
        rows = conn.execute(sql, params).fetchall()
    finally:
        pool.release(conn)
    n_jobs = len(rows)
    if n_jobs < cfg.min_jobs_for_frequency:
        return {}, n_jobs
    counts: Counter = Counter()
    for row in rows:
        # a skill listed twice in one posting still counts once for that posting
        for skill in {x.strip().lower() for x in row["skills_required"].split(",") if x.strip()}:
            counts[skill] += 1
    return dict(counts), n_jobs


def get_skill_frequencies(user_id: str | None = None) -> dict[str, int]:
    """Back-compat wrapper around get_skill_demand()."""
    return get_skill_demand(user_id)[0]


def get_all_jobs() -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    finally:
        pool.release(conn)
    return [{**dict(r), "applied": bool(r["applied"])} for r in rows]


def add_job(title: str, company: str, experience_required: str,
            skills_required: str, description: str, source_url: str = "",
            user_id: str = "") -> int:
    def _insert(conn):
        cur = conn.execute(
            "INSERT INTO jobs (user_id, title, company, experience_required, skills_required, description, source_url, applied, created_at) VALUES (?,?,?,?,?,?,?,0,?)",
            (user_id, title, company, experience_required, skills_required, description, source_url, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid
    return _get_pool().execute_tx(_insert)


def delete_job(job_id: int, user_id: str = "") -> bool:
    def _delete(conn):
        cur = conn.execute(
            "DELETE FROM jobs WHERE job_id=? AND user_id=?", (job_id, user_id)
        )
        return cur.rowcount > 0
    return _get_pool().execute_tx(_delete)


def toggle_job_applied(job_id: int, applied: bool, user_id: str = "") -> bool:
    def _update(conn):
        cur = conn.execute(
            "UPDATE jobs SET applied=? WHERE job_id=? AND user_id=?",
            (1 if applied else 0, job_id, user_id)
        )
        return cur.rowcount > 0
    return _get_pool().execute_tx(_update)


# ── Labels ────────────────────────────────────────────────────────────────────

def add_label(resume_id: str, job_id: str, fit_label: int, golden: int = 0):
    def _insert(conn):
        conn.execute(
            "INSERT INTO labels (resume_id, job_id, fit_label, golden, created_at) VALUES (?,?,?,?,?)",
            (resume_id, job_id, fit_label, golden, datetime.utcnow().isoformat()),
        )
    _get_pool().execute_tx(_insert)


def get_labels() -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        rows = conn.execute("SELECT * FROM labels ORDER BY created_at DESC").fetchall()
    finally:
        pool.release(conn)
    return [dict(r) for r in rows]


def save_run(run_id: str, model: str, resume_text: str, job_desc: str,
             pipeline_result: dict, ats_result: dict | None = None,
             candidate_name: str = "", job_title: str = "", user_id: str = ""):
    overall = pipeline_result.get("overall", {})
    cs = overall.get("component_scores", {})

    # Extract summary fields so list_runs never needs pipeline_json
    skills_data    = pipeline_result.get("skills",    {}).get("scores", {})
    exp_data       = pipeline_result.get("experience",{}).get("scores", {})
    seniority_data = pipeline_result.get("seniority", {}).get("scores", {})
    relevancy_data = pipeline_result.get("relevancy", {}).get("scores", {})
    impact_data    = pipeline_result.get("impact",    {}).get("scores", {})
    coach          = pipeline_result.get("coach", {})
    coach_recs     = (coach.get("data") or {}).get("improvement_analysis", []) if coach.get("success") else []

    run_status = overall.get("status") or "complete"
    run_version = overall.get("pipeline_version")

    def _write(conn):
        conn.execute("""
            INSERT OR REPLACE INTO runs
            (id, user_id, created_at, model, candidate_name, job_title,
             resume_text, job_desc,
             overall_score, skills_score, experience_score,
             seniority_score, relevancy_score, impact_score,
             ats_score, ats_grade, skills_gate_applied, pipeline_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id,
            user_id,
            datetime.utcnow().isoformat(),
            model,
            candidate_name,
            job_title,
            resume_text,
            job_desc,
            overall.get("overall_score"),
            cs.get("skills"),
            cs.get("experience"),
            cs.get("seniority"),
            cs.get("relevancy"),
            cs.get("impact"),
            ats_result.get("score") if ats_result else None,
            ats_result.get("grade") if ats_result else None,
            1 if overall.get("skills_gate_applied") else 0,
            json.dumps(_slim_pipeline(pipeline_result), separators=(',', ':')),
        ))
        conn.execute("""
            INSERT OR REPLACE INTO run_summaries
            (run_id, missing_critical_skills, unmet_responsibilities,
             absent_seniority_signals, irrelevant_statements,
             unquantified_statements, coach_recommendations)
            VALUES (?,?,?,?,?,?,?)
        """, (
            run_id,
            json.dumps(skills_data.get("missing_skills", [])),
            json.dumps(exp_data.get("unmet_responsibilities", [])),
            json.dumps(seniority_data.get("absent_signals", [])),
            json.dumps(relevancy_data.get("irrelevant_statements", [])),
            json.dumps(impact_data.get("unquantified_statements", [])),
            json.dumps(coach_recs),
        ))
        conn.execute("UPDATE runs SET status=?, pipeline_version=? WHERE id=?",
                     (run_status, run_version, run_id))

    _get_pool().execute_tx(_write)


def update_run_tailored(run_id: str, tailored_text: str, ats_validation: dict | None = None):
    def _update(conn):
        conn.execute(
            "UPDATE runs SET tailored_text=?, ats_validation_json=? WHERE id=?",
            (tailored_text, json.dumps(ats_validation) if ats_validation else None, run_id),
        )
    _get_pool().execute_tx(_update)


def list_runs(user_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        if user_id is not None:
            rows = conn.execute("""
                SELECT r.id, r.created_at, r.model,
                       r.candidate_name, r.job_title,
                       substr(r.resume_text, 1, 120) AS resume_snippet,
                       substr(r.job_desc,    1, 120) AS job_snippet,
                       r.overall_score, r.skills_score, r.experience_score,
                       r.seniority_score, r.relevancy_score, r.impact_score,
                       r.ats_score, r.ats_grade, r.skills_gate_applied,
                       s.missing_critical_skills, s.unmet_responsibilities,
                       s.absent_seniority_signals, s.irrelevant_statements,
                       s.unquantified_statements,  s.coach_recommendations
                FROM runs r
                LEFT JOIN run_summaries s ON s.run_id = r.id
                WHERE r.user_id = ?
                ORDER BY r.created_at DESC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT r.id, r.created_at, r.model,
                       r.candidate_name, r.job_title,
                       substr(r.resume_text, 1, 120) AS resume_snippet,
                       substr(r.job_desc,    1, 120) AS job_snippet,
                       r.overall_score, r.skills_score, r.experience_score,
                       r.seniority_score, r.relevancy_score, r.impact_score,
                       r.ats_score, r.ats_grade, r.skills_gate_applied,
                       s.missing_critical_skills, s.unmet_responsibilities,
                       s.absent_seniority_signals, s.irrelevant_statements,
                       s.unquantified_statements,  s.coach_recommendations
                FROM runs r
                LEFT JOIN run_summaries s ON s.run_id = r.id
                ORDER BY r.created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
    finally:
        pool.release(conn)

    result = []
    for row in rows:
        result.append({
            "result_id": row["id"],
            "timestamp": row["created_at"],
            "model": row["model"],
            "resume_name": row["candidate_name"] or "",
            "job_title": row["job_title"] or "",
            "resume_snippet": row["resume_snippet"],
            "job_snippet": row["job_snippet"],
            "match_score": round((row["overall_score"] or 0) * 100, 1),
            "skills_gate_applied": bool(row["skills_gate_applied"]),
            "ats_score": row["ats_score"],
            "ats_grade": row["ats_grade"],
            "metrics": {
                "skills_coverage":       round((row["skills_score"]      or 0) * 100, 1),
                "experience_depth":      round((row["experience_score"]   or 0) * 100, 1),
                "seniority_calibration": round((row["seniority_score"]    or 0) * 100, 1),
                "statement_relevancy":   round((row["relevancy_score"]    or 0) * 100, 1),
                "impact_quantification": round((row["impact_score"]       or 0) * 100, 1),
            },
            "analysis": {
                "missing_critical_skills":  json.loads(row["missing_critical_skills"] or "[]"),
                "unmet_responsibilities":   json.loads(row["unmet_responsibilities"]   or "[]"),
                "absent_seniority_signals": json.loads(row["absent_seniority_signals"] or "[]"),
                "irrelevant_statements":    json.loads(row["irrelevant_statements"]    or "[]"),
                "unquantified_statements":  json.loads(row["unquantified_statements"]  or "[]"),
                "coach_recommendations":    json.loads(row["coach_recommendations"]    or "[]"),
            },
        })
    return result


def get_run(run_id: str, user_id: str = "") -> dict | None:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute(
            "SELECT * FROM runs WHERE id=? AND user_id=?", (run_id, user_id)
        ).fetchone()
    finally:
        pool.release(conn)
    if not row:
        return None
    return {
        "result_id": row["id"],
        "created_at": row["created_at"],
        "model": row["model"],
        "resume_text": row["resume_text"],
        "job_description": row["job_desc"],
        "pipeline_result": json.loads(row["pipeline_json"] or "{}"),
        "tailored_text": row["tailored_text"],
        "ats_score": row["ats_score"],
        "ats_grade": row["ats_grade"],
        "ats_validation": json.loads(row["ats_validation_json"]) if row["ats_validation_json"] else None,
    }


# ── Score cache ───────────────────────────────────────────────────────────────

def get_cached_run_id(cache_key: str, max_age_days: float | None = None) -> str | None:
    """Return a cached run id, honouring a TTL and ignoring runs that were not complete."""
    if max_age_days is None:
        max_age_days = float(os.getenv("SCORE_CACHE_TTL_DAYS", "30"))
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    pool = _get_pool()
    conn = pool.acquire()
    try:
        row = conn.execute(
            """SELECT c.run_id FROM score_cache c
               JOIN runs r ON r.id = c.run_id
               WHERE c.cache_key=? AND c.created_at >= ?
                 AND COALESCE(r.status, 'complete') = 'complete'""",
            (cache_key, cutoff),
        ).fetchone()
    finally:
        pool.release(conn)
    return row["run_id"] if row else None


def set_cached_run(cache_key: str, run_id: str, user_id: str = "") -> None:
    def _insert(conn):
        conn.execute(
            "INSERT OR REPLACE INTO score_cache (cache_key, run_id, user_id, created_at) VALUES (?,?,?,?)",
            (cache_key, run_id, user_id, datetime.utcnow().isoformat()),
        )
    _get_pool().execute_tx(_insert)


# ── Outcomes ──────────────────────────────────────────────────────────────────

def save_outcome(result_id: str, predicted_score: float,
                 got_job: bool | None, got_interview: bool | None,
                 interview_stage: str | None = None,
                 rejection_reason: str | None = None,
                 user_id: str = "") -> bool:
    """Store an outcome. Returns True if this is the first outcome for (user, run)."""
    def _insert(conn):
        # One outcome per (user, run): re-submitting replaces rather than multiplying a
        # single data point into many (which would let one user steer learned weights).
        existed = conn.execute(
            "SELECT 1 FROM outcomes WHERE result_id=? AND COALESCE(user_id,'')=?",
            (result_id, user_id)).fetchone() is not None
        conn.execute("DELETE FROM outcomes WHERE result_id=? AND COALESCE(user_id,'')=?",
                     (result_id, user_id))
        conn.execute(
            """INSERT INTO outcomes
               (result_id, predicted_score, got_job, got_interview,
                interview_stage, rejection_reason, recorded_at, user_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                result_id,
                predicted_score,
                None if got_job is None else int(got_job),
                None if got_interview is None else int(got_interview),
                interview_stage,
                rejection_reason,
                datetime.utcnow().isoformat(),
                user_id,
            ),
        )
        return not existed
    return _get_pool().execute_tx(_insert)


def get_outcomes_all(user_id: str | None = None) -> list[dict]:
    pool = _get_pool()
    conn = pool.acquire()
    try:
        if user_id is None:
            rows = conn.execute("SELECT * FROM outcomes ORDER BY recorded_at DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM outcomes WHERE user_id=? ORDER BY recorded_at DESC",
                                (user_id,)).fetchall()
    finally:
        pool.release(conn)
    return [
        {
            "result_id":        r["result_id"],
            "predicted_score":  r["predicted_score"],
            "got_job":          None if r["got_job"] is None else bool(r["got_job"]),
            "got_interview":    None if r["got_interview"] is None else bool(r["got_interview"]),
            "interview_stage":  r["interview_stage"],
            "rejection_reason": r["rejection_reason"],
            "timestamp":        r["recorded_at"],
        }
        for r in rows
    ]
