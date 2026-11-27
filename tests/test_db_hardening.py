"""SQLite behaviour: write contention, cache semantics, tenant scoping, outcome integrity."""
import threading
from datetime import datetime, timedelta

import pytest

import db


@pytest.fixture(scope="module", autouse=True)
def _init():
    db.init_db()


def _pipeline(status="complete", score=0.8):
    return {"overall": {"overall_score": score, "component_scores": {}, "status": status,
                        "pipeline_version": "test"},
            "skills": {"scores": {}}, "coach": {}}


def test_concurrent_writers_never_hit_database_is_locked():
    errors = []
    def worker(w):
        try:
            for i in range(15):
                db.save_run(f"analysis_w{w}_{i}", "m", "resume", "jd", _pipeline(), None, "", "", f"u{w}")
                db.save_outcome(f"analysis_w{w}_{i}", 80.0, True, None, user_id=f"u{w}")
        except Exception as e:                    # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert errors == []
    assert len(db.get_outcomes_all()) >= 8 * 15


def test_transactions_take_the_write_lock_up_front():
    seen = []
    def probe(conn):
        seen.append(conn.in_transaction)
        # An IMMEDIATE transaction already holds the RESERVED lock: a second writer on another
        # connection must be refused immediately (busy) rather than succeed or deadlock on upgrade.
        import sqlite3
        other = sqlite3.connect(db.DB_PATH, timeout=0.05)
        try:
            other.execute("BEGIN IMMEDIATE")
            seen.append("second-writer-allowed")
        except sqlite3.OperationalError:
            seen.append("second-writer-blocked")
        finally:
            other.close()
    db._get_pool().execute_tx(probe)
    assert seen == [True, "second-writer-blocked"]


def test_run_status_is_persisted_and_only_complete_runs_are_cacheable():
    db.save_run("analysis_ok", "m", "r", "j", _pipeline("complete"), None, "", "", "u1")
    db.save_run("analysis_deg", "m", "r", "j", _pipeline("degraded"), None, "", "", "u1")
    db.set_cached_run("k_ok", "analysis_ok", "u1")
    db.set_cached_run("k_deg", "analysis_deg", "u1")
    assert db.get_cached_run_id("k_ok") == "analysis_ok"
    assert db.get_cached_run_id("k_deg") is None            # degraded results are never served from cache
    assert db.get_cached_run_id("missing") is None


def test_cache_entries_expire():
    db.save_run("analysis_old", "m", "r", "j", _pipeline(), None, "", "", "u1")
    db.set_cached_run("k_old", "analysis_old", "u1")
    def age(conn):
        conn.execute("UPDATE score_cache SET created_at=? WHERE cache_key='k_old'",
                     ((datetime.utcnow() - timedelta(days=45)).isoformat(),))
    db._get_pool().execute_tx(age)
    assert db.get_cached_run_id("k_old") is None
    assert db.get_cached_run_id("k_old", max_age_days=60) == "analysis_old"


def test_skill_demand_is_tenant_scoped_deduplicated_and_needs_enough_jobs():
    for _ in range(6):
        db.add_job("t", "c", "", "Python, python, AWS", "d", user_id="tenantA")
    for _ in range(2):
        db.add_job("t", "c", "", "Rust, Rust", "d", user_id="tenantB")
    freqs, n = db.get_skill_demand("tenantA")
    assert n == 6 and freqs == {"python": 6, "aws": 6}      # duplicate within a posting counts once
    assert "rust" not in freqs                              # another tenant's jobs are invisible
    assert db.get_skill_demand("tenantB") == ({}, 2)        # < min_jobs_for_frequency -> no market signal


def test_outcomes_are_deduplicated_per_user_and_run():
    db.save_run("analysis_dup", "m", "r", "j", _pipeline(), None, "", "", "u9")
    assert db.save_outcome("analysis_dup", 80.0, True, None, user_id="u9") is True
    assert db.save_outcome("analysis_dup", 10.0, False, None, user_id="u9") is False   # replaced, not added
    mine = [o for o in db.get_outcomes_all("u9") if o["result_id"] == "analysis_dup"]
    assert len(mine) == 1 and mine[0]["got_job"] is False
    assert db.get_outcomes_all("someone-else") == []
