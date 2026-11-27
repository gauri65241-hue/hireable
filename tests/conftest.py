import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Isolate ALL mutable state before any application module is imported: the tests must
# never create or touch the developer's real runs.db / feedback file / JWT secret.
_STATE = tempfile.mkdtemp(prefix="hireable-test-state-")
os.environ["HIREABLE_STATE_DIR"] = _STATE
os.environ["HIREABLE_DB_PATH"] = os.path.join(_STATE, "runs.db")
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")
os.environ["ADMIN_EMAILS"] = "admin@example.com"
os.environ.pop("API_KEY", None)
