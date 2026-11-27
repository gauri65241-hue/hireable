"""
Where mutable runtime state lives (SQLite DB, learned-weight feedback, JWT secret).

Default is <repo>/data, which is fine locally. On a host that mounts a persistent volume
(Render disk, Docker volume) mount it at a DIFFERENT path and point HIREABLE_STATE_DIR at it:
a volume mounted over /app/data hides everything baked into the image there (skills
taxonomy, sample CSVs), so those files silently disappear at runtime.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.getenv("HIREABLE_STATE_DIR") or os.path.join(BASE_DIR, "data")
