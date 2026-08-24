"""
Re-export shim for chunker.py.

Sub-packages (skills, experience, etc.) import from here instead of directly
from the root-level chunker.py. This decouples them from the project root being
on sys.path, making each sub-package independently importable as a standalone
module (e.g., in a Lambda function or separate container).

When migrating to microservices: replace this file's body with the actual
chunker logic or a thin HTTP client that calls a dedicated chunker service.

Security layer: all formatter wrappers apply sanitize_chunk() to the resume
block (index 1 of the returned tuple) before it reaches any prompt builder.
This closes the post-chunker injection surface — fragment evasion and tag
break-out attacks that survive the raw-text sanitize_user_content() check.
"""
import functools
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from chunker import (  # noqa: E402
    parse_jd      as _parse_jd_raw,
    parse_resume  as _parse_resume_raw,
    format_for_skills_judge      as _fmt_skills,
    format_for_experience_judge  as _fmt_experience,
    format_for_seniority_judge   as _fmt_seniority,
    format_for_relevancy_judge   as _fmt_relevancy,
    format_for_impact_judge      as _fmt_impact,
    format_for_coach_judge       as _fmt_coach,
    JDChunks,
    ResumeChunks,
)
from analysers.base import sanitize_chunk

# ── Parse cache ───────────────────────────────────────────────────────────────
# Each analysis passes the same resume_text and job_description to all 6 judges.
# Without caching, parse_jd and parse_resume run 5–6× on identical strings.
# LRU cache with maxsize=8 covers 3 concurrent analyses (3 JDs + 3 resumes = 6
# slots) with 2 buffer slots for repeated JDs across different resume batches.
# The cache holds JDChunks/ResumeChunks dataclass references (~8–15KB each).
# All callers treat these as read-only — mutating a cached chunk is a bug.

@functools.lru_cache(maxsize=8)
def parse_jd(text: str):
    return _parse_jd_raw(text)


@functools.lru_cache(maxsize=8)
def parse_resume(text: str, name: str = ""):
    return _parse_resume_raw(text, name)


def _sanitize_formatter(fn, name: str):
    """Wrap a formatter to sanitize the resume-derived block(s) in its output."""
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, tuple):
            # Convention: index 0 = JD block (employer-supplied, lower risk)
            #             index 1+ = resume block(s) (candidate-supplied, untrusted)
            return tuple(
                sanitize_chunk(part, f"{name}[{i}]") if isinstance(part, str) and i >= 1 else part
                for i, part in enumerate(result)
            )
        if isinstance(result, str):
            return sanitize_chunk(result, name)
        return result
    return wrapper


format_for_skills_judge     = _sanitize_formatter(_fmt_skills,     "skills_formatter")
format_for_experience_judge = _sanitize_formatter(_fmt_experience,  "experience_formatter")
format_for_seniority_judge  = _sanitize_formatter(_fmt_seniority,   "seniority_formatter")
format_for_relevancy_judge  = _sanitize_formatter(_fmt_relevancy,   "relevancy_formatter")
format_for_impact_judge     = _sanitize_formatter(_fmt_impact,      "impact_formatter")
format_for_coach_judge      = _sanitize_formatter(_fmt_coach,       "coach_formatter")

__all__ = [
    "parse_jd",
    "parse_resume",
    "format_for_skills_judge",
    "format_for_experience_judge",
    "format_for_seniority_judge",
    "format_for_relevancy_judge",
    "format_for_impact_judge",
    "format_for_coach_judge",
    "JDChunks",
    "ResumeChunks",
]
