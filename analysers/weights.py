"""
Learned-weight computation from real outcome feedback.
Reads judge_feedback.jsonl and derives per-analyser weights from prediction accuracy.
Falls back to config defaults when fewer than MIN_FEEDBACK_SAMPLES samples exist per analyser.
"""

import json
import math
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone

from analysers.config import cfg

# Exponential decay half-life in days — entries older than this contribute ~50%
_DECAY_TAU = 90.0


def _decay_weight(ts: str, now: datetime) -> float:
    """Weight for a feedback entry based on age. Returns 1.0 when timestamp is missing."""
    try:
        age_days = (now - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 86400
        return math.exp(-age_days / _DECAY_TAU)
    except Exception:
        return 1.0

# Single authoritative path — written by record_outcome, read by load_calibration
from paths import STATE_DIR
FEEDBACK_FILE = os.path.join(STATE_DIR, "judge_feedback.jsonl")

# Which analysers each real-world signal trains
INTERVIEW_ANALYSERS = ["skills", "experience", "seniority", "relevancy"]
JOB_ANALYSERS = ["impact", "overall"]


def _read_entries(path: str) -> list[dict]:
    entries: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # one torn/corrupt line must not disable learning
    except FileNotFoundError:
        pass
    return entries


def compute_learned_weights(feedback_file: str = None) -> dict:
    """
    Derive analyser weights from outcome feedback.

    Each analyser's configured prior is scaled by a multiplier derived from its decayed
    accuracy, shrunk toward 1.0 when evidence is thin:

        lam        = n / (n + cfg.feedback_prior_strength)
        multiplier = clamp(1 + lam * (2*accuracy - 1), 0.25, 2.0)
        raw        = prior * multiplier

    The old version replaced the prior with the raw accuracy (0.5-1.0) for analysers with
    >= 3 samples while leaving the others at their priors (0.10-0.35), then normalised:
    mixing two different scales, so three lucky outcomes could move one judge from 0.35 to
    ~0.64 of the total weight. Here accuracy 0.5 (coin flip) leaves the prior unchanged.
    Returns {"weights": {...}, "source": "learned"|"default", "analyser_stats": {...}}
    """
    path = feedback_file or FEEDBACK_FILE
    entries = _read_entries(path)
    if not entries:
        return {"weights": cfg.default_weights, "source": "default", "analyser_stats": {}}

    component_analysers = list(cfg.default_weights.keys())
    by_analyser = defaultdict(list)
    for e in entries:
        if e.get("judge") in component_analysers:
            by_analyser[e["judge"]].append(e)

    now = datetime.now(timezone.utc)
    raw, stats = {}, {}
    for name in component_analysers:
        entries_for = by_analyser[name]
        prior = cfg.default_weights[name]
        n = len(entries_for)
        if n >= cfg.min_feedback_samples:
            weights = [_decay_weight(e.get("timestamp", ""), now) for e in entries_for]
            total_w = sum(weights)
            correct_w = sum(w for e, w in zip(entries_for, weights) if e.get("correct"))
            accuracy = correct_w / total_w if total_w > 0 else 0.5
            lam = n / (n + cfg.feedback_prior_strength)
            multiplier = min(2.0, max(0.25, 1.0 + lam * (2.0 * accuracy - 1.0)))
            raw[name] = prior * multiplier
            stats[name] = {"accuracy": round(accuracy * 100, 1), "n": n, "source": "learned",
                           "multiplier": round(multiplier, 3)}
        else:
            raw[name] = prior
            stats[name] = {"n": n, "source": "default", "accuracy": None}

    total = sum(raw.values())
    normalized = {k: round(v / total, 4) for k, v in raw.items()}
    diff = round(1.0 - sum(normalized.values()), 4)
    if diff:
        top = max(normalized, key=normalized.get)
        normalized[top] = round(normalized[top] + diff, 4)

    any_learned = any(s.get("source") == "learned" for s in stats.values())
    return {
        "weights": normalized,
        "source": "learned" if any_learned else "default",
        "analyser_stats": stats,
    }


_APPEND_LOCK = threading.Lock()


def append_feedback(entries: list[dict]) -> None:
    """Append feedback records. One write() per call under a lock (and O_APPEND), so
    concurrent requests cannot interleave partial lines."""
    if not entries:
        return
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    blob = "".join(json.dumps(e) + "\n" for e in entries)
    with _APPEND_LOCK:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(blob)
