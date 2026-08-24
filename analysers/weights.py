"""
Learned-weight computation from real outcome feedback.
Reads judge_feedback.jsonl and derives per-analyser weights from prediction accuracy.
Falls back to config defaults when fewer than MIN_FEEDBACK_SAMPLES samples exist per analyser.
"""

import json
import math
import os
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
FEEDBACK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "judge_feedback.jsonl",
)

# Which analysers each real-world signal trains
INTERVIEW_ANALYSERS = ["skills", "experience", "seniority", "relevancy"]
JOB_ANALYSERS = ["impact", "overall"]


def compute_learned_weights(feedback_file: str = None) -> dict:
    """
    Derive analyser weights from outcome feedback.
    Falls back to cfg.default_weights per analyser when samples < cfg.min_feedback_samples.
    Returns {"weights": {...}, "source": "learned"|"default", "analyser_stats": {...}}
    """
    path = feedback_file or FEEDBACK_FILE
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except FileNotFoundError:
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
        if len(entries_for) >= cfg.min_feedback_samples:
            weights = [_decay_weight(e.get("timestamp", ""), now) for e in entries_for]
            total_w = sum(weights)
            correct_w = sum(w for e, w in zip(entries_for, weights) if e["correct"])
            accuracy = correct_w / total_w if total_w > 0 else 0.5
            raw[name] = max(0.1, accuracy)
            stats[name] = {"accuracy": round(accuracy * 100, 1), "n": len(entries_for), "source": "learned"}
        else:
            raw[name] = cfg.default_weights[name]
            stats[name] = {"n": len(entries_for), "source": "default", "accuracy": None}

    total = sum(raw.values())
    normalized = {k: round(v / total, 4) for k, v in raw.items()}
    # Fix floating-point rounding drift
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


def append_feedback(entries: list[dict]) -> None:
    """Append feedback records to the shared feedback file."""
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
