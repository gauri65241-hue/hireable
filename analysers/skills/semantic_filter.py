"""
Post-filter for LLM-extracted skill analysis.

Two jobs applied after the LLM returns, before scoring:

1. Spurious skill removal — the LLM sometimes extracts JD boilerplate phrases
   like "solid enterprise experience" or "meaningful impact" as if they were
   technical skills. These are detected via word-length + regex patterns and
   dropped so they cannot count as missing skills.

2. Alias reclassification — a skill marked MISSING that is semantically
   near-identical (cos_sim >= ALIAS_THRESHOLD) to an already-PRESENT skill
   is reclassified to partial (is_present=0.5) instead of 0.
   Catches: "React.js" vs "React", "AWS Lambda" vs "Amazon Lambda", etc.

Neither change modifies the raw LLM output stored in the DB — filter_and_reclassify
returns a new list used only by the score() function.
"""

import logging
import re

import numpy as np

logger = logging.getLogger("AnalyserEngine")

# Cosine sim above this → treat as alias of an already-present skill
ALIAS_THRESHOLD = 0.85

# Regex for common JD boilerplate that the LLM mistakes for skills.
# These are phrase-level patterns that never represent a discrete technical skill.
_BOILERPLATE_RE = re.compile(
    r'\b('
    r'ability\s+to|passion(ate)?\s+(for|about)|focus\s+on|'
    r'meaningful\s+(impact|contribution)|thrive\s+in|'
    r'solid\s+\w+\s+(experience|background)|strong\s+(background|interpersonal)|'
    r'comfortable\s+(with|working)|proven\s+track\s+record|'
    r'self.motivated|results?.driven|problem.solver|team\s+player|'
    r'detail.oriented|forward.thinking|fast.paced|'
    r'collaborative\s+(environment|team)|ownership\s+mentality|'
    r'growth\s+mindset|can.do\s+attitude|hungry\s+to\s+learn'
    r')\b',
    re.I,
)


def _is_spurious(skill_name: str) -> bool:
    """Return True if the skill name looks like JD boilerplate, not a real skill."""
    if not skill_name:
        return True
    # Explicit pattern match first (cheap, catches most Revolut-style boilerplate)
    if _BOILERPLATE_RE.search(skill_name):
        return True
    # Long phrases (5+ words) with no pattern match above are still suspicious.
    # Real technical skills are almost always 1-4 words:
    #   "Python", "REST APIs", "Google Cloud Platform", "Apache Kafka Streams"
    # Sentences like "a focus on X in Y environment" have 6+ words.
    words = skill_name.split()
    if len(words) >= 5:
        logger.info(
            "semantic_filter: long phrase '%s' (%d words) — likely boilerplate, excluding",
            skill_name, len(words),
        )
        return True
    return False


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def filter_and_reclassify(skill_analysis: list[dict]) -> list[dict]:
    """
    Returns a filtered/adjusted copy of skill_analysis items.
    Spurious items are dropped. Alias MISSING items become partial (0.5).
    Falls back to original list unchanged on any error.
    """
    if not skill_analysis:
        return skill_analysis

    skill_names = [
        (item.get("skill") or item.get("skill_requirement") or "").strip()
        for item in skill_analysis
    ]

    # Pass 1: drop boilerplate phrases (no SBERT needed)
    filtered_items = []
    filtered_names = []
    for i, item in enumerate(skill_analysis):
        if _is_spurious(skill_names[i]):
            logger.info(
                "semantic_filter: spurious skill '%s' — excluded from scoring", skill_names[i]
            )
            continue
        filtered_items.append(item)
        filtered_names.append(skill_names[i])

    if not filtered_items:
        return filtered_items

    dropped = len(skill_analysis) - len(filtered_items)
    if dropped:
        logger.info("semantic_filter: dropped %d spurious skill(s)", dropped)

    # Pass 2: alias reclassification using SBERT (only for MISSING items)
    missing_indices = [
        i for i, item in enumerate(filtered_items) if item.get("is_present") == 0
    ]
    present_indices = [
        i for i, item in enumerate(filtered_items) if item.get("is_present") == 1
    ]

    # Skip SBERT entirely if no alias check is possible
    if not missing_indices or not present_indices:
        return filtered_items

    try:
        enc = _get_encoder()
        all_embs = enc.encode(
            filtered_names, normalize_embeddings=True, show_progress_bar=False
        )
        present_embs = all_embs[present_indices]  # shape (n_present, dim)

        result = list(filtered_items)  # copy
        for mi in missing_indices:
            alias_sims = present_embs @ all_embs[mi]
            max_alias_sim = float(alias_sims.max())
            if max_alias_sim >= ALIAS_THRESHOLD:
                best = int(alias_sims.argmax())
                alias_name = filtered_names[present_indices[best]]
                logger.info(
                    "semantic_filter: '%s' is alias of present '%s' (sim=%.2f) → partial",
                    filtered_names[mi], alias_name, max_alias_sim,
                )
                item = dict(result[mi])
                item["is_present"] = 0.5
                item["_alias_of"] = alias_name
                item["_alias_sim"] = round(max_alias_sim, 3)
                result[mi] = item

        return result

    except Exception as exc:
        logger.warning(
            "semantic_filter alias pass failed (%s) — returning spurious-filtered list", exc
        )
        return filtered_items
