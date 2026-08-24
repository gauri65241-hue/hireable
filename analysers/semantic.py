"""
SBERT-based semantic similarity between resume and job description.
Used as a sanity layer: if embedding similarity is high but an LLM judge
scores very low, that judge's result is flagged as suspect.
"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# SBERT max input is 512 tokens ≈ 2000 chars; truncate before encoding.
_MAX_CHARS = 2000


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def resume_jd_similarity(resume_text: str, job_description: str) -> float | None:
    """
    Returns cosine similarity [0,1] between resume and JD embeddings.
    Returns None if sentence-transformers is unavailable or encoding fails.
    """
    try:
        import numpy as np
        model = _encoder()
        emb = model.encode(
            [resume_text[:_MAX_CHARS], job_description[:_MAX_CHARS]],
            convert_to_numpy=True,
        )
        norm = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
        if norm == 0:
            return None
        return round(float(np.dot(emb[0], emb[1]) / norm), 3)
    except Exception as e:
        logger.warning("SBERT similarity failed: %s", e)
        return None


def flag_suspect_scores(
    scores: dict[str, float],
    semantic_similarity: float | None,
    *,
    sim_threshold: float = 0.45,
    score_threshold: float = 0.25,
) -> list[str]:
    """
    Returns list of judge names whose LLM score is suspiciously low
    relative to the semantic similarity between resume and JD.
    """
    if semantic_similarity is None or semantic_similarity < sim_threshold:
        return []
    return [
        judge
        for judge, score in scores.items()
        if score is not None and score < score_threshold
    ]
