"""
Central configuration for all analysers.
Every tunable value lives here and can be overridden via environment variables.
"""

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default

def _float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default

def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class AnalyserConfig:
    # LLM retry behaviour
    confidence_threshold: int    = _int("ANALYSER_CONFIDENCE_THRESHOLD", 75)
    max_retries: int             = _int("ANALYSER_MAX_RETRIES", 3)

    # Scoring
    skills_gate_threshold: float = _float("ANALYSER_SKILLS_GATE", 0.40)
    default_weights: dict        = field(default_factory=lambda: {
        "skills":     _float("ANALYSER_WEIGHT_SKILLS",     0.35),
        "experience": _float("ANALYSER_WEIGHT_EXPERIENCE", 0.30),
        "seniority":  _float("ANALYSER_WEIGHT_SENIORITY",  0.10),
        "relevancy":  _float("ANALYSER_WEIGHT_RELEVANCY",  0.15),
        "impact":     _float("ANALYSER_WEIGHT_IMPACT",     0.10),
    })

    # Concurrency
    # ollama_semaphore_size: max concurrent LLM calls across ALL judges.
    # A local 14B model processes ~2 requests in parallel before queuing hurts
    # latency. Default 2 prevents pile-up; set ANALYSER_OLLAMA_SEM=5 if running
    # a larger GPU or an API backend where concurrency is free.
    ollama_semaphore_size: int        = _int("ANALYSER_OLLAMA_SEM", 2)
    # max_concurrent_analyses: router-level gate. Returns 503 when at capacity.
    # A local 14B model has no GPU headroom for concurrent analyses — queue them.
    # Set ANALYSER_MAX_CONCURRENT=3+ only when using an API backend.
    max_concurrent_analyses: int      = _int("ANALYSER_MAX_CONCURRENT", 1)
    # How long to wait for the analysis slot before returning 503.
    # With max_concurrent=1 and 7-job batch runs, each analysis can take ~10min —
    # set high enough that the batch loop queues through without 503ing.
    sem_acquire_timeout: float        = _float("ANALYSER_SEM_TIMEOUT", 3600.0)

    # Infrastructure — override these for containerised / cloud deployments
    ollama_base_url: str              = _str("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    # Input limits
    max_resume_chars: int             = _int("ANALYSER_MAX_RESUME_CHARS", 20_000)
    max_job_description_chars: int    = _int("ANALYSER_MAX_JD_CHARS", 10_000)

    # Defaults
    default_model: str                = _str("ANALYSER_DEFAULT_MODEL", "cogito:14b")
    # Fallback used by run_with_retry when primary model exhausts all retries.
    # Set to "" to disable. Override via ANALYSER_FALLBACK_MODEL env var.
    fallback_model: str               = _str("ANALYSER_FALLBACK_MODEL", "phi4-mini:latest")
    # Fast model for lightweight judges (seniority, relevancy, impact).
    # These judges do binary classification / pattern matching — they don't need a 14B model.
    # Set to "" to use default_model for all judges.
    fast_model: str                   = _str("ANALYSER_FAST_MODEL", "phi4-mini:latest")
    min_feedback_samples: int         = _int("ANALYSER_MIN_FEEDBACK_SAMPLES", 3)

    # OpenRouter — auto-activated when OPENROUTER_API_KEY is set
    # Heavy judges (skills, experience, coach) use the default model.
    # Fast judges (seniority, relevancy, impact) use the fast model.
    openrouter_api_key: str           = _str("OPENROUTER_API_KEY", "")
    openrouter_base_url: str          = _str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_default_model: str     = _str("OPENROUTER_DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    openrouter_fast_model: str        = _str("OPENROUTER_FAST_MODEL", "nvidia/nemotron-nano-9b-v2:free")

    # Two-phase judge mode (GrandCode-style hypothesis → challenge).
    # Only effective when using an API backend — local models are too slow.
    # Set ANALYSER_TWO_PHASE=true to enable when ready.
    two_phase_enabled: bool           = _str("ANALYSER_TWO_PHASE", "false").lower() == "true"


# Module-level singleton — import this everywhere
cfg = AnalyserConfig()
