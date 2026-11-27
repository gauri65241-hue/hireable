"""
Shared dependencies: constants, singletons, and utility functions
imported by all routers. Import from here — never from linkedin_api.py.
"""
import asyncio
import os
import re
import json
import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from analysers import AnalyserEngine, calculate_overall, DEFAULT_WEIGHTS, sanitize_user_content
from analysers.config import cfg as _analyser_cfg
from analysers.skills.analyser     import score as calculate_skills_score
from analysers.experience.analyser import score as calculate_experience_score
from analysers.seniority.analyser  import score as calculate_seniority_score
from analysers.relevancy.analyser  import score as calculate_relevancy_score
from analysers.impact.analyser     import score as calculate_impact_score
from analysers.weights import (
    compute_learned_weights,
    append_feedback,
    INTERVIEW_ANALYSERS as INTERVIEW_JUDGES,
    JOB_ANALYSERS as JOB_JUDGES,
)

from ats_checker import score_ats, ats_to_dict
import db as run_db
from auth_utils import get_current_user

calculate_overall_match = calculate_overall

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA_BASE = _analyser_cfg.ollama_base_url   # set OLLAMA_BASE_URL env var to override

judge_engine  = AnalyserEngine()
_analyze_sem  = asyncio.Semaphore(_analyser_cfg.max_concurrent_analyses)


def make_cache_key(user_id: str, resume_text: str, job_description: str,
                   model: str, weights: dict | None) -> str:
    """Everything that can change a score must be in the key, otherwise a stale result
    is served after a model / weights / prompt / pipeline change."""
    import hashlib
    from analysers.engine import PIPELINE_VERSION
    sig = json.dumps(weights or {}, sort_keys=True)
    payload = "\x1f".join([PIPELINE_VERSION, user_id, model or "", sig, resume_text, job_description])
    return hashlib.sha256(payload.encode()).hexdigest()


def get_llm_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def _get_model_key(model_name: str) -> str:
    """Return dedicated key for model_name from OPENROUTER_MODEL_N / OPENROUTER_KEY_N pairs."""
    i = 1
    while True:
        m = os.getenv(f"OPENROUTER_MODEL_{i}")
        if m is None:
            break
        if m == model_name:
            return os.getenv(f"OPENROUTER_KEY_{i}", "")
        i += 1
    return ""


def _is_local_ollama_host(url: str) -> bool:
    """Exact-hostname match. (The old substring test also matched http://evil.com/localhost.)"""
    from urllib.parse import urlsplit
    try:
        return (urlsplit(url or "").hostname or "") in {"localhost", "127.0.0.1", "::1"}
    except ValueError:
        return False


def _validate_custom_base_url(url: str) -> str:
    """A caller-supplied LLM base URL makes this server POST resume text to that URL.
    Require https + a public address, and optionally an explicit host allow-list
    (ALLOWED_API_HOSTS=openrouter.ai,api.openai.com)."""
    from urllib.parse import urlsplit
    from fastapi import HTTPException
    from net_safety import UnsafeURLError, validate_public_url
    allow = {h.strip().lower() for h in os.getenv("ALLOWED_API_HOSTS", "").split(",") if h.strip()}
    try:
        host = (urlsplit(url).hostname or "").lower()
        if allow and host not in allow:
            raise UnsafeURLError("api_base_url host is not on the allow-list")
        return validate_public_url(url, allowed_schemes=("https",))
    except UnsafeURLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid api_base_url: {e}")


def get_api_config(request_api_key: str = "", request_api_base_url: str = "",
                   model: str = "") -> dict:
    """
    Resolve api_key, api_base_url, llm_model, fast_model for a request.

    Priority:
      1. Caller-supplied credentials — used as-is.
      2. OPENROUTER_API_KEY in env — auto-route to OpenRouter with per-model key lookup.
      3. Fall back to local Ollama with no key.
    """
    _local_ollama = not request_api_key and _is_local_ollama_host(request_api_base_url)
    if (request_api_key or request_api_base_url) and not _local_ollama:
        if request_api_base_url:
            request_api_base_url = _validate_custom_base_url(request_api_base_url)
        return {
            "api_key":      request_api_key,
            "api_base_url": request_api_base_url,
            "llm_model":    model or _analyser_cfg.default_model,
            "fast_model":   _analyser_cfg.fast_model,
        }

    if _analyser_cfg.openrouter_api_key:
        llm_model  = _analyser_cfg.openrouter_default_model
        fast_model = _analyser_cfg.openrouter_fast_model
        api_key    = _get_model_key(llm_model) or _analyser_cfg.openrouter_api_key
        return {
            "api_key":      api_key,
            "api_base_url": _analyser_cfg.openrouter_base_url,
            "llm_model":    llm_model,
            "fast_model":   fast_model,
        }

    return {
        "api_key":      "",
        "api_base_url": "",
        "llm_model":    model or _analyser_cfg.default_model,
        "fast_model":   _analyser_cfg.fast_model,
    }


def strip_html(text: str) -> str:
    if not text:
        return text
    match = re.search(r'"description":\s*"((?:[^\\"]|\\.)*)"', text, re.DOTALL)
    if match:
        try:
            extracted = json.loads('"' + match.group(1) + '"').strip()
            if len(extracted) > 100:
                return extracted
        except Exception:
            pass
    text = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = (text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"'))
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 6:
            continue
        if re.search(r'(function\s*\(|window\.|NREUM|gtm\.js|\.push\s*\(|var\s+[a-zA-Z_$]|document\.|__cf_|dataLayer)', line):
            continue
        if re.match(r'^[{}\[\];:,\s\d.%#]+$', line):
            continue
        clean_lines.append(line)
    result = ' '.join(clean_lines).strip()
    return result if len(result) > 50 else text


def get_free_model() -> str:
    import httpx
    try:
        busy = set()
        ps = httpx.get(f"{OLLAMA_BASE}/api/ps", timeout=3).json()
        for m in ps.get("models", []):
            busy.add(m["name"].split(":")[0])

        EMBEDDING_MODELS = {"nomic-embed-text", "nomic-embed", "mxbai-embed-large", "all-minilm", "bge-m3"}
        tags = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3).json()
        for m in tags.get("models", []):
            name = m["name"]
            base = name.split(":")[0]
            if base in EMBEDDING_MODELS:
                continue
            if base not in busy:
                return name

        models = tags.get("models", [])
        if models:
            return models[0]["name"]
    except Exception as e:
        logger.warning("get_free_model failed: %s", e)
    return "deepseek-r1:14b"


async def execute_llm_call(system_or_prompt: str, prompt_or_model: str = None, model: str = None) -> dict:
    """Legacy wrapper — supports both 2-arg (prompt, model) and 3-arg (system, prompt, model) forms."""
    if model is not None:
        system_prompt = system_or_prompt
        prompt = prompt_or_model
    else:
        system_prompt = ""
        prompt = system_or_prompt
        model = prompt_or_model

    try:
        from analysers.base import call_llm_async
        result = await call_llm_async(system_prompt, prompt, model, temperature=0.1)
        if not result["success"]:
            fallback = get_free_model()
            if fallback != model:
                result = await call_llm_async("", prompt, fallback, temperature=0.1)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
