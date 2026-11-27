import asyncio
import os
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from routers.deps import (
    OLLAMA_BASE, BASE_DIR, run_db,
    _analyser_cfg, _analyze_sem,
    compute_learned_weights, get_api_config,
)

router = APIRouter()


@router.get("/health")
async def health_check():
    import httpx as _httpx

    ollama_status = "disconnected"
    ollama_detail = ""
    available_models: list[str] = []
    try:
        resp = await asyncio.to_thread(
            lambda: _httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3.0)
        )
        if resp.status_code == 200:
            available_models = [m["name"] for m in resp.json().get("models", [])]
            ollama_status = "connected"
            ollama_detail = f"{len(available_models)} model(s) available"
        else:
            ollama_status = "error"
            ollama_detail = f"HTTP {resp.status_code}"
    except Exception as e:
        ollama_status = "error"
        ollama_detail = "unreachable"

    # OpenRouter status — read env directly; frozen cfg captures value at import time
    api_cfg = get_api_config()
    _or_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    openrouter_active = bool(_or_key or api_cfg["api_key"])
    openrouter_status = "connected" if openrouter_active else "not_configured"
    openrouter_detail = f"{api_cfg['llm_model']} / {api_cfg['fast_model']}" if openrouter_active else "Set OPENROUTER_API_KEY to enable"

    db_status = "connected"
    db_detail = ""
    try:
        runs = run_db.list_runs(limit=1)
        db_detail = f"{len(runs)} recent run(s)"
    except Exception as e:
        db_status = "error"
        db_detail = "error"

    overall = "healthy" if db_status == "connected" and (ollama_status == "connected" or openrouter_active) else "degraded"

    return {
        "status": overall,
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "concurrency": {
            "max_concurrent_analyses": _analyser_cfg.max_concurrent_analyses,
            "analyses_in_flight": _analyser_cfg.max_concurrent_analyses - _analyze_sem._value,
        },
        "services": {
            "ollama": ollama_status,
            "ollama_detail": ollama_detail,
            "available_models": available_models,
            "openrouter": openrouter_status,
            "openrouter_detail": openrouter_detail,
            "openrouter_model": api_cfg["llm_model"],
            "openrouter_fast_model": api_cfg["fast_model"],
            "database": db_status,
            "database_detail": db_detail,
        }
    }


@router.get("/openrouter-models")
async def list_openrouter_models():
    """Return all OPENROUTER_MODEL_N entries from env, each with its dedicated key."""
    models = []
    i = 1
    while True:
        model = os.getenv(f"OPENROUTER_MODEL_{i}")
        if model is None:
            break
        key = os.getenv(f"OPENROUTER_KEY_{i}", _analyser_cfg.openrouter_api_key)
        models.append({"model": model, "key": key, "index": i})
        i += 1
    return {
        "models": models,
        "default_model": _analyser_cfg.openrouter_default_model,
        "fast_model":    _analyser_cfg.openrouter_fast_model,
        "base_url":      _analyser_cfg.openrouter_base_url,
        "configured":    bool(_analyser_cfg.openrouter_api_key),
    }


@router.get("/ollama-models")
async def list_ollama_models():
    import httpx
    EMBEDDING_MODELS = {"nomic-embed-text", "nomic-embed", "mxbai-embed-large", "all-minilm", "bge-m3"}
    try:
        tags = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3).json()
        models = [
            m["name"] for m in tags.get("models", [])
            if m["name"].split(":")[0] not in EMBEDDING_MODELS
        ]
        return {"models": models, "default": "cogito:14b"}
    except Exception as e:
        return {"models": ["cogito:14b"], "default": "cogito:14b", "error": "model list unavailable"}


@router.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    with open(dashboard_path, encoding="utf-8") as f:
        return f.read()
