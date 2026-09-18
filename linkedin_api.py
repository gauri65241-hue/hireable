import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routers.deps import get_llm_api_key, run_db
from routers import admin, data, analysis, career, tools, benchmark, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting hireable API...")
    if not get_llm_api_key():
        print("WARNING: No LLM API Key detected in .env file.")
    _key = os.getenv("API_KEY", "")
    if _key:
        print("Auth: API key protection ENABLED (X-API-Key header required).")
    else:
        print("Auth: DISABLED — set API_KEY env var to enable.")
    run_db.init_db()
    print("SQLite DB initialised.")
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="hireable API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]
if "*" in _allowed_origins:
    import warnings
    warnings.warn(
        "ALLOWED_ORIGINS=* with allow_credentials=True is unsafe in production. "
        "Set ALLOWED_ORIGINS to your frontend domain.",
        stacklevel=2,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths that must remain public so the dashboard page itself can load.
_PUBLIC_PATHS = {"/", "/favicon.ico"}


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith("/auth/")

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    api_key = os.getenv("API_KEY", "")
    # Auth disabled or public path — pass through.
    if not api_key or _is_public(request.url.path):
        return await call_next(request)
    provided = (
        request.headers.get("X-API-Key") or
        request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    if not provided:
        return JSONResponse(
            status_code=401,
            content={"detail": "API key required. Send X-API-Key header."},
        )
    if provided != api_key:
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid API key."},
        )
    return await call_next(request)

# Minimal SVG favicon — silences browser 404 noise
_FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#6366f1"/>'
    b'<text x="16" y="23" font-size="20" text-anchor="middle" fill="white" '
    b'font-family="sans-serif" font-weight="bold">h</text>'
    b'</svg>'
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(data.router)
app.include_router(analysis.router)
app.include_router(career.router)
app.include_router(tools.router)
app.include_router(benchmark.router)

# Serve React SPA from frontend_dist/ — must come after all API routers
_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend_dist")
if os.path.isdir(_FRONTEND):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # Serve known static files from frontend_dist root
        candidate = os.path.join(_FRONTEND, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        # All other paths → SPA index.html (client-side routing)
        # no-cache so browsers always refetch the HTML entry point (hashed assets are fine to cache)
        return FileResponse(
            os.path.join(_FRONTEND, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
