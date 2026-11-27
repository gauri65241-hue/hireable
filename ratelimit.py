"""
Shared rate limiter.

Key = authenticated user id when a valid JWT is presented, otherwise the client IP.

Behind a reverse proxy (Render, nginx) the socket peer is the proxy, so every
visitor would share one bucket. Set TRUSTED_PROXY_COUNT to the number of proxies
you control in front of the app (Render: 1) and the client IP is taken from the
matching position of X-Forwarded-For (counted from the right, which is the part a
client cannot forge). With the default of 0 the header is ignored entirely.
"""
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_ip(request: Request) -> str:
    try:
        n = int(os.getenv("TRUSTED_PROXY_COUNT", "0"))
    except ValueError:
        n = 0
    if n > 0:
        xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(xff) >= n:
            return xff[-n]
    return get_remote_address(request)


def rate_limit_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        from auth_utils import decode_token  # local import: avoids import cycle at startup
        user = decode_token(auth[7:].strip())
        if user:
            return f"user:{user['user_id']}"
    return f"ip:{_client_ip(request)}"


limiter = Limiter(key_func=rate_limit_key)
