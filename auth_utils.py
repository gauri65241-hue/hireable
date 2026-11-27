"""
Auth utilities: password hashing (bcrypt) + JWT token creation/verification.
JWT_SECRET env var controls signing key. Falls back to a generated key stored
in data/.jwt_secret so tokens survive restarts without manual configuration.
"""

import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

_ALGORITHM = "HS256"
_TOKEN_EXPIRE_DAYS = 7

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _get_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if secret:
        return secret
    from paths import STATE_DIR
    secret_file = Path(STATE_DIR) / ".jwt_secret"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    if secret_file.exists():
        return secret_file.read_text().strip()
    generated = secrets.token_hex(32)
    secret_file.write_text(generated)
    try:
        secret_file.chmod(0o600)
    except OSError:
        pass
    return generated


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_TOKEN_EXPIRE_DAYS)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, _get_secret(), algorithm=_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[_ALGORITHM])
        return {"user_id": payload["sub"], "email": payload["email"]}
    except JWTError:
        return None


async def get_current_user(token: str = Depends(_oauth2_scheme)) -> dict:
    user = decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user


def _admin_emails() -> set[str]:
    return {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Gate for operational endpoints (benchmarks etc.).

    Admins are listed in ADMIN_EMAILS (comma separated). If the variable is unset
    nobody is an admin and the endpoints are disabled — fail closed.
    """
    if user.get("email", "").lower() not in _admin_emails():
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


def api_keys_match(provided: str, expected: str) -> bool:
    """Constant-time comparison for the optional global API_KEY gate."""
    return hmac.compare_digest(provided.encode(), expected.encode())
