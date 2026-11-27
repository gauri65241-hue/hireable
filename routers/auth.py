"""
Auth endpoints: register, login, me.
Exempt from API key middleware — /auth/* is in _PUBLIC_PATHS in linkedin_api.py.
"""

import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr

import db as run_db
from auth_utils import (
    api_keys_match, hash_password, verify_password, create_access_token, get_current_user,
    _admin_emails,
)
from concurrency import run_blocking
from ratelimit import limiter

router = APIRouter(prefix="/auth")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    # Only needed to register an address listed in ADMIN_EMAILS (see register()).
    bootstrap_token: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# Hash used to burn the same CPU time when the account does not exist, so response time does
# not reveal which emails are registered.
_DUMMY_HASH = hash_password("not-a-real-password")


@router.post("/register")
@limiter.limit("10/hour")
async def register(request: Request, req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    # ADMIN_EMAILS grants admin on an *unverified* email string, so anyone could otherwise
    # pre-register an admin address. Those addresses can only be registered with the
    # out-of-band ADMIN_BOOTSTRAP_TOKEN; with no token configured they are reserved.
    if req.email.lower() in _admin_emails():
        expected = os.getenv("ADMIN_BOOTSTRAP_TOKEN", "")
        if not expected or not api_keys_match(req.bootstrap_token, expected):
            raise HTTPException(status_code=403, detail="This address is reserved.")
    if run_db.get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    user_id = str(uuid.uuid4())
    hashed = await run_blocking(hash_password, req.password, timeout=15.0)
    run_db.create_user(user_id, req.email, hashed)
    return {
        "access_token": create_access_token(user_id, req.email),
        "token_type": "bearer",
        "email": req.email,
    }


@router.post("/login")
@limiter.limit("10/minute")
async def login(request: Request, req: LoginRequest):
    user = run_db.get_user_by_email(req.email)
    # bcrypt is CPU-bound: off the event loop, and always run (dummy hash if no such user).
    ok = await run_blocking(verify_password, req.password,
                            user["hashed_password"] if user else _DUMMY_HASH, timeout=15.0)
    if not user or not ok:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {
        "access_token": create_access_token(user["user_id"], user["email"]),
        "token_type": "bearer",
        "email": user["email"],
    }


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"user_id": current_user["user_id"], "email": current_user["email"]}
