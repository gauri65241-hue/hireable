"""
Auth endpoints: register, login, me.
Exempt from API key middleware — /auth/* is in _PUBLIC_PATHS in linkedin_api.py.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

import db as run_db
from auth_utils import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
async def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if run_db.get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    user_id = str(uuid.uuid4())
    run_db.create_user(user_id, req.email, hash_password(req.password))
    return {
        "access_token": create_access_token(user_id, req.email),
        "token_type": "bearer",
        "email": req.email,
    }


@router.post("/login")
async def login(req: LoginRequest):
    user = run_db.get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {
        "access_token": create_access_token(user["user_id"], user["email"]),
        "token_type": "bearer",
        "email": user["email"],
    }


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"user_id": current_user["user_id"], "email": current_user["email"]}
