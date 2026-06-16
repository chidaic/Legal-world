from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash


class AuthError(RuntimeError):
    """Raised when auth credentials or tokens are invalid."""


_PASSWORD_HASHER = PasswordHash.recommended()
_JWT_ALGORITHM = "HS256"


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        raise AuthError("JWT_SECRET environment variable is required")
    return secret


def get_jwt_expiry_minutes() -> int:
    raw_value = os.getenv("JWT_EXPIRES_MINUTES", "60").strip()
    try:
        expiry_minutes = int(raw_value)
    except ValueError as exc:
        raise AuthError("JWT_EXPIRES_MINUTES must be an integer") from exc
    if expiry_minutes <= 0:
        raise AuthError("JWT_EXPIRES_MINUTES must be positive")
    return expiry_minutes


def hash_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password, password_hash)
    except Exception:
        return False


def create_access_token(*, user_id: str, token_version: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "token_version": token_version,
        "iat": now,
        "exp": now + timedelta(minutes=get_jwt_expiry_minutes()),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=_JWT_ALGORITHM)


def get_access_token_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=get_jwt_expiry_minutes())


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[_JWT_ALGORITHM])
    except InvalidTokenError as exc:
        raise AuthError("Invalid access token") from exc

    user_id = str(payload.get("user_id") or "").strip()
    token_version = payload.get("token_version")
    if not user_id or not isinstance(token_version, int):
        raise AuthError("Access token is missing required claims")

    return {
        "user_id": user_id,
        "token_version": token_version,
    }
