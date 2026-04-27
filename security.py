"""
JWT verification reusing the same HS256 secret Spring Boot signs with.
Rejects anything that isn't a SUPERVISOR or ADMIN token because the
insights endpoints are gated to those two roles.
"""
from __future__ import annotations

import os
from typing import Iterable

from fastapi import Header, HTTPException, status
from jose import JWTError, jwt


def _expected_secret() -> bytes:
    """The secret is base64-encoded in `application.properties`; jjwt
    decodes it before signing, so we mirror that here.
    """
    import base64

    raw = os.getenv(
        "JWT_SECRET",
        "ZGV2LXdvcmtmbG93LWVuZ2luZS1zZWNyZXQta2V5LWNoYW5nZS1tZS1pbi1wcm9k",
    )
    return base64.b64decode(raw)


def _expected_issuer() -> str:
    return os.getenv("JWT_ISSUER", "workflow-engine")


def require_role(allowed: Iterable[str]):
    """FastAPI dependency factory: reads the Authorization header,
    validates the JWT and ensures the role claim is in `allowed`.
    """
    allowed_set = {r.upper() for r in allowed}

    async def checker(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            print("[Insights] 401 — missing or non-bearer Authorization header")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        try:
            # Spring's jjwt picks the HMAC variant based on the decoded
            # secret's bit length (HS256 for 256–383 bits, HS384 for
            # 384–511, HS512 for ≥ 512). We accept all three so the
            # sidecar verifies whichever Spring happens to issue.
            claims = jwt.decode(
                token,
                _expected_secret(),
                algorithms=["HS256", "HS384", "HS512"],
                issuer=_expected_issuer(),
            )
        except JWTError as exc:
            # Show the underlying jose error in the FastAPI log so the
            # operator can tell apart "wrong secret", "expired token",
            # "wrong issuer", etc. The HTTP body intentionally stays
            # generic — we don't want to leak details over the wire.
            print(f"[Insights] 401 — JWT decode failed: {exc}")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc
        role = (claims.get("role") or "").upper()
        if role not in allowed_set:
            print(f"[Insights] 403 — role {role!r} not in {sorted(allowed_set)}")
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role {role!r} is not allowed on this endpoint",
            )
        return claims

    return checker
