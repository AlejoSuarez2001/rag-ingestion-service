import logging
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError, PyJWKClient

from rag_ingestion.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=8)
def _get_jwk_client(jwks_uri: str) -> PyJWKClient:
    return PyJWKClient(jwks_uri)


def _ensure_auth_config(settings: Settings) -> None:
    missing = [
        name
        for name, value in (
            ("auth_certs", settings.auth_certs),
            ("auth_server_issuer", settings.auth_server_issuer),
            ("keycloak_clientid", settings.keycloak_clientid),
        )
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Configuración de autenticación incompleta. "
                f"Faltan: {', '.join(missing)}."
            ),
        )


async def get_current_token_payload(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación no encontrado.",
        )

    _ensure_auth_config(settings)
    token = credentials.credentials

    try:
        signing_key = _get_jwk_client(settings.auth_certs).get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "RS384", "RS512"],
            issuer=settings.auth_server_issuer,
            options={"verify_aud": False},
        )
        return payload
    except InvalidTokenError as exc:
        logger.warning("JWT inválido recibido: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token JWT inválido.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error al validar token JWT")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al validar token JWT: {exc}",
        ) from exc


def require_client_roles(*allowed_roles: str):
    async def dependency(
        payload: dict[str, Any] = Depends(get_current_token_payload),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        client_roles = (
            payload.get("resource_access", {})
            .get(settings.keycloak_clientid, {})
            .get("roles", [])
        )

        if not any(role in client_roles for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permisos para acceder a este recurso.",
            )

        return payload

    return dependency


require_ingestion_role = require_client_roles("ROLE_RAG_INGESTION_COMPLETA")
