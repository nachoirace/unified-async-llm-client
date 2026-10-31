"""Traducción de excepciones de los SDKs a errores controlados (`ErrorInfo`)."""

from __future__ import annotations

from types import ModuleType

from .schemas import ErrorInfo, ErrorType


class LLMConfigurationError(Exception):
    """Configuración inválida detectada antes de llamar a la API (p. ej. falta la API key)."""


def classify_sdk_error(exc: BaseException, sdk: ModuleType) -> ErrorInfo | None:
    """Convierte una excepción de `openai` o `anthropic` en un `ErrorInfo`.

    Ambos SDKs exponen la misma jerarquía de excepciones, así que una sola
    función sirve para los dos. Devuelve `None` si la excepción no viene del
    SDK (un bug propio, por ejemplo), para que esos casos no se oculten.
    """
    if not isinstance(exc, sdk.APIError):
        return None

    message = _api_message(exc)
    status = getattr(exc, "status_code", None)

    def info(type_: ErrorType, retryable: bool = False) -> ErrorInfo:
        return ErrorInfo(type=type_, message=message, retryable=retryable, status_code=status)

    # El orden importa: APITimeoutError hereda de APIConnectionError.
    if isinstance(exc, sdk.APITimeoutError):
        return info(ErrorType.TIMEOUT, retryable=True)
    if isinstance(exc, sdk.APIConnectionError):
        return info(ErrorType.CONNECTION, retryable=True)
    if isinstance(exc, sdk.AuthenticationError):
        return info(ErrorType.AUTHENTICATION)
    if isinstance(exc, sdk.PermissionDeniedError):
        return info(ErrorType.PERMISSION)
    if isinstance(exc, sdk.RateLimitError):
        # OpenAI usa 429 tanto para rate limit como para cuota agotada;
        # reintentar una cuota agotada no tiene sentido.
        if getattr(exc, "code", None) == "insufficient_quota":
            return info(ErrorType.QUOTA_EXCEEDED)
        return info(ErrorType.RATE_LIMIT, retryable=True)
    if isinstance(exc, sdk.NotFoundError):
        return info(ErrorType.NOT_FOUND)
    if isinstance(exc, sdk.BadRequestError | sdk.UnprocessableEntityError):
        # Anthropic informa el saldo insuficiente como 400.
        if "credit balance" in message.lower():
            return info(ErrorType.QUOTA_EXCEEDED)
        return info(ErrorType.BAD_REQUEST)
    if isinstance(exc, sdk.APIStatusError) and status is not None and status >= 500:
        # Incluye 529 (overloaded) de Anthropic.
        return info(ErrorType.SERVER, retryable=True)
    return info(ErrorType.UNKNOWN)


def _api_message(exc: BaseException) -> str:
    """Extrae el mensaje legible del body (`{"error": {"message": ...}}`) si existe."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return inner["message"]
    return getattr(exc, "message", None) or str(exc)
