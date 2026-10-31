"""Modelos Pydantic compartidos por todos los proveedores.

Definir estas estructuras primero evita pasar diccionarios anidados entre capas:
todo lo que entra (mensajes, configuración) y todo lo que sale (respuestas,
fragmentos de streaming, errores) está tipado y validado.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["openai", "anthropic"]


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role=Role.ASSISTANT, content=content)


class ModelConfig(BaseModel):
    """Parámetros de entrada del modelo, validados antes de llamar a la API.

    `temperature` y `top_p` los aplica OpenAI; la API actual de Anthropic no
    acepta parámetros de sampling, así que `AnthropicClient` los ignora.
    """

    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=128_000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    timeout: float = Field(default=60.0, gt=0, description="Segundos por request.")
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0, description="Backoff inicial (s).")


class ErrorType(StrEnum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    SERVER = "server"
    UNKNOWN = "unknown"


class ErrorInfo(BaseModel):
    """Error controlado: lo que se devuelve en lugar de propagar la excepción."""

    type: ErrorType
    message: str
    retryable: bool = False
    status_code: int | None = None
    attempts: int = 1


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Respuesta normalizada, idéntica para cualquier proveedor."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = None
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento emitido por `stream()`. Si falla, el último chunk trae `error`."""

    text: str = ""
    error: ErrorInfo | None = None
