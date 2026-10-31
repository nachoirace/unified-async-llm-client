"""Clase base abstracta con la lógica común: validación, reintentos y errores."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from types import ModuleType
from typing import Any, ClassVar, Self

from .errors import LLMConfigurationError, classify_sdk_error
from .schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse, Provider, StreamChunk

logger = logging.getLogger(__name__)

Messages = str | Sequence[ChatMessage]

MAX_BACKOFF_SECONDS = 30.0


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores.

    Las subclases sólo implementan la llamada "cruda" al SDK (`_generate`,
    `_stream`). `generate()` y `stream()` agregan, para todos por igual:
    normalización de mensajes, reintentos con backoff exponencial y la
    conversión de excepciones en un `ErrorInfo` en lugar de un crash.
    """

    provider: ClassVar[Provider]
    api_key_env: ClassVar[str]
    sdk: ClassVar[ModuleType]

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        if config.provider != self.provider:
            raise LLMConfigurationError(
                f"{type(self).__name__} requiere provider={self.provider!r}, "
                f"recibió {config.provider!r}"
            )
        api_key = api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise LLMConfigurationError(f"Falta la variable de entorno {self.api_key_env}")
        self.config = config
        self._client = self._build_client(api_key)

    # --- Lo que implementa cada proveedor -------------------------------------

    @abstractmethod
    def _build_client(self, api_key: str) -> Any:
        """Crea el cliente asíncrono del SDK oficial."""

    @abstractmethod
    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Una única llamada no streaming. Puede lanzar excepciones del SDK."""

    @abstractmethod
    def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono de fragmentos de texto. Puede lanzar excepciones del SDK."""

    # --- API pública ----------------------------------------------------------

    async def generate(self, messages: Messages) -> ModelResponse:
        """Devuelve la respuesta completa. Nunca lanza errores de la API: los informa en `.error`."""
        msgs = self._normalize(messages)
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._generate(msgs)
                return response.model_copy(update={"latency_ms": _elapsed_ms(start)})
            except Exception as exc:
                error = self._to_error(exc, attempt)
                if self._should_retry(error, attempt):
                    await self._backoff(exc, attempt)
                    continue
                return ModelResponse(
                    provider=self.provider,
                    model=self.config.model,
                    error=error,
                    latency_ms=_elapsed_ms(start),
                )

    async def stream(self, messages: Messages) -> AsyncIterator[StreamChunk]:
        """Emite `StreamChunk`s a medida que llegan los tokens.

        Si algo falla antes del primer token se reintenta; si falla a mitad
        del stream no (se duplicaría texto ya emitido). En ambos casos el
        error llega como último chunk con `.error`, sin lanzar excepción.
        """
        msgs = self._normalize(messages)
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                # aclosing: si quien consume corta el loop, se cierra la conexión.
                async with aclosing(self._stream(msgs)) as texts:
                    async for text in texts:
                        if text:
                            emitted = True
                            yield StreamChunk(text=text)
                return
            except Exception as exc:
                error = self._to_error(exc, attempt)
                if not emitted and self._should_retry(error, attempt):
                    await self._backoff(exc, attempt)
                    continue
                yield StreamChunk(error=error)
                return

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- Helpers internos -----------------------------------------------------

    @staticmethod
    def _normalize(messages: Messages) -> list[ChatMessage]:
        if isinstance(messages, str):
            return [ChatMessage.user(messages)]
        msgs = list(messages)
        if not msgs:
            raise ValueError("Se necesita al menos un mensaje")
        return msgs

    def _to_error(self, exc: Exception, attempt: int) -> ErrorInfo:
        error = classify_sdk_error(exc, self.sdk)
        if error is None:
            # No es un error de la API sino un bug: no lo escondemos.
            raise exc
        logger.warning(
            "%s intento %d/%d falló (%s): %s",
            self.provider, attempt, self.config.max_retries + 1, error.type, error.message,
        )
        return error.model_copy(update={"attempts": attempt})

    def _should_retry(self, error: ErrorInfo, attempt: int) -> bool:
        return error.retryable and attempt <= self.config.max_retries

    async def _backoff(self, exc: Exception, attempt: int) -> None:
        delay = _retry_after(exc)
        if delay is None:
            # Backoff exponencial con jitter: 1s, 2s, 4s... ± aleatorio.
            delay = self.config.retry_base_delay * 2 ** (attempt - 1)
            delay += random.uniform(0, self.config.retry_base_delay)
        await asyncio.sleep(min(delay, MAX_BACKOFF_SECONDS))


def _retry_after(exc: Exception) -> float | None:
    """Lee el header `Retry-After` (segundos) si la API lo envió."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after", ""))
    except ValueError:
        return None


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
