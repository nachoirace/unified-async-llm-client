"""`AsyncLLMManager`: elige el proveedor a partir de la configuración."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Self

from pydantic import ValidationError

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient, Messages
from .errors import LLMConfigurationError
from .openai_client import OpenAIClient
from .schemas import ModelConfig, ModelResponse, Provider, StreamChunk

CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
}


def create_client(config: ModelConfig, api_key: str | None = None) -> BaseLLMClient:
    """Fábrica: devuelve el cliente concreto para `config.provider`."""
    return CLIENTS[config.provider](config, api_key=api_key)


def config_from_env(provider: str | None = None) -> ModelConfig:
    """Arma un `ModelConfig` validado a partir de variables de entorno.

    `provider` tiene prioridad sobre `LLM_PROVIDER`. El modelo se toma de
    `OPENAI_MODEL` / `ANTHROPIC_MODEL` o, si no están, de `DEFAULT_MODELS`.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if provider not in CLIENTS:
        raise LLMConfigurationError(
            f"LLM_PROVIDER inválido: {provider!r} (opciones: {', '.join(CLIENTS)})"
        )

    raw = {
        "provider": provider,
        "model": os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider],
        "temperature": os.getenv("LLM_TEMPERATURE"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS"),
        "timeout": os.getenv("LLM_TIMEOUT"),
        "max_retries": os.getenv("LLM_MAX_RETRIES"),
    }
    try:
        # Pydantic convierte los strings del entorno a float/int y valida rangos.
        return ModelConfig(**{k: v for k, v in raw.items() if v not in (None, "")})
    except ValidationError as exc:
        raise LLMConfigurationError(f"Configuración inválida:\n{exc}") from exc


class AsyncLLMManager:
    """Punto de entrada único: el resto del código no sabe qué proveedor hay detrás.

    Uso:
        async with AsyncLLMManager.from_env() as llm:
            response = await llm.generate("¿Qué es la entropía?")
            async for chunk in llm.stream("¿Qué es la entropía?"):
                print(chunk.text, end="")
    """

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        self.config = config
        self.client = create_client(config, api_key=api_key)

    @classmethod
    def from_env(cls, provider: str | None = None) -> Self:
        return cls(config_from_env(provider))

    @property
    def provider(self) -> Provider:
        return self.config.provider

    async def generate(self, messages: Messages) -> ModelResponse:
        return await self.client.generate(messages)

    async def stream(self, messages: Messages) -> AsyncIterator[StreamChunk]:
        async for chunk in self.client.stream(messages):
            yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
