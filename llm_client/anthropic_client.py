"""Proveedor Anthropic usando el SDK oficial asíncrono (`AsyncAnthropic`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelResponse, Role, TokenUsage


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    sdk = anthropic

    _client: AsyncAnthropic

    def _build_client(self, api_key: str) -> AsyncAnthropic:
        # max_retries=0: los reintentos los maneja BaseLLMClient.
        return AsyncAnthropic(api_key=api_key, timeout=self.config.timeout, max_retries=0)

    def _request_params(self, messages: list[ChatMessage]) -> dict[str, Any]:
        # A diferencia de OpenAI, Anthropic recibe el prompt de sistema como
        # parámetro aparte y no dentro de la lista de mensajes.
        system = "\n\n".join(m.content for m in messages if m.role is Role.SYSTEM)
        chat = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role is not Role.SYSTEM
        ]
        if not chat:
            raise ValueError("Anthropic requiere al menos un mensaje de user")

        # temperature y top_p no se envían: la Messages API actual de Anthropic
        # (SDK >= 2.x) ya no acepta parámetros de sampling.
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": chat,
            "max_tokens": self.config.max_tokens,
        }
        if system:
            params["system"] = system
        return params

    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        message = await self._client.messages.create(**self._request_params(messages))
        return ModelResponse(
            provider=self.provider,
            model=message.model,
            content="".join(block.text for block in message.content if block.type == "text"),
            finish_reason=message.stop_reason,
            usage=TokenUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._request_params(messages)) as stream:
            async for text in stream.text_stream:
                yield text
