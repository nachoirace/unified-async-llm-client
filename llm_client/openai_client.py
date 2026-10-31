"""Proveedor OpenAI usando el SDK oficial asíncrono (`AsyncOpenAI`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelResponse, TokenUsage


class OpenAIClient(BaseLLMClient):
    provider = "openai"
    api_key_env = "OPENAI_API_KEY"
    sdk = openai

    _client: AsyncOpenAI

    def _build_client(self, api_key: str) -> AsyncOpenAI:
        # max_retries=0: los reintentos los maneja BaseLLMClient, con la misma
        # política para todos los proveedores.
        return AsyncOpenAI(api_key=api_key, timeout=self.config.timeout, max_retries=0)

    def _request_params(self, messages: list[ChatMessage]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        return params

    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        completion = await self._client.chat.completions.create(**self._request_params(messages))
        choice = completion.choices[0]
        usage = completion.usage
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens
            )
            if usage
            else None,
        )

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._request_params(messages), stream=True
        )
        async with stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
