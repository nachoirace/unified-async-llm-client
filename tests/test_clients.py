"""Tests sin red: las APIs se simulan con `httpx.MockTransport`.

Así se ejercita el SDK real (parseo de respuestas, SSE, excepciones) sin
gastar tokens ni necesitar API keys.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

try:  # Las versiones nuevas de los SDKs usan httpx2; las anteriores, httpx.
    import httpx2 as httpx
except ImportError:
    import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    ChatMessage,
    ErrorType,
    LLMConfigurationError,
    ModelConfig,
    OpenAIClient,
)

Handler = Callable[[httpx.Request], httpx.Response]


def openai_client(handler: Handler, **config) -> OpenAIClient:
    client = OpenAIClient(
        ModelConfig(provider="openai", model="gpt-test", retry_base_delay=0, **config),
        api_key="sk-test",
    )
    client._client = AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client


def anthropic_client(handler: Handler, **config) -> AnthropicClient:
    client = AnthropicClient(
        ModelConfig(provider="anthropic", model="claude-test", retry_base_delay=0, **config),
        api_key="sk-ant-test",
    )
    client._client = AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client


def sse(events: list[tuple[str | None, dict | str]]) -> httpx.Response:
    lines = []
    for event, data in events:
        if event:
            lines.append(f"event: {event}")
        lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}\n")
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content="\n".join(lines) + "\n"
    )


OPENAI_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-test",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "La entropía mide el desorden."},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
}


def openai_chunk(content: str | None) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


# --- Schemas -------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_out_of_range_is_rejected(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_max_tokens_must_be_positive(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai", model="m", max_tokens=max_tokens)


def test_invalid_role_and_empty_content_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hola")
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


# --- Manager -------------------------------------------------------------------


def test_manager_picks_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    manager = AsyncLLMManager.from_env()
    assert isinstance(manager.client, AnthropicClient)
    assert manager.config.temperature == 0.3


def test_manager_missing_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        AsyncLLMManager.from_env("openai")


def test_manager_invalid_env_value_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "5")
    with pytest.raises(LLMConfigurationError, match="temperature"):
        AsyncLLMManager.from_env("openai")


# --- OpenAI --------------------------------------------------------------------


async def test_openai_generate() -> None:
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=OPENAI_COMPLETION)

    response = await openai_client(handler, temperature=0.2).generate("¿Qué es la entropía?")

    assert response.ok
    assert response.content == "La entropía mide el desorden."
    assert response.usage.total_tokens == 16
    assert sent["temperature"] == 0.2
    assert sent["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]


async def test_openai_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse(
            [(None, openai_chunk(t)) for t in ["La ", "entropía", None]] + [(None, "[DONE]")]
        )

    chunks = [c async for c in openai_client(handler).stream("hola")]
    assert "".join(c.text for c in chunks) == "La entropía"
    assert all(c.error is None for c in chunks)


async def test_rate_limit_is_retried_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=OPENAI_COMPLETION)

    response = await openai_client(handler, max_retries=3).generate("hola")
    assert response.ok
    assert calls == 3


async def test_rate_limit_exhausts_retries_without_crashing() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    response = await openai_client(handler, max_retries=2).generate("hola")
    assert not response.ok
    assert response.error.type is ErrorType.RATE_LIMIT
    assert response.error.attempts == 3
    assert calls == 3


async def test_invalid_api_key_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "Incorrect API key"}})

    response = await openai_client(handler).generate("hola")
    assert response.error.type is ErrorType.AUTHENTICATION
    assert response.error.status_code == 401
    assert calls == 1


async def test_insufficient_quota_is_not_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"message": "quota", "code": "insufficient_quota"}}
        )

    response = await openai_client(handler).generate("hola")
    assert response.error.type is ErrorType.QUOTA_EXCEEDED
    assert response.error.attempts == 1


async def test_network_error_in_stream_becomes_error_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red")

    chunks = [c async for c in openai_client(handler, max_retries=1).stream("hola")]
    assert len(chunks) == 1
    assert chunks[0].error.type is ErrorType.CONNECTION
    assert chunks[0].error.attempts == 2


# --- Anthropic -----------------------------------------------------------------


async def test_anthropic_generate_sends_system_separately() -> None:
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "Es una medida del desorden."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 7},
            },
        )

    messages = [ChatMessage.system("Sé breve."), ChatMessage.user("¿Qué es la entropía?")]
    response = await anthropic_client(handler).generate(messages)

    assert response.content == "Es una medida del desorden."
    assert response.finish_reason == "end_turn"
    assert sent["system"] == "Sé breve."
    assert sent["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert "temperature" not in sent


async def test_anthropic_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        delta = lambda t: {  # noqa: E731
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": t},
        }
        return sse(
            [
                ("message_start", {"type": "message_start", "message": message}),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                ("content_block_delta", delta("Es ")),
                ("content_block_delta", delta("desorden.")),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 3},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )

    chunks = [c async for c in anthropic_client(handler).stream("hola")]
    assert "".join(c.text for c in chunks) == "Es desorden."


async def test_anthropic_overloaded_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )

    response = await anthropic_client(handler, max_retries=2).generate("hola")
    assert response.error.type is ErrorType.SERVER
    assert calls == 3
