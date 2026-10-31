# Unified Async LLM Client

Cliente asíncrono unificado para **OpenAI** y **Anthropic** en Python 3.12: una sola
interfaz para generar respuestas completas o en streaming, con validación Pydantic,
reintentos automáticos y errores controlados en lugar de crashes.

```python
async with AsyncLLMManager.from_env() as llm:          # proveedor según LLM_PROVIDER
    response = await llm.generate("¿Qué es la entropía?")
    print(response.content if response.ok else response.error)

    async for chunk in llm.stream("¿Qué es la entropía?"):
        print(chunk.text, end="", flush=True)
```

## Estructura

```
├── llm_client/
│   ├── schemas.py           # Pydantic: ChatMessage, ModelConfig, ModelResponse, StreamChunk, ErrorInfo
│   ├── base.py              # BaseLLMClient (ABC): generate(), stream(), reintentos, manejo de errores
│   ├── openai_client.py     # OpenAIClient    → AsyncOpenAI
│   ├── anthropic_client.py  # AnthropicClient → AsyncAnthropic
│   ├── manager.py           # AsyncLLMManager + config_from_env() + create_client()
│   └── errors.py            # excepciones de los SDKs → ErrorInfo
├── tests/test_clients.py    # tests sin red (APIs simuladas)
├── main.py                  # script de validación: modo normal + streaming
├── .env.example
└── requirements.txt
```

## Instalación

Requiere **Python 3.12+**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y completá tus API keys
```

## Variables de entorno

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `OPENAI_API_KEY` | si usás OpenAI | – | API key de OpenAI |
| `ANTHROPIC_API_KEY` | si usás Anthropic | – | API key de Anthropic |
| `LLM_PROVIDER` | no | `openai` | Proveedor que carga `AsyncLLMManager.from_env()`: `openai` o `anthropic` |
| `OPENAI_MODEL` | no | `gpt-4.1-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | no | `claude-haiku-5-5` | Modelo de Anthropic |
| `LLM_TEMPERATURE` | no | `0.7` | Entre 0 y 2 (ver nota sobre Anthropic) |
| `LLM_MAX_TOKENS` | no | `1024` | Máximo de tokens de salida (> 0) |
| `LLM_TIMEOUT` | no | `60` | Timeout por request, en segundos |
| `LLM_MAX_RETRIES` | no | `3` | Reintentos ante errores transitorios (0–10) |

Los valores se validan con Pydantic al arrancar: una temperatura de `5` o un
`max_tokens` negativo se rechazan con un mensaje claro antes de llamar a la API.

## Ejecutar el script de prueba

`main.py` pregunta *"¿Qué es la entropía?"* en modo normal y en streaming.

```bash
python main.py                         # proveedor de LLM_PROVIDER
python main.py --provider anthropic    # forzar un proveedor
python main.py --provider all          # ambos (el modo normal corre en paralelo)
python main.py --provider all -v       # además muestra los reintentos en el log
```

Si falta la key de un proveedor, se omite con un aviso y se sigue con el otro.

## Tests

No usan red ni API keys: simulan las respuestas HTTP (incluido el SSE del
streaming) y cubren validación, streaming, reintentos y manejo de errores.

```bash
pip install -r requirements-dev.txt
pytest
```

## Decisiones de diseño

**Interfaz común.** `BaseLLMClient` define `generate()` y `stream()` una sola vez.
Cada proveedor implementa sólo `_generate()` y `_stream()` (la llamada cruda al SDK),
así que reintentos y manejo de errores se comportan igual en OpenAI y Anthropic.
`AsyncLLMManager` elige la clase concreta a partir de `ModelConfig.provider`.

**Sin bloquear el event loop.** Se usan exclusivamente `AsyncOpenAI` y
`AsyncAnthropic` con `await`. En `main.py` las dos llamadas en modo normal se
lanzan con `asyncio.gather` y corren a la vez.

**Streaming.** `stream()` es un generador asíncrono: recorre el stream del SDK con
`async for` y hace `yield` de cada fragmento como `StreamChunk(text=...)`. Si el
consumidor corta el loop antes de terminar, la conexión se cierra (`aclosing`).

**Errores controlados.** Las excepciones de los SDKs se traducen a un `ErrorInfo`
tipado (`authentication`, `rate_limit`, `quota_exceeded`, `timeout`, `connection`,
`server`, ...). `generate()` devuelve un `ModelResponse` con `.error` y `.ok == False`;
`stream()` emite un último `StreamChunk` con `.error`. Ninguno lanza la excepción.

**Reintentos.** Rate limit (429), timeouts, errores de red y 5xx (incluido el 529
*overloaded* de Anthropic) se reintentan con backoff exponencial + jitter,
respetando el header `Retry-After` si viene. Errores permanentes (API key inválida,
cuota agotada, request mal formado) se devuelven al primer intento. En streaming
sólo se reintenta si todavía no se emitió ningún token, para no duplicar texto.
Los reintentos internos de los SDKs están desactivados (`max_retries=0`) para que
haya una sola política.

**Mensajes de sistema.** OpenAI los recibe dentro de `messages`; Anthropic como
parámetro `system` aparte. `AnthropicClient` hace esa conversión.

**Nota sobre Anthropic y `temperature`.** La Messages API actual de Anthropic
(SDK `anthropic` 2.x) ya no acepta parámetros de sampling (`temperature`, `top_p`).
`ModelConfig` los sigue validando y OpenAI los aplica, pero `AnthropicClient`
no los envía.
