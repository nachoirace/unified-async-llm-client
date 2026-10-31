"""Script de validación: pregunta "¿Qué es la entropía?" en modo normal y en streaming.

    python main.py                       # usa LLM_PROVIDER del .env
    python main.py --provider anthropic
    python main.py --provider all        # ambos proveedores, en paralelo
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, LLMConfigurationError, ModelResponse

QUESTION = "¿Qué es la entropía?"
MESSAGES = [
    ChatMessage.system("Respondé en español, en un párrafo breve (máximo 80 palabras)."),
    ChatMessage.user(QUESTION),
]


def print_response(response: ModelResponse) -> None:
    if not response.ok:
        err = response.error
        print(f"  ✗ Error controlado [{err.type}] tras {err.attempts} intento(s): {err.message}")
        return
    print(f"  {response.content}\n")
    usage = response.usage
    tokens = f"{usage.input_tokens} in / {usage.output_tokens} out" if usage else "n/d"
    print(f"  ({response.model} · {response.latency_ms} ms · tokens: {tokens})")


async def run_normal(llm: AsyncLLMManager) -> ModelResponse:
    return await llm.generate(MESSAGES)


async def run_stream(llm: AsyncLLMManager) -> None:
    print(f"\n── [{llm.provider}] modo streaming ──")
    print("  ", end="", flush=True)
    async for chunk in llm.stream(MESSAGES):
        if chunk.error:
            print(f"\n  ✗ Error controlado [{chunk.error.type}]: {chunk.error.message}")
            return
        print(chunk.text, end="", flush=True)
    print()


def build_managers(providers: list[str]) -> list[AsyncLLMManager]:
    managers = []
    for provider in providers:
        try:
            managers.append(AsyncLLMManager.from_env(provider))
        except LLMConfigurationError as exc:
            # Falta una key o hay un valor inválido: avisamos y seguimos con el resto.
            print(f"⚠ [{provider}] omitido: {exc}")
    return managers


async def main(provider: str) -> int:
    providers = ["openai", "anthropic"] if provider == "all" else [provider]
    managers = build_managers(providers)
    if not managers:
        print("No hay ningún proveedor configurado. Revisá tu .env (ver .env.example).")
        return 1

    print(f"Pregunta: {QUESTION}")
    try:
        # Modo normal: con varios proveedores las llamadas corren en paralelo,
        # porque ninguna bloquea el event loop.
        responses = await asyncio.gather(*(run_normal(m) for m in managers))
        for llm, response in zip(managers, responses, strict=True):
            print(f"\n── [{llm.provider}] modo normal ──")
            print_response(response)

        # Streaming: secuencial, para que los tokens no se mezclen en la consola.
        for llm in managers:
            await run_stream(llm)
    finally:
        await asyncio.gather(*(m.aclose() for m in managers))

    return 0


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "all"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="Proveedor a probar (por defecto: LLM_PROVIDER o 'openai').",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra los reintentos.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main(args.provider)))
