"""
Agent / supervisor / classifier LLM calls.

Uses :class:`OllamaClient` from ``llm_client.py``:
- When ``LLM_BASE_URL`` is set → OpenAI-compatible ``/v1/chat/completions`` (Modal, vLLM, etc.)
- Otherwise → legacy local Ollama ``/api/generate`` (``OLLAMA_BASE_URL``).

Do not use ``langchain_ollama.ChatOllama`` here — it only spoke the Ollama wire protocol.
"""

from __future__ import annotations

from core.generation.llm_client import OllamaClient
from config.settings import settings

_client: OllamaClient | None = None


def get_agent_llm_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def agent_llm_complete(
    user: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """
    Single-turn completion for JSON / extraction prompts.
    ``enable_thinking=False`` on Ollama path to keep output parseable.
    """
    c = get_agent_llm_client()
    mt = max_tokens if max_tokens is not None else settings.llm_max_tokens
    out = c.generate(
        prompt=user,
        system=system,
        temperature=temperature,
        max_tokens=mt,
        include_reasoning=False,
        enable_thinking=False,
    )
    return (out.content or "").strip()
