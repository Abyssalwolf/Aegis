"""
LLM client: Ollama /api/generate and optional OpenAI-compatible HTTP API.

Thinking models often ignore or mishandle `think: false`. We use **one** setting
(`OLLAMA_ENABLE_THINKING`) for the Ollama `think` flag on every call, then **always**
split JSON `thinking` + `</think>` blocks into content vs reasoning.

- **Answers** (`include_reasoning=True`): return both `content` and `reasoning` (UI).
- **Query rewrite**: `include_reasoning=False`, optional `enable_thinking=False` on Ollama;
  rewrites use strict JSON in `content`. Empty `content` is still back-filled from
  `reasoning_content` / `thinking` when the provider splits fields.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LLMOutput:
    """Visible answer text and optional model reasoning (chain-of-thought)."""

    content: str
    reasoning: Optional[str] = None


def split_reasoning_from_tags(text: str) -> tuple[Optional[str], str]:
    """
    Split Qwen-style `</think>` blocks from the rest of the string.
    Returns (reasoning_or_none, content).
    """
    if not text or "</think>" not in text:
        return None, text
    reasoning_parts: list[str] = []
    out_parts: list[str] = []
    pos = 0
    for m in _THINK_BLOCK.finditer(text):
        reasoning_parts.append(m.group(1).strip())
        out_parts.append(text[pos : m.start()])
        pos = m.end()
    out_parts.append(text[pos:])
    reasoning = "\n\n".join(p for p in reasoning_parts if p) or None
    content = "".join(out_parts).strip()
    return reasoning, content


def strip_all_thinking_tags(text: str) -> str:
    """Remove every `</think>` block (for rewrite path safety)."""
    return _THINK_BLOCK.sub("", text).strip()


def _normalize_chat_content(val: object) -> str:
    """OpenAI-style message.content may be a string or a list of {type, text} parts."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        parts: list[str] = []
        for block in val:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif "text" in block and block["text"]:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(val).strip()


def _first_openai_reasoning_aux(msg: dict, choice: dict) -> str | None:
    """Collect chain-of-thought / auxiliary text from common provider shapes."""
    chunks: list[str] = []
    for key in (
        "reasoning_content",
        "reasoning",
        "thinking",
        "thought",
        "analysis",
        "model_thought",
    ):
        s = _normalize_chat_content(msg.get(key))
        if s:
            chunks.append(s)
    for key in ("reasoning_content", "reasoning", "thinking"):
        s = _normalize_chat_content(choice.get(key))
        if s:
            chunks.append(s)
    if not chunks:
        return None
    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "\n\n".join(out) if out else None


class OllamaClient:

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        if (settings.llm_base_url or "").strip():
            self._mode = "openai"
            self.base_url = (base_url or settings.llm_base_url).rstrip("/")
            self.model = (model or settings.llm_model or settings.ollama_model).strip()
            self._api_key = (settings.llm_api_key or "").strip()
        else:
            self._mode = "ollama"
            self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
            self.model = model or settings.ollama_model
            self._api_key = ""

    def _auth_headers(self) -> dict[str, str]:
        if self._mode == "openai" and self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int | None = None,
        *,
        include_reasoning: bool = False,
        enable_thinking: bool | None = None,
    ) -> LLMOutput:
        """
        Non-streaming generation.

        `include_reasoning` controls whether **reasoning** is returned.

        `enable_thinking` overrides `OLLAMA_ENABLE_THINKING` for this call only
        (Ollama `/api/generate` `think` flag). Use `False` for query rewrite to
        avoid chain-of-thought leaking into the completion. Ignored for OpenAI mode.
        """
        mt = max_tokens if max_tokens is not None else settings.llm_max_tokens
        think_flag: bool | None = enable_thinking
        if self._mode == "openai":
            return self._generate_openai(prompt, system, temperature, mt, include_reasoning)
        return self._generate_ollama(
            prompt, system, temperature, mt, include_reasoning, think_override=think_flag
        )

    def _generate_openai(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        include_reasoning: bool,
    ) -> LLMOutput:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._auth_headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            if not isinstance(msg, dict):
                msg = {}
            raw = _normalize_chat_content(msg.get("content"))
            if not raw:
                raw = _normalize_chat_content(choice.get("text"))
            extra = _first_openai_reasoning_aux(msg, choice)
            tag_r, c_from_tags = split_reasoning_from_tags(raw)
            reasoning = (extra or tag_r) or None
            content = strip_all_thinking_tags(raw)
            if not content.strip() and c_from_tags.strip():
                content = c_from_tags.strip()
            # vLLM / Modal / etc.: assistant text sometimes only in reasoning_content.
            if not include_reasoning and not content.strip() and extra:
                content = strip_all_thinking_tags(extra).strip()
                if not content.strip():
                    _, tail = split_reasoning_from_tags(extra)
                    content = tail.strip()
            # include_reasoning=True: still surface answer text in .content when providers
            # only stream "thinking" fields (rewriter merges both, RAG UI uses content).
            if include_reasoning and not content.strip() and reasoning:
                content = strip_all_thinking_tags(reasoning).strip()
                if not content.strip():
                    _, tail = split_reasoning_from_tags(reasoning)
                    content = tail.strip()
            if not content.strip() and not (reasoning or "").strip():
                logger.warning(
                    "OpenAI chat completion: empty assistant text "
                    "(finish_reason=%s, message_keys=%s). "
                    "For rewrites: raise QUERY_REWRITE_MAX_TOKENS or set OLLAMA_ENABLE_THINKING=false.",
                    choice.get("finish_reason"),
                    list(msg.keys()) if msg else None,
                )
            if include_reasoning:
                return LLMOutput(content=content, reasoning=reasoning)
            return LLMOutput(content=content, reasoning=None)
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach LLM at {self.base_url}. Check LLM_BASE_URL and network."
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM API error ({e.response.status_code}): {e.response.text}"
            )

    def _generate_ollama(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        include_reasoning: bool,
        *,
        think_override: bool | None = None,
    ) -> LLMOutput:
        if think_override is not None:
            think_flag = bool(think_override)
        else:
            think_flag = bool(settings.ollama_enable_thinking)
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "think": think_flag,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            raw = (data.get("response") or "").strip()
            api_thinking = (data.get("thinking") or "").strip() or None

            tag_reasoning, content_from_tags = split_reasoning_from_tags(raw)
            reasoning = (api_thinking or tag_reasoning) or None
            content = strip_all_thinking_tags(raw)
            if not content.strip() and content_from_tags.strip():
                content = content_from_tags.strip()
            # Rare: visible completion only in `thinking` while `response` is empty.
            if not include_reasoning and not content.strip() and api_thinking:
                content = strip_all_thinking_tags(api_thinking).strip()
                if not content.strip():
                    _, tail = split_reasoning_from_tags(api_thinking)
                    content = tail.strip()

            if include_reasoning and not content.strip() and reasoning:
                content = strip_all_thinking_tags(reasoning).strip()
                if not content.strip():
                    _, tail = split_reasoning_from_tags(reasoning)
                    content = tail.strip()
            if not content.strip() and not (reasoning or "").strip():
                logger.warning(
                    "Ollama /api/generate: empty response and thinking "
                    "(raise QUERY_REWRITE_MAX_TOKENS or set OLLAMA_ENABLE_THINKING=false). "
                    "Top-level keys: %s",
                    list(data.keys()) if isinstance(data, dict) else None,
                )

            if include_reasoning:
                return LLMOutput(content=content, reasoning=reasoning)
            return LLMOutput(content=content, reasoning=None)

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is Ollama running? Try: `ollama serve`"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.text}")

    def stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Streaming generation (Ollama). Yields content tokens only; thinking not split."""
        import json

        mt = max_tokens if max_tokens is not None else settings.llm_max_tokens

        if self._mode == "openai":
            yield self._generate_openai(
                prompt, system, temperature, mt, include_reasoning=False
            ).content
            return

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": True,
                "think": bool(settings.ollama_enable_thinking),
                "options": {
                    "temperature": temperature,
                    "num_predict": mt,
                },
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    def is_available(self) -> bool:
        try:
            if self._mode == "openai":
                r = httpx.get(
                    f"{self.base_url}/v1/models",
                    headers=self._auth_headers(),
                    timeout=5.0,
                )
                return r.status_code < 500
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
