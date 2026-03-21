"""
LLM client backed by a remote Ollama server (Modal-hosted Qwen 3 30b).
Uses the native Ollama /api/chat endpoint with thinking enabled.
Returns both content and reasoning separately so callers can choose
what to display.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove any residual <think>...</think> blocks from content."""
    return _THINK_RE.sub("", text).strip()


@dataclass
class LLMResponse:
    content: str
    reasoning: str


class LLMClient:

    def __init__(
        self,
        base_url: str = settings.llm_base_url,
        model: str = settings.llm_model,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        messages: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """
        Chat generation via Ollama native /api/chat with thinking enabled.
        Returns an LLMResponse with separate content and reasoning fields.
        """
        chat_messages: list[dict] = []

        if system:
            chat_messages.append({"role": "system", "content": system})

        if messages:
            chat_messages.extend(messages)

        chat_messages.append({"role": "user", "content": prompt})

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": chat_messages,
                    "stream": False,
                    "think": True,
                    "keep_alive": -1,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})
            content = _strip_thinking(msg.get("content", ""))
            reasoning = (msg.get("thinking", "") or "").strip()

            return LLMResponse(content=content, reasoning=reasoning)

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach LLM at {self.base_url}. "
                "Is the Ollama server running?"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM API error: {e.response.text}")

    def is_available(self) -> bool:
        """Health check — returns True if the Ollama server is reachable."""
        try:
            resp = httpx.get(self.base_url, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
