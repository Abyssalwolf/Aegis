"""
LLM client backed by an OpenAI-compatible API (Modal-hosted Qwen 3 30b).
Supports conversation history for multi-turn chat and single-shot generation
for internal tasks like query rewriting.

Qwen3 models default to "thinking mode" which puts reasoning in a separate
field and can exhaust the token budget before producing an answer.
We append /nothink to user prompts to disable this behaviour.
"""

import logging
import re
from typing import Optional

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3-style reasoning models."""
    return _THINK_RE.sub("", text).strip()


class LLMClient:

    def __init__(
        self,
        base_url: str = settings.llm_base_url,
        model: str = settings.llm_model,
        api_key: str = settings.llm_api_key,
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        messages: Optional[list[dict]] = None,
    ) -> str:
        """
        Chat-completion generation.

        If *messages* is provided the call becomes multi-turn:
            [system] + messages + [user: prompt + /nothink]
        Otherwise it is a single-turn call:
            [system] + [user: prompt + /nothink]
        """
        chat_messages: list[dict] = []

        if system:
            chat_messages.append({"role": "system", "content": system})

        if messages:
            chat_messages.extend(messages)

        chat_messages.append({"role": "user", "content": prompt + " /nothink"})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"keep_alive": -1},
            )
            raw = response.choices[0].message.content or ""
            return _strip_thinking(raw)

        except Exception as exc:
            raise RuntimeError(
                f"LLM API error ({self.client.base_url}): {exc}"
            ) from exc

    def is_available(self) -> bool:
        """Health check — returns True if the LLM endpoint is reachable."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
