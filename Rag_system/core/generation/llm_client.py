"""
Local LLM client backed by Ollama.
Used for both query rewriting and final answer generation.
Streams responses for better perceived latency.

Note on Qwen3 / thinking models:
  Qwen3.x models output <think>...</think> reasoning chains before every answer.
  We disable this with "think": false so responses are direct and fast.
  A cleanup pass strips any residual thinking blocks just in case.
"""

import logging
import re
from typing import Iterator

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Matches <think>...</think> blocks including multi-line content.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3-style reasoning models."""
    return _THINK_RE.sub("", text).strip()


class OllamaClient:

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
    ):
        self.base_url = base_url
        self.model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Non-streaming generation. Returns full response string."""
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    # think:false disables the reasoning chain on Qwen3.x models,
                    # giving faster and cleaner responses.
                    "think": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=300.0,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            return _strip_thinking(raw)

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
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Streaming generation. Yields text tokens as they arrive."""
        import json

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": True,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
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
        """Health check — returns True if Ollama is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
