"""
Query rewriter.
Uses a local Ollama LLM to generate N alternative phrasings of the
user's query to improve retrieval recall via multi-query expansion.

The original query is always included, so retrieval always covers
the user's exact intent.
"""

import logging
import re
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
You are helping improve a document retrieval system for police case files.

Given the following search query, generate {n} alternative phrasings that
capture the same information need but use different words and structure.
This helps retrieve relevant documents that may use different terminology.

Rules:
- Keep each rewrite concise (under 20 words)
- Focus on the core information need
- Do NOT add assumptions or new facts
- Output ONLY the rewrites, one per line, no numbering or bullet points

Original query: {query}

Alternative phrasings:"""


class QueryRewriter:

    def __init__(
        self,
        ollama_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
        n_rewrites: int = settings.query_rewrite_count,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.n_rewrites = n_rewrites

    def rewrite(self, query: str) -> list[str]:
        """
        Returns a list of query variants: [original] + [rewrites].
        Falls back gracefully to [original] if Ollama is unavailable.
        """
        rewrites = self._call_ollama(query)
        all_queries = [query] + rewrites

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for q in all_queries:
            normalized = q.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(q.strip())

        logger.info(f"Query rewriting: 1 original → {len(unique)} total variants.")
        return unique

    def _call_ollama(self, query: str) -> list[str]:
        prompt = REWRITE_PROMPT.format(n=self.n_rewrites, query=query)

        try:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,     # Low temp for focused rewrites
                        "num_predict": 200,
                    },
                },
                timeout=30.0,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
            return self._parse_rewrites(raw_text)

        except httpx.ConnectError:
            logger.warning("Ollama not reachable. Proceeding with original query only.")
            return []
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            return []

    def _parse_rewrites(self, raw: str) -> list[str]:
        """Parse LLM output into a clean list of query strings."""
        lines = raw.strip().split("\n")
        rewrites: list[str] = []

        for line in lines:
            # Strip numbering, bullets, quotes
            cleaned = re.sub(r'^[\d\.\-\*\"\'\s]+', '', line).strip()
            cleaned = cleaned.strip('"\'')
            if cleaned and len(cleaned) > 5:
                rewrites.append(cleaned)

        return rewrites[:self.n_rewrites]
