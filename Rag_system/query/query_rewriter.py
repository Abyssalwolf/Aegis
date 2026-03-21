"""
Query rewriter.
Uses the LLM to generate N alternative phrasings of the user's query
to improve retrieval recall via multi-query expansion.

The original query is always included, so retrieval always covers
the user's exact intent.
"""

import logging
import re

from core.generation.llm_client import LLMClient
from config.settings import settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
Given the following search query, generate {n} alternative phrasings that
capture the same information need but use different words and structure.

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
        llm: LLMClient | None = None,
        n_rewrites: int = settings.query_rewrite_count,
    ):
        self.llm = llm or LLMClient()
        self.n_rewrites = n_rewrites

    def rewrite(self, query: str) -> list[str]:
        """
        Returns a list of query variants: [original] + [rewrites].
        Falls back gracefully to [original] if the LLM is unavailable.
        """
        rewrites = self._call_llm(query)
        all_queries = [query] + rewrites

        seen: set[str] = set()
        unique: list[str] = []
        for q in all_queries:
            normalized = q.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(q.strip())

        logger.info(f"Query rewriting: 1 original → {len(unique)} total variants.")
        return unique

    def _call_llm(self, query: str) -> list[str]:
        prompt = REWRITE_PROMPT.format(n=self.n_rewrites, query=query)

        try:
            result = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2048,
            )
            return self._parse_rewrites(result.content)

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            return []

    def _parse_rewrites(self, raw: str) -> list[str]:
        """Parse LLM output into a clean list of query strings."""
        lines = raw.strip().split("\n")
        rewrites: list[str] = []

        for line in lines:
            cleaned = re.sub(r'^[\d\.\-\*\"\'\s]+', '', line).strip()
            cleaned = cleaned.strip('"\'')
            if cleaned and len(cleaned) > 5:
                rewrites.append(cleaned)

        return rewrites[:self.n_rewrites]
