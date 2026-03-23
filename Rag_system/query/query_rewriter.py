"""
Query rewriter.
Uses the configured LLM to produce N alternative search phrasings (multi-query expansion).

We ask for **strict JSON** so chain-of-thought is not mistaken for queries. On Ollama we
pass `enable_thinking=False` so reasoning is less likely to fill the completion. We only
parse **assistant `content`** (never merge a separate reasoning field — that was CoT).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from config.settings import settings
from core.generation.llm_client import OllamaClient, strip_all_thinking_tags

logger = logging.getLogger(__name__)

# JSON-only: avoids "We are given the original query..." being split into fake rewrites.
REWRITE_PROMPT = """You improve search queries for a police case-file retrieval system.

Task: For the query below, produce exactly {n} alternative search phrases (same information need, different wording). Each phrase must be under 20 words. Do not add new facts.

Output rules:
- Respond with a single JSON object only.
- No markdown code fences, no text before or after the JSON.
- Use key "rewrites" with an array of exactly {n} strings (each string = one search query).
- Example shape only: {{"rewrites": ["alt phrase one", "alt phrase two"]}} — use {n} items.

Original query: {query}
"""


# Lines that are clearly model chain-of-thought / instructions, not search queries.
_META_LINE = re.compile(
    r"^\s*("
    r"we['\u2019]?re\b|we are\b|we need\b|we must\b|we will\b|we have\b|"
    r"let'?s\b|let me\b|first,|second,|third,|the user\b|given that\b|"
    r"okay,|so,|note:|alternatively,|i need\b|i will\b|i'?ll\b|"
    r"to answer\b|looking at\b|original query\b|alternative phrasings\b|"
    r"respond with\b|you must\b|task:|here are\b|output rules\b|"
    r"the object must\b|each string is\b"
    r")",
    re.I | re.UNICODE,
)


def _word_count(s: str) -> int:
    return len(re.findall(r"\S+", s))


def _looks_like_search_query(s: str, *, max_words: int = 22) -> bool:
    s = s.strip()
    if not s or _word_count(s) > max_words:
        return False
    if _META_LINE.match(s):
        return False
    low = s.lower()
    if "original query" in low or "alternative phrasing" in low:
        return False
    # CoT often repeats prompt fragments in quotes
    if low.startswith("we ") and "given" in low[:40]:
        return False
    return True


def _try_parse_json_rewrites(raw: str) -> list[str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    text = strip_all_thinking_tags(text)
    # Strip ```json ... ``` if the model disobeys
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    # Some models wrap JSON in other text — find outermost {...}
    if "{" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    arr = data.get("rewrites")
    if not isinstance(arr, list):
        return None
    out: list[str] = []
    for x in arr:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out if out else None


class QueryRewriter:

    def __init__(
        self,
        n_rewrites: int = settings.query_rewrite_count,
        llm: Optional[OllamaClient] = None,
    ):
        self._llm = llm or OllamaClient()
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
            out = self._llm.generate(
                prompt,
                system="",
                temperature=0.2,
                max_tokens=settings.query_rewrite_max_tokens,
                include_reasoning=False,
                enable_thinking=False,
            )
            raw_text = (out.content or "").strip()
            if not raw_text:
                logger.warning(
                    "Query rewriter: empty assistant content "
                    "(increase QUERY_REWRITE_MAX_TOKENS or check provider)."
                )
                return []

            parsed = _try_parse_json_rewrites(raw_text)
            if parsed is not None:
                cleaned = [q for q in parsed if _looks_like_search_query(q)]
                cleaned = cleaned[: self.n_rewrites]
                if len(cleaned) >= self.n_rewrites:
                    return cleaned[: self.n_rewrites]
                if cleaned:
                    logger.info(
                        "Query rewriter: JSON parse ok but filtered %d → %d lines",
                        len(parsed),
                        len(cleaned),
                    )
                    return cleaned[: self.n_rewrites]
                # JSON parsed but every string looked like CoT — retry line filter on raw
                logger.debug(
                    "Query rewriter: JSON strings failed heuristic filter, using line fallback"
                )

            logger.debug("Query rewriter: JSON parse failed, falling back to line filter")
            return self._parse_rewrites_lines(raw_text, original_query=query)
        except Exception as e:
            logger.warning(
                "Query rewriter LLM call failed (%s). Using original query only.",
                e,
            )
            return []

    def _parse_rewrites_lines(self, raw: str, *, original_query: str) -> list[str]:
        """Last resort: newline split with strong CoT filtering."""
        text = strip_all_thinking_tags(raw.strip())
        if not text:
            return []

        if "\n" not in text and (text.count(";") >= self.n_rewrites - 1 or text.count("|") >= 1):
            parts = re.split(r"[;|]+", text)
            lines = [p.strip() for p in parts if p.strip()]
        else:
            lines = text.split("\n")

        rewrites: list[str] = []
        orig_norm = original_query.strip().lower()

        for line in lines:
            cleaned = re.sub(r'^[\d\.\-\*\"\'\s]+', "", line).strip()
            cleaned = cleaned.strip("\"'")
            if not cleaned or cleaned.lower() == orig_norm:
                continue
            if not _looks_like_search_query(cleaned):
                continue
            if len(cleaned) < 4:
                continue
            rewrites.append(cleaned)

        return rewrites[: self.n_rewrites]
