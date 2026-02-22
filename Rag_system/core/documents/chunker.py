"""
Semantic chunker.

Strategy:
  1. Split document text into sentences using a simple sentence tokenizer.
  2. Embed each sentence using the provided embedder.
  3. Compute cosine similarity between adjacent sentence embeddings.
  4. Find breakpoints where similarity drops below (mean - threshold * std).
  5. Group sentences between breakpoints into chunks.
  6. Enforce min/max token limits — split oversized chunks recursively,
     merge undersized chunks with their neighbour.
  7. Attach parent_text: the broader section a chunk came from, for
     richer context at generation time.
"""

import re
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Embedder protocol (avoids circular import with local_embedder)
# -------------------------------------------------------------------
class SentenceEmbedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — no NLTK dependency."""
    # Split on . ! ? followed by whitespace or end-of-string,
    # but keep the delimiter attached to the preceding sentence.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _count_tokens(text: str) -> int:
    """Rough token count: words * 1.3 (good enough for guards)."""
    return int(len(text.split()) * 1.3)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -------------------------------------------------------------------
# Core chunker
# -------------------------------------------------------------------

@dataclass
class SemanticChunk:
    text: str
    sentences: list[str]
    parent_text: str        # broader section this chunk belongs to
    token_count: int


class SemanticChunker:
    def __init__(
        self,
        embedder: SentenceEmbedder,
        max_tokens: int = settings.chunk_max_tokens,
        min_tokens: int = settings.chunk_min_tokens,
        similarity_threshold: float = settings.semantic_similarity_threshold,
    ):
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> list[SemanticChunk]:
        """Main entry point. Returns a list of SemanticChunks."""
        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Single sentence — return as-is if above min
        if len(sentences) == 1:
            tc = _count_tokens(sentences[0])
            if tc >= self.min_tokens:
                return [SemanticChunk(
                    text=sentences[0],
                    sentences=sentences,
                    parent_text=sentences[0],
                    token_count=tc,
                )]
            return []

        # Embed all sentences in one batch
        logger.debug(f"Embedding {len(sentences)} sentences for semantic chunking.")
        embeddings: np.ndarray = self.embedder.encode(sentences)  # (N, D)

        # Compute adjacent cosine similarities
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Find breakpoints: where sim drops below (mean - threshold * std)
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        cutoff = mean_sim - self.similarity_threshold * std_sim

        breakpoints: set[int] = set()
        for i, sim in enumerate(similarities):
            if sim < cutoff:
                breakpoints.add(i + 1)   # break BEFORE sentence i+1

        # Group sentences into raw sections
        sections: list[list[str]] = []
        current: list[str] = []
        for i, sentence in enumerate(sentences):
            if i in breakpoints and current:
                sections.append(current)
                current = []
            current.append(sentence)
        if current:
            sections.append(current)

        # Build chunks with token enforcement + parent assignment
        chunks: list[SemanticChunk] = []
        for section in sections:
            parent_text = " ".join(section)
            section_chunks = self._enforce_limits(section, parent_text)
            chunks.extend(section_chunks)

        # Merge chunks that are below min_tokens into their neighbour
        chunks = self._merge_small_chunks(chunks)

        logger.info(f"Produced {len(chunks)} semantic chunks from {len(sentences)} sentences.")
        return chunks

    def _enforce_limits(self, sentences: list[str], parent_text: str) -> list[SemanticChunk]:
        """
        Given sentences from one semantic section, produce chunks that
        respect max_tokens. If a section is too long, split greedily.
        """
        chunks: list[SemanticChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            st = _count_tokens(sentence)
            if current_tokens + st > self.max_tokens and current_sentences:
                # Flush current
                text = " ".join(current_sentences)
                chunks.append(SemanticChunk(
                    text=text,
                    sentences=list(current_sentences),
                    parent_text=parent_text,
                    token_count=current_tokens,
                ))
                current_sentences = []
                current_tokens = 0
            current_sentences.append(sentence)
            current_tokens += st

        if current_sentences:
            text = " ".join(current_sentences)
            chunks.append(SemanticChunk(
                text=text,
                sentences=list(current_sentences),
                parent_text=parent_text,
                token_count=current_tokens,
            ))

        return chunks

    def _merge_small_chunks(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        """Merge chunks below min_tokens into their right neighbour."""
        if not chunks:
            return chunks

        merged: list[SemanticChunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if chunk.token_count < self.min_tokens and i + 1 < len(chunks):
                # Merge with next
                next_chunk = chunks[i + 1]
                combined_text = chunk.text + " " + next_chunk.text
                merged.append(SemanticChunk(
                    text=combined_text,
                    sentences=chunk.sentences + next_chunk.sentences,
                    parent_text=chunk.parent_text,   # keep original parent
                    token_count=chunk.token_count + next_chunk.token_count,
                ))
                i += 2  # skip next since we consumed it
            else:
                merged.append(chunk)
                i += 1

        return merged
