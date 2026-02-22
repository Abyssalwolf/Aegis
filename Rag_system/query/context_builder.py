"""
Context builder.
Assembles reranked chunks into a structured prompt for the LLM,
with source citations embedded so every claim is traceable to a document.

Uses parent_text when available for richer context, falling back to
chunk.text for the actual passage sent to the LLM.
"""

from core.documents.models import RetrievedChunk

MAX_CONTEXT_TOKENS = 2048   # Conservative limit for a 3B model


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


SYSTEM_PROMPT = """You are a precise assistant helping police officers retrieve information from case files.

Rules:
- Answer ONLY based on the provided context passages. Never use outside knowledge.
- Every factual claim in your answer must cite the source using [Source N].
- If the context does not contain enough information to answer, say so explicitly.
- Be concise and factual. Avoid speculation.
- If evidence is ambiguous or contradictory, say so and cite both sources."""


def build_prompt(
    query: str,
    reranked_chunks: list[RetrievedChunk],
) -> tuple[str, list[dict]]:
    """
    Build a (prompt, sources) tuple.

    Returns:
        prompt: Full prompt string to send to the LLM.
        sources: List of source dicts for the API response (for citation rendering).
    """
    context_blocks: list[str] = []
    sources: list[dict] = []
    total_tokens = 0

    for i, retrieved in enumerate(reranked_chunks, start=1):
        chunk = retrieved.chunk

        # Use parent_text for context if significantly larger than chunk text
        display_text = chunk.text
        if (
            chunk.parent_text
            and len(chunk.parent_text) > len(chunk.text) * 1.3
        ):
            display_text = chunk.parent_text

        block_tokens = _estimate_tokens(display_text)
        if total_tokens + block_tokens > MAX_CONTEXT_TOKENS:
            break   # Context window full

        # Build source metadata
        source_info = {
            "index": i,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_path": chunk.metadata.get("source_path", "unknown"),
            "page_number": chunk.page_number,
            "case_id": chunk.metadata.get("case_id"),
            "relevance_score": round(retrieved.score, 4),
            "chunk_type": chunk.chunk_type.value,
        }
        sources.append(source_info)

        # Format context block
        page_ref = f"(Page {chunk.page_number})" if chunk.page_number else ""
        filename = source_info["source_path"].split("/")[-1]
        header = f"[Source {i}] {filename} {page_ref}".strip()
        context_blocks.append(f"{header}\n{display_text}")
        total_tokens += block_tokens

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""CONTEXT PASSAGES:

{context_text}

---

QUESTION: {query}

ANSWER (cite sources as [Source N]):"""

    return prompt, sources
