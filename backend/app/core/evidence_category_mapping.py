"""
Map UI / API evidence category values to Insights LangGraph FILE_TYPE strings.

Rag_system/orchestration/graph/state.py expects exactly:
  fir | case_diary | statement | scene_of_crime | forensic | seizure | arrest_remand

The upload modal (and optional DB column) may use longer slugs; normalize here so
backend → RAG Insights / Celery always receives a valid agent file_type when possible.

When mapping returns None (e.g. unknown slug), unified document upload rejects with 422 — the UI must
use one of the seven investigation types (no "Other") so both RAG ingest and the agent queue get an explicit type.
"""

from __future__ import annotations

# Must stay in sync with Rag_system/orchestration/graph/state.py FILE_TYPE
AGENT_FILE_TYPES: frozenset[str] = frozenset(
    {
        "fir",
        "case_diary",
        "statement",
        "scene_of_crime",
        "forensic",
        "seizure",
        "arrest_remand",
    }
)

# Keys = values accepted from clients (lowercase). Values = agent file_type.
_EVIDENCE_CATEGORY_ALIASES: dict[str, str] = {
    # Canonical (same as agent id)
    "fir": "fir",
    "case_diary": "case_diary",
    "statement": "statement",
    "scene_of_crime": "scene_of_crime",
    "forensic": "forensic",
    "seizure": "seizure",
    "arrest_remand": "arrest_remand",
    # Common UI / form variants (upload modal style)
    "fir_file": "fir",
    "statement_file": "statement",
    "forensic_evidence": "forensic",
    "property_seizure": "seizure",
    "scene_of_crime_file": "scene_of_crime",
    "case_diary_file": "case_diary",
    "arrest_remand_file": "arrest_remand",
}


def map_evidence_category_to_agent_file_type(evidence_category: str | None) -> str | None:
    """
    Return the LangGraph / specialist agent file_type, or None.

    None means: no reliable mapping (e.g. missing, unknown, "other") —
    downstream should classify from document text or skip specialist routing.
    """
    if not evidence_category:
        return None
    key = evidence_category.strip().lower()
    if key in ("other", "unknown", ""):
        return None
    if key in AGENT_FILE_TYPES:
        return key
    return _EVIDENCE_CATEGORY_ALIASES.get(key)


def is_valid_agent_file_type(value: str | None) -> bool:
    return bool(value and value in AGENT_FILE_TYPES)
