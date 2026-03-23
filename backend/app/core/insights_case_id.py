"""
Map backend Case.id (UUID) → integer case_id for RAG Insights /agents + Redis blackboard.

The multi-agent service uses int paths and Redis keys `case:{id}:...`. This module
provides a **stable, deterministic** mapping so the main app never stores a second key.

Algorithm: UUID.int % (2**31 - 1), with 0 remapped to 1. Same as
`Rag_system/core/insights_case_id.py` — keep both files in sync.
"""

from __future__ import annotations

import uuid

# Mersenne-style modulus; keeps values in 1 .. 2**31-2 (fits signed 32-bit int).
_MOD = 2**31 - 1


def uuid_to_insights_case_id(case_id: uuid.UUID) -> int:
    """
    Deterministic 1:1-style namespace for a UUID into a positive int.

    Collision: two different UUIDs could map to the same int (birthday bound).
    For typical case counts this is negligible; if it ever matters, store an
    explicit mapping column on `case` instead.
    """
    n = case_id.int % _MOD
    return n if n != 0 else 1


def uuid_str_to_insights_case_id(case_id: str) -> int:
    """Accept canonical UUID string from path/query."""
    return uuid_to_insights_case_id(uuid.UUID(case_id))
