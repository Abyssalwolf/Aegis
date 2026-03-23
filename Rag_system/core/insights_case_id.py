"""
UUID → int case_id for /agents HTTP API and Redis blackboard.

**Must match** `backend/app/core/insights_case_id.py` byte-for-byte logic.
The main AEGIS backend computes this when proxying to RAG; direct curl testers
can use the same formula for a given case UUID.
"""

from __future__ import annotations

import uuid

_MOD = 2**31 - 1


def uuid_to_insights_case_id(case_id: uuid.UUID) -> int:
    n = case_id.int % _MOD
    return n if n != 0 else 1


def uuid_str_to_insights_case_id(case_id: str) -> int:
    return uuid_to_insights_case_id(uuid.UUID(case_id))
