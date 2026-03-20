"""
MemoryManager — reads and writes the per-agent markdown memory file.

Each agent has ONE .md file per case that acts as its persistent knowledge base.

Structure of each file:
  # Case <case_id> — <agent_type> Memory
  ## Extracted Facts
  ## Timeline
  ## Key Entities
  ## Insights
  ## Inconsistencies Detected
"""

from __future__ import annotations
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "./memory_store"))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

SECTIONS = [
    "Extracted Facts",
    "Timeline",
    "Key Entities",
    "Insights",
    "Inconsistencies Detected",
]


def _memory_path(case_id: str, agent_type: str) -> Path:
    safe_case = re.sub(r"[^\w\-]", "_", case_id)
    return MEMORY_DIR / f"{safe_case}__{agent_type}.md"


# ── Read ──────────────────────────────────────────────────────────────────────

def read_memory(case_id: str, agent_type: str) -> dict[str, str]:
    """
    Returns a dict keyed by section name → section content string.
    Returns empty strings for sections not yet written.
    """
    path = _memory_path(case_id, agent_type)
    if not path.exists():
        return {s: "" for s in SECTIONS}

    text = path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for section in SECTIONS:
        pattern = rf"## {re.escape(section)}\n(.*?)(?=\n## |\Z)"
        m = re.search(pattern, text, re.DOTALL)
        result[section] = m.group(1).strip() if m else ""
    return result


def read_full_memory(case_id: str, agent_type: str) -> str:
    """Returns the raw markdown string (used as LLM context)."""
    path = _memory_path(case_id, agent_type)
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ── Write ─────────────────────────────────────────────────────────────────────

def update_memory(
    case_id: str,
    agent_type: str,
    *,
    new_facts: list[str] | None = None,
    timeline_entries: list[str] | None = None,
    entities: list[str] | None = None,
    insights: list[str] | None = None,
    inconsistencies: list[str] | None = None,
) -> None:
    """
    Reads existing memory, appends new items to relevant sections,
    and rewrites the file. Nothing is ever overwritten — only appended.
    """
    path = _memory_path(case_id, agent_type)
    existing = read_memory(case_id, agent_type)

    def _append(section_text: str, new_items: list[str] | None) -> str:
        if not new_items:
            return section_text
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        added = "\n".join(f"- [{ts}] {item}" for item in new_items)
        return (section_text + "\n" + added).strip()

    updated = {
        "Extracted Facts":        _append(existing["Extracted Facts"], new_facts),
        "Timeline":               _append(existing["Timeline"], timeline_entries),
        "Key Entities":           _append(existing["Key Entities"], entities),
        "Insights":               _append(existing["Insights"], insights),
        "Inconsistencies Detected": _append(existing["Inconsistencies Detected"], inconsistencies),
    }

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md = f"# Case {case_id} — {agent_type} Memory\n\n"
    md += f"_Last updated: {ts}_\n\n"
    for section in SECTIONS:
        md += f"## {section}\n{updated[section]}\n\n"

    path.write_text(md, encoding="utf-8")


# ── Inconsistency check ───────────────────────────────────────────────────────

def check_inconsistencies(
    case_id: str,
    agent_type: str,
    new_data: dict[str, Any],
) -> list[str]:
    """
    Lightweight heuristic check of new_data against stored memory.
    The LLM does deeper analysis — this catches obvious conflicts fast.
    """
    issues: list[str] = []
    existing = read_memory(case_id, agent_type)

    known_entities_raw = existing["Key Entities"]
    known_timeline_raw = existing["Timeline"]

    if "suspect_name" in new_data and known_entities_raw:
        candidate = new_data["suspect_name"].strip().lower()
        if candidate and candidate not in known_entities_raw.lower():
            issues.append(
                f"New suspect name '{new_data['suspect_name']}' not seen in prior records."
            )

    if "incident_date" in new_data and known_timeline_raw:
        candidate_date = str(new_data.get("incident_date", ""))
        if candidate_date and candidate_date not in known_timeline_raw:
            issues.append(
                f"Incident date '{candidate_date}' differs from previously recorded timeline."
            )

    return issues
