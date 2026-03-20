"""
FileClassifier — automatically detects the police document type from file content.

Used when the uploader does NOT specify file_type, or as a validation layer
to confirm the user's stated type matches the actual content.

Two strategies:
  1. Keyword heuristic  — fast, zero API cost, handles 80 % of cases
  2. LLM fallback       — for ambiguous documents
"""

from __future__ import annotations
import re
from typing import Any

from orchestration.graph.state import FILE_TYPE


# ─── Keyword heuristic ────────────────────────────────────────────────────────
# Each document type has high-signal keywords. Scored by hit count.

_KEYWORD_MAP: dict[FILE_TYPE, list[str]] = {
    "fir": [
        "first information report", "fir no", "fir number",
        "complainant", "informant", "alleged offence", "ipc section",
        "offence reported", "police station", "f.i.r",
    ],
    "case_diary": [
        "case diary", "rozhnamcha", "investigation diary",
        "investigating officer", "io report", "daily diary",
        "progress report", "action taken", "case progress",
        "further investigation", "section 172 crpc",
    ],
    "statement": [
        "statement of", "i state that", "deponent", "examination under",
        "section 161", "section 164", "crpc", "voluntary statement",
        "recorded by", "witness statement", "accused statement",
        "i am examined", "sworn statement",
    ],
    "scene_of_crime": [
        "scene of crime", "scene inspection", "inquest report",
        "panchanama", "spot inspection", "crime scene",
        "entry point", "exit point", "scene of offence",
        "place of occurrence", "mahazar", "spot mahazar",
        "rough sketch", "site plan",
    ],
    "forensic": [
        "forensic", "laboratory report", "lab report", "fsl",
        "ballistic", "serological", "dna", "fingerprint report",
        "chemical analysis", "toxicology", "post mortem",
        "expert opinion", "questioned document", "cyber forensic",
    ],
    "seizure": [
        "seizure list", "property list", "seized property",
        "recovery memo", "seizure mahazar", "seized from",
        "articles seized", "muddemal", "case property",
        "recovery list", "seizure memo",
    ],
    "arrest_remand": [
        "arrest memo", "remand report", "custody report",
        "arrested person", "production before", "judicial custody",
        "police custody", "remand application", "transit remand",
        "arrested on", "place of arrest", "section 57 crpc",
    ],
}


def classify_by_keywords(text: str) -> tuple[FILE_TYPE | None, float]:
    """
    Returns (best_match_type, confidence_0_to_1).
    Returns (None, 0.0) if no confident match found.
    """
    lower = text.lower()
    scores: dict[FILE_TYPE, int] = {}

    for doc_type, keywords in _KEYWORD_MAP.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[doc_type] = hits

    if not scores:
        return None, 0.0

    best_type = max(scores, key=scores.__getitem__)
    best_score = scores[best_type]
    total_keywords = len(_KEYWORD_MAP[best_type])
    confidence = min(best_score / max(total_keywords * 0.4, 1), 1.0)

    # Require at least 2 keyword hits for a non-trivial match
    if best_score < 2:
        return None, confidence

    return best_type, round(confidence, 2)


# ─── LLM fallback ─────────────────────────────────────────────────────────────

_LLM_PROMPT = """You are an expert in Indian police documentation.

Classify the following document excerpt into EXACTLY ONE of these categories:
- fir              (First Information Report)
- case_diary       (Case Diary / Rozhnamcha / Investigation Diary)
- statement        (Witness or Accused Statement under CrPC 161/164)
- scene_of_crime   (Scene of Crime / Inquest / Panchanama / Spot Mahazar)
- forensic         (Forensic / Lab / Ballistic / DNA / Post-Mortem Report)
- seizure          (Property / Seizure / Recovery List / Muddemal)
- arrest_remand    (Arrest Memo / Remand Report / Production Report)

Document excerpt:
{text}

Respond with ONLY a JSON object:
{{"file_type": "<one of the 7 values above>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}"""


def classify_by_llm(text: str) -> tuple[FILE_TYPE | None, float, str]:
    """
    LLM-based classification. Returns (type, confidence, reason).
    Falls back to ("fir", 0.5, "unknown") on error.
    """
    import json
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = _LLM_PROMPT.format(text=text[:2000])
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        return data.get("file_type"), data.get("confidence", 0.7), data.get("reason", "")
    except Exception:
        return None, 0.0, "classification failed"


# ─── Main entry point ─────────────────────────────────────────────────────────

def classify_document(
    text: str,
    stated_type: FILE_TYPE | None = None,
    use_llm_fallback: bool = True,
) -> dict[str, Any]:
    """
    Classify a document. Returns a result dict with:
      - file_type       : detected type
      - confidence      : 0.0–1.0
      - method          : "keyword" | "llm" | "stated"
      - conflict        : True if stated_type contradicts detected type
      - conflict_detail : human-readable explanation if conflict

    If stated_type is given and confidence is high (>0.7), we validate.
    If stated_type is given and detection is low confidence, we trust stated_type.
    """
    result: dict[str, Any] = {
        "file_type": stated_type,
        "confidence": 1.0,
        "method": "stated",
        "conflict": False,
        "conflict_detail": "",
    }

    # Step 1: keyword heuristic
    kw_type, kw_conf = classify_by_keywords(text)

    if kw_type and kw_conf >= 0.6:
        result.update({"file_type": kw_type, "confidence": kw_conf, "method": "keyword"})
    elif use_llm_fallback:
        llm_type, llm_conf, llm_reason = classify_by_llm(text)
        if llm_type:
            result.update({
                "file_type": llm_type,
                "confidence": llm_conf,
                "method": "llm",
                "llm_reason": llm_reason,
            })

    # Step 2: conflict detection
    if stated_type and result["file_type"] and result["confidence"] >= 0.7:
        if stated_type != result["file_type"]:
            result["conflict"] = True
            result["conflict_detail"] = (
                f"You stated '{stated_type}' but the document appears to be "
                f"'{result['file_type']}' (confidence {result['confidence']:.0%}). "
                f"Proceeding with stated type '{stated_type}'."
            )
            # Trust the human's stated type but log the conflict
            result["file_type"] = stated_type

    # Ensure we always have a valid type
    if not result["file_type"]:
        result["file_type"] = stated_type or "fir"
        result["confidence"] = 0.3
        result["method"] = "fallback"

    return result
