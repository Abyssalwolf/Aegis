"""
BaseAgent — common logic for all 7 specialist document agents.

LLM: :func:`core.generation.agent_chat.agent_llm_complete` → ``OllamaClient``
(``LLM_BASE_URL`` OpenAI-compatible API when set, else Ollama).
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

from core.generation.agent_chat import agent_llm_complete
from orchestration.graph.state import BlackboardMessage, InvestigationState
from core.documents.manager import (
    read_full_memory,
    update_memory,
    check_inconsistencies,
)
from orchestration.blackboard import post_finding
from core.retrieval.agent_retriever import query_rag


# ── Extraction prompts per file type ─────────────────────────────────────────

EXTRACTION_PROMPTS: dict[str, str] = {
    "fir": """You are an expert police analyst processing a First Information Report (FIR).
Extract the following fields as JSON:
{
  "complainant_name": "",
  "suspect_name": "",
  "incident_date": "",
  "incident_location": "",
  "offence_sections": [],
  "allegations": "",
  "witnesses": [],
  "summary": ""
}
Return ONLY valid JSON, no explanation.""",

    "case_diary": """You are an expert police analyst processing a Case Diary entry.
Extract as JSON:
{
  "entry_date": "",
  "investigating_officer": "",
  "actions_taken": [],
  "leads_followed": [],
  "pending_actions": [],
  "timeline_events": [],
  "summary": ""
}
Return ONLY valid JSON.""",

    "statement": """You are an expert police analyst processing a Witness/Accused Statement.
Extract as JSON:
{
  "deponent_name": "",
  "statement_date": "",
  "deponent_role": "",
  "key_claims": [],
  "alibi_if_any": "",
  "corroborating_persons": [],
  "contradictions": [],
  "summary": ""
}
Return ONLY valid JSON.""",

    "scene_of_crime": """You are an expert police analyst processing a Scene of Crime report.
Extract as JSON:
{
  "scene_location": "",
  "date_of_visit": "",
  "investigating_officers": [],
  "physical_evidence_noted": [],
  "entry_exit_points": [],
  "fingerprints_found": false,
  "cctv_available": false,
  "summary": ""
}
Return ONLY valid JSON.""",

    "forensic": """You are an expert police analyst processing a Forensic/Evidence report.
Extract as JSON:
{
  "lab_reference": "",
  "report_date": "",
  "samples_examined": [],
  "findings": [],
  "conclusion": "",
  "matching_suspects": [],
  "summary": ""
}
Return ONLY valid JSON.""",

    "seizure": """You are an expert police analyst processing a Property/Seizure list.
Extract as JSON:
{
  "seizure_date": "",
  "seizing_officer": "",
  "seized_from": "",
  "items": [],
  "location_of_seizure": "",
  "mahazar_witnesses": [],
  "summary": ""
}
Return ONLY valid JSON.""",

    "arrest_remand": """You are an expert police analyst processing an Arrest/Remand document.
Extract as JSON:
{
  "arrested_person": "",
  "arrest_date": "",
  "arrest_location": "",
  "arresting_officer": "",
  "offences_cited": [],
  "remand_period": "",
  "custody_type": "",
  "summary": ""
}
Return ONLY valid JSON.""",
}

INCONSISTENCY_PROMPT = """You are a senior forensic analyst reviewing police documents.

=== AGENT MEMORY (previous knowledge) ===
{memory}

=== NEW DOCUMENT CONTENT ===
{document}

=== EXTRACTED DATA ===
{extracted}

Tasks:
1. Compare the new document against the agent memory.
2. Identify ANY factual inconsistencies or contradictions.
3. List critical missing information.
4. Provide 2-5 insight observations.

Respond as JSON:
{{
  "inconsistencies": [],
  "missing_information": [],
  "insights": [],
  "needs_rag": false,
  "rag_questions": []
}}
Return ONLY valid JSON."""


class BaseAgent:
    agent_id: str
    file_type: str

    def __init__(self) -> None:
        assert hasattr(self, "agent_id"), "Subclass must define agent_id"
        assert hasattr(self, "file_type"), "Subclass must define file_type"

    def __call__(self, state: InvestigationState) -> dict[str, Any]:
        """LangGraph node entry point."""
        case_id = state.case_id
        doc_text = state.file_content
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # 1. Read agent memory
        memory_md = read_full_memory(case_id, self.file_type)

        # 2. Extract structured data from document
        extracted = self._extract(doc_text)

        # 3. Heuristic inconsistency check against memory
        heuristic_issues = check_inconsistencies(case_id, self.file_type, extracted)

        # 4. Deep LLM analysis — inconsistencies + insights
        analysis = self._analyse(memory_md, doc_text, extracted)

        all_inconsistencies = heuristic_issues + analysis.get("inconsistencies", [])
        insights = analysis.get("insights", [])
        needs_rag = analysis.get("needs_rag", False)
        rag_questions = analysis.get("rag_questions", [])
        rag_context_used: list[str] = []

        # 5. Query RAG only if needed
        if needs_rag and rag_questions:
            for question in rag_questions[:3]:
                ctx = query_rag(case_id, question)
                if ctx:
                    rag_context_used.append(question)
                    extra = self._analyse_with_rag(ctx, extracted)
                    insights.extend(extra.get("insights", []))
                    all_inconsistencies.extend(extra.get("inconsistencies", []))

        # 6. Update memory file
        update_memory(
            case_id,
            self.file_type,
            new_facts=[extracted.get("summary", "")],
            timeline_entries=self._timeline_entries(extracted),
            entities=self._entity_list(extracted),
            insights=insights,
            inconsistencies=all_inconsistencies,
        )

        # 7. Post to blackboard
        post_finding(
            case_id=int(case_id) if str(case_id).isdigit() else case_id,
            agent=self.agent_id,
            file_type=self.file_type,
            payload={
                "summary": extracted.get("summary", ""),
                "key_entities": self._entity_list(extracted),
                "inconsistencies": all_inconsistencies,
                "insights": insights,
                "rag_queries_made": rag_context_used,
                "raw_extracted": extracted,
            }
        )

        # 8. Return LangGraph state update
        msg = BlackboardMessage(
            agent_id=self.agent_id,
            file_type=self.file_type,
            case_id=str(case_id),
            timestamp=ts,
            summary=extracted.get("summary", ""),
            key_entities=self._entity_list(extracted),
            inconsistencies=all_inconsistencies,
            rag_queries_made=rag_context_used,
            insights=insights,
            raw_extracted=extracted,
        )

        return {
            "blackboard": [msg],
            "needs_rag": needs_rag,
            "assigned_agent": self.agent_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract(self, doc_text: str) -> dict[str, Any]:
        prompt = EXTRACTION_PROMPTS.get(self.file_type, "Extract key information as JSON.")
        content = agent_llm_complete(doc_text[:6000], system=prompt, temperature=0)
        try:
            # strip markdown fences if model adds them
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"summary": content, "parse_error": True}

    def _analyse(self, memory_md: str, doc_text: str,
                 extracted: dict[str, Any]) -> dict[str, Any]:
        prompt = INCONSISTENCY_PROMPT.format(
            memory=memory_md or "(No prior memory for this case)",
            document=doc_text[:3000],
            extracted=json.dumps(extracted, indent=2),
        )
        content = agent_llm_complete(prompt, system="", temperature=0)
        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"inconsistencies": [], "insights": [],
                    "needs_rag": False, "rag_questions": []}

    def _analyse_with_rag(self, rag_context: str,
                          extracted: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""Given this additional context from the case knowledge base:

{rag_context}

And this extracted data:
{json.dumps(extracted, indent=2)}

Identify any new insights or inconsistencies.
Return ONLY JSON: {{"inconsistencies": [], "insights": []}}"""
        content = agent_llm_complete(prompt, system="", temperature=0)
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"inconsistencies": [], "insights": []}

    def _entity_list(self, extracted: dict[str, Any]) -> list[str]:
        entities = []
        for key in ["complainant_name", "suspect_name", "arrested_person",
                    "deponent_name", "investigating_officer", "seizing_officer"]:
            val = extracted.get(key)
            if val and isinstance(val, str):
                entities.append(val)
        for key in ["witnesses", "corroborating_persons", "mahazar_witnesses",
                    "investigating_officers", "matching_suspects"]:
            val = extracted.get(key)
            if val and isinstance(val, list):
                entities.extend(str(v) for v in val if v)
        return list(dict.fromkeys(entities))

    def _timeline_entries(self, extracted: dict[str, Any]) -> list[str]:
        entries = []
        for key in ["incident_date", "entry_date", "statement_date",
                    "date_of_visit", "report_date", "seizure_date", "arrest_date"]:
            val = extracted.get(key)
            if val and isinstance(val, str):
                entries.append(f"{key.replace('_', ' ').title()}: {val}")
        for event in extracted.get("timeline_events", []):
            entries.append(str(event))
        return entries