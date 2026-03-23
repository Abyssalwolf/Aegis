"""
SupervisorAgent — reads the full blackboard and produces
a consolidated investigation report (same LLM stack as RAG: LLM_BASE_URL or Ollama).
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

from orchestration.graph.state import InvestigationState
from orchestration.blackboard import format_brief
from core.generation.agent_chat import agent_llm_complete


SUPERVISOR_SYSTEM = """You are the Chief Investigating Officer supervising a criminal case.

You will receive a full blackboard brief with findings from multiple specialist agents:
- FIR Agent, Case Diary Agent, Statement Agent, Scene of Crime Agent,
  Forensic Agent, Seizure Agent, Arrest/Remand Agent

Your tasks:
1. CROSS-CHECK all findings against each other. Flag inconsistencies BETWEEN agents
   (e.g. suspect name in FIR differs from arrested person in remand order).
2. Synthesize a coherent CASE NARRATIVE.
3. Identify GAPS — information missing across reports.
4. Highlight CRITICAL LEADS and pending actions.
5. Flag anything needing immediate legal attention.

Respond as JSON:
{
  "case_narrative": "",
  "cross_inconsistencies": [],
  "information_gaps": [],
  "critical_leads": [],
  "legal_flags": [],
  "recommended_actions": [],
  "overall_assessment": "",
  "risk_level": "low|medium|high|critical"
}
Return ONLY valid JSON."""


class SupervisorAgent:

    def __call__(self, state: InvestigationState) -> dict[str, Any]:
        case_id = state.case_id
        brief = format_brief(case_id)
        analysis = self._analyse(brief)
        report = self._build_report(case_id, analysis)

        return {
            "supervisor_report": report,
            "cross_inconsistencies": analysis.get("cross_inconsistencies", []),
            "final_status": (
                "needs_review" if analysis.get("cross_inconsistencies") else "analysed"
            ),
        }

    def _analyse(self, brief: str) -> dict[str, Any]:
        content = agent_llm_complete(brief, system=SUPERVISOR_SYSTEM, temperature=0)
        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {
                "case_narrative": content,
                "cross_inconsistencies": [],
                "information_gaps": [],
                "critical_leads": [],
                "recommended_actions": [],
                "overall_assessment": "Could not parse structured response.",
                "risk_level": "medium",
            }

    def _build_report(self, case_id: str, analysis: dict[str, Any]) -> str:
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        risk = analysis.get("risk_level", "unknown").upper()
        risk_emoji = {
            "LOW": "🟢", "MEDIUM": "🟡",
            "HIGH": "🟠", "CRITICAL": "🔴"
        }.get(risk, "⚪")

        lines = [
            f"# Consolidated Investigation Report — Case {case_id}",
            f"_Generated: {ts}_  |  Risk: {risk_emoji} {risk}\n",
            "---\n",
            "## Case Narrative",
            analysis.get("case_narrative", ""),
            "\n## Overall Assessment",
            analysis.get("overall_assessment", ""),
        ]

        if analysis.get("cross_inconsistencies"):
            lines += ["\n## ⚠ Cross-Agent Inconsistencies"]
            lines += [f"- {i}" for i in analysis["cross_inconsistencies"]]

        if analysis.get("legal_flags"):
            lines += ["\n## 🚨 Legal Flags"]
            lines += [f"- {f}" for f in analysis["legal_flags"]]

        if analysis.get("information_gaps"):
            lines += ["\n## 🔍 Information Gaps"]
            lines += [f"- {g}" for g in analysis["information_gaps"]]

        if analysis.get("critical_leads"):
            lines += ["\n## 🎯 Critical Leads"]
            lines += [f"- {l}" for l in analysis["critical_leads"]]

        if analysis.get("recommended_actions"):
            lines += ["\n## ✅ Recommended Actions"]
            lines += [f"{i+1}. {a}" for i, a in enumerate(analysis["recommended_actions"])]

        return "\n".join(lines)