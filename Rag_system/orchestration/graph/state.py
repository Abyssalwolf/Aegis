"""
LangGraph state schema for the Police Investigation Multi-Agent System.
"""

from __future__ import annotations
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
import operator


FILE_TYPE = Literal[
    "fir",
    "case_diary",
    "statement",
    "scene_of_crime",
    "forensic",
    "seizure",
    "arrest_remand",
]

AGENT_FOR_FILE: dict[str, FILE_TYPE] = {
    "fir": "fir",
    "case_diary": "case_diary",
    "statement": "statement",
    "scene_of_crime": "scene_of_crime",
    "forensic": "forensic",
    "seizure": "seizure",
    "arrest_remand": "arrest_remand",
}


class BlackboardMessage(BaseModel):
    agent_id: str
    file_type: FILE_TYPE
    case_id: str
    timestamp: str
    summary: str
    key_entities: list[str] = Field(default_factory=list)
    inconsistencies: list[str] = Field(default_factory=list)
    rag_queries_made: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    raw_extracted: dict[str, Any] = Field(default_factory=dict)


class InvestigationState(BaseModel):
    """
    The single shared state object that every LangGraph node reads and writes.
    blackboard uses operator.add so each agent APPENDS rather than replaces.
    """

    # Inputs
    case_id: str = ""
    file_type: FILE_TYPE = "fir"
    file_path: str = ""
    file_content: str = ""

    # Routing
    assigned_agent: str = ""
    needs_rag: bool = False

    # Blackboard — append-only, all agents write here
    blackboard: Annotated[list[BlackboardMessage], operator.add] = Field(
        default_factory=list
    )

    # Supervisor output
    supervisor_report: str = ""
    cross_inconsistencies: list[str] = Field(default_factory=list)
    final_status: Literal["pending", "analysed", "needs_review"] = "pending"

    class Config:
        arbitrary_types_allowed = True
