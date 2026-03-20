"""
Specialist agents — one per document type.
Each is a thin subclass of BaseAgent with its own agent_id and file_type.
All heavy logic lives in BaseAgent (base_agent.py).
"""

from agents.base_agent import BaseAgent


class FIRAgent(BaseAgent):
    agent_id = "fir_agent"
    file_type = "fir"


class CaseDiaryAgent(BaseAgent):
    agent_id = "case_diary_agent"
    file_type = "case_diary"


class StatementAgent(BaseAgent):
    agent_id = "statement_agent"
    file_type = "statement"


class SceneOfCrimeAgent(BaseAgent):
    agent_id = "scene_of_crime_agent"
    file_type = "scene_of_crime"


class ForensicAgent(BaseAgent):
    agent_id = "forensic_agent"
    file_type = "forensic"


class SeizureAgent(BaseAgent):
    agent_id = "seizure_agent"
    file_type = "seizure"


class ArrestRemandAgent(BaseAgent):
    agent_id = "arrest_remand_agent"
    file_type = "arrest_remand"


# Maps file_type string → agent instance
AGENT_REGISTRY: dict[str, BaseAgent] = {
    "fir": FIRAgent(),
    "case_diary": CaseDiaryAgent(),
    "statement": StatementAgent(),
    "scene_of_crime": SceneOfCrimeAgent(),
    "forensic": ForensicAgent(),
    "seizure": SeizureAgent(),
    "arrest_remand": ArrestRemandAgent(),
}
