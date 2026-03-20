from orchestration.celery_app import celery_app
from orchestration.blackboard import (
    read_messages,
    read_anomalies,
    post_message
)

from core.generation.llm_client import LLMClient


llm = LLMClient()


@celery_app.task
def supervisor_agent(case_id: str):

    messages = read_messages(case_id)
    anomalies = read_anomalies(case_id)

    combined_messages = "\n".join([m["content"] for m in messages])
    combined_anomalies = "\n".join([a["content"] for a in anomalies])

    prompt = f"""
You are supervising an AI investigation system.

Observations from agents:
{combined_messages}

Detected anomalies:
{combined_anomalies}

Provide a concise investigation summary and highlight key findings.
"""

    summary = llm.generate(prompt)

    post_message(
        case_id,
        "SupervisorAgent",
        summary,
        confidence=0.95
    )