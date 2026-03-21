from orchestration.celery_app import celery_app
from orchestration.blackboard import (
    read_messages,
    post_anomaly
)

from core.generation.llm_client import LLMClient


llm = LLMClient()


@celery_app.task
def timeline_agent(case_id: str):

    messages = read_messages(case_id)

    if not messages:
        return

    combined = "\n".join([m["content"] for m in messages])

    prompt = f"""
Analyze the following investigation observations.

Identify timeline inconsistencies or contradictions.

Observations:
{combined}

If there are contradictions or suspicious timeline issues, explain them.
If everything is consistent, say "No anomaly detected".
"""

    analysis = llm.generate(prompt).content

    if "no anomaly" not in analysis.lower():

        post_anomaly(
            case_id,
            "TimelineAgent",
            analysis,
            confidence=0.9
        )