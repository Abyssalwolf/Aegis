from orchestration.celery_app import celery_app
from agents.witness_agent import witness_agent
from agents.suspect_agent import suspect_agent
from agents.cctv_agent import cctv_agent
from agents.timeline_agent import timeline_agent
from agents.supervisor_agent import supervisor_agent


@celery_app.task
def analyze_case(case_id: str):

    witness_agent.delay(case_id)
    suspect_agent.delay(case_id)
    cctv_agent.delay(case_id)

    timeline_agent.delay(case_id)

    supervisor_agent.delay(case_id)