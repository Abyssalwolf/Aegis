import json
from datetime import datetime
from typing import Iterator

import redis

from config.settings import settings

redis_client = redis.from_url(
    settings.redis_url,
    decode_responses=True,
)


def _now():
    return datetime.utcnow().isoformat()


def post_message(case_id: int, agent: str, content: str, confidence: float):
    """Post an observation to the blackboard."""
    message = {
        "agent": agent,
        "type": "observation",
        "content": content,
        "confidence": confidence,
        "timestamp": _now()
    }
    key = f"case:{case_id}:messages"
    redis_client.rpush(key, json.dumps(message))
    # also broadcast live so SSE picks it up
    redis_client.publish(f"blackboard:{case_id}", json.dumps(message))


def post_anomaly(case_id: int, agent: str, content: str, confidence: float):
    """Post an anomaly to the blackboard."""
    anomaly = {
        "agent": agent,
        "type": "anomaly",
        "content": content,
        "confidence": confidence,
        "timestamp": _now()
    }
    key = f"case:{case_id}:anomalies"
    redis_client.rpush(key, json.dumps(anomaly))
    redis_client.publish(f"blackboard:{case_id}", json.dumps(anomaly))


def post_finding(case_id: int, agent: str, file_type: str, payload: dict):
    """
    Called by document agents (FIR, Forensic, Statement, etc.)
    after analysing an uploaded file.
    Stores structured finding AND bridges into messages/anomalies
    so everything stays in one unified feed.
    """
    finding = {
        "agent": agent,
        "type": "finding",
        "file_type": file_type,
        "content": payload.get("summary", ""),
        "confidence": 0.9,
        "timestamp": _now(),
        "key_entities": payload.get("key_entities", []),
        "inconsistencies": payload.get("inconsistencies", []),
        "insights": payload.get("insights", []),
        "rag_queries_made": payload.get("rag_queries_made", []),
        "raw_extracted": payload.get("raw_extracted", {}),
    }

    # persist to findings list
    key = f"case:{case_id}:findings"
    redis_client.rpush(key, json.dumps(finding))
    redis_client.expire(key, 60 * 60 * 24 * 7)

    # broadcast live
    redis_client.publish(f"blackboard:{case_id}", json.dumps(finding))

    # bridge into existing channels
    post_message(case_id, agent, finding["content"], finding["confidence"])
    for issue in finding["inconsistencies"]:
        post_anomaly(case_id, agent, issue, 0.85)


def post_insight(case_id: int, agent: str, content: str, confidence: float):
    """Post a cross-agent insight from the supervisor."""
    insight = {
        "agent": agent,
        "type": "insight",
        "content": content,
        "confidence": confidence,
        "timestamp": _now()
    }
    key = f"case:{case_id}:insights"
    redis_client.rpush(key, json.dumps(insight))
    redis_client.publish(f"blackboard:{case_id}", json.dumps(insight))


def read_messages(case_id: int):
    key = f"case:{case_id}:messages"
    return [json.loads(m) for m in redis_client.lrange(key, 0, -1)]


def read_anomalies(case_id: int):
    key = f"case:{case_id}:anomalies"
    return [json.loads(m) for m in redis_client.lrange(key, 0, -1)]


def read_findings(case_id: int):
    key = f"case:{case_id}:findings"
    return [json.loads(m) for m in redis_client.lrange(key, 0, -1)]


def read_insights(case_id: int):
    key = f"case:{case_id}:insights"
    return [json.loads(m) for m in redis_client.lrange(key, 0, -1)]


def read_all(case_id: int) -> dict:
    """Read everything from the blackboard — used by dashboard."""
    return {
        "messages":  read_messages(case_id),
        "anomalies": read_anomalies(case_id),
        "findings":  read_findings(case_id),
        "insights":  read_insights(case_id),
    }


def set_case_status(case_id: int, status: str):
    redis_client.set(f"case:{case_id}:status", status)


def get_case_status(case_id: int):
    return redis_client.get(f"case:{case_id}:status")


def subscribe_to_case(case_id: int) -> Iterator[dict]:
    """
    Live generator — yields every message published to the blackboard channel.
    Used by the SSE endpoint so the dashboard updates in real time.
    Run in a background thread.
    """
    r = redis.from_url(settings.redis_url, decode_responses=True)
    ps = r.pubsub()
    ps.subscribe(f"blackboard:{case_id}")
    for raw in ps.listen():
        if raw["type"] != "message":
            continue
        try:
            yield json.loads(raw["data"])
        except Exception:
            continue


def format_brief(case_id: int) -> str:
    """
    Markdown summary of the full blackboard for a case.
    Fed to the supervisor agent as context.
    """
    lines = [f"# Blackboard — Case {case_id}\n"]

    findings = read_findings(case_id)
    if findings:
        lines.append("## Document Findings\n")
        for f in findings:
            lines.append(f"### [{f['agent']}] {f['file_type']}  —  {f['timestamp']}")
            lines.append(f"{f['content']}\n")
            if f.get("key_entities"):
                lines.append("**Entities:** " + ", ".join(f["key_entities"]))
            if f.get("insights"):
                lines.extend(f"- {i}" for i in f["insights"])
            if f.get("inconsistencies"):
                lines.extend(f"- ⚠ {i}" for i in f["inconsistencies"])
            lines.append("---")

    messages = read_messages(case_id)
    if messages:
        lines.append("\n## Observations\n")
        for m in messages:
            lines.append(f"- [{m['agent']}] {m['content']}  (conf: {m['confidence']})")

    anomalies = read_anomalies(case_id)
    if anomalies:
        lines.append("\n## Anomalies\n")
        for a in anomalies:
            lines.append(f"- ⚠ [{a['agent']}] {a['content']}  (conf: {a['confidence']})")

    return "\n".join(lines)