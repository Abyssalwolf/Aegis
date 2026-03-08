import json
from datetime import datetime
import redis

# Redis client (simple, local)
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)


def _now():
    return datetime.utcnow().isoformat()


def post_message(case_id: int, agent: str, content: str, confidence: float):
    """
    Post an observation message to the blackboard
    """
    message = {
        "agent": agent,
        "type": "observation",
        "content": content,
        "confidence": confidence,
        "timestamp": _now()
    }

    key = f"case:{case_id}:messages"
    redis_client.rpush(key, json.dumps(message))


def post_anomaly(case_id: int, agent: str, content: str, confidence: float):
    """
    Post an anomaly message to the blackboard
    """
    anomaly = {
        "agent": agent,
        "type": "anomaly",
        "content": content,
        "confidence": confidence,
        "timestamp": _now()
    }

    key = f"case:{case_id}:anomalies"
    redis_client.rpush(key, json.dumps(anomaly))


def read_messages(case_id: int):
    """
    Read all observation messages
    """
    key = f"case:{case_id}:messages"
    raw = redis_client.lrange(key, 0, -1)
    return [json.loads(msg) for msg in raw]


def read_anomalies(case_id: int):
    """
    Read all anomaly messages
    """
    key = f"case:{case_id}:anomalies"
    raw = redis_client.lrange(key, 0, -1)
    return [json.loads(msg) for msg in raw]


def set_case_status(case_id: int, status: str):
    """
    Set case analysis status (running / completed)
    """
    key = f"case:{case_id}:status"
    redis_client.set(key, status)


def get_case_status(case_id: int):
    key = f"case:{case_id}:status"
    return redis_client.get(key)