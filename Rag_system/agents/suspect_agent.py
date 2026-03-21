from orchestration.celery_app import celery_app
from orchestration.blackboard import post_message

from core.retrieval.hybrid_retriever import HybridRetriever
from core.embeddings.local_embedder import LocalEmbedder
from stores.qdrant_store import QdrantStore
from core.retrieval.bm25_retriever import BM25Retriever

from query.context_builder import build_prompt, SYSTEM_PROMPT
from core.generation.llm_client import LLMClient


embedder = LocalEmbedder()
qdrant = QdrantStore()
bm25 = BM25Retriever()

retriever = HybridRetriever(embedder, qdrant, bm25)
llm = LLMClient()


@celery_app.task
def suspect_agent(case_id: str):

    query = "What information do case documents provide about the suspect's actions, movements, or identity?"

    chunks = retriever.search(query=query, case_id=case_id)

    if not chunks:
        post_message(case_id, "SuspectAgent", "No suspect information found.", 0.3)
        return

    prompt, sources = build_prompt(query, chunks)

    answer = llm.generate(prompt).content

    post_message(
        case_id,
        "SuspectAgent",
        answer,
        confidence=0.8
    )