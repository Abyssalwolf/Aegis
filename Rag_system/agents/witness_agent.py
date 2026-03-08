from orchestration.celery_app import celery_app
from orchestration.blackboard import post_message

from core.retrieval.hybrid_retriever import HybridRetriever
from core.embeddings.local_embedder import LocalEmbedder
from stores.qdrant_store import QdrantStore
from core.retrieval.bm25_retriever import BM25Retriever

from query.context_builder import build_prompt, SYSTEM_PROMPT
from core.generation.llm_client import OllamaClient


# Initialize components once
embedder = LocalEmbedder()
qdrant = QdrantStore()
bm25 = BM25Retriever()

retriever = HybridRetriever(
    embedder=embedder,
    qdrant=qdrant,
    bm25=bm25,
)

llm = OllamaClient()


@celery_app.task
def witness_agent(case_id: str):

    query = "What do witness statements say about the suspect or timeline?"

    # 1️⃣ Retrieve evidence from RAG
    chunks = retriever.search(
        query=query,
        case_id=case_id
    )

    if not chunks:
        post_message(case_id, "WitnessAgent", "No witness information found.", 0.3)
        return

    # 2️⃣ Build prompt
    prompt, sources = build_prompt(query, chunks)

    # 3️⃣ Generate analysis
    answer = llm.generate(prompt=prompt, system=SYSTEM_PROMPT)

    # 4️⃣ Post to blackboard
    post_message(
        case_id,
        "WitnessAgent",
        answer,
        confidence=0.8
    )