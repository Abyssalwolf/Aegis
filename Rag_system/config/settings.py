from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # --- Qdrant ---
    # Set QDRANT_URL to a full URL (e.g. ngrok https URL) — overrides host+port when present
    qdrant_url: str = Field(default="", env="QDRANT_URL")
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")
    qdrant_text_collection: str = Field(default="case_text_chunks", env="QDRANT_TEXT_COLLECTION")
    qdrant_image_collection: str = Field(default="case_image_chunks", env="QDRANT_IMAGE_COLLECTION")

    # --- Embedder ---
    embedder_model: str = Field(default="BAAI/bge-base-en-v1.5", env="EMBEDDER_MODEL")
    embedder_device: str = Field(default="cpu", env="EMBEDDER_DEVICE")
    embedding_dim: int = Field(default=768, env="EMBEDDING_DIM")

    # --- Reranker ---
    reranker_model: str = Field(default="BAAI/bge-reranker-base", env="RERANKER_MODEL")
    reranker_top_k: int = Field(default=7, env="RERANKER_TOP_K")
    retrieval_top_k: int = Field(default=50, env="RETRIEVAL_TOP_K")  # before reranking

    # --- Chunking ---
    chunk_max_tokens: int = Field(default=512, env="CHUNK_MAX_TOKENS")
    chunk_min_tokens: int = Field(default=50, env="CHUNK_MIN_TOKENS")
    semantic_similarity_threshold: float = Field(default=0.3, env="SEMANTIC_SIMILARITY_THRESHOLD")
    # Drop threshold: if cosine similarity between adjacent sentences drops below
    # (mean - threshold * std), cut a chunk boundary there.

    # --- Legacy Ollama HTTP API (used only when LLM_BASE_URL is unset) ---
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:3b", env="OLLAMA_MODEL")
    query_rewrite_count: int = Field(default=2, env="QUERY_REWRITE_COUNT")
    # Max tokens for final RAG answer (answer + chain-of-thought in output)
    llm_max_tokens: int = Field(default=8192, env="LLM_MAX_TOKENS")
    # Ollama `think` flag for all /api/generate calls (rewrite + answer). Rewrites still
    # only use split `content`; reasoning is omitted unless include_reasoning=True.
    ollama_enable_thinking: bool = Field(default=True, env="OLLAMA_ENABLE_THINKING")
    # Rewrites are short, but thinking models use the same completion budget for reasoning
    # *and* visible lines (Ollama `num_predict`; some APIs split reasoning into a separate field).
    query_rewrite_max_tokens: int = Field(default=2048, env="QUERY_REWRITE_MAX_TOKENS")
    # Max prior chat turns (user+assistant messages) injected into the answer prompt.
    chat_history_max_messages: int = Field(default=16, env="CHAT_HISTORY_MAX_MESSAGES")
    # Soft token budget for user prompt: prior chat + document passages (+ headers).
    # Chunks are added until this budget (minus history) is reached.
    rag_combined_context_budget_tokens: int = Field(
        default=16000, env="RAG_COMBINED_CONTEXT_BUDGET_TOKENS"
    )
    # Never shrink passage area below this many (estimated) tokens if history is huge.
    rag_context_passages_min_tokens: int = Field(
        default=3500, env="RAG_CONTEXT_PASSAGES_MIN_TOKENS"
    )

    # --- Primary LLM: OpenAI-compatible API (RAG query, agents, supervisor, classifier) ---
    # When set, overrides Ollama for all OllamaClient usage. e.g. Modal, vLLM — POST {base}/v1/chat/completions
    llm_base_url: str = Field(default="", env="LLM_BASE_URL")
    llm_model: str = Field(default="", env="LLM_MODEL")
    llm_api_key: str = Field(default="", env="LLM_API_KEY")

    # --- Redis (Celery broker + Insights blackboard) ---
    # Examples: redis://localhost:6379/0
    # Remote: redis://:PASSWORD@192.168.1.50:6379/0  (use TLS if your host provides rediss://)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # --- Storage ---
    document_store_path: Path = Field(default=Path("data/document_store.db"), env="DOCUMENT_STORE_PATH")
    bm25_index_path: Path = Field(default=Path("data/bm25_index.pkl"), env="BM25_INDEX_PATH")
    rag_database_url: str = Field(default="", env="RAG_DATABASE_URL")

    # --- Ingestion ---
    docling_device: str = Field(default="cpu", env="DOCLING_DEVICE")
    image_dpi: int = Field(default=200, env="IMAGE_DPI")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")

    memory_dir: str = Field(default="data/memory_store", env="MEMORY_DIR")
    upload_dir: str = Field(default="data/uploads", env="UPLOAD_DIR")


settings = Settings()
