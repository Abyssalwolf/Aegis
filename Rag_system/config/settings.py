from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_text_collection: str = Field(default="case_text_chunks", env="QDRANT_TEXT_COLLECTION")
    qdrant_image_collection: str = Field(default="case_image_chunks", env="QDRANT_IMAGE_COLLECTION")

    # --- Embedder ---
    embedder_model: str = Field(default="BAAI/bge-base-en-v1.5", env="EMBEDDER_MODEL")
    embedder_device: str = Field(default="cpu", env="EMBEDDER_DEVICE")
    embedding_dim: int = Field(default=768, env="EMBEDDING_DIM")

    # --- Reranker ---
    reranker_model: str = Field(default="BAAI/bge-reranker-base", env="RERANKER_MODEL")
    reranker_top_k: int = Field(default=5, env="RERANKER_TOP_K")
    retrieval_top_k: int = Field(default=50, env="RETRIEVAL_TOP_K")  # before reranking

    # --- Chunking ---
    chunk_max_tokens: int = Field(default=512, env="CHUNK_MAX_TOKENS")
    chunk_min_tokens: int = Field(default=50, env="CHUNK_MIN_TOKENS")
    semantic_similarity_threshold: float = Field(default=0.3, env="SEMANTIC_SIMILARITY_THRESHOLD")
    # Drop threshold: if cosine similarity between adjacent sentences drops below
    # (mean - threshold * std), cut a chunk boundary there.

    # --- Ollama (local LLM for query rewriting + generation) ---
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:3b", env="OLLAMA_MODEL")
    query_rewrite_count: int = Field(default=2, env="QUERY_REWRITE_COUNT")

    # --- Storage ---
    document_store_path: Path = Field(default=Path("data/document_store.db"), env="DOCUMENT_STORE_PATH")
    bm25_index_path: Path = Field(default=Path("data/bm25_index.pkl"), env="BM25_INDEX_PATH")

    # --- Ingestion ---
    docling_device: str = Field(default="cpu", env="DOCLING_DEVICE")
    image_dpi: int = Field(default=200, env="IMAGE_DPI")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
