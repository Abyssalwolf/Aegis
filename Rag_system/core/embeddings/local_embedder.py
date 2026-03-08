"""
Local embedder backed by sentence-transformers.
Lazy-loads the model on first use to avoid consuming RAM during import.
"""

import logging
from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Thin wrapper around SentenceTransformer for batch encoding.
    Default model: BAAI/bge-base-en-v1.5 (768-dim, CPU-friendly).
    """

    def __init__(
        self,
        model_name: str = settings.embedder_model,
        device: str = settings.embedder_device,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedder model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into a (N, D) float32 numpy array.
        Normalizes embeddings by default (required for cosine similarity).
        """
        if not texts:
            return np.array([])

        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience method for a single string."""
        return self.encode([text])[0]

    def unload(self) -> None:
        """
        Explicitly free the model from memory.
        Call this after bulk ingestion to reclaim RAM before loading
        the reranker or LLM.
        """
        if self._model is not None:
            logger.info("Unloading embedder model to free RAM.")
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    @property
    def dim(self) -> int:
        return settings.embedding_dim
