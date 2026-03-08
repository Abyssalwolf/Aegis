"""
Qdrant vector store.
Manages two collections:
  - case_text_chunks: for text chunks from PDFs
  - case_image_chunks: for OCR/caption text from images

Payload stored alongside each vector:
  chunk_id, document_id, chunk_type, text, page_number,
  chunk_index, token_count, parent_text, case_id, source_path
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

from config.settings import settings
from core.documents.models import Chunk, ChunkType, RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantStore:

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._ensure_collections()

    def _collection_for(self, chunk_type: ChunkType) -> str:
        if chunk_type == ChunkType.IMAGE:
            return settings.qdrant_image_collection
        return settings.qdrant_text_collection

    def _ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        for collection_name in [
            settings.qdrant_text_collection,
            settings.qdrant_image_collection,
        ]:
            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name not in existing:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """
        Upsert a list of chunks into the appropriate collection.
        Each chunk must have an embedding set.
        """
        # Group by collection
        text_points: list[PointStruct] = []
        image_points: list[PointStruct] = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding — skipping.")
                continue

            point = PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_type": chunk.chunk_type.value,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "parent_text": chunk.parent_text,
                    **chunk.metadata,   # case_id, source_path, etc.
                },
            )

            if chunk.chunk_type == ChunkType.IMAGE:
                image_points.append(point)
            else:
                text_points.append(point)

        if text_points:
            self.client.upsert(
                collection_name=settings.qdrant_text_collection,
                points=text_points,
            )
            logger.info(f"Upserted {len(text_points)} text chunks to Qdrant.")

        if image_points:
            self.client.upsert(
                collection_name=settings.qdrant_image_collection,
                points=image_points,
            )
            logger.info(f"Upserted {len(image_points)} image chunks to Qdrant.")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = settings.retrieval_top_k,
        chunk_type: ChunkType = ChunkType.TEXT,
        case_id: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Dense vector search against the specified collection.
        Optionally filter by case_id.
        """
        collection = self._collection_for(chunk_type)

        query_filter = None
        if case_id:
            query_filter = Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            )

        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        retrieved = []
        for r in results:
            payload = r.payload or {}
            chunk = Chunk(
                chunk_id=payload.get("chunk_id", str(r.id)),
                document_id=payload.get("document_id", ""),
                chunk_type=ChunkType(payload.get("chunk_type", "text")),
                text=payload.get("text", ""),
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index", 0),
                token_count=payload.get("token_count", 0),
                parent_text=payload.get("parent_text"),
                metadata={k: v for k, v in payload.items()
                           if k not in {"chunk_id", "document_id", "chunk_type",
                                        "text", "page_number", "chunk_index",
                                        "token_count", "parent_text"}},
            )
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=r.score,
                retrieval_method="dense",
            ))

        return retrieved

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        from qdrant_client.models import FilterSelector

        for collection in [settings.qdrant_text_collection, settings.qdrant_image_collection]:
            self.client.delete(
                collection_name=collection,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )]
                    )
                ),
            )
        logger.info(f"Deleted all chunks for document_id: {document_id}")

    def get_all_texts(self, case_id: Optional[str] = None) -> list[tuple[str, str]]:
        """
        Scroll through all text chunks and return (chunk_id, text) pairs.
        Used to build/rebuild the BM25 index.
        """
        pairs: list[tuple[str, str]] = []
        scroll_filter = None
        if case_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            )

        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=settings.qdrant_text_collection,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in results:
                payload = r.payload or {}
                pairs.append((payload.get("chunk_id", str(r.id)), payload.get("text", "")))

            if next_offset is None:
                break
            offset = next_offset

        return pairs
