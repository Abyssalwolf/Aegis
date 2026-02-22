"""
Ingestion pipeline.
Orchestrates: load → clean → chunk → embed → store (Qdrant + SQLite).

Handles both PDF and image files. For PDFs, also processes any
embedded images found within the document.

Memory note: embedder is loaded for the duration of ingestion then
explicitly unloaded to free RAM before the API serves queries.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.processors.cleaner import TextCleaner
from core.documents.chunker import SemanticChunker
from core.embeddings.local_embedder import LocalEmbedder
from core.documents.models import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
    RetrievedChunk,
)
from stores.qdrant_store import QdrantStore
from stores.document_store import DocumentStore

logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.image_loader = ImageLoader()
        self.cleaner = TextCleaner()
        self.embedder = LocalEmbedder()
        self.chunker = SemanticChunker(embedder=self.embedder)
        self.qdrant = QdrantStore()
        self.doc_store = DocumentStore()

    def ingest_file(
        self,
        file_path: str | Path,
        case_id: Optional[str] = None,
        officer_id: Optional[str] = None,
        skip_if_exists: bool = True,
    ) -> DocumentRecord:
        """
        Ingest a single PDF or image file.
        Returns the DocumentRecord with final status.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Deduplicate
        if skip_if_exists and self.doc_store.exists_by_path(str(path)):
            logger.info(f"File already ingested, skipping: {path.name}")
            existing = [r for r in self.doc_store.list_all()
                        if r.metadata.source_path == str(path)]
            if existing:
                return existing[0]

        # Determine file type
        if suffix == ".pdf":
            file_type = "pdf"
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
            file_type = "image"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Create document record
        metadata = DocumentMetadata(
            source_path=str(path),
            filename=path.name,
            file_type=file_type,
            case_id=case_id,
            officer_id=officer_id,
        )
        record = DocumentRecord(metadata=metadata, status=DocumentStatus.PROCESSING)
        self.doc_store.create(record)

        try:
            chunks: list[Chunk] = []

            if file_type == "pdf":
                chunks = self._process_pdf(path, record.document_id, case_id)
                # Update page count
                record.metadata.page_count = getattr(
                    self._last_pdf_result, "page_count", None
                )
            else:
                chunks = self._process_image(path, record.document_id, case_id)

            # Embed all chunks in one batch pass
            chunks = self._embed_chunks(chunks)

            # Store in Qdrant
            self.qdrant.upsert_chunks(chunks)

            # Mark complete
            self.doc_store.update_status(
                record.document_id,
                status=DocumentStatus.COMPLETED,
                chunk_count=len(chunks),
            )
            record.status = DocumentStatus.COMPLETED
            record.chunk_count = len(chunks)

            logger.info(
                f"Ingestion complete: '{path.name}' → "
                f"{len(chunks)} chunks stored."
            )

        except Exception as e:
            logger.error(f"Ingestion failed for '{path.name}': {e}", exc_info=True)
            self.doc_store.update_status(
                record.document_id,
                status=DocumentStatus.FAILED,
                error_message=str(e),
            )
            record.status = DocumentStatus.FAILED
            record.error_message = str(e)

        return record

    def ingest_directory(
        self,
        directory: str | Path,
        case_id: Optional[str] = None,
        officer_id: Optional[str] = None,
        recursive: bool = False,
    ) -> list[DocumentRecord]:
        """Ingest all PDFs and images in a directory."""
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

        files = [f for f in directory.glob(pattern) if f.suffix.lower() in supported]
        logger.info(f"Found {len(files)} files to ingest in {directory}.")

        results = []
        for file in files:
            result = self.ingest_file(file, case_id=case_id, officer_id=officer_id)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_pdf(
        self, path: Path, document_id: str, case_id: Optional[str]
    ) -> list[Chunk]:
        result = self.pdf_loader.load(path)
        self._last_pdf_result = result
        chunks: list[Chunk] = []

        # --- Main text ---
        clean_text = self.cleaner.clean(result.text)
        if clean_text:
            text_chunks = self._text_to_chunks(
                clean_text,
                document_id=document_id,
                chunk_type=ChunkType.TEXT,
                case_id=case_id,
                source_path=str(path),
            )
            chunks.extend(text_chunks)

        # --- Embedded images ---
        for i, (img_bytes, page_no) in enumerate(
            zip(result.images, result.image_page_numbers)
        ):
            label = f"embedded_image_page_{page_no}_{i}"
            img_result = self.image_loader.load_bytes(img_bytes, source_label=label)
            if img_result.text:
                clean_img_text = self.cleaner.clean(img_result.text)
                if clean_img_text:
                    img_chunk = Chunk(
                        document_id=document_id,
                        chunk_type=ChunkType.IMAGE,
                        text=clean_img_text,
                        page_number=page_no,
                        chunk_index=len(chunks),
                        token_count=len(clean_img_text.split()),
                        metadata={
                            "case_id": case_id,
                            "source_path": str(path),
                            "image_label": label,
                        },
                    )
                    chunks.append(img_chunk)

        return chunks

    def _process_image(
        self, path: Path, document_id: str, case_id: Optional[str]
    ) -> list[Chunk]:
        result = self.image_loader.load_file(path)
        chunks: list[Chunk] = []

        if result.text:
            clean_text = self.cleaner.clean(result.text)
            if clean_text:
                chunks = self._text_to_chunks(
                    clean_text,
                    document_id=document_id,
                    chunk_type=ChunkType.IMAGE,
                    case_id=case_id,
                    source_path=str(path),
                )

        return chunks

    def _text_to_chunks(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType,
        case_id: Optional[str],
        source_path: str,
    ) -> list[Chunk]:
        semantic_chunks = self.chunker.chunk(text)
        chunks: list[Chunk] = []

        for i, sc in enumerate(semantic_chunks):
            chunk = Chunk(
                document_id=document_id,
                chunk_type=chunk_type,
                text=sc.text,
                chunk_index=i,
                token_count=sc.token_count,
                parent_text=sc.parent_text,
                metadata={
                    "case_id": case_id,
                    "source_path": source_path,
                },
            )
            chunks.append(chunk)

        return chunks

    def _embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Batch embed all chunks."""
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        return chunks
