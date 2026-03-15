from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# Let's create a quick dummy case document to ingest 
# so our agents have something to read and work with.
# We'll write it out and then ingest it.

sample_text = """
CASE REPORT - Incident #992

Date: October 24, 2025
Location: First National Bank, Downtown Branch
Suspect: John Doe, approx 6ft tall, wearing a black hoodie and blue jeans.
Actions: The suspect entered the bank at 14:32. He approached the teller and demanded cash.
CCTV: Security cameras caught the suspect entering a silver sedan with license plate XYZ-123 at 14:38.

Timeline:
14:30 - Suspect parks car outside.
14:32 - Suspect enters bank.
14:35 - Suspect receives cash.
14:38 - Suspect flees in silver sedan.

Witness Statement: The bank teller mentioned the suspect seemed nervous and kept checking his watch.
"""

with open("dummy_case.txt", "w") as f:
    f.write(sample_text)

# We can ingest it using the pipeline, but the pipeline requires PDF or Images according to the code.
# Let's use the underlying text_to_chunks and upsert logic directly for this quick test
from core.documents.models import DocumentMetadata, DocumentRecord, DocumentStatus, ChunkType
from stores.document_store import DocumentStore

metadata = DocumentMetadata(
    source_path="dummy_case.txt",
    filename="dummy_case.txt",
    file_type="txt",
    case_id="1"
)
record = DocumentRecord(metadata=metadata, status=DocumentStatus.COMPLETED)

doc_store = DocumentStore()
doc_store.create(record)

chunks = pipeline._text_to_chunks(
    text=sample_text,
    document_id=record.document_id,
    chunk_type=ChunkType.TEXT,
    case_id="1",
    source_path="dummy_case.txt"
)

chunks = pipeline._embed_chunks(chunks)
pipeline.qdrant.upsert_chunks(chunks)

print(f"Successfully ingested dummy case document with {len(chunks)} chunks!")
