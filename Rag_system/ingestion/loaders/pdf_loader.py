"""
PDF loader using Docling.
Extracts structured text (with layout, tables, reading order) and
embedded images from PDF files. Images are returned as separate
raw bytes for the image loader to handle.
"""

import logging
from pathlib import Path
from dataclasses import dataclass

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

logger = logging.getLogger(__name__)


@dataclass
class PDFLoadResult:
    text: str                           # Full extracted markdown text
    page_count: int
    images: list[bytes]                 # Raw bytes of embedded images
    image_page_numbers: list[int]       # Which page each image came from
    metadata: dict


class PDFLoader:
    """
    Loads a PDF using Docling. Returns clean markdown text and
    any embedded images as raw bytes.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._converter: DocumentConverter | None = None

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            logger.info("Initializing Docling DocumentConverter (CPU mode).")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True           # OCR for scanned pages
            pipeline_options.do_table_structure = True
            pipeline_options.images_scale = 2.0      # Higher quality image extraction

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        return self._converter

    def load(self, file_path: str | Path) -> PDFLoadResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        logger.info(f"Loading PDF: {path.name}")
        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document

        # Export to markdown — preserves headings, tables, reading order
        markdown_text = doc.export_to_markdown()
        page_count = len(doc.pages) if hasattr(doc, "pages") else 0

        # Extract embedded images
        images: list[bytes] = []
        image_page_numbers: list[int] = []
        if hasattr(doc, "pictures"):
            for picture in doc.pictures:
                try:
                    img_bytes = picture.get_image(doc)
                    if img_bytes:
                        images.append(img_bytes)
                        # Page number from picture location
                        page_no = getattr(picture.prov[0], "page_no", 0) if picture.prov else 0
                        image_page_numbers.append(page_no)
                except Exception as e:
                    logger.warning(f"Failed to extract image from PDF: {e}")

        logger.info(
            f"PDF loaded: {page_count} pages, "
            f"{len(markdown_text)} chars, "
            f"{len(images)} embedded images."
        )

        return PDFLoadResult(
            text=markdown_text,
            page_count=page_count,
            images=images,
            image_page_numbers=image_page_numbers,
            metadata={
                "filename": path.name,
                "source_path": str(path),
                "file_size_bytes": path.stat().st_size,
            },
        )
