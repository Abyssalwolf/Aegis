"""
Image loader using Docling.
Accepts image files (PNG, JPG, TIFF, BMP) or raw image bytes
(from embedded PDF images) and returns extracted text via OCR.

For images that are primarily visual (photos, diagrams) with little
text, Docling will return minimal text. Future enhancement: plug in
a local VLM caption model here.
"""

import io
import logging
from pathlib import Path
from dataclasses import dataclass

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions
from docling.document_converter import ImageFormatOption
from docling.datamodel.pipeline_options import EasyOcrOptions

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@dataclass
class ImageLoadResult:
    text: str               # OCR-extracted text (or empty if purely visual)
    source: str             # filename or "embedded_image_page_{n}"
    width: int
    height: int
    metadata: dict


class ImageLoader:
    """
    Loads image files or raw image bytes using Docling's image pipeline.
    Returns extracted text. Images with no meaningful text will return
    an empty string — the pipeline handles this gracefully.
    """

    def __init__(self):
        self._converter: DocumentConverter | None = None

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            logger.info("Initializing Docling image converter.")

            # Use EasyOCR for local OCR without external API calls
            ocr_options = EasyOcrOptions(force_full_page_ocr=True)
            pipeline_options = PipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = ocr_options

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        return self._converter

    def load_file(self, file_path: str | Path) -> ImageLoadResult:
        """Load an image from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        logger.info(f"Loading image file: {path.name}")
        return self._process(str(path), source=path.name)

    def load_bytes(self, image_bytes: bytes, source_label: str = "embedded_image") -> ImageLoadResult:
        """
        Load an image from raw bytes (e.g. extracted from a PDF).
        Writes to a temp file since Docling requires a file path.
        """
        import tempfile, os

        # Detect format from magic bytes
        suffix = self._detect_format(image_bytes)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            return self._process(tmp_path, source=source_label)
        finally:
            os.unlink(tmp_path)

    def _process(self, file_path: str, source: str) -> ImageLoadResult:
        converter = self._get_converter()
        result = converter.convert(file_path)
        doc = result.document

        text = doc.export_to_markdown().strip()

        # Get image dimensions if available
        width, height = 0, 0
        if hasattr(doc, "pages") and doc.pages:
            page = doc.pages[0]
            if hasattr(page, "size"):
                width = int(page.size.width)
                height = int(page.size.height)

        logger.info(f"Image loaded: '{source}' → {len(text)} chars extracted.")

        return ImageLoadResult(
            text=text,
            source=source,
            width=width,
            height=height,
            metadata={"source": source, "width": width, "height": height},
        )

    @staticmethod
    def _detect_format(data: bytes) -> str:
        """Detect image format from magic bytes."""
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ".png"
        if data[:3] == b'\xff\xd8\xff':
            return ".jpg"
        if data[:4] in (b'II*\x00', b'MM\x00*'):
            return ".tiff"
        return ".png"  # safe default
