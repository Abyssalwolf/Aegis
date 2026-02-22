"""
Text cleaner for post-OCR and post-Docling text normalization.
Handles common artifacts from PDF extraction and OCR.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw extracted text before chunking.
    Conservative — preserves content, only removes clear noise.
    """

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_whitespace(text)
        text = self._remove_repeated_chars(text)
        text = self._deduplicate_lines(text)

        return text.strip()

    def _normalize_unicode(self, text: str) -> str:
        """Normalize to NFC form — handles accented chars from OCR."""
        return unicodedata.normalize("NFC", text)

    def _remove_control_chars(self, text: str) -> str:
        """Remove non-printable control characters except newlines and tabs."""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _fix_hyphenation(self, text: str) -> str:
        """
        Rejoin words split by end-of-line hyphens (common in PDFs).
        e.g. "inves-\ntigation" → "investigation"
        """
        return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Collapse multiple spaces/tabs to single space.
        Preserve paragraph breaks (double newlines) but collapse
        single newlines within paragraphs.
        """
        # Preserve double newlines as paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse single newlines within paragraphs (not paragraph breaks)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    def _remove_repeated_chars(self, text: str) -> str:
        """
        Remove sequences of repeated special chars — common Docling artifact.
        e.g. "........" or "________" or "- - - - -"
        Preserves ellipsis (...) and similar intentional punctuation.
        """
        # Remove 4+ repeated non-alphanumeric chars (except spaces)
        text = re.sub(r'([^\w\s])\1{3,}', '', text)
        # Remove lines that are purely dashes, underscores, dots (page separators)
        text = re.sub(r'^[\s\-_=.]{4,}$', '', text, flags=re.MULTILINE)
        return text

    def _deduplicate_lines(self, text: str) -> str:
        """
        Remove consecutive duplicate lines — can happen with OCR on
        documents with repeated headers/footers.
        """
        lines = text.split('\n')
        deduped = []
        prev = None
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev:
                continue
            deduped.append(line)
            prev = stripped if stripped else prev
        return '\n'.join(deduped)
