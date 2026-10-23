"""
jp_fin_parser/extraction/exceptions.py

Exceptions raised by the extraction layer.
"""

from __future__ import annotations


class NeedsOCR(Exception):
    """
    Raised when the PDF page has an insufficient text layer and cannot be
    parsed with pdfplumber alone.

    Downstream code should catch this and route the page to the VLM fallback.
    """

    def __init__(self, page_number: int, word_count: int, threshold: int) -> None:
        self.page_number = page_number
        self.word_count = word_count
        self.threshold = threshold
        super().__init__(
            f"Page {page_number} has only {word_count} words "
            f"(threshold={threshold}); OCR / VLM fallback required."
        )
