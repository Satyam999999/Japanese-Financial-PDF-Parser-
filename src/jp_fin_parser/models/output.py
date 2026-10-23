"""
jp_fin_parser/models/output.py

Top-level output envelope for a single PDF page parse run.
This is what gets serialised to JSON / CSV.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .table import FinancialTable


class ParsedPage(BaseModel):
    """
    The root output object emitted by the pipeline for one PDF page.

    Fields
    ------
    source_file     – absolute path (or URL) of the source PDF
    page_number     – 0-based page index
    parsed_at       – UTC timestamp of when the parse completed
    parser_version  – semver string of the jp-fin-parser release
    tables          – usually one table per page, but multi-table pages
                      (e.g. continued BS) may produce more
    overall_confidence – weighted average of all row confidences
    warnings        – human-readable warnings that don't block output
    """

    source_file: Optional[str] = None
    page_number: int = Field(..., ge=0)
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parser_version: str = "0.1.0"
    tables: list[FinancialTable] = Field(default_factory=list)
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    def compute_overall_confidence(self) -> None:
        """
        Recompute ``overall_confidence`` from all row scores.
        Call this after all tables have been assembled and validated.
        """
        scores = [
            row.confidence.overall
            for table in self.tables
            for row in table.rows
        ]
        self.overall_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0
