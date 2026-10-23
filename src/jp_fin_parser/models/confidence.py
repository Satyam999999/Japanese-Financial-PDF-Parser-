"""
jp_fin_parser/models/confidence.py

Confidence scoring and validation flag types.
These are attached to every FinancialRow so downstream consumers
can filter by quality without re-running any logic.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Sequence

from pydantic import BaseModel, Field, field_validator


class ValidationFlag(StrEnum):
    # Extraction-layer signals
    LOW_EXTRACTION_CONFIDENCE = "low_extraction_confidence"
    OVERLAPPING_BBOX = "overlapping_bbox"
    MERGED_CELL_SUSPECTED = "merged_cell_suspected"

    # Preprocessing signals
    AMBIGUOUS_SIGN = "ambiguous_sign"          # could not resolve △/▲ direction
    FULL_WIDTH_DIGITS = "full_width_digits"    # informational: was normalized
    UNKNOWN_NIL_MARKER = "unknown_nil_marker"  # nil-like char not in known set

    # Layout signals
    INDENTATION_AMBIGUOUS = "indentation_ambiguous"
    COLUMN_MISALIGNED = "column_misaligned"
    HEADER_NOT_FOUND = "header_not_found"

    # Parsing signals
    VALUE_PARSE_FAILED = "value_parse_failed"
    PERIOD_LABEL_MISSING = "period_label_missing"
    ROW_TYPE_INFERRED = "row_type_inferred"    # classifier fell back to heuristic

    # Validation signals
    ARITHMETIC_MISMATCH = "arithmetic_mismatch"
    BALANCE_SHEET_IMBALANCE = "balance_sheet_imbalance"
    KNOWN_SIGN_VIOLATED = "known_sign_violated"

    # Meta
    CORRECTION_MARKER_PRESENT = "correction_marker_present"
    VLM_FALLBACK_USED = "vlm_fallback_used"


class ConfidenceScore(BaseModel):
    """
    Composite confidence for a single row.
    Each sub-score is in [0, 1]; the overall score is their weighted average.
    Weights are intentionally visible so they can be tuned via config.
    """

    extraction: float = Field(1.0, ge=0.0, le=1.0)
    layout: float = Field(1.0, ge=0.0, le=1.0)
    parsing: float = Field(1.0, ge=0.0, le=1.0)
    validation: float = Field(1.0, ge=0.0, le=1.0)

    # Weights (must sum to 1 — validated below)
    w_extraction: float = Field(0.25, ge=0.0, le=1.0, exclude=True)
    w_layout: float = Field(0.25, ge=0.0, le=1.0, exclude=True)
    w_parsing: float = Field(0.30, ge=0.0, le=1.0, exclude=True)
    w_validation: float = Field(0.20, ge=0.0, le=1.0, exclude=True)

    flags: list[ValidationFlag] = Field(default_factory=list)

    @field_validator("flags", mode="before")
    @classmethod
    def deduplicate_flags(cls, v: Sequence[ValidationFlag]) -> list[ValidationFlag]:
        seen: set[ValidationFlag] = set()
        return [f for f in v if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

    @property
    def overall(self) -> float:
        return round(
            self.extraction * self.w_extraction
            + self.layout * self.w_layout
            + self.parsing * self.w_parsing
            + self.validation * self.w_validation,
            4,
        )

    def penalise(self, flag: ValidationFlag, amount: float = 0.15) -> "ConfidenceScore":
        """Return a new score with the validation sub-score reduced and the flag recorded."""
        new_flags = list(self.flags)
        if flag not in new_flags:
            new_flags.append(flag)
        return self.model_copy(
            update={
                "validation": max(0.0, self.validation - amount),
                "flags": new_flags,
            }
        )
