"""
jp_fin_parser/models/row.py

Domain model for a single row of a Japanese financial statement.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .cell import BBox
from .confidence import ConfidenceScore


class RowRole(StrEnum):
    """
    Semantic role of a row in the statement.

    header    – a non-data row that titles a section (e.g. 「流動資産」)
    item      – a leaf data row with actual values
    subtotal  – 合計 / 計 row; values should equal sum of sibling items
    total     – grand total at the bottom of a section (e.g. 資産合計)
    net       – pure net row (純額 = gross + depreciation)
    note      – footnote / annotation row, values are usually absent
    """

    HEADER = "header"
    ITEM = "item"
    SUBTOTAL = "subtotal"
    TOTAL = "total"
    NET = "net"
    NOTE = "note"


class FinancialRow(BaseModel):
    """
    One logical row of a financial statement table.

    Design notes
    ------------
    * ``values`` maps a period label (e.g. "前連結会計年度") to its Decimal
      value **in the original unit** (千円, 百万円, etc.).  ``None`` means the
      cell was explicitly marked as nil (－ / ー); a missing key means the
      column was not found on this page.
    * ``is_nil`` mirrors the same nil distinction at the row level when *all*
      value columns are nil.
    * ``source_bbox`` is the union of all bboxes that contributed to this row;
      useful for debug overlays and VLM fallback crops.
    * ``confidence`` is required — there is no such thing as an
      "unscored" row in the output.
    """

    id: UUID = Field(default_factory=uuid4)
    label_raw: str = Field(..., description="Text exactly as extracted, before any normalisation.")
    label_normalized: str = Field(..., description="Cleaned label: NFKC, spaces stripped, etc.")
    level: int = Field(
        0, ge=0, description="Indentation depth; 0 = top-level account group."
    )
    parent_id: Optional[UUID] = Field(
        None, description="ID of the parent FinancialRow, or None for top-level rows."
    )
    row_type: RowRole = RowRole.ITEM
    formula_of: list[int] = Field(
        default_factory=list,
        description="1-based row indices that contribute to this formula/subtotal.",
    )
    values: dict[str, Optional[Decimal]] = Field(
        default_factory=dict,
        description=(
            "Maps period label → value in original filing unit. "
            "None = explicit nil (－). Missing key = column absent."
        ),
    )
    is_nil: bool = Field(
        False,
        description="True when every value column is explicitly nil (all None).",
    )
    source_bbox: Optional[BBox] = Field(
        None,
        description="Union bounding box of all raw cells that formed this row.",
    )
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)

    model_config = {"frozen": False}  # mutable during pipeline construction

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_value(self, period: str) -> Optional[Decimal]:
        """Return value for *period*, or None if nil/missing."""
        return self.values.get(period)

    def is_subtotal_like(self) -> bool:
        return self.row_type in (RowRole.SUBTOTAL, RowRole.TOTAL)
