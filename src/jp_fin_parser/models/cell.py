"""
jp_fin_parser/models/cell.py

Low-level building blocks produced by the extraction layer.
These are intentionally thin — no business logic, no interpretation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class CellType(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    HEADER = "header"
    UNKNOWN = "unknown"


class BBox(BaseModel):
    """Axis-aligned bounding box in PDF point coordinates (origin = bottom-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2.0


class RawCell(BaseModel):
    """
    A single text cell as returned by the extraction layer.
    All coordinates are in the extractor's native space (pt).
    No interpretation has been applied yet.
    """

    text: str = Field(..., description="Raw Unicode text as extracted from the PDF.")
    bbox: BBox
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    page_number: int = Field(..., ge=0)
    cell_type: CellType = CellType.UNKNOWN

    model_config = {"frozen": True}
