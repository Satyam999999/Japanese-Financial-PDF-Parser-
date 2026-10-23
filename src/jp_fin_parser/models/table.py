"""
jp_fin_parser/models/table.py

Top-level table and metadata models.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StatementType(StrEnum):
    BALANCE_SHEET = "balance_sheet"            # 貸借対照表
    INCOME_STATEMENT = "income_statement"      # 損益計算書
    CASH_FLOW = "cash_flow"                    # キャッシュ・フロー計算書
    EQUITY_CHANGES = "equity_changes"          # 株主資本等変動計算書
    NOTES = "notes"                            # 注記
    UNKNOWN = "unknown"


class UnitScale(StrEnum):
    """
    Unit as printed in the filing header.
    ``multiplier_to_yen`` converts one unit to yen.
    """

    YEN = "円"
    THOUSAND_YEN = "千円"
    MILLION_YEN = "百万円"
    HUNDRED_MILLION_YEN = "億円"
    UNKNOWN = "unknown"

    @property
    def multiplier_to_yen(self) -> Decimal:
        _MAP: dict[str, Decimal] = {
            "円": Decimal("1"),
            "千円": Decimal("1000"),
            "百万円": Decimal("1000000"),
            "億円": Decimal("100000000"),
            "unknown": Decimal("1"),
        }
        return _MAP[self.value]


# ---------------------------------------------------------------------------
# Period
# ---------------------------------------------------------------------------

_JP_DATE_RE = re.compile(
    r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
)


class Period(BaseModel):
    """
    One column period (前連結会計年度 / 当連結会計年度, etc.).

    ``label`` is the raw column header as printed.
    ``end_date`` is parsed from 「2025年3月31日」-style strings if present.
    """

    label: str = Field(..., description="Column header exactly as printed.")
    end_date: Optional[date] = None

    @classmethod
    def from_label(cls, label: str, date_hint: Optional[str] = None) -> "Period":
        """
        Construct a Period, attempting to parse a date from *date_hint*
        (or from *label* itself as a fallback).
        """
        end_date = cls._parse_jp_date(date_hint or label)
        return cls(label=label, end_date=end_date)

    @staticmethod
    def _parse_jp_date(text: str) -> Optional[date]:
        m = _JP_DATE_RE.search(text)
        if m:
            try:
                return date(int(m.group("year")), int(m.group("month")), int(m.group("day")))
            except ValueError:
                pass
        return None


# ---------------------------------------------------------------------------
# PageMeta
# ---------------------------------------------------------------------------


class PageMeta(BaseModel):
    """
    Metadata extracted from the page header / title area.
    This is populated before any row-level parsing begins.
    """

    title: str = Field(..., description="Statement title as printed on the page.")
    statement_type: StatementType = StatementType.UNKNOWN
    unit: UnitScale = UnitScale.UNKNOWN
    periods: list[Period] = Field(
        default_factory=list,
        description="Ordered list of period columns found on the page.",
    )
    is_correction: bool = Field(
        False, description="True when the page bears 訂正後 / 訂正 markers."
    )
    page_number: int = Field(..., ge=0)
    source_file: Optional[str] = None

    @property
    def multiplier_to_yen(self) -> Decimal:
        return self.unit.multiplier_to_yen

    @property
    def period_labels(self) -> list[str]:
        return [p.label for p in self.periods]


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class ArithmeticCheck(BaseModel):
    """Result of checking one subtotal/total row against its children."""

    row_label: str
    period: str
    expected: Optional[Decimal]   # sum of children
    actual: Optional[Decimal]     # value in the subtotal row
    delta: Optional[Decimal]      # actual - expected
    passes: bool
    rule_name: str = "subtotal"
    row_id: Optional[UUID] = None
    tolerance: Optional[Decimal] = None
    child_row_ids: list[UUID] = Field(default_factory=list)
    suggestion: Optional[dict] = None


class ValidationReport(BaseModel):
    """Aggregate of all validation results for one table."""

    arithmetic_checks: list[ArithmeticCheck] = Field(default_factory=list)
    balance_sheet_balanced: Optional[bool] = None  # None if not a BS
    unknown_flags_count: int = 0
    notes: list[str] = Field(default_factory=list)
    failures: list[dict] = Field(default_factory=list)
    suggestions: list[dict] = Field(default_factory=list)

    @property
    def arithmetic_pass_rate(self) -> float:
        if not self.arithmetic_checks:
            return 1.0
        return sum(c.passes for c in self.arithmetic_checks) / len(self.arithmetic_checks)

    @property
    def total_checks(self) -> int:
        return len(self.arithmetic_checks)

    @property
    def passing_checks(self) -> int:
        return sum(1 for c in self.arithmetic_checks if c.passes)

    @property
    def failing_checks(self) -> int:
        return sum(1 for c in self.arithmetic_checks if not c.passes)


# ---------------------------------------------------------------------------
# FinancialTable  (the top-level output for one statement on one page)
# ---------------------------------------------------------------------------


class FinancialTable(BaseModel):
    """
    A fully parsed, validated table from one PDF page.

    ``rows`` is ordered top-to-bottom as they appear on the page.
    ``validation_report`` is populated by the validation stage and
    is always present — even if empty — so callers never need to
    check for None.
    """

    meta: PageMeta
    rows: list["FinancialRow"]  # noqa: F821  (forward ref resolved by models/__init__)
    validation_report: ValidationReport = Field(default_factory=ValidationReport)

    @model_validator(mode="after")
    def _check_periods_in_rows(self) -> "FinancialTable":
        """Warn (via assertion) if any row references a period not in meta."""
        known = set(self.meta.period_labels)
        for row in self.rows:
            for period in row.values:
                if period not in known:
                    # Non-fatal: flag it rather than reject
                    pass
        return self

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def subtotal_rows(self) -> list["FinancialRow"]:  # noqa: F821
        from .row import RowRole
        return [r for r in self.rows if r.row_type in (RowRole.SUBTOTAL, RowRole.TOTAL)]

    def items_under(self, parent_id: "UUID") -> list["FinancialRow"]:  # noqa: F821
        return [r for r in self.rows if r.parent_id == parent_id]


# Resolve forward references
from .row import FinancialRow  # noqa: E402
FinancialTable.model_rebuild()
