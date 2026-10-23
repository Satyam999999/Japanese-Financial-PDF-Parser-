"""
tests/conftest.py

Shared pytest fixtures.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from uuid import uuid4

from jp_fin_parser.models.row import FinancialRow, RowRole
from jp_fin_parser.models.table import (
    FinancialTable,
    PageMeta,
    Period,
    StatementType,
    UnitScale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(
    page: int = 0,
    unit: UnitScale = UnitScale.THOUSAND_YEN,
    periods: list[str] | None = None,
) -> PageMeta:
    period_labels = periods or ["前連結会計年度", "当連結会計年度"]
    return PageMeta(
        title="連結貸借対照表",
        statement_type=StatementType.BALANCE_SHEET,
        unit=unit,
        periods=[Period(label=p) for p in period_labels],
        page_number=page,
    )


def _row(
    label: str,
    values: dict,
    level: int = 0,
    row_type: RowRole = RowRole.ITEM,
    parent_id=None,
) -> FinancialRow:
    return FinancialRow(
        label_raw=label,
        label_normalized=label,
        level=level,
        row_type=row_type,
        values=values,
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


PERIODS = ["前連結会計年度", "当連結会計年度"]


@pytest.fixture
def simple_meta() -> PageMeta:
    return _meta()


@pytest.fixture
def simple_expected_table() -> FinancialTable:
    """
    A tiny synthetic balance-sheet fragment:

    現金及び預金         1,000 | 1,200
    売掛金                 500 |   600
    流動資産合計          1,500 | 1,800   (subtotal)
    """
    p0, p1 = PERIODS
    cash = _row("現金及び預金", {p0: Decimal("1000"), p1: Decimal("1200")}, level=1)
    receivable = _row("売掛金", {p0: Decimal("500"), p1: Decimal("600")}, level=1)
    subtotal = _row(
        "流動資産合計",
        {p0: Decimal("1500"), p1: Decimal("1800")},
        level=0,
        row_type=RowRole.SUBTOTAL,
    )
    # Wire parent_id so arithmetic_consistency can find children
    subtotal2 = subtotal.model_copy(update={})
    cash2 = cash.model_copy(update={"parent_id": subtotal.id})
    recv2 = receivable.model_copy(update={"parent_id": subtotal.id})

    return FinancialTable(meta=_meta(), rows=[cash2, recv2, subtotal2])


@pytest.fixture
def simple_actual_table(simple_expected_table) -> FinancialTable:
    """A perfect copy of the expected table — all metrics should be 1.0."""
    return simple_expected_table.model_copy(deep=True)
