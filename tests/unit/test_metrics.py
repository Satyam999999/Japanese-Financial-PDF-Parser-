"""
tests/unit/test_metrics.py

Pytest tests for jp_fin_parser.metrics.

Each test uses a small synthetic table (2–5 rows) so that every metric
value can be calculated by hand and verified exactly.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from jp_fin_parser.metrics import (
    ArithmeticConsistencyResult,
    CellAccuracyResult,
    MetricsReport,
    StructureAccuracyResult,
    arithmetic_consistency,
    cell_accuracy,
    evaluate_table,
    report,
    structure_accuracy,
)
from jp_fin_parser.models.confidence import ConfidenceScore
from jp_fin_parser.models.row import FinancialRow, RowRole
from jp_fin_parser.models.table import (
    FinancialTable,
    PageMeta,
    Period,
    StatementType,
    UnitScale,
)

# ---------------------------------------------------------------------------
# Helpers (kept local so each test is self-contained)
# ---------------------------------------------------------------------------

P0 = "前連結会計年度"
P1 = "当連結会計年度"


def _meta(page: int = 0) -> PageMeta:
    return PageMeta(
        title="連結貸借対照表",
        statement_type=StatementType.BALANCE_SHEET,
        unit=UnitScale.THOUSAND_YEN,
        periods=[Period(label=P0), Period(label=P1)],
        page_number=page,
    )


def _row(
    label: str,
    values: dict,
    level: int = 1,
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


def _make_basic_table() -> FinancialTable:
    """
    現金及び預金  1000 | 1200   (item, level=1)
    売掛金         500 |  600   (item, level=1)
    流動資産合計  1500 | 1800   (subtotal, level=0)
    """
    subtotal = _row(
        "流動資産合計",
        {P0: Decimal("1500"), P1: Decimal("1800")},
        level=0,
        row_type=RowRole.SUBTOTAL,
    )
    cash = _row("現金及び預金", {P0: Decimal("1000"), P1: Decimal("1200")}, parent_id=subtotal.id)
    recv = _row("売掛金", {P0: Decimal("500"), P1: Decimal("600")}, parent_id=subtotal.id)
    return FinancialTable(meta=_meta(), rows=[cash, recv, subtotal])


# ===========================================================================
# 1. cell_accuracy
# ===========================================================================


class TestCellAccuracy:
    def test_perfect_match(self):
        tbl = _make_basic_table()
        result = cell_accuracy(actual=tbl, expected=tbl)
        # 3 rows × 2 periods = 6 cells
        assert result.total_cells == 6
        assert result.matched_values == 6
        assert result.value_accuracy == pytest.approx(1.0)
        assert result.missing_labels == 0
        assert result.mismatched_values == 0

    def test_one_value_wrong(self):
        expected = _make_basic_table()
        actual = _make_basic_table()
        # Corrupt 現金及び預金 P1 value in actual
        cash_row = actual.rows[0]
        bad_cash = cash_row.model_copy(
            update={"values": {P0: Decimal("1000"), P1: Decimal("9999")}}
        )
        actual = FinancialTable(meta=_meta(), rows=[bad_cash, actual.rows[1], actual.rows[2]])

        result = cell_accuracy(actual=actual, expected=expected)
        assert result.total_cells == 6
        assert result.matched_values == 5
        assert result.mismatched_values == 1
        assert result.value_accuracy == pytest.approx(5 / 6)

    def test_missing_row_in_actual(self):
        expected = _make_basic_table()
        # Actual is missing 売掛金
        actual = FinancialTable(
            meta=_meta(),
            rows=[expected.rows[0], expected.rows[2]],  # cash + subtotal only
        )
        result = cell_accuracy(actual=actual, expected=expected)
        # 売掛金 has 2 periods → 2 missing
        assert result.missing_labels == 2
        assert result.total_cells == 6
        assert result.value_accuracy == pytest.approx(4 / 6)

    def test_extra_row_in_actual(self):
        expected = _make_basic_table()
        extra = _row("その他", {P0: Decimal("100"), P1: Decimal("200")})
        actual = FinancialTable(
            meta=_meta(),
            rows=expected.rows + [extra],
        )
        result = cell_accuracy(actual=actual, expected=expected)
        assert result.extra_labels == 1
        assert result.value_accuracy == pytest.approx(1.0)  # expected cells all match

    def test_nil_value_matches_nil(self):
        """None (nil) in both actual and expected should count as a match."""
        expected = FinancialTable(
            meta=_meta(),
            rows=[_row("売掛金", {P0: None, P1: Decimal("600")})],
        )
        actual = FinancialTable(
            meta=_meta(),
            rows=[_row("売掛金", {P0: None, P1: Decimal("600")})],
        )
        result = cell_accuracy(actual=actual, expected=expected)
        assert result.matched_values == 2
        assert result.value_accuracy == pytest.approx(1.0)

    def test_nil_vs_value_is_mismatch(self):
        """None in actual but a real value in expected → mismatch."""
        expected = FinancialTable(
            meta=_meta(),
            rows=[_row("売掛金", {P0: Decimal("500"), P1: Decimal("600")})],
        )
        actual = FinancialTable(
            meta=_meta(),
            rows=[_row("売掛金", {P0: None, P1: Decimal("600")})],
        )
        result = cell_accuracy(actual=actual, expected=expected)
        assert result.mismatched_values == 1
        assert result.matched_values == 1


# ===========================================================================
# 2. structure_accuracy
# ===========================================================================


class TestStructureAccuracy:
    def test_perfect_structure(self):
        tbl = _make_basic_table()
        result = structure_accuracy(actual=tbl, expected=tbl)
        assert result.total_rows == 3
        assert result.full_accuracy == pytest.approx(1.0)

    def test_wrong_row_type(self):
        expected = _make_basic_table()
        # Mark 流動資産合計 as TOTAL instead of SUBTOTAL in actual
        subtotal = expected.rows[2]
        wrong_type = subtotal.model_copy(update={"row_type": RowRole.TOTAL})
        actual = FinancialTable(
            meta=_meta(),
            rows=[expected.rows[0], expected.rows[1], wrong_type],
        )
        result = structure_accuracy(actual=actual, expected=expected)
        assert result.row_type_matches == 2  # cash and receivable correct
        assert result.full_matches == 2
        assert result.full_accuracy == pytest.approx(2 / 3)

    def test_wrong_level(self):
        expected = _make_basic_table()
        # Give 現金及び預金 the wrong level in actual
        wrong_level = expected.rows[0].model_copy(update={"level": 0})
        actual = FinancialTable(
            meta=_meta(),
            rows=[wrong_level, expected.rows[1], expected.rows[2]],
        )
        result = structure_accuracy(actual=actual, expected=expected)
        assert result.level_matches == 2  # 売掛金 and subtotal correct
        assert result.full_matches == 2

    def test_swapped_row_order(self):
        expected = _make_basic_table()
        # Swap 現金及び預金 and 売掛金 in actual
        actual = FinancialTable(
            meta=_meta(),
            rows=[expected.rows[1], expected.rows[0], expected.rows[2]],
        )
        result = structure_accuracy(actual=actual, expected=expected)
        # Both items are at wrong positions; subtotal is still at index 2 → correct
        assert result.order_matches == 1   # only subtotal
        assert result.level_matches == 3   # all levels still correct
        assert result.full_matches == 1    # only subtotal passes all three

    def test_missing_row_in_actual(self):
        expected = _make_basic_table()
        # Drop 売掛金 from actual
        actual = FinancialTable(
            meta=_meta(),
            rows=[expected.rows[0], expected.rows[2]],
        )
        result = structure_accuracy(actual=actual, expected=expected)
        assert result.total_rows == 3   # still counts 3 expected rows


# ===========================================================================
# 3. arithmetic_consistency
# ===========================================================================


class TestArithmeticConsistency:
    def test_perfect_arithmetic(self):
        tbl = _make_basic_table()
        result = arithmetic_consistency(tbl)
        # Subtotal has 2 periods → 2 checks
        assert result.total_checks == 2
        assert result.passing_checks == 2
        assert result.consistency_rate == pytest.approx(1.0)
        assert result.failures == []

    def test_arithmetic_failure(self):
        """Subtotal deliberately wrong for P0."""
        subtotal = _row(
            "流動資産合計",
            {P0: Decimal("9999"), P1: Decimal("1800")},  # 9999 ≠ 1500
            level=0,
            row_type=RowRole.SUBTOTAL,
        )
        cash = _row("現金及び預金", {P0: Decimal("1000"), P1: Decimal("1200")}, parent_id=subtotal.id)
        recv = _row("売掛金", {P0: Decimal("500"), P1: Decimal("600")}, parent_id=subtotal.id)
        tbl = FinancialTable(meta=_meta(), rows=[cash, recv, subtotal])

        result = arithmetic_consistency(tbl, tolerance=Decimal("1"))
        assert result.total_checks == 2
        assert result.passing_checks == 1  # P1 passes
        assert result.consistency_rate == pytest.approx(0.5)
        assert len(result.failures) == 1
        assert result.failures[0]["period"] == P0

    def test_tolerance_allows_rounding(self):
        """Delta of exactly 1 (千円) should pass with default tolerance."""
        subtotal = _row(
            "流動資産合計",
            {P0: Decimal("1501"), P1: Decimal("1800")},  # off by 1
            level=0,
            row_type=RowRole.SUBTOTAL,
        )
        cash = _row("現金及び預金", {P0: Decimal("1000"), P1: Decimal("1200")}, parent_id=subtotal.id)
        recv = _row("売掛金", {P0: Decimal("500"), P1: Decimal("600")}, parent_id=subtotal.id)
        tbl = FinancialTable(meta=_meta(), rows=[cash, recv, subtotal])

        result = arithmetic_consistency(tbl, tolerance=Decimal("1"))
        assert result.passing_checks == 2  # both pass within tolerance

    def test_no_subtotals(self):
        """A table with only ITEM rows should give consistency_rate == 1.0 (vacuously true)."""
        tbl = FinancialTable(
            meta=_meta(),
            rows=[
                _row("現金及び預金", {P0: Decimal("1000"), P1: Decimal("1200")}),
                _row("売掛金", {P0: Decimal("500"), P1: Decimal("600")}),
            ],
        )
        result = arithmetic_consistency(tbl)
        assert result.total_checks == 0
        assert result.consistency_rate == pytest.approx(1.0)

    def test_nil_children_skipped(self):
        """Nil (None) child values are excluded from the sum; check still runs if any non-nil."""
        subtotal = _row(
            "流動資産合計",
            {P0: Decimal("1000")},
            level=0,
            row_type=RowRole.SUBTOTAL,
        )
        cash = _row("現金及び預金", {P0: Decimal("1000")}, parent_id=subtotal.id)
        recv = _row("売掛金", {P0: None}, parent_id=subtotal.id)  # nil
        tbl = FinancialTable(meta=_meta(), rows=[cash, recv, subtotal])

        result = arithmetic_consistency(tbl, tolerance=Decimal("1"))
        # Only cash contributes → sum = 1000 == subtotal 1000 → pass
        assert result.passing_checks == 1


# ===========================================================================
# 4. evaluate_table + report (smoke tests)
# ===========================================================================


class TestEvaluateAndReport:
    def test_evaluate_table_returns_metrics_report(self):
        tbl = _make_basic_table()
        r = evaluate_table(actual=tbl, expected=tbl)
        assert isinstance(r, MetricsReport)
        assert r.cell.value_accuracy == pytest.approx(1.0)
        assert r.structure.full_accuracy == pytest.approx(1.0)
        assert r.arithmetic.consistency_rate == pytest.approx(1.0)

    def test_report_does_not_raise(self, capsys):
        """report() should print without errors even for empty input."""
        report([], show_failures=False)

    def test_report_with_single_page(self, capsys):
        tbl = _make_basic_table()
        r = evaluate_table(actual=tbl, expected=tbl)
        # Should not raise
        report([r], show_failures=True)
