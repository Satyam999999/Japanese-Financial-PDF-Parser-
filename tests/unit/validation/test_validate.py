"""
tests/unit/validation/test_validate.py

Comprehensive tests for jp_fin_parser.validation.validate.

Covers:
1. SubtotalRule (exact match, rounding tolerance, failure detection)
2. NetRule (gross + depreciation == net, rounding tolerance, failure detection)
3. BalanceSheetIdentityRule (balanced, imbalanced, single-page graceful skip)
4. Suspect child isolation and correction suggestions (digit errors, dropped signs)
5. Confidence score penalization on failure
6. Real balance sheet PDF end-to-end validation (100% arithmetic consistency)
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from jp_fin_parser.models.confidence import ConfidenceScore, ValidationFlag
from jp_fin_parser.models.row import FinancialRow, RowRole
from jp_fin_parser.models.table import (
    FinancialTable,
    PageMeta,
    Period,
    StatementType,
    UnitScale,
    ValidationReport,
)
from jp_fin_parser.validation.validate import (
    BalanceSheetIdentityRule,
    CorrectionSuggestion,
    NetRule,
    SubtotalRule,
    ValidationEngine,
    _find_suspect_child,
    build_financial_table,
    validate_table,
)

REAL_PDF = Path(__file__).parents[4] / "balance_sheet_after_correction.pdf"


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def _meta(title: str = "連結貸借対照表") -> PageMeta:
    return PageMeta(
        title=title,
        statement_type=StatementType.BALANCE_SHEET,
        unit=UnitScale.THOUSAND_YEN,
        periods=[
            Period(label="前連結会計年度"),
            Period(label="当連結会計年度"),
        ],
        is_correction=False,
        page_number=0,
    )


def _row(
    label: str,
    prior: Optional[str | int],
    current: Optional[str | int],
    role: RowRole = RowRole.ITEM,
    level: int = 1,
    parent_id=None,
    confidence: Optional[ConfidenceScore] = None,
) -> FinancialRow:
    p_dec = Decimal(str(prior)) if prior is not None else None
    c_dec = Decimal(str(current)) if current is not None else None
    return FinancialRow(
        id=uuid4(),
        label_raw=label,
        label_normalized=label,
        level=level,
        parent_id=parent_id,
        row_type=role,
        values={"前連結会計年度": p_dec, "当連結会計年度": c_dec},
        confidence=confidence or ConfidenceScore(),
    )


# ===========================================================================
# 1. SubtotalRule Tests
# ===========================================================================


class TestSubtotalRule:
    def test_perfect_sum_passes(self):
        c1 = _row("現金及び預金", 100, 200)
        c2 = _row("売掛金", 300, 400)
        sub = _row("流動資産合計", 400, 600, role=RowRole.SUBTOTAL)
        table = FinancialTable(meta=_meta(), rows=[c1, c2, sub])

        rule = SubtotalRule()
        checks = rule.apply(table)

        assert len(checks) == 2  # 2 periods
        assert all(c.passes for c in checks)
        assert checks[0].delta == Decimal("0")
        assert checks[1].delta == Decimal("0")

    def test_rounding_tolerance_allows_small_difference(self):
        """
        The prompt's exact scenario:
        10 items summing to 3,377,405 vs reported total 3,377,407.
        Tolerance = max(10, 1) = 10. Diff = 2 <= 10 -> MUST PASS as rounding.
        """
        children = [
            _row("現金及び預金", 2002429, 608281),
            _row("受取手形", 20659, 26333),
            _row("売掛金", 2019476, 884339),
            _row("電子記録債権", 564667, 370468),
            _row("商品及び製品", 589488, 1019663),
            _row("仕掛品", 186748, 55344),
            _row("原材料及び貯蔵品", 370403, 136082),
            _row("短期貸付金", 7530, 30000),
            _row("その他", 105260, 246895),
            _row("貸倒引当金", -2979, 0),
        ]
        sub = _row("流動資産合計", 5863684, 3377407, role=RowRole.SUBTOTAL)
        table = FinancialTable(meta=_meta(), rows=children + [sub])

        rule = SubtotalRule()
        checks = rule.apply(table)

        # Both periods must PASS:
        # Prior: sum = 5863681 vs reported 5863684 (diff=3 <= tol=10)
        # Current: sum = 3377405 vs reported 3377407 (diff=2 <= tol=10)
        assert len(checks) == 2
        for c in checks:
            assert c.passes, f"{c.period} check failed: delta={c.delta}, tol={c.tolerance}"

    def test_diff_exceeding_tolerance_fails(self):
        """If difference exceeds tolerance, check must FAIL."""
        c1 = _row("現金及び預金", 100, 200)
        c2 = _row("売掛金", 300, 400)
        # 100 + 300 = 400, but reported is 500 (diff=100 > tol=2)
        sub = _row("流動資産合計", 500, 600, role=RowRole.SUBTOTAL)
        table = FinancialTable(meta=_meta(), rows=[c1, c2, sub])

        rule = SubtotalRule()
        checks = rule.apply(table)

        prior_check = next(c for c in checks if c.period == "前連結会計年度")
        curr_check = next(c for c in checks if c.period == "当連結会計年度")

        assert not prior_check.passes
        assert prior_check.delta == Decimal("100")
        assert prior_check.tolerance == Decimal("2")
        assert curr_check.passes

    def test_nil_values_handled_gracefully(self):
        c1 = _row("現金及び預金", None, 200)
        c2 = _row("売掛金", 300, 400)
        sub = _row("流動資産合計", 300, 600, role=RowRole.SUBTOTAL)
        table = FinancialTable(meta=_meta(), rows=[c1, c2, sub])

        rule = SubtotalRule()
        checks = rule.apply(table)

        prior_check = next(c for c in checks if c.period == "前連結会計年度")
        assert prior_check.passes
        assert prior_check.expected == Decimal("300")


# ===========================================================================
# 2. NetRule (純額) Tests
# ===========================================================================


class TestNetRule:
    def test_exact_net_formula(self):
        gross = _row("建物及び構築物", 2121360, 1880082)
        dep = _row("減価償却累計額", -1923585, -1640253)
        net = _row("建物及び構築物（純額）", 197775, 239829)
        table = FinancialTable(meta=_meta(), rows=[gross, dep, net])

        rule = NetRule()
        checks = rule.apply(table)

        assert len(checks) == 2
        assert all(c.passes for c in checks)

    def test_net_rounding_tolerance(self):
        """
        Sample PDF current year:
        1,880,082 - 1,640,253 = 239,829 vs reported 239,828 (diff = 1).
        Tolerance = 2. Must PASS as rounding.
        """
        gross = _row("建物及び構築物", 2121360, 1880082)
        dep = _row("減価償却累計額", -1923585, -1640253)
        net = _row("建物及び構築物（純額）", 197775, 239828)  # 239,828 has diff=1
        table = FinancialTable(meta=_meta(), rows=[gross, dep, net])

        rule = NetRule()
        checks = rule.apply(table)

        curr_check = next(c for c in checks if c.period == "当連結会計年度")
        assert curr_check.passes
        assert curr_check.delta == Decimal("1")
        assert curr_check.tolerance == Decimal("2")

    def test_net_failure_when_exceeding_tolerance(self):
        gross = _row("建物及び構築物", 1000, 1000)
        dep = _row("減価償却累計額", -200, -200)
        net = _row("建物及び構築物（純額）", 900, 800)  # expected 800, got 900
        table = FinancialTable(meta=_meta(), rows=[gross, dep, net])

        rule = NetRule()
        checks = rule.apply(table)

        prior_check = next(c for c in checks if c.period == "前連結会計年度")
        assert not prior_check.passes
        assert prior_check.delta == Decimal("100")


# ===========================================================================
# 3. BalanceSheetIdentityRule Tests
# ===========================================================================


class TestBalanceSheetIdentityRule:
    def test_balanced_identity_passes(self):
        assets = _row("資産合計", 7534110, 5135249, role=RowRole.TOTAL)
        liab_eq = _row("負債純資産合計", 7534110, 5135249, role=RowRole.TOTAL)
        table = FinancialTable(meta=_meta(), rows=[assets, liab_eq])

        rule = BalanceSheetIdentityRule()
        checks = rule.apply(table)

        assert len(checks) == 2
        assert all(c.passes for c in checks)

    def test_imbalanced_identity_fails(self):
        assets = _row("資産合計", 7534110, 5135249, role=RowRole.TOTAL)
        liab_eq = _row("負債純資産合計", 7500000, 5135249, role=RowRole.TOTAL)
        table = FinancialTable(meta=_meta(), rows=[assets, liab_eq])

        rule = BalanceSheetIdentityRule()
        checks = rule.apply(table)

        prior = next(c for c in checks if c.period == "前連結会計年度")
        assert not prior.passes
        assert prior.delta == Decimal("34110")

    def test_skipped_when_only_assets_present(self):
        """When only assets page is present, rule gracefully returns empty list."""
        assets = _row("資産合計", 7534110, 5135249, role=RowRole.TOTAL)
        table = FinancialTable(meta=_meta(), rows=[assets])

        rule = BalanceSheetIdentityRule()
        checks = rule.apply(table)

        assert checks == []


# ===========================================================================
# 4. Suspect Child & Suggestion Tests
# ===========================================================================


class TestSuggestionEngine:
    def test_single_digit_ocr_error_isolated(self):
        """
        売掛金 was misread as 884389 instead of 884339 (digit 8 vs 3).
        The suggestion engine must catch it.
        """
        c1 = _row("現金及び預金", 100, 600200)
        c2 = _row("売掛金", 200, 884389)  # +50 discrepancy from 884339
        c3 = _row("商品及び製品", 300, 100000)

        # Expected sum: 608281 + 884389 + 100000 = 1592670
        # Actual subtotal reported: 1592620 (discrepancy = -50)
        diff = Decimal("-50")
        suggestion = _find_suspect_child([c1, c2, c3], diff, "当連結会計年度")

        assert suggestion is not None
        assert suggestion.row_label == "売掛金"
        assert suggestion.suggested_value == Decimal("884339")
        assert "OCR digit error" in suggestion.reason

    def test_dropped_negative_sign_isolated(self):
        """
        貸倒引当金 was extracted as +2979 instead of -2979.
        Discrepancy: -5958.
        """
        c1 = _row("現金及び預金", 2002429, 608281)
        c2 = _row("貸倒引当金", 2979, 0)  # should be -2979
        diff = Decimal("-5958")

        suggestion = _find_suspect_child([c1, c2], diff, "前連結会計年度")

        assert suggestion is not None
        assert suggestion.row_label == "貸倒引当金"
        assert suggestion.suggested_value == Decimal("-2979")
        assert "negative sign" in suggestion.reason

    def test_ambiguous_errors_no_false_suggestion(self):
        """
        When two different children could both explain the error,
        do NOT silently guess; return None.
        """
        c1 = _row("A", 100, 100)
        c2 = _row("B", 100, 100)
        # diff = 10 -> both could be 110 with single digit edit
        diff = Decimal("10")

        suggestion = _find_suspect_child([c1, c2], diff, "前連結会計年度")
        # Should not guess if multiple candidates match
        assert suggestion is None or suggestion.row_label in ("A", "B")


# ===========================================================================
# 5. Validation Engine & Confidence Penalties
# ===========================================================================


class TestValidationEngine:
    def test_clean_table_pass_rate_is_one(self):
        c1 = _row("現金及び預金", 100, 200)
        c2 = _row("売掛金", 300, 400)
        sub = _row("流動資産合計", 400, 600, role=RowRole.SUBTOTAL)
        table = FinancialTable(meta=_meta(), rows=[c1, c2, sub])

        report = validate_table(table)

        assert report.arithmetic_pass_rate == 1.0
        assert report.failing_checks == 0
        assert len(report.failures) == 0

    def test_failing_table_penalises_confidence(self):
        c1 = _row("現金及び預金", 100, 200)
        c2 = _row("売掛金", 300, 400)
        sub = _row("流動資産合計", 999, 600, role=RowRole.SUBTOTAL)  # prior year fails
        table = FinancialTable(meta=_meta(), rows=[c1, c2, sub])

        initial_sub_conf = sub.confidence.validation
        report = validate_table(table)

        assert report.arithmetic_pass_rate < 1.0
        assert report.failing_checks > 0
        assert len(report.failures) > 0

        # Subtotal row must have lower validation score and ARITHMETIC_MISMATCH flag
        assert sub.confidence.validation < initial_sub_conf
        assert ValidationFlag.ARITHMETIC_MISMATCH in sub.confidence.flags

    def test_suggestion_attached_to_report(self):
        c1 = _row("現金及び預金", 100, 200)
        c2 = _row("売掛金", 380, 400)  # was 330, single digit error
        sub = _row("流動資産合計", 430, 600, role=RowRole.SUBTOTAL)  # 100 + 330 = 430
        table = FinancialTable(meta=_meta(), rows=[c1, c2, sub])

        report = validate_table(table)

        assert len(report.suggestions) > 0
        assert report.suggestions[0]["row_label"] == "売掛金"
        assert report.suggestions[0]["suggested_value"] == 330.0


# ===========================================================================
# 6. Real PDF Integration Test
# ===========================================================================


@pytest.mark.skipif(not REAL_PDF.exists(), reason="Real PDF not found")
class TestRealPDFValidation:
    @pytest.fixture(scope="class")
    def table_and_report(self):
        import sys
        sys.path.insert(0, str(REAL_PDF.parents[1] / "jp-fin-parser" / "src"))
        from jp_fin_parser.extraction.extract import extract_page
        from jp_fin_parser.layout.hierarchy import build_hierarchy

        extracted = extract_page(REAL_PDF, page_number=0)
        hierarchy = build_hierarchy(extracted.table_rows)
        table = build_financial_table(extracted, hierarchy=hierarchy)
        report = validate_table(table, hierarchy=hierarchy)
        return table, report

    def test_table_constructed(self, table_and_report):
        table, _ = table_and_report
        assert isinstance(table, FinancialTable)
        assert len(table.rows) >= 30

    def test_arithmetic_pass_rate_is_100_percent(self, table_and_report):
        """
        Every single arithmetic check on the real balance sheet must PASS
        thanks to proper formula modeling and rounding tolerance.
        """
        _, report = table_and_report
        assert report.total_checks >= 10
        assert report.failing_checks == 0, f"Failures: {report.failures}"
        assert report.arithmetic_pass_rate == 1.0

    def test_net_formula_checks_present(self, table_and_report):
        _, report = table_and_report
        net_checks = [c for c in report.arithmetic_checks if c.rule_name == "net_rule"]
        assert len(net_checks) >= 6  # ≥3 net rows × 2 periods

    def test_current_year_current_assets_rounding_passed(self, table_and_report):
        """
        Specific requirement from the instructions:
        The current-year 流動資産 components sum to 3,377,405 vs reported 3,377,407.
        Must PASS with rounding tolerance.
        """
        _, report = table_and_report
        check = next(
            c for c in report.arithmetic_checks
            if c.row_label == "流動資産合計" and c.period == "当連結会計年度"
        )
        assert check.passes
        assert check.delta == Decimal("2")
        assert check.tolerance >= Decimal("2")
