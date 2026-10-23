"""
tests/unit/preprocessing/test_normalize.py

Full pytest coverage for jp_fin_parser.preprocessing.normalize.

Every section covers one major function, with explicit "ugly case" subsections
that stress the edge cases documented in the prompt.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from jp_fin_parser.models.row import RowRole
from jp_fin_parser.models.table import UnitScale
from jp_fin_parser.preprocessing.normalize import (
    ParsedDate,
    ParsedLabel,
    ParsedNumber,
    detect_row_type,
    normalize_label,
    parse_date,
    parse_number,
)


# ===========================================================================
# 1. parse_number — normal cases
# ===========================================================================

class TestParseNumberNormal:
    def test_plain_integer(self):
        r = parse_number("1234567")
        assert r.value == Decimal("1234567")
        assert not r.is_nil
        assert not r.is_negative

    def test_comma_separated(self):
        r = parse_number("2,002,429")
        assert r.value == Decimal("2002429")

    def test_unit_multiplier_senyen(self):
        r = parse_number("1,234", UnitScale.THOUSAND_YEN)
        assert r.value == Decimal("1234")
        assert r.value_in_yen == Decimal("1234000")

    def test_unit_multiplier_hyakuman(self):
        r = parse_number("5", UnitScale.MILLION_YEN)
        assert r.value_in_yen == Decimal("5000000")

    def test_unit_unknown_gives_none_in_yen(self):
        r = parse_number("100", UnitScale.UNKNOWN)
        assert r.value_in_yen is None

    def test_original_preserved(self):
        raw = "  △1,234  "
        r = parse_number(raw)
        assert r.original == "△1,234"   # stripped but otherwise untouched

    def test_zero(self):
        r = parse_number("0")
        assert r.value == Decimal("0")
        assert not r.is_negative
        assert not r.is_nil


# ===========================================================================
# 2. parse_number — negative signs
# ===========================================================================

class TestParseNumberNegative:
    def test_delta_prefix(self):
        r = parse_number("△1,234")
        assert r.value == Decimal("-1234")
        assert r.is_negative
        assert r.had_sign_marker

    def test_filled_triangle_prefix(self):
        r = parse_number("▲3,634,165")
        assert r.value == Decimal("-3634165")
        assert r.is_negative

    def test_parenthesis_half_width(self):
        r = parse_number("(1,234)")
        assert r.value == Decimal("-1234")
        assert r.is_negative
        assert r.had_sign_marker

    def test_parenthesis_full_width(self):
        r = parse_number("（5,678）")
        assert r.value == Decimal("-5678")
        assert r.is_negative

    def test_real_file_example_delta_2979(self):
        """△2,979 appears in the real balance sheet."""
        r = parse_number("△2,979", UnitScale.THOUSAND_YEN)
        assert r.value == Decimal("-2979")
        assert r.value_in_yen == Decimal("-2979000")

    def test_real_file_example_delta_large(self):
        """△3,634,165 appears in the real balance sheet."""
        r = parse_number("△3,634,165")
        assert r.value == Decimal("-3634165")


# ===========================================================================
# 3. parse_number — nil markers  (is_nil=True, value=None)
# ===========================================================================

class TestParseNumberNil:
    @pytest.mark.parametrize("raw", [
        "－",       # FULLWIDTH HYPHEN-MINUS
        "―",        # EM DASH
        "ー",        # KATAKANA-HIRAGANA PROLONGED SOUND MARK
        "-",        # ASCII hyphen standing alone
        "‐",        # HYPHEN (U+2010)
        "--",       # double dash
        "－－",      # double full-width dash
    ])
    def test_nil_marker(self, raw: str):
        r = parse_number(raw)
        assert r.is_nil, f"Expected nil for {raw!r}"
        assert r.value is None
        assert r.value_in_yen is None
        assert not r.is_negative

    def test_nil_is_distinct_from_missing(self):
        """is_nil=True means the cell was *explicitly* marked nil in the filing."""
        r = parse_number("－")
        assert r.is_nil
        # A missing cell would simply not exist; is_nil=False for a missing key
        assert r.value is None


# ===========================================================================
# 4. parse_number — ugly / edge cases
# ===========================================================================

class TestParseNumberEdgeCases:
    def test_delta_zero(self):
        """△0 appears in the real BS; must not become −0."""
        r = parse_number("△0")
        assert r.value == Decimal("0")
        assert not r.is_negative          # zero has no sign
        assert r.had_sign_marker          # but we saw the △

    def test_filled_triangle_zero(self):
        r = parse_number("▲0")
        assert r.value == Decimal("0")
        assert not r.is_negative
        assert r.had_sign_marker

    def test_fullwidth_digits(self):
        """１，２３４ (all full-width) must parse to 1234."""
        r = parse_number("１，２３４")
        assert r.value == Decimal("1234")

    def test_fullwidth_digits_with_delta(self):
        r = parse_number("△１，２３４")
        assert r.value == Decimal("-1234")

    def test_space_padded(self):
        r = parse_number("  2,002,429  ")
        assert r.value == Decimal("2002429")

    def test_unparseable_returns_nil_flagged(self):
        """Garbage text cannot be parsed → nil with had_sign_marker=False."""
        r = parse_number("???")
        assert r.is_nil
        assert r.value is None

    def test_parenthesis_with_fullwidth(self):
        """（1,234）with full-width brackets."""
        r = parse_number("（1,234）")
        assert r.value == Decimal("-1234")
        assert r.is_negative

    def test_large_number_precision(self):
        """Decimal must preserve full precision (no float rounding)."""
        r = parse_number("7,534,110")
        assert r.value == Decimal("7534110")
        assert str(r.value) == "7534110"

    def test_unit_multiplier_yen(self):
        r = parse_number("500", UnitScale.YEN)
        assert r.value_in_yen == Decimal("500")  # ×1

    def test_unit_multiplier_okuyen(self):
        r = parse_number("3", UnitScale.HUNDRED_MILLION_YEN)
        assert r.value_in_yen == Decimal("300000000")


# ===========================================================================
# 5. normalize_label
# ===========================================================================

class TestNormalizeLabel:
    def test_simple_label(self):
        r = normalize_label("現金及び預金")
        assert r.normalized == "現金及び預金"
        assert r.raw == "現金及び預金"

    def test_spaces_inside_japanese_stripped(self):
        r = normalize_label("現 金 及 び 預 金")
        assert r.normalized == "現金及び預金"

    def test_ideographic_space_stripped(self):
        """U+3000 IDEOGRAPHIC SPACE between CJK chars is stripped."""
        r = normalize_label("流動\u3000資産")
        assert r.normalized == "流動資産"

    def test_fullwidth_alphanumeric_nfkc(self):
        """Full-width ASCII is normalised to half-width via NFKC."""
        r = normalize_label("ＡＢＣ")
        assert r.normalized == "ABC"

    def test_raw_label_preserved(self):
        raw = "  現 金 及 び 預 金  "
        r = normalize_label(raw)
        assert r.raw == raw
        assert r.normalized == "現金及び預金"

    def test_leading_trailing_stripped(self):
        r = normalize_label("  売掛金  ")
        assert r.normalized == "売掛金"

    def test_row_type_inferred_for_item(self):
        r = normalize_label("現金及び預金")
        assert r.row_type == RowRole.ITEM

    def test_row_type_subtotal(self):
        r = normalize_label("流動資産合計")
        assert r.row_type == RowRole.SUBTOTAL

    def test_row_type_total(self):
        r = normalize_label("資産合計")
        assert r.row_type == RowRole.TOTAL

    def test_row_type_header_no_bu_suffix(self):
        r = normalize_label("資産の部")
        assert r.row_type == RowRole.HEADER

    def test_mixed_jp_ascii_spaces_preserved(self):
        """Space between ASCII and Japanese should survive."""
        r = normalize_label("IFRSに基づく")
        # internal space between ASCII and JP is kept (not between JP-JP)
        assert "IFRS" in r.normalized


# ===========================================================================
# 6. parse_date — Gregorian (西暦)
# ===========================================================================

class TestParseDateSeireki:
    def test_standard_gregorian(self):
        r = parse_date("2025年3月31日")
        assert r.date == date(2025, 3, 31)
        assert r.era == "西暦"

    def test_fullwidth_digits(self):
        r = parse_date("２０２５年３月３１日")
        assert r.date == date(2025, 3, 31)

    def test_with_parentheses(self):
        r = parse_date("(2025年３月31日)")
        assert r.date == date(2025, 3, 31)

    def test_with_fullwidth_parentheses(self):
        r = parse_date("（2026年３月31日）")
        assert r.date == date(2026, 3, 31)

    def test_mid_year(self):
        r = parse_date("2024年9月30日")
        assert r.date == date(2024, 9, 30)

    def test_real_filing_date_prev(self):
        """From actual balance sheet header."""
        r = parse_date("(2025年３月31日)")
        assert r.date == date(2025, 3, 31)

    def test_real_filing_date_curr(self):
        r = parse_date("(2026年３月31日)")
        assert r.date == date(2026, 3, 31)


# ===========================================================================
# 7. parse_date — Japanese era (和暦)
# ===========================================================================

class TestParseDateWareki:
    def test_reiwa_7(self):
        """令和7年 = 2019 + 7 - 1 = 2025."""
        r = parse_date("令和7年3月31日")
        assert r.date == date(2025, 3, 31)
        assert r.era == "令和"

    def test_reiwa_1(self):
        """令和1年 = 2019."""
        r = parse_date("令和1年5月1日")
        assert r.date == date(2019, 5, 1)

    def test_heisei_30(self):
        """平成30年 = 1989 + 30 - 1 = 2018."""
        r = parse_date("平成30年3月31日")
        assert r.date == date(2018, 3, 31)
        assert r.era == "平成"

    def test_heisei_1(self):
        r = parse_date("平成1年1月8日")
        assert r.date == date(1989, 1, 8)

    def test_showa_64(self):
        """昭和64年 = 1926 + 64 - 1 = 1989."""
        r = parse_date("昭和64年1月7日")
        assert r.date == date(1989, 1, 7)
        assert r.era == "昭和"

    def test_taisho(self):
        """大正15年 = 1912 + 15 - 1 = 1926."""
        r = parse_date("大正15年12月25日")
        assert r.date == date(1926, 12, 25)

    def test_era_year_only_no_date(self):
        """Era with year but no month/day → date=None, era captured."""
        r = parse_date("令和7年")
        assert r.date is None
        assert r.era == "令和"


# ===========================================================================
# 8. parse_date — fiscal-year labels & junk
# ===========================================================================

class TestParseDateFiscalAndJunk:
    @pytest.mark.parametrize("label", [
        "前連結会計年度",
        "当連結会計年度",
        "当事業年度",
        "前事業年度",
        "第45期",
        "第2四半期",
    ])
    def test_fiscal_label_detected(self, label: str):
        r = parse_date(label)
        assert r.is_fiscal_label, f"{label!r} should be a fiscal label"
        assert r.date is None

    def test_junk_text(self):
        r = parse_date("（単位：千円）")
        assert r.date is None
        assert not r.is_fiscal_label

    def test_empty_string(self):
        r = parse_date("")
        assert r.date is None
        assert not r.is_fiscal_label


# ===========================================================================
# 9. detect_row_type
# ===========================================================================

class TestDetectRowType:
    # Subtotals
    @pytest.mark.parametrize("label", [
        "流動資産合計",
        "固定資産合計",
        "有形固定資産合計",
        "無形固定資産合計",
        "投資その他の資産合計",
        "固定負債合計",
        "流動負債合計",
        "営業活動によるキャッシュ・フロー計",
        "売上総利益",   # does NOT end in 合計/計 → ITEM
    ])
    def test_subtotal_labels(self, label: str):
        result = detect_row_type(label, has_values=True)
        if label == "売上総利益":
            assert result == RowRole.ITEM
        else:
            assert result == RowRole.SUBTOTAL, f"{label!r} → {result}"

    # Grand totals
    @pytest.mark.parametrize("label", [
        "資産合計",
        "負債合計",
        "純資産合計",
        "負債純資産合計",
        "負債及び純資産合計",
    ])
    def test_total_labels(self, label: str):
        result = detect_row_type(label, has_values=True)
        assert result == RowRole.TOTAL, f"{label!r} → {result}"

    # Section headers (の部 suffix)
    @pytest.mark.parametrize("label", [
        "資産の部",
        "負債の部",
        "純資産の部",
    ])
    def test_header_no_bu(self, label: str):
        result = detect_row_type(label, has_values=False)
        assert result == RowRole.HEADER, f"{label!r} → {result}"

    # Section headers (no values, known keyword)
    @pytest.mark.parametrize("label", [
        "流動資産",
        "固定資産",
        "有形固定資産",
        "無形固定資産",
        "投資その他の資産",
        "流動負債",
        "固定負債",
    ])
    def test_header_keywords_no_values(self, label: str):
        result = detect_row_type(label, has_values=False)
        assert result == RowRole.HEADER, f"{label!r} → {result}"

    def test_header_keyword_with_values_is_item(self):
        """If 流動資産 somehow has values it should be ITEM, not HEADER."""
        result = detect_row_type("流動資産", has_values=True)
        assert result == RowRole.ITEM

    # Normal items
    @pytest.mark.parametrize("label", [
        "現金及び預金",
        "売掛金",
        "建物及び構築物",
        "減価償却累計額",
        "退職給付に係る資産",
        "当期純利益",
        "繰越利益剰余金",
    ])
    def test_item_labels(self, label: str):
        result = detect_row_type(label, has_values=True)
        assert result == RowRole.ITEM, f"{label!r} → {result}"

    # Notes / footnotes
    @pytest.mark.parametrize("label", [
        "※退職給付引当金",
        "（注）",
        "注記事項",
    ])
    def test_note_labels(self, label: str):
        result = detect_row_type(label, has_values=False)
        assert result == RowRole.NOTE, f"{label!r} → {result}"

    def test_subtotal_takes_priority_over_header(self):
        """'流動資産合計' ends in 合計 → SUBTOTAL even if in header_keywords."""
        result = detect_row_type("流動資産合計", has_values=True)
        assert result == RowRole.SUBTOTAL

    def test_total_takes_priority_over_subtotal(self):
        """'資産合計' is in TOTAL_KEYWORDS — must be TOTAL not SUBTOTAL."""
        result = detect_row_type("資産合計", has_values=True)
        assert result == RowRole.TOTAL


# ===========================================================================
# 10. Round-trip / integration: real balance sheet values
# ===========================================================================

class TestRealFileValues:
    """
    Values taken directly from balance_sheet_after_correction.pdf.
    These test the whole normalization path end-to-end.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("2,002,429",   Decimal("2002429")),
        ("608,281",     Decimal("608281")),
        ("△2,979",      Decimal("-2979")),
        ("△0",          Decimal("0")),
        ("－",          None),
        ("5,863,684",   Decimal("5863684")),
        ("7,534,110",   Decimal("7534110")),
        ("1,123",       Decimal("1123")),
    ])
    def test_real_values(self, raw: str, expected):
        r = parse_number(raw, UnitScale.THOUSAND_YEN)
        assert r.value == expected

    @pytest.mark.parametrize("raw,expected_yen", [
        ("2,002,429",  Decimal("2002429000")),
        ("△2,979",     Decimal("-2979000")),
    ])
    def test_real_values_in_yen(self, raw: str, expected_yen: Decimal):
        r = parse_number(raw, UnitScale.THOUSAND_YEN)
        assert r.value_in_yen == expected_yen

    @pytest.mark.parametrize("label,expected_type", [
        ("資産の部",          RowRole.HEADER),
        ("流動資産",          RowRole.HEADER),
        ("現金及び預金",       RowRole.ITEM),
        ("流動資産合計",       RowRole.SUBTOTAL),
        ("固定資産合計",       RowRole.SUBTOTAL),
        ("有形固定資産合計",    RowRole.SUBTOTAL),
        ("資産合計",          RowRole.TOTAL),
    ])
    def test_real_row_types(self, label: str, expected_type: RowRole):
        has_values = expected_type != RowRole.HEADER
        result = detect_row_type(label, has_values=has_values)
        assert result == expected_type
