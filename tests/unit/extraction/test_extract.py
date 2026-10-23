"""
tests/unit/extraction/test_extract.py

Unit + integration tests for the extraction layer.

Unit tests use a tiny mock pdfplumber page so they run fast and with no PDF
needed.  The integration test (marked `real_pdf`) runs against the actual
balance_sheet_after_correction.pdf in the repo root and is skipped if the
file is absent.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jp_fin_parser.extraction.exceptions import NeedsOCR
from jp_fin_parser.extraction.extract import (
    ExtractionResult,
    LineBand,
    Word,
    _cluster_into_bands,
    _detect_column_right_edges,
    _filter_page_numbers,
    _merge_wrapped_labels,
    _nearest_column,
    extract_page,
)

# ---------------------------------------------------------------------------
# Path to the real PDF (may not exist in CI)
# ---------------------------------------------------------------------------
REAL_PDF = Path(__file__).parents[4] / "balance_sheet_after_correction.pdf"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _word(text: str, x0: float, top: float, x1: float, bottom: float) -> Word:
    return Word(text=text, x0=x0, top=top, x1=x1, bottom=bottom, font_size=9.0)


def _make_pdfplumber_word(text, x0, top, x1, bottom):
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom, "size": 9.0}


def _mock_page(words_dicts, width=595, height=842):
    """Return a mock pdfplumber page that returns the supplied word dicts."""
    page = MagicMock()
    page.width = width
    page.height = height
    page.extract_words.return_value = words_dicts
    return page


def _mock_pdf(page):
    """Return a mock pdfplumber PDF context manager."""
    pdf = MagicMock()
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    pdf.pages = [page]
    return pdf


# ---------------------------------------------------------------------------
# 1. _filter_page_numbers
# ---------------------------------------------------------------------------


class TestFilterPageNumbers:
    def test_removes_bottom_page_number_strip(self):
        """Words '-', '4', '-' near the very bottom are filtered out."""
        normal = _word("現金", 88, 172, 142, 181)
        dash1 = _word("-", 287, 820, 291, 829)
        num = _word("4", 295, 820, 300, 829)
        dash2 = _word("-", 304, 820, 309, 829)

        result = _filter_page_numbers([normal, dash1, num, dash2], page_height=842)
        assert len(result) == 1
        assert result[0].text == "現金"

    def test_keeps_normal_words_near_bottom(self):
        """Words that don't match the page-number pattern are kept."""
        w = _word("注記", 88, 800, 140, 810)
        result = _filter_page_numbers([w], page_height=842)
        assert len(result) == 1

    def test_empty_input(self):
        assert _filter_page_numbers([], 842) == []


# ---------------------------------------------------------------------------
# 2. _cluster_into_bands
# ---------------------------------------------------------------------------


class TestClusterIntoBands:
    def test_same_y_words_grouped(self):
        words = [
            _word("現金", 88, 172, 142, 181),
            _word("1,000", 357, 172, 397, 181),
            _word("2,000", 490, 172, 522, 181),
        ]
        bands = _cluster_into_bands(words, y_tol=5.0)
        assert len(bands) == 1
        assert len(bands[0].words) == 3

    def test_different_y_makes_separate_bands(self):
        w1 = _word("現金", 88, 172, 142, 181)
        w2 = _word("売掛金", 88, 184, 124, 193)
        bands = _cluster_into_bands([w1, w2], y_tol=5.0)
        assert len(bands) == 2

    def test_bands_sorted_top_to_bottom(self):
        w_low = _word("B", 88, 200, 100, 209)
        w_high = _word("A", 88, 100, 100, 109)
        bands = _cluster_into_bands([w_low, w_high], y_tol=5.0)
        assert bands[0].words[0].text == "A"
        assert bands[1].words[0].text == "B"

    def test_tolerance_boundary(self):
        """Words exactly y_tol apart should land in the same band."""
        w1 = _word("A", 0, 100, 10, 109)
        w2 = _word("B", 20, 109, 30, 118)  # y_center delta = ~9 ≤ tol 10
        bands = _cluster_into_bands([w1, w2], y_tol=10.0)
        assert len(bands) == 1


# ---------------------------------------------------------------------------
# 3. _detect_column_right_edges
# ---------------------------------------------------------------------------


class TestDetectColumnRightEdges:
    def test_two_distinct_columns(self):
        """Two clusters of numeric x1s well apart → two column slots."""
        # Col A: x1 around 397-398
        # Col B: x1 around 522-523
        bands = [
            LineBand(words=[
                _word("1,000", 357, 172, 397, 181),
                _word("2,000", 490, 172, 522, 181),
            ]),
            LineBand(words=[
                _word("500", 370, 184, 397, 193),
                _word("600", 495, 184, 522, 193),
            ]),
        ]
        cols = _detect_column_right_edges(bands, gap_threshold=20.0)
        assert len(cols) == 2
        assert cols[0] < cols[1]
        assert abs(cols[0] - 397) < 5
        assert abs(cols[1] - 522) < 5

    def test_no_numeric_words(self):
        bands = [LineBand(words=[_word("現金", 88, 172, 142, 181)])]
        assert _detect_column_right_edges(bands, 20.0) == []


# ---------------------------------------------------------------------------
# 4. _merge_wrapped_labels
# ---------------------------------------------------------------------------


class TestMergeWrappedLabels:
    def test_two_pure_label_bands_close_x0_are_merged(self):
        b1 = LineBand(words=[_word("建物及び", 88, 100, 140, 109)])
        b2 = LineBand(words=[_word("構築物", 90, 112, 130, 121)])  # x0 within 5
        result = _merge_wrapped_labels([b1, b2])
        assert len(result) == 1
        assert "建物及び" in result[0].text
        assert "構築物" in result[0].text

    def test_label_then_numeric_not_merged(self):
        b1 = LineBand(words=[_word("現金", 88, 172, 142, 181)])
        b2 = LineBand(words=[
            _word("売掛金", 88, 184, 124, 193),
            _word("1,000", 357, 184, 397, 193),
        ])
        result = _merge_wrapped_labels([b1, b2])
        assert len(result) == 2

    def test_different_x0_not_merged(self):
        """Labels at different indentation levels should NOT be merged."""
        b1 = LineBand(words=[_word("流動資産", 75, 100, 111, 109)])
        b2 = LineBand(words=[_word("現金", 88, 112, 124, 121)])  # x0 diff = 13 > 5
        result = _merge_wrapped_labels([b1, b2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. _nearest_column
# ---------------------------------------------------------------------------


class TestNearestColumn:
    def test_exact_match(self):
        assert _nearest_column(397.5, [397.5, 522.6]) == 0

    def test_closer_to_second(self):
        assert _nearest_column(520.0, [397.5, 522.6]) == 1

    def test_empty_columns(self):
        assert _nearest_column(500.0, []) == 0


# ---------------------------------------------------------------------------
# 6. NeedsOCR exception
# ---------------------------------------------------------------------------


class TestNeedsOCR:
    def test_raised_when_too_few_words(self):
        """A page with 3 words (< threshold 10) should raise NeedsOCR."""
        sparse_words = [
            _make_pdfplumber_word("A", 0, 0, 10, 10),
            _make_pdfplumber_word("B", 20, 0, 30, 10),
            _make_pdfplumber_word("C", 40, 0, 50, 10),
        ]
        mock_page = _mock_page(sparse_words)
        mock_pdf = _mock_pdf(mock_page)

        with patch("pdfplumber.open", return_value=mock_pdf):
            with pytest.raises(NeedsOCR) as exc_info:
                extract_page("fake.pdf", page_number=0, ocr_threshold=10)

        assert exc_info.value.page_number == 0
        assert exc_info.value.word_count == 3
        assert "OCR" in str(exc_info.value)

    def test_not_raised_when_enough_words(self):
        """A page with sufficient words should not raise."""
        words = [
            _make_pdfplumber_word(f"w{i}", i * 10, 100, i * 10 + 8, 110)
            for i in range(15)
        ]
        mock_page = _mock_page(words)
        mock_pdf = _mock_pdf(mock_page)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract_page("fake.pdf", page_number=0, ocr_threshold=10)
        assert isinstance(result, ExtractionResult)


# ---------------------------------------------------------------------------
# 7. Full mock extraction (small synthetic balance sheet)
# ---------------------------------------------------------------------------


class TestMockExtraction:
    """
    Simulate a minimal BS page with:
      Header: correction marker, title, unit, two period columns + dates
      Body:   4 data rows (2 items + 1 subtotal label-only, 1 item)
    """

    @pytest.fixture
    def mock_result(self):
        raw = [
            # Header
            _make_pdfplumber_word("（訂正後）",       54,  63, 104,  73),
            _make_pdfplumber_word("連結貸借対照表",    63,  91, 153, 100),
            _make_pdfplumber_word("（単位：千円）",   461, 104, 524, 113),
            _make_pdfplumber_word("前連結会計年度",   305, 122, 368, 131),
            _make_pdfplumber_word("当連結会計年度",   430, 122, 493, 131),
            _make_pdfplumber_word("(2025年３月31日)", 297, 131, 376, 140),
            _make_pdfplumber_word("(2026年３月31日)", 422, 131, 501, 140),
            # Body
            _make_pdfplumber_word("現金及び預金",      88, 172, 142, 181),
            _make_pdfplumber_word("2,002,429",       356, 172, 397, 181),
            _make_pdfplumber_word("608,281",         490, 172, 522, 181),
            _make_pdfplumber_word("売掛金",            88, 184, 115, 193),
            _make_pdfplumber_word("2,019,476",       356, 184, 397, 193),
            _make_pdfplumber_word("884,339",         490, 184, 522, 193),
            _make_pdfplumber_word("流動資産合計",       88, 196, 142, 205),
            _make_pdfplumber_word("5,863,684",       356, 196, 397, 205),
            _make_pdfplumber_word("3,377,407",       490, 196, 522, 205),
            # Page number
            _make_pdfplumber_word("-",               287, 820, 291, 829),
            _make_pdfplumber_word("4",               295, 820, 300, 829),
            _make_pdfplumber_word("-",               304, 820, 309, 829),
        ]
        mock_page = _mock_page(raw)
        mock_pdf = _mock_pdf(mock_page)

        with patch("pdfplumber.open", return_value=mock_pdf):
            return extract_page("fake.pdf", page_number=0)

    def test_page_number_tokens_excluded(self, mock_result):
        all_labels = [r.label for r in mock_result.table_rows]
        assert not any("-" == lbl.strip() for lbl in all_labels)
        assert not any("4" == lbl.strip() for lbl in all_labels)

    def test_two_column_slots_detected(self, mock_result):
        assert len(mock_result.column_x_rights) == 2

    def test_header_correction_marker(self, mock_result):
        assert mock_result.header.correction_marker is not None
        assert "訂正後" in mock_result.header.correction_marker

    def test_header_unit_line(self, mock_result):
        assert mock_result.header.unit_line is not None
        assert "単位" in mock_result.header.unit_line

    def test_header_period_labels(self, mock_result):
        assert len(mock_result.header.period_labels) >= 2

    def test_table_rows_present(self, mock_result):
        assert len(mock_result.table_rows) >= 3

    def test_each_row_has_two_values(self, mock_result):
        # Every data row (not pure header) should have 2 values
        data_rows = [r for r in mock_result.table_rows if r.values]
        for row in data_rows:
            assert len(row.values) == 2, f"Row '{row.label}' has {len(row.values)} values"

    def test_label_x0_captured(self, mock_result):
        labels = {r.label: r.label_x0 for r in mock_result.table_rows}
        # All labels should have a positive x0
        for label, x0 in labels.items():
            assert x0 > 0, f"label_x0 for '{label}' is {x0}"

    def test_subtotal_row_present(self, mock_result):
        labels = [r.label for r in mock_result.table_rows]
        assert any("合計" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# 8. Integration test on the real PDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_PDF.exists(), reason="Real PDF not found")
class TestRealPDF:
    @pytest.fixture(scope="class")
    def result(self):
        return extract_page(REAL_PDF, page_number=0)

    def test_no_ocr_needed(self, result):
        """Real PDF has a text layer — should not raise NeedsOCR."""
        assert isinstance(result, ExtractionResult)

    def test_correction_marker_detected(self, result):
        assert result.header.correction_marker is not None
        assert "訂正後" in result.header.correction_marker

    def test_unit_line_detected(self, result):
        assert result.header.unit_line is not None
        assert "千円" in result.header.unit_line

    def test_two_period_columns(self, result):
        assert len(result.column_x_rights) == 2

    def test_sensible_row_count(self, result):
        # The real page has ~40 data rows
        assert len(result.table_rows) >= 20

    def test_page_number_excluded(self, result):
        """The '- 4 -' footer must not appear as a table row."""
        for row in result.table_rows:
            assert not (row.label.strip() == "4"), "page number leaked into table rows"

    def test_label_x0_hierarchy(self, result):
        """
        At least three distinct x0 positions should exist in label_x0,
        reflecting the BS indentation hierarchy
        (資産の部=62, 流動資産=75, 現金=88, 建物=101, 減価償却=114).
        """
        x0s = {round(r.label_x0) for r in result.table_rows if r.label}
        assert len(x0s) >= 3, f"Only {len(x0s)} distinct x0 positions found: {sorted(x0s)}"

    def test_known_value_present(self, result):
        """2,002,429 (現金及び預金 前年度) must appear somewhere in extracted values."""
        all_vals = [v for row in result.table_rows for v in row.values]
        assert "2,002,429" in all_vals

    def test_print_rows(self, result, capsys):
        """Print all rows so the user can visually inspect the output."""
        print(f"\n{'label':45s} {'label_x0':>8s}  {'values'}")
        print("-" * 80)
        for row in result.table_rows:
            print(f"{row.label:45s} {row.label_x0:8.1f}  {row.values}")
        captured = capsys.readouterr()
        assert len(captured.out) > 0
