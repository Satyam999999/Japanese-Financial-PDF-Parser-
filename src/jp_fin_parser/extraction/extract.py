"""
jp_fin_parser/extraction/extract.py

Stage 1 – Raw text extraction and 2-D clustering from a PDF page.

Pipeline
--------
1. Open the page with pdfplumber; extract words with bboxes.
2. Guard: if too few words → raise NeedsOCR.
3. Filter out stray page-number tokens ("- 4 -" pattern at bottom).
4. Cluster all remaining words into horizontal *line bands* using a
   y-tolerance derived from the dominant font size.
5. Split line bands into a *header block* (title + unit + period headers)
   and *table body* rows using heuristics (unit marker "単位", period label
   keywords, position above first data row).
6. For each table row, detect column slots from the x-coordinate
   distribution of all numeric words on the page, then assign each word
   to its column slot.
7. Merge *wrapped labels*: when two consecutive rows both lack numeric
   values and the second row's x0 is close to the first, join them.
8. Emit an ``ExtractionResult`` with header lines, table rows (each
   carrying a ``label_x0`` for later hierarchy inference), and all raw words.

All coordinate spaces are pdfplumber's native space (pt, origin top-left).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

import pdfplumber

from .exceptions import NeedsOCR

# ---------------------------------------------------------------------------
# Constants / patterns
# ---------------------------------------------------------------------------

# Words per page below this → raise NeedsOCR
_OCR_WORD_THRESHOLD = 10

# Page-number pattern: optional spaces, dash, digits, dash  e.g. "- 4 -"
_PAGE_NUM_RE = re.compile(r"^[-－]\s*\d+\s*[-－]$")
_PAGE_NUM_TOKENS = re.compile(r"^[-－\d\s]+$")

# Unit marker used to recognise the unit header line
_UNIT_RE = re.compile(r"単位")

# Period header keywords
_PERIOD_RE = re.compile(r"(前|当|年度|会計|第\d+期|期末|期首)")

# Numeric value pattern (handles △▲ prefix, commas, hyphens for nil)
_NUMBER_RE = re.compile(r"^[△▲－\-]?[\d,，]+$|^[△▲][\d,，]+$|^[-－]$")


# ---------------------------------------------------------------------------
# Data structures returned by this layer
# ---------------------------------------------------------------------------


@dataclass
class Word:
    """A single word as extracted from pdfplumber, in pt coordinates."""

    text: str
    x0: float
    top: float   # y from top of page (pdfplumber convention)
    x1: float
    bottom: float
    font_size: float = 9.0

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def is_numeric(self) -> bool:
        return bool(_NUMBER_RE.match(self.text.strip()))


@dataclass
class LineBand:
    """A cluster of Words that share approximately the same vertical position."""

    words: list[Word] = field(default_factory=list)

    @property
    def y_center(self) -> float:
        if not self.words:
            return 0.0
        return median(w.y_center for w in self.words)

    @property
    def x0_min(self) -> float:
        return min(w.x0 for w in self.words) if self.words else 0.0

    @property
    def x1_max(self) -> float:
        return max(w.x1 for w in self.words) if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.x0))

    def label_words(self) -> list[Word]:
        return [w for w in self.words if not w.is_numeric()]

    def numeric_words(self) -> list[Word]:
        return [w for w in self.words if w.is_numeric()]

    def is_purely_text(self) -> bool:
        return len(self.numeric_words()) == 0


@dataclass
class ExtractedRow:
    """
    One logical table row, post-clustering and column assignment.

    ``label_x0`` is the left edge of the leftmost label word — used by the
    layout stage to infer indentation depth / hierarchy level.
    """

    label: str                          # concatenated label text
    label_x0: float                     # left edge of label (hierarchy proxy)
    label_y_center: float               # vertical centre of label band
    values: list[str]                   # value strings ordered left→right
    value_x0s: list[float]              # x0 of each value word
    column_indices: list[int]           # which column slot each value maps to
    source_band_y: float                # y_center of originating LineBand


@dataclass
class HeaderBlock:
    """Structured contents of the header region above the data table."""

    correction_marker: Optional[str] = None  # e.g. "（訂正後）"
    title_lines: list[str] = field(default_factory=list)
    unit_line: Optional[str] = None
    period_labels: list[str] = field(default_factory=list)
    period_dates: list[str] = field(default_factory=list)
    raw_bands: list[LineBand] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """
    Complete output of the extraction stage for one PDF page.

    Attributes
    ----------
    page_number     – 0-based page index
    page_width      – page width in pt
    page_height     – page height in pt
    header          – parsed header block
    table_rows      – data rows extracted from the table body
    all_words       – every word extracted (pre-filter); useful for debugging
    column_x_rights – right-edge x positions of detected numeric columns
    """

    page_number: int
    page_width: float
    page_height: float
    header: HeaderBlock
    table_rows: list[ExtractedRow]
    all_words: list[Word]
    column_x_rights: list[float]       # right-edge x of each numeric column


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


def extract_page(
    pdf_path: str | Path,
    page_number: int = 0,
    *,
    ocr_threshold: int = _OCR_WORD_THRESHOLD,
    y_tolerance_factor: float = 0.6,   # fraction of font_size used for row clustering
    col_gap_threshold: float = 20.0,   # minimum x-gap (pt) to split column clusters
) -> ExtractionResult:
    """
    Extract and cluster all words on *page_number* of *pdf_path*.

    Parameters
    ----------
    pdf_path            – path to the PDF file
    page_number         – 0-based page index
    ocr_threshold       – minimum word count; below this NeedsOCR is raised
    y_tolerance_factor  – controls how tightly words are grouped into rows
    col_gap_threshold   – x-gap (pt) that separates distinct numeric columns

    Returns
    -------
    ExtractionResult

    Raises
    ------
    NeedsOCR   – if the page text layer is too sparse
    IndexError – if page_number is out of range
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_number]
        raw_words = page.extract_words(x_tolerance=3, y_tolerance=3)

    words = [
        Word(
            text=w["text"],
            x0=w["x0"],
            top=w["top"],
            x1=w["x1"],
            bottom=w["bottom"],
            font_size=w.get("size", 9.0) or 9.0,
        )
        for w in raw_words
    ]

    # --- OCR guard ------------------------------------------------------------
    if len(words) < ocr_threshold:
        raise NeedsOCR(page_number, len(words), ocr_threshold)

    # --- Filter page-number tokens at the very bottom -------------------------
    page_height = page.height
    words = _filter_page_numbers(words, page_height)

    # --- Determine y-tolerance from dominant font size -----------------------
    dominant_font = median(w.font_size for w in words) if words else 9.0
    y_tol = dominant_font * y_tolerance_factor

    # --- Cluster into horizontal line bands ----------------------------------
    bands = _cluster_into_bands(words, y_tol)

    # --- Detect numeric column slots across the whole page -------------------
    col_x_rights = _detect_column_right_edges(bands, col_gap_threshold)

    # --- Split header vs table body -----------------------------------------
    header_bands, body_bands = _split_header_body(bands)

    # --- Parse header block --------------------------------------------------
    header = _parse_header(header_bands)

    # --- Build table rows from body bands ------------------------------------
    body_bands = _merge_wrapped_labels(body_bands)
    table_rows = _build_table_rows(body_bands, col_x_rights)

    return ExtractionResult(
        page_number=page_number,
        page_width=page.width,
        page_height=page_height,
        header=header,
        table_rows=table_rows,
        all_words=words,
        column_x_rights=col_x_rights,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_page_numbers(words: list[Word], page_height: float) -> list[Word]:
    """
    Remove page-number tokens like "-", "4", "-" that appear near the bottom
    of the page (top > 90 % of page_height).
    """
    bottom_threshold = page_height * 0.90
    filtered: list[Word] = []
    # Collect candidate bottom-strip tokens
    bottom_tokens = [w for w in words if w.top >= bottom_threshold]
    joined = " ".join(w.text for w in bottom_tokens).strip()
    is_page_num_strip = bool(_PAGE_NUM_RE.match(joined.replace(" ", ""))) or (
        len(bottom_tokens) <= 5
        and all(_PAGE_NUM_TOKENS.match(w.text) for w in bottom_tokens)
    )
    for w in words:
        if w.top >= bottom_threshold and is_page_num_strip:
            continue
        filtered.append(w)
    return filtered


def _cluster_into_bands(words: list[Word], y_tol: float) -> list[LineBand]:
    """
    Group words into horizontal bands where every word's y_center is within
    y_tol of the band's median y_center (iterative, single-pass greedy).
    """
    sorted_words = sorted(words, key=lambda w: (w.y_center, w.x0))
    bands: list[LineBand] = []
    for word in sorted_words:
        placed = False
        for band in reversed(bands):  # check recent bands first
            if abs(word.y_center - band.y_center) <= y_tol:
                band.words.append(word)
                placed = True
                break
        if not placed:
            bands.append(LineBand(words=[word]))
    # Sort bands top → bottom
    bands.sort(key=lambda b: b.y_center)
    return bands


def _detect_column_right_edges(
    bands: list[LineBand], gap_threshold: float
) -> list[float]:
    """
    Collect the x1 (right edge) of every numeric word across all bands,
    cluster them by proximity, and return the representative right-edge for
    each cluster (sorted left → right).

    This gives us stable column positions regardless of number width variation.
    """
    all_x1s = sorted(
        w.x1 for band in bands for w in band.words if w.is_numeric()
    )
    if not all_x1s:
        return []

    clusters: list[list[float]] = [[all_x1s[0]]]
    for x in all_x1s[1:]:
        if x - clusters[-1][-1] <= gap_threshold:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    return [median(c) for c in clusters]


def _split_header_body(
    bands: list[LineBand],
) -> tuple[list[LineBand], list[LineBand]]:
    """
    Separate header bands (title, unit, period columns) from table body bands.

    The header block contains metadata (title, correction marker, unit line)
    and column headers (period labels and dates). The table body starts at the
    first row beneath the period column headers/dates (typically a section
    header like 資産の部, or the first data item).
    """
    last_header_idx = -1
    for i, band in enumerate(bands):
        text = band.text.strip()
        if _is_period_header_band(band):
            last_header_idx = i
        elif re.search(r"\(?\s*\d{4}年\s*\d{1,2}月\s*\d{1,2}日\s*\)?", text) and not band.numeric_words():
            last_header_idx = i

    if last_header_idx >= 0:
        return bands[:last_header_idx + 1], bands[last_header_idx + 1:]

    # Fallback: first band that contains numeric words
    first_data_idx = 0
    for i, band in enumerate(bands):
        if band.numeric_words() and not _is_period_header_band(band):
            first_data_idx = i
            break
    return bands[:first_data_idx], bands[first_data_idx:]


def _is_period_header_band(band: LineBand) -> bool:
    """Return True if this band looks like a period header row."""
    text = band.text
    return bool(_PERIOD_RE.search(text)) and not band.numeric_words()


def _parse_header(bands: list[LineBand]) -> HeaderBlock:
    """
    Extract structured info from header bands.
    """
    hb = HeaderBlock(raw_bands=bands)
    for band in bands:
        text = band.text.strip()
        if "訂正後" in text or "訂正前" in text:
            hb.correction_marker = text
        elif _UNIT_RE.search(text):
            hb.unit_line = text
        elif _PERIOD_RE.search(text):
            # Period label row (e.g. 前連結会計年度 / 当連結会計年度)
            # or date row (e.g. (2025年3月31日))
            if "年" in text and "月" in text and "日" in text:
                hb.period_dates.extend(
                    w.text for w in band.words if "年" in w.text or "月" in w.text
                )
            else:
                hb.period_labels.extend(
                    w.text for w in band.words
                    if _PERIOD_RE.search(w.text)
                )
        else:
            hb.title_lines.append(text)
    return hb


def _merge_wrapped_labels(bands: list[LineBand]) -> list[LineBand]:
    """
    Merge consecutive bands where:
    - The first band has no numeric values (pure label / partial label), AND
    - The second band also has no numeric values, AND
    - Their x0_min values are within 5 pt of each other (same indent level).

    This handles labels that wrap to a second line.
    """
    if not bands:
        return bands

    merged: list[LineBand] = [bands[0]]
    for band in bands[1:]:
        prev = merged[-1]
        prev_has_nums = bool(prev.numeric_words())
        curr_has_nums = bool(band.numeric_words())
        x0_close = abs(band.x0_min - prev.x0_min) <= 5.0

        if (not prev_has_nums) and (not curr_has_nums) and x0_close:
            # Absorb into previous band
            merged[-1] = LineBand(words=prev.words + band.words)
        else:
            merged.append(band)
    return merged


def _build_table_rows(
    bands: list[LineBand], col_x_rights: list[float]
) -> list[ExtractedRow]:
    """
    Convert each body LineBand into an ExtractedRow.

    For each numeric word, assign it to the nearest column slot by comparing
    its x1 against each slot's right-edge and picking the closest.
    """
    rows: list[ExtractedRow] = []
    for band in bands:
        label_words = sorted(band.label_words(), key=lambda w: w.x0)
        numeric_words = sorted(band.numeric_words(), key=lambda w: w.x0)

        label = " ".join(w.text for w in label_words).strip()
        label_x0 = label_words[0].x0 if label_words else band.x0_min

        values: list[str] = []
        value_x0s: list[float] = []
        col_indices: list[int] = []

        for nw in numeric_words:
            col_idx = _nearest_column(nw.x1, col_x_rights)
            values.append(nw.text)
            value_x0s.append(nw.x0)
            col_indices.append(col_idx)

        rows.append(
            ExtractedRow(
                label=label,
                label_x0=label_x0,
                label_y_center=band.y_center,
                values=values,
                value_x0s=value_x0s,
                column_indices=col_indices,
                source_band_y=band.y_center,
            )
        )
    return rows


def _nearest_column(x1: float, col_x_rights: list[float]) -> int:
    """Return the index of the column slot whose right-edge is nearest to x1."""
    if not col_x_rights:
        return 0
    return min(range(len(col_x_rights)), key=lambda i: abs(col_x_rights[i] - x1))
