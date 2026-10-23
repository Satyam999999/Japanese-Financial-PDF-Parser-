"""
jp_fin_parser/preprocessing/normalize.py

Stage 2 – Normalization of raw extracted text into typed domain values.

Responsibilities
----------------
1. Number parsing
   - Full-width digits / commas → ASCII
   - △ / ▲ prefix or ( ) wrapping → negative
   - Lone－/―/ー/- → nil (None, is_nil=True), distinct from a missing cell
   - △0 / ▲0 edge case → Decimal("0"), flagged as originally signed
   - Apply unit multiplier (千円→×1000) while keeping the original value

2. Label normalization
   - NFKC unicode normalisation
   - Strip spaces *inside* Japanese labels (e.g. "現 金 及 び 預 金")
   - Preserve the original raw label unchanged

3. Date parsing
   - 西暦 (Gregorian): 2025年3月31日, (2025年３月31日)
   - 和暦 (Japanese eras): 令和7年、平成30年、昭和64年、大正x年、明治x年
   - Fiscal-year labels: 前連結会計年度, 当第X期, 第X四半期 etc.

4. Row-type detection
   - Ending in 合計 / 計 / 小計  → SUBTOTAL
   - Grand-total keywords (資産合計, 負債純資産合計, …) → TOTAL
   - Section-header phrases (XXの部, 流動資産, 固定資産, …) + no values → HEADER
   - Anything else with values → ITEM
   - Footnote / annotation rows → NOTE
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

from jp_fin_parser.models.row import RowRole
from jp_fin_parser.models.table import UnitScale

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Lone nil markers (全角 dash variants + ASCII hyphen standing alone)
_NIL_RE = re.compile(r"^[－―ーｰ\-‐‑–—]+$")

# Negative sign prefixes
_NEG_PREFIX_RE = re.compile(r"^[△▲]")

# Parenthesised negative: (1,234)  or （1,234）
_PAREN_RE = re.compile(r"^[（(](.+)[）)]$")

# Full-width digit / comma → half-width mapping (done via NFKC + translate)
_FULLWIDTH_COMMA_RE = re.compile(r"[，、]")

# Gregorian date: 2025年3月31日  (digits may be full-width after NFKC)
_SEIREKI_RE = re.compile(
    r"[（(]?(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日[）)]?"
)

# Japanese era table  {era_kanji: (gregorian_year_of_era_1, max_year)}
_ERA_TABLE: dict[str, int] = {
    "明治": 1868,
    "大正": 1912,
    "昭和": 1926,
    "平成": 1989,
    "令和": 2019,
}
_ERA_RE = re.compile(
    r"(?P<era>明治|大正|昭和|平成|令和)(?P<nen>\d{1,2})年"
    r"(?:(?P<month>\d{1,2})月(?P<day>\d{1,2})日)?"
)

# Fiscal-year label patterns
_FISCAL_LABEL_RE = re.compile(
    r"(?:前|当|翌|次|第\d+)?(?:連結)?(?:会計年度|事業年度|期)|"
    r"第\d+(?:期|四半期)"
)

# ---- Row-type rules --------------------------------------------------------

# Labels that end with these suffixes are SUBTOTAL (unless overridden by TOTAL)
_SUBTOTAL_SUFFIX_RE = re.compile(r"(?:合計|小計|計)$")

# Grand-total labels (exact or substring match) → TOTAL
_TOTAL_KEYWORDS = frozenset(
    [
        "資産合計",
        "負債合計",
        "純資産合計",
        "負債純資産合計",
        "負債及び純資産合計",
        "資本合計",
        "株主資本合計",
        "その他の包括利益累計額合計",
        "非支配株主持分",   # sometimes treated as total-equivalent
    ]
)

# Section-header labels (have NO numeric values; signal a new section)
_HEADER_SUFFIX_RE = re.compile(r"の部$")
_HEADER_KEYWORDS = frozenset(
    [
        "資産の部",
        "負債の部",
        "純資産の部",
        "株主資本",
        "その他の包括利益累計額",
        "非支配株主持分",
        "流動資産",
        "固定資産",
        "有形固定資産",
        "無形固定資産",
        "投資その他の資産",
        "流動負債",
        "固定負債",
        "繰延税金資産",   # sometimes appears as section header
    ]
)

# Footnote / annotation rows
_NOTE_KEYWORDS = frozenset(["※", "注", "（注）", "注記"])


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParsedNumber:
    """Result of parsing one raw numeric cell."""

    original: str                       # exactly as extracted
    value: Optional[Decimal]            # None → nil cell
    value_in_yen: Optional[Decimal]     # value × unit multiplier (None if nil)
    is_nil: bool = False                # True when cell explicitly marked nil
    is_negative: bool = False           # True when original had sign marker
    had_sign_marker: bool = False       # True even when value==0 (△0 edge case)


@dataclass
class ParsedLabel:
    """Result of normalising one label string."""

    raw: str                            # original extraction text
    normalized: str                     # NFKC + space-stripped
    row_type: RowRole = RowRole.ITEM    # inferred; may be overridden later


@dataclass
class ParsedDate:
    """Result of parsing a date or fiscal-year label."""

    original: str
    date: Optional[date] = None
    era: Optional[str] = None           # e.g. "令和", "平成", "西暦"
    is_fiscal_label: bool = False       # True if it's a period label, not a date


# ---------------------------------------------------------------------------
# 1. Number parsing
# ---------------------------------------------------------------------------


def parse_number(raw: str, unit: UnitScale = UnitScale.UNKNOWN) -> ParsedNumber:
    """
    Parse a raw extracted numeric cell into a :class:`ParsedNumber`.

    Parameters
    ----------
    raw   – text exactly as extracted from the PDF
    unit  – unit scale; used to compute ``value_in_yen``

    Returns
    -------
    ParsedNumber

    Edge cases handled
    ------------------
    * △0 / ▲0  → value = Decimal("0"), had_sign_marker = True, is_negative = False
    * Lone dash variants (－ ― ー - etc.) → nil (value = None, is_nil = True)
    * Full-width digits and commas → converted via NFKC
    * Parenthesised values (1,234) / （1,234） → negative
    """
    text = raw.strip()
    original = text

    # --- Nil detection (must come before sign stripping) ------------------
    if _NIL_RE.match(text):
        return ParsedNumber(
            original=original,
            value=None,
            value_in_yen=None,
            is_nil=True,
            is_negative=False,
            had_sign_marker=False,
        )

    # --- NFKC normalise (full-width → half-width digits, commas, etc.) ----
    text = unicodedata.normalize("NFKC", text)
    # Remove any remaining full-width commas
    text = _FULLWIDTH_COMMA_RE.sub("", text)

    # --- Detect sign -------------------------------------------------------
    had_sign = False
    is_neg = False

    if _NEG_PREFIX_RE.match(text):
        had_sign = True
        is_neg = True
        text = text[1:]  # strip △/▲

    paren_m = _PAREN_RE.match(text)
    if paren_m:
        had_sign = True
        is_neg = True
        text = paren_m.group(1)

    # --- Strip commas and parse decimal ------------------------------------
    text = text.replace(",", "").strip()

    try:
        value = Decimal(text)
    except InvalidOperation:
        # Unparseable → treat as nil with a flag
        return ParsedNumber(
            original=original,
            value=None,
            value_in_yen=None,
            is_nil=True,
            is_negative=False,
            had_sign_marker=had_sign,
        )

    # --- △0 edge case: 0 is not negative regardless of sign marker --------
    if value == Decimal("0"):
        is_neg = False

    if is_neg:
        value = -value

    multiplier = unit.multiplier_to_yen
    value_in_yen = value * multiplier if unit != UnitScale.UNKNOWN else None

    return ParsedNumber(
        original=original,
        value=value,
        value_in_yen=value_in_yen,
        is_nil=False,
        is_negative=is_neg,
        had_sign_marker=had_sign,
    )


# ---------------------------------------------------------------------------
# 2. Label normalisation
# ---------------------------------------------------------------------------

# Ranges of CJK / Japanese character categories (for space-stripping)
_JP_CHAR_RE = re.compile(
    r"[\u3000-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF\u30A0-\u30FF\u3040-\u309F]"
)


def _strip_internal_spaces(text: str) -> str:
    """
    Remove ASCII/ideographic spaces that appear *between* Japanese characters.

    Spaces between ASCII words or at boundaries with ASCII tokens are kept.
    """
    # Replace ideographic space with ASCII space first
    text = text.replace("\u3000", " ")
    # If the text is purely Japanese (no ASCII words), strip all spaces
    without_spaces = text.replace(" ", "")
    if _JP_CHAR_RE.sub("", without_spaces) == "":
        # Entirely CJK → strip all spaces
        return without_spaces
    # Mixed: strip spaces only between two consecutive CJK characters
    result = re.sub(r"(?<=[\u3040-\u9FFF])\s+(?=[\u3040-\u9FFF])", "", text)
    return result


def normalize_label(raw: str) -> ParsedLabel:
    """
    Normalise a raw label string extracted from the PDF.

    Steps
    -----
    1. NFKC unicode normalisation (full-width → half-width, compat forms)
    2. Strip leading/trailing whitespace
    3. Remove spaces inside Japanese-only labels
    4. Infer row_type (section header, subtotal, total, item)
    """
    nfkc = unicodedata.normalize("NFKC", raw)
    nfkc = nfkc.replace("(", "（").replace(")", "）")
    normalized = _strip_internal_spaces(nfkc).strip()
    row_type = detect_row_type(normalized, has_values=None)  # type: ignore[arg-type]
    return ParsedLabel(raw=raw, normalized=normalized, row_type=row_type)


# ---------------------------------------------------------------------------
# 3. Date parsing
# ---------------------------------------------------------------------------


def parse_date(text: str) -> ParsedDate:
    """
    Parse a date string in Gregorian (西暦) or Japanese era (和暦) format,
    or recognise it as a fiscal-year label.

    Supported inputs
    ----------------
    * "2025年3月31日"
    * "（2025年３月31日）"
    * "令和7年3月31日"
    * "平成30年9月30日"
    * "昭和64年1月7日"
    * "前連結会計年度" → is_fiscal_label=True, date=None
    * Junk text → date=None, is_fiscal_label=False
    """
    original = text
    # NFKC first (converts full-width digits)
    text = unicodedata.normalize("NFKC", text).strip()

    # --- Gregorian ---------------------------------------------------------
    m = _SEIREKI_RE.search(text)
    if m:
        try:
            d = date(int(m.group("year")), int(m.group("month")), int(m.group("day")))
            return ParsedDate(original=original, date=d, era="西暦")
        except ValueError:
            pass

    # --- Japanese era (和暦) ------------------------------------------------
    m = _ERA_RE.search(text)
    if m:
        era = m.group("era")
        nen = int(m.group("nen"))
        base_year = _ERA_TABLE.get(era, 1868)
        gregorian_year = base_year + nen - 1
        month_str = m.group("month")
        day_str = m.group("day")
        if month_str and day_str:
            try:
                d = date(gregorian_year, int(month_str), int(day_str))
                return ParsedDate(original=original, date=d, era=era)
            except ValueError:
                pass
        # Year-only match (no month/day)
        return ParsedDate(original=original, date=None, era=era, is_fiscal_label=False)

    # --- Fiscal-year label (no concrete date) ------------------------------
    if _FISCAL_LABEL_RE.search(text):
        return ParsedDate(original=original, date=None, is_fiscal_label=True)

    return ParsedDate(original=original)


# ---------------------------------------------------------------------------
# 4. Row-type detection
# ---------------------------------------------------------------------------


def detect_row_type(label_normalized: str, has_values: Optional[bool]) -> RowRole:
    """
    Infer the :class:`RowRole` of a row from its normalised label and whether
    it carries numeric values.

    Priority order
    --------------
    1. Grand-total keywords  → TOTAL
    2. Subtotal suffix (合計/計/小計)  → SUBTOTAL
    3. Section-header keywords *and* no values  → HEADER
    4. Header-suffix (の部) regardless of values  → HEADER
    5. Note keywords  → NOTE
    6. Default  → ITEM

    Parameters
    ----------
    label_normalized – the NFKC-normalised label
    has_values       – True if the row carries numeric values; False/None if not
    """
    label = label_normalized.strip()

    # --- Grand totals -------------------------------------------------------
    if label in _TOTAL_KEYWORDS:
        return RowRole.TOTAL

    # --- Subtotals ----------------------------------------------------------
    if _SUBTOTAL_SUFFIX_RE.search(label):
        return RowRole.SUBTOTAL

    # --- Section headers ----------------------------------------------------
    if _HEADER_SUFFIX_RE.search(label):
        return RowRole.HEADER

    if label in _HEADER_KEYWORDS and not has_values:
        return RowRole.HEADER

    # --- Pure net rows (純額) -----------------------------------------------
    if "純額" in label:
        return RowRole.NET

    # --- Footnotes ----------------------------------------------------------
    if any(kw in label for kw in _NOTE_KEYWORDS):
        return RowRole.NOTE

    return RowRole.ITEM
