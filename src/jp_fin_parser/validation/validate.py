"""
jp_fin_parser/validation/validate.py

Stage 4 – Rule-based validation engine for financial statements.

Responsibilities
----------------
1. Rule engine with extensible validation rules:
   - SubtotalRule: subtotal == sum(children) per period within tolerance
   - NetRule (純額): gross + accumulated depreciation (negative) == net within tolerance
   - BalanceSheetIdentityRule: 資産合計 == 負債純資産合計 when both sides are present
2. Rounding tolerance:
   abs(diff) <= max(n_children, 1) * tolerance_unit (default 1 unit of rounding precision)
   Accounts for filings rounded to thousands (千円) where separately rounded items accumulate diffs.
3. Diagnostic failure reporting:
   Each failure produces a flag with row ids, expected, actual, diff, and period.
4. Single-suspect child isolation & OCR error correction suggestion:
   Detects single-digit substitutions, sign flips, or low-confidence cells and suggests
   the likely correction without applying it automatically.
5. Per-row confidence adjustments:
   Reduces validation confidence score and attaches ValidationFlag.ARITHMETIC_MISMATCH
   to failing rows and suspect children.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Protocol, Sequence
from uuid import UUID

from jp_fin_parser.layout.hierarchy import HierarchyNode, HierarchyResult, RelationType
from jp_fin_parser.models.confidence import ValidationFlag
from jp_fin_parser.models.row import FinancialRow, RowRole
from jp_fin_parser.models.table import (
    ArithmeticCheck,
    FinancialTable,
    PageMeta,
    Period,
    StatementType,
    UnitScale,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE_UNIT = Decimal("1")  # 1 unit (1 thousand yen, 1 million yen, etc.)


# ---------------------------------------------------------------------------
# Correction Suggestion Model
# ---------------------------------------------------------------------------


@dataclass
class CorrectionSuggestion:
    """A suggested correction for a single suspect cell that would reconcile arithmetic."""

    row_id: Optional[UUID]
    row_label: str
    period: str
    current_value: Optional[Decimal]
    suggested_value: Decimal
    discrepancy: Decimal  # actual - expected
    reason: str

    def to_dict(self) -> dict:
        return {
            "row_id": str(self.row_id) if self.row_id else None,
            "row_label": self.row_label,
            "period": self.period,
            "current_value": float(self.current_value) if self.current_value is not None else None,
            "suggested_value": float(self.suggested_value),
            "discrepancy": float(self.discrepancy),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Levenshtein / Edit distance helper
# ---------------------------------------------------------------------------


def _levenshtein(s1: str, s2: str) -> int:
    """Compute string edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _find_suspect_child(
    children: Sequence[FinancialRow],
    diff: Decimal,  # subtotal_actual - expected_sum
    period: str,
) -> Optional[CorrectionSuggestion]:
    """
    Given an arithmetic discrepancy (diff = actual - expected), check if
    EXACTLY ONE child row is suspect for an OCR / parsing error.

    If candidate = child_value + diff:
    1. Sign error: child_value was positive when it should be negative (or vice versa).
    2. Single digit substitution error: string edit distance is exactly 1 (e.g. 3 vs 8, 1 vs 7).
    3. Low extraction confidence cell.
    """
    candidates: list[CorrectionSuggestion] = []

    for child in children:
        c_val = child.values.get(period)
        if c_val is None:
            continue

        cand_val = c_val + diff

        # 1. Sign check: candidate is the exact negative of current value
        if c_val != 0 and cand_val == -c_val:
            candidates.append(
                CorrectionSuggestion(
                    row_id=child.id,
                    row_label=child.label_normalized,
                    period=period,
                    current_value=c_val,
                    suggested_value=cand_val,
                    discrepancy=diff,
                    reason=(
                        f"Possible inverted/dropped negative sign (△/▲): "
                        f"current {c_val} vs suggested {cand_val}"
                    ),
                )
            )
            continue

        # 2. Single digit OCR substitution
        s_cur = str(abs(c_val))
        s_cand = str(abs(cand_val))

        if len(s_cur) == len(s_cand):
            mismatches = [
                (idx, c1, c2)
                for idx, (c1, c2) in enumerate(zip(s_cur, s_cand))
                if c1 != c2
            ]
            if len(mismatches) == 1:
                idx, orig_char, cand_char = mismatches[0]
                candidates.append(
                    CorrectionSuggestion(
                        row_id=child.id,
                        row_label=child.label_normalized,
                        period=period,
                        current_value=c_val,
                        suggested_value=cand_val,
                        discrepancy=diff,
                        reason=(
                            f"Possible OCR digit error: '{s_cur}' vs suggested '{s_cand}' "
                            f"(position {idx}: '{orig_char}' misread as '{cand_char}')"
                        ),
                    )
                )
                continue

        # 3. Single insertion or deletion (Levenshtein == 1)
        if abs(len(s_cur) - len(s_cand)) == 1 and _levenshtein(s_cur, s_cand) == 1:
            candidates.append(
                CorrectionSuggestion(
                    row_id=child.id,
                    row_label=child.label_normalized,
                    period=period,
                    current_value=c_val,
                    suggested_value=cand_val,
                    discrepancy=diff,
                    reason=(
                        f"Possible OCR character insertion/omission: "
                        f"'{s_cur}' vs suggested '{s_cand}'"
                    ),
                )
            )
            continue

        # 4. Low extraction confidence flag
        if child.confidence.extraction < 0.85 or ValidationFlag.LOW_EXTRACTION_CONFIDENCE in child.confidence.flags:
            candidates.append(
                CorrectionSuggestion(
                    row_id=child.id,
                    row_label=child.label_normalized,
                    period=period,
                    current_value=c_val,
                    suggested_value=cand_val,
                    discrepancy=diff,
                    reason=(
                        f"Suspect row with low extraction confidence ({child.confidence.extraction:.2f}): "
                        f"current {c_val} vs suggested {cand_val}"
                    ),
                )
            )

    # Only return a suggestion when exactly ONE candidate child is isolated
    if len(candidates) == 1:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Validation Rule Interface
# ---------------------------------------------------------------------------


class ValidationRule(Protocol):
    """Protocol for pluggable validation rules."""

    name: str

    def apply(
        self,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult] = None,
        tolerance_unit: Decimal = DEFAULT_TOLERANCE_UNIT,
    ) -> list[ArithmeticCheck]:
        ...


# ---------------------------------------------------------------------------
# Rule 1: SubtotalRule
# ---------------------------------------------------------------------------


class SubtotalRule:
    """
    Validates that each subtotal / total equals the sum of its children per period.

    Tolerance:
        tolerance = max(n_children, 1) * tolerance_unit
        abs(diff) <= tolerance passes (filings round to thousands/millions).
    """

    name: str = "subtotal_rule"

    def apply(
        self,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult] = None,
        tolerance_unit: Decimal = DEFAULT_TOLERANCE_UNIT,
    ) -> list[ArithmeticCheck]:
        checks: list[ArithmeticCheck] = []

        subtotals = [
            r
            for r in table.rows
            if r.row_type in (RowRole.SUBTOTAL, RowRole.TOTAL)
            and not (r.label_normalized.endswith("（純額）") or r.label_normalized.endswith("(純額)"))
        ]

        for sub in subtotals:
            children = self._get_contributing_children(sub, table, hierarchy)
            if not children:
                continue

            for period, sub_val in sub.values.items():
                if sub_val is None:
                    continue

                valid_children = [
                    c for c in children if period in c.values and c.values[period] is not None
                ]
                if not valid_children:
                    continue

                child_sum: Decimal = sum(
                    (c.values[period] for c in valid_children),  # type: ignore[arg-type]
                    Decimal(0),
                )
                delta = abs(sub_val - child_sum)
                signed_diff = sub_val - child_sum
                n_children = len(valid_children)
                tolerance = max(Decimal(n_children), Decimal(1)) * tolerance_unit
                passes = delta <= tolerance

                suggestion_dict: Optional[dict] = None
                if not passes:
                    suggestion = _find_suspect_child(valid_children, signed_diff, period)
                    if suggestion:
                        suggestion_dict = suggestion.to_dict()

                checks.append(
                    ArithmeticCheck(
                        rule_name=self.name,
                        row_id=sub.id,
                        row_label=sub.label_normalized,
                        period=period,
                        expected=child_sum,
                        actual=sub_val,
                        delta=delta,
                        tolerance=tolerance,
                        passes=passes,
                        child_row_ids=[c.id for c in valid_children],
                        suggestion=suggestion_dict,
                    )
                )

        return checks

    def _get_contributing_children(
        self,
        sub: FinancialRow,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult],
    ) -> list[FinancialRow]:
        """
        Identify leaf items or sub-subtotals that contribute to this subtotal.
        """
        # 0. Explicit formula_of on row (1-based indices)
        if sub.formula_of:
            valid = [
                table.rows[idx - 1]
                for idx in sub.formula_of
                if 1 <= idx <= len(table.rows)
            ]
            if valid:
                return valid

        sub_idx = next((i for i, r in enumerate(table.rows) if r.id == sub.id), None)
        if sub_idx is None:
            return []

        # 0b. Hierarchy formula_components (0-based indices)
        if hierarchy and sub_idx < len(hierarchy.nodes):
            node = hierarchy.nodes[sub_idx]
            if node.formula_components:
                valid = [
                    table.rows[idx]
                    for idx in node.formula_components
                    if 0 <= idx < len(table.rows)
                ]
                if valid:
                    return valid

        # 1. Direct children via parent_id if explicitly wired
        direct = [r for r in table.rows if r.parent_id == sub.id and r.id != sub.id]
        if direct and not any(r.row_type == RowRole.SUBTOTAL for r in direct):
            return direct

        # Case A: Grand total like 資産合計 -> sum of prior section subtotals
        if sub.label_normalized in ("資産合計", "負債純資産合計", "負債及び純資産合計", "資産の部合計"):
            section_subtotals = [
                r for r in table.rows[:sub_idx]
                if r.row_type == RowRole.SUBTOTAL and r.label_normalized in ("流動資産合計", "固定資産合計", "流動負債合計", "固定負債合計", "純資産合計")
            ]
            if section_subtotals:
                return section_subtotals

        # Case B: Section subtotal like 固定資産合計 -> sum of sub-section subtotals
        if sub.label_normalized == "固定資産合計":
            sub_subtotals = [
                r for r in table.rows[:sub_idx]
                if r.label_normalized in ("有形固定資産合計", "無形固定資産合計", "投資その他の資産合計")
            ]
            if sub_subtotals:
                return sub_subtotals

        # Case C: Section items (e.g. 流動資産, 有形固定資産, 無形固定資産, 投資その他の資産)
        start_idx = 0
        for i in range(sub_idx - 1, -1, -1):
            if table.rows[i].row_type in (RowRole.SUBTOTAL, RowRole.TOTAL):
                start_idx = i + 1
                break
            if table.rows[i].row_type == RowRole.HEADER and not table.rows[i].values:
                start_idx = i + 1
                break

        section_rows = table.rows[start_idx:sub_idx]

        # In tangible assets, net rows (純額) represent the actual net value.
        # Exclude their corresponding gross and depreciation rows from the section sum.
        junmei_indices = {i for i, r in enumerate(section_rows) if "純額" in r.label_normalized}
        excluded_indices: set[int] = set()
        for j_idx in junmei_indices:
            if j_idx >= 1 and "減価償却" in section_rows[j_idx - 1].label_normalized:
                excluded_indices.add(j_idx - 1)  # exclude depreciation
                if j_idx >= 2:
                    excluded_indices.add(j_idx - 2)  # exclude gross

        collected = [
            r for i, r in enumerate(section_rows)
            if i not in excluded_indices and r.row_type in (RowRole.ITEM, RowRole.NET)
        ]
        return collected


# ---------------------------------------------------------------------------
# Rule 2: NetRule (純額 formula relation)
# ---------------------------------------------------------------------------


class NetRule:
    """
    Validates formula-type relations where net amount is gross plus accumulated depreciation.

    Formula:
        net == gross + depreciation (where depreciation is negative)
    Tolerance:
        tolerance = max(2, 1) * tolerance_unit
    """

    name: str = "net_rule"

    def apply(
        self,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult] = None,
        tolerance_unit: Decimal = DEFAULT_TOLERANCE_UNIT,
    ) -> list[ArithmeticCheck]:
        checks: list[ArithmeticCheck] = []

        for idx, row in enumerate(table.rows):
            is_junmei = "純額" in row.label_normalized
            if not is_junmei:
                continue

            gross_row, dep_row = self._find_gross_and_dep(idx, table, hierarchy)
            if not gross_row or not dep_row:
                continue

            for period, net_val in row.values.items():
                if net_val is None:
                    continue

                gross_val = gross_row.values.get(period)
                dep_val = dep_row.values.get(period)

                if gross_val is None or dep_val is None:
                    continue

                # In filings, accumulated depreciation is negative (△ / ▲).
                # Net rule: net == gross + accumulated depreciation (negative)
                expected_net = gross_val + dep_val
                delta = abs(net_val - expected_net)
                signed_diff = net_val - expected_net
                tolerance = Decimal("2") * tolerance_unit  # 2 components: gross + dep
                passes = delta <= tolerance

                suggestion_dict: Optional[dict] = None
                if not passes:
                    suggestion = _find_suspect_child(
                        [gross_row, dep_row], signed_diff, period
                    )
                    if suggestion:
                        suggestion_dict = suggestion.to_dict()

                checks.append(
                    ArithmeticCheck(
                        rule_name=self.name,
                        row_id=row.id,
                        row_label=row.label_normalized,
                        period=period,
                        expected=expected_net,
                        actual=net_val,
                        delta=delta,
                        tolerance=tolerance,
                        passes=passes,
                        child_row_ids=[gross_row.id, dep_row.id],
                        suggestion=suggestion_dict,
                    )
                )

        return checks

    def _find_gross_and_dep(
        self,
        net_idx: int,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult],
    ) -> tuple[Optional[FinancialRow], Optional[FinancialRow]]:
        """
        Locate the matching gross item and depreciation sibling for a net row.
        """
        net_row = table.rows[net_idx]

        # 0. Use net_row.formula_of if available (1-based indices)
        if len(net_row.formula_of) >= 2:
            g_idx = net_row.formula_of[0] - 1
            d_idx = net_row.formula_of[1] - 1
            if 0 <= g_idx < len(table.rows) and 0 <= d_idx < len(table.rows):
                return table.rows[g_idx], table.rows[d_idx]

        # 1. Use hierarchy formula_components if available
        if hierarchy and net_idx < len(hierarchy.nodes):
            node = hierarchy.nodes[net_idx]
            if len(node.formula_components) >= 2:
                g_idx = node.formula_components[0]
                d_idx = node.formula_components[1]
                if g_idx < len(table.rows) and d_idx < len(table.rows):
                    return table.rows[g_idx], table.rows[d_idx]

        # 2. Directly preceding rows (gross -> dep -> net)
        if net_idx >= 2 and "減価償却" in table.rows[net_idx - 1].label_normalized:
            return table.rows[net_idx - 2], table.rows[net_idx - 1]

        # 3. Sequential backward search fallback
        dep_row: Optional[FinancialRow] = None
        gross_row: Optional[FinancialRow] = None

        for r in reversed(table.rows[:net_idx]):
            if "減価償却" in r.label_normalized and dep_row is None:
                dep_row = r
                continue
            if dep_row is not None and gross_row is None:
                gross_row = r
                break

        return gross_row, dep_row


# ---------------------------------------------------------------------------
# Rule 3: BalanceSheetIdentityRule
# ---------------------------------------------------------------------------


class BalanceSheetIdentityRule:
    """
    Validates the fundamental balance sheet equation:
        資産合計 == 負債純資産合計 (Total Assets == Total Liabilities + Equity)
    when both sides are present.
    """

    name: str = "balance_sheet_identity"

    _ASSETS_TOTAL_NAMES = frozenset(["資産合計", "資産の部合計"])
    _LIABILITIES_EQUITY_TOTAL_NAMES = frozenset([
        "負債純資産合計",
        "負債及び純資産合計",
        "負債・純資産合計",
        "負債及び純資産の部合計",
    ])

    def apply(
        self,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult] = None,
        tolerance_unit: Decimal = DEFAULT_TOLERANCE_UNIT,
    ) -> list[ArithmeticCheck]:
        checks: list[ArithmeticCheck] = []

        assets_row = next(
            (r for r in table.rows if r.label_normalized in self._ASSETS_TOTAL_NAMES),
            None,
        )
        liab_equity_row = next(
            (r for r in table.rows if r.label_normalized in self._LIABILITIES_EQUITY_TOTAL_NAMES),
            None,
        )

        if not assets_row or not liab_equity_row:
            # Only one page / side present; identity cannot be checked
            return []

        for period, a_val in assets_row.values.items():
            if a_val is None:
                continue
            le_val = liab_equity_row.values.get(period)
            if le_val is None:
                continue

            delta = abs(a_val - le_val)
            tolerance = Decimal("1") * tolerance_unit
            passes = delta <= tolerance

            checks.append(
                ArithmeticCheck(
                    rule_name=self.name,
                    row_id=assets_row.id,
                    row_label="資産合計 vs 負債純資産合計",
                    period=period,
                    expected=le_val,
                    actual=a_val,
                    delta=delta,
                    tolerance=tolerance,
                    passes=passes,
                    child_row_ids=[liab_equity_row.id],
                )
            )

        return checks


# ---------------------------------------------------------------------------
# Validation Engine
# ---------------------------------------------------------------------------


class ValidationEngine:
    """
    Orchestrates validation rules, collects diagnostic reports,
    and applies confidence penalties.
    """

    def __init__(self, rules: Optional[list[ValidationRule]] = None):
        self.rules: list[ValidationRule] = rules or [
            SubtotalRule(),
            NetRule(),
            BalanceSheetIdentityRule(),
        ]

    def validate(
        self,
        table: FinancialTable,
        hierarchy: Optional[HierarchyResult] = None,
        tolerance_unit: Decimal = DEFAULT_TOLERANCE_UNIT,
    ) -> ValidationReport:
        """
        Run all validation rules on *table*.

        Updates row confidence scores and attaches flags in-place.
        Returns a comprehensive ValidationReport.
        """
        report = ValidationReport()
        all_checks: list[ArithmeticCheck] = []
        failures: list[dict] = []
        suggestions: list[dict] = []

        # 1. Run all rules
        for rule in self.rules:
            checks = rule.apply(table, hierarchy=hierarchy, tolerance_unit=tolerance_unit)
            all_checks.extend(checks)

        report.arithmetic_checks = all_checks

        # 2. Check balance sheet identity status
        bs_checks = [c for c in all_checks if c.rule_name == BalanceSheetIdentityRule.name]
        if bs_checks:
            report.balance_sheet_balanced = all(c.passes for c in bs_checks)
        else:
            report.balance_sheet_balanced = None
            if table.meta.statement_type == StatementType.BALANCE_SHEET:
                report.notes.append(
                    "Balance sheet identity skipped: liabilities & equity side not present."
                )

        # 3. Process failures and adjust row confidence
        row_by_id = {r.id: r for r in table.rows}

        for check in all_checks:
            if check.passes:
                continue

            failure_info = {
                "rule_name": check.rule_name,
                "row_label": check.row_label,
                "row_id": str(check.row_id) if check.row_id else None,
                "period": check.period,
                "expected": float(check.expected) if check.expected is not None else None,
                "actual": float(check.actual) if check.actual is not None else None,
                "delta": float(check.delta) if check.delta is not None else None,
                "tolerance": float(check.tolerance) if check.tolerance is not None else None,
                "suggestion": check.suggestion,
            }
            failures.append(failure_info)

            if check.suggestion:
                suggestions.append(check.suggestion)

            # Penalise failing subtotal row
            if check.row_id and check.row_id in row_by_id:
                sub_row = row_by_id[check.row_id]
                sub_row.confidence = sub_row.confidence.penalise(
                    ValidationFlag.ARITHMETIC_MISMATCH, amount=0.30
                )

            # Penalise suspect child if isolated, or all children lightly
            if check.suggestion and check.suggestion.get("row_id"):
                s_id = UUID(check.suggestion["row_id"])
                if s_id in row_by_id:
                    child_row = row_by_id[s_id]
                    child_row.confidence = child_row.confidence.penalise(
                        ValidationFlag.ARITHMETIC_MISMATCH, amount=0.35
                    )
            else:
                # Distribute light penalty to contributing children
                for c_id in check.child_row_ids:
                    if c_id in row_by_id:
                        row_by_id[c_id].confidence = row_by_id[c_id].confidence.penalise(
                            ValidationFlag.ARITHMETIC_MISMATCH, amount=0.10
                        )

        report.failures = failures
        report.suggestions = suggestions
        table.validation_report = report
        return report


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def validate_table(
    table: FinancialTable,
    hierarchy: Optional[HierarchyResult] = None,
    tolerance_unit: Decimal = DEFAULT_TOLERANCE_UNIT,
) -> ValidationReport:
    """Run full validation on a FinancialTable."""
    engine = ValidationEngine()
    return engine.validate(table, hierarchy=hierarchy, tolerance_unit=tolerance_unit)


# ---------------------------------------------------------------------------
# Helper: Build FinancialTable from Extraction and Normalization
# ---------------------------------------------------------------------------


def build_financial_table(
    extracted,  # ExtractionResult
    hierarchy: Optional[HierarchyResult] = None,
    unit: UnitScale = UnitScale.THOUSAND_YEN,
) -> FinancialTable:
    """
    Construct a validated FinancialTable from an ExtractionResult and optional HierarchyResult.
    """
    from jp_fin_parser.preprocessing.normalize import (
        detect_row_type,
        normalize_label,
        parse_date,
        parse_number,
    )

    # 1. PageMeta
    meta_unit = unit
    if extracted.header.unit_line:
        if "千円" in extracted.header.unit_line:
            meta_unit = UnitScale.THOUSAND_YEN
        elif "百万円" in extracted.header.unit_line:
            meta_unit = UnitScale.MILLION_YEN
        elif "円" in extracted.header.unit_line:
            meta_unit = UnitScale.YEN

    periods: list[Period] = []
    labels = extracted.header.period_labels or ["前連結会計年度", "当連結会計年度"]
    dates = extracted.header.period_dates or []

    for i, lbl in enumerate(labels):
        d_hint = dates[i] if i < len(dates) else None
        p_obj = Period.from_label(lbl, date_hint=d_hint)
        periods.append(p_obj)

    period_names = [p.label for p in periods]
    is_corr = bool(extracted.header.correction_marker)

    title = "連結貸借対照表"
    statement_type = StatementType.BALANCE_SHEET
    if extracted.header.title_lines:
        full_title = " ".join(extracted.header.title_lines)
        if "損益計算書" in full_title:
            statement_type = StatementType.INCOME_STATEMENT
            title = "連結損益計算書"
        elif "キャッシュ・フロー" in full_title:
            statement_type = StatementType.CASH_FLOW
            title = "連結キャッシュ・フロー計算書"

    meta = PageMeta(
        title=title,
        statement_type=statement_type,
        unit=meta_unit,
        periods=periods,
        is_correction=is_corr,
        page_number=extracted.page_number,
    )

    # 2. Build rows
    rows: list[FinancialRow] = []
    h_nodes = hierarchy.nodes if hierarchy else []

    for i, ext_row in enumerate(extracted.table_rows):
        p_label = normalize_label(ext_row.label)
        level = 0
        h_node: Optional[HierarchyNode] = None
        if i < len(h_nodes):
            h_node = h_nodes[i]
            level = h_node.level

        # Values mapping: period -> Decimal or None
        values: dict[str, Optional[Decimal]] = {}
        for p_idx, p_name in enumerate(period_names):
            if p_idx < len(ext_row.values):
                raw_val = ext_row.values[p_idx]
                parsed = parse_number(raw_val, meta_unit)
                values[p_name] = parsed.value
            else:
                values[p_name] = None

        all_nil = bool(values and all(v is None for v in values.values()))

        # Determine row role
        has_vals = any(v is not None for v in values.values())
        role = detect_row_type(p_label.normalized, has_values=has_vals)

        formula_of: list[int] = []
        if h_node and h_node.formula_components:
            formula_of = [c_idx + 1 for c_idx in h_node.formula_components]

        fin_row = FinancialRow(
            label_raw=ext_row.label,
            label_normalized=p_label.normalized,
            level=level,
            parent_id=None,  # will link below
            row_type=role,
            formula_of=formula_of,
            values=values,
            is_nil=all_nil,
        )
        rows.append(fin_row)

    # 3. Link parent_ids using hierarchy if available
    if hierarchy:
        for i, h_node in enumerate(hierarchy.nodes):
            if i < len(rows) and h_node.parent_idx is not None and h_node.parent_idx < len(rows):
                rows[i].parent_id = rows[h_node.parent_idx].id

    table = FinancialTable(meta=meta, rows=rows)
    return table
