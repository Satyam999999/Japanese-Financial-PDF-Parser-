"""
jp_fin_parser/metrics.py

Three independent accuracy metrics for the Japanese financial PDF parser.

Metrics
-------
1. cell_accuracy         – exact-match fraction over (label, value) cells vs ground truth
2. structure_accuracy    – exact-match fraction over (row_order, level, row_type) tuples
3. arithmetic_consistency – fraction of subtotal/total rows whose values reconcile

All metrics operate on ``FinancialTable`` objects so they can be used both
in offline evaluation (vs ground-truth JSON) and as live pipeline health checks.

Usage
-----
    from jp_fin_parser.metrics import MetricsReport, evaluate_table, report

    actual   = FinancialTable(...)   # from pipeline
    expected = FinancialTable(...)   # from ground_truth JSON
    result   = evaluate_table(actual, expected)
    report([result])                 # pretty-print to stdout
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence

from rich.console import Console
from rich.table import Table as RichTable

from jp_fin_parser.models.row import FinancialRow, RowRole
from jp_fin_parser.models.table import FinancialTable

_console = Console()

# ---------------------------------------------------------------------------
# Tolerance for arithmetic checks
# ---------------------------------------------------------------------------
DEFAULT_ARITHMETIC_TOLERANCE: Decimal = Decimal("1")  # 1 unit (千円 or 百万円)


# ---------------------------------------------------------------------------
# 1. Cell accuracy
# ---------------------------------------------------------------------------


@dataclass
class CellAccuracyResult:
    """Breakdown of exact-match cell hits and misses."""

    total_cells: int = 0          # expected label×period combos
    matched_labels: int = 0       # labels that exist in actual
    matched_values: int = 0       # (label, period, value) exact matches
    mismatched_values: int = 0    # label found but value differs
    missing_labels: int = 0       # label absent from actual
    extra_labels: int = 0         # labels in actual not in expected

    @property
    def label_recall(self) -> float:
        if self.total_cells == 0:
            return 1.0
        return self.matched_labels / self.total_cells

    @property
    def value_accuracy(self) -> float:
        """Fraction of expected cells where both label AND value match exactly."""
        denom = self.total_cells
        return self.matched_values / denom if denom else 1.0


def cell_accuracy(
    actual: FinancialTable,
    expected: FinancialTable,
) -> CellAccuracyResult:
    """
    Compare every (label_normalized, period, value) triple in *expected*
    against *actual*.

    Label matching is done on ``label_normalized``. In cases of repeated labels
    (e.g. 減価償却累計額 across assets), rows are matched in document FIFO order.
    Value matching is exact Decimal equality; None (nil) == None.
    """
    result = CellAccuracyResult()

    act_rows_by_lbl: dict[str, list[FinancialRow]] = {}
    for r in actual.rows:
        act_rows_by_lbl.setdefault(r.label_normalized.strip(), []).append(r)

    exp_counts: dict[str, int] = {}
    for r in expected.rows:
        exp_counts[r.label_normalized.strip()] = exp_counts.get(r.label_normalized.strip(), 0) + 1

    # Extra labels in actual
    act_counts: dict[str, int] = {}
    for r in actual.rows:
        act_counts[r.label_normalized.strip()] = act_counts.get(r.label_normalized.strip(), 0) + 1

    extra = 0
    for lbl, count in act_counts.items():
        exp_c = exp_counts.get(lbl, 0)
        if count > exp_c:
            extra += (count - exp_c)
    result.extra_labels = extra

    for exp_row in expected.rows:
        key = exp_row.label_normalized.strip()
        queue = act_rows_by_lbl.get(key)
        act_row = queue.pop(0) if queue else None

        for period, exp_val in exp_row.values.items():
            result.total_cells += 1
            if act_row is None:
                result.missing_labels += 1
                continue

            result.matched_labels += 1
            act_val = act_row.values.get(period)

            if act_val == exp_val:
                result.matched_values += 1
            else:
                result.mismatched_values += 1

    return result


# ---------------------------------------------------------------------------
# 2. Structure accuracy
# ---------------------------------------------------------------------------


@dataclass
class StructureAccuracyResult:
    """Comparison of row order, level, row_type, and parent_index."""

    total_rows: int = 0
    order_matches: int = 0       # same relative position
    level_matches: int = 0       # level depth correct
    row_type_matches: int = 0    # RowRole correct
    parent_matches: int = 0      # parent_index matches
    full_matches: int = 0        # all four correct

    @property
    def order_accuracy(self) -> float:
        return self.order_matches / self.total_rows if self.total_rows else 1.0

    @property
    def level_accuracy(self) -> float:
        return self.level_matches / self.total_rows if self.total_rows else 1.0

    @property
    def row_type_accuracy(self) -> float:
        return self.row_type_matches / self.total_rows if self.total_rows else 1.0

    @property
    def parent_accuracy(self) -> float:
        return self.parent_matches / self.total_rows if self.total_rows else 1.0

    @property
    def full_accuracy(self) -> float:
        """All structural properties (order, level, row_type, parent) correct for the row."""
        return self.full_matches / self.total_rows if self.total_rows else 1.0


def structure_accuracy(
    actual: FinancialTable,
    expected: FinancialTable,
) -> StructureAccuracyResult:
    """
    Compares structural properties for rows that can be aligned by label.

    Order is judged by the rank (0-based index) of a row in the
    expected list vs the actual list; a match means the rank is identical.
    Level, row_type, and parent_index are verified.
    """
    result = StructureAccuracyResult()

    act_indices: dict[str, list[int]] = {}
    act_rows_by_lbl: dict[str, list[FinancialRow]] = {}
    for i, r in enumerate(actual.rows):
        k = r.label_normalized.strip()
        act_indices.setdefault(k, []).append(i)
        act_rows_by_lbl.setdefault(k, []).append(r)

    exp_id_to_idx = {r.id: i + 1 for i, r in enumerate(expected.rows)}
    act_id_to_idx = {r.id: i + 1 for i, r in enumerate(actual.rows)}

    for exp_rank, exp_row in enumerate(expected.rows):
        key = exp_row.label_normalized.strip()
        idx_queue = act_indices.get(key)
        row_queue = act_rows_by_lbl.get(key)

        if not idx_queue or not row_queue:
            # Missing row: all sub-metrics fail for this row
            result.total_rows += 1
            continue

        result.total_rows += 1
        act_rank = idx_queue.pop(0)
        act_row = row_queue.pop(0)

        order_ok = act_rank == exp_rank
        level_ok = act_row.level == exp_row.level
        type_ok = act_row.row_type == exp_row.row_type

        exp_parent_idx = exp_id_to_idx.get(exp_row.parent_id) if exp_row.parent_id else None
        act_parent_idx = act_id_to_idx.get(act_row.parent_id) if act_row.parent_id else None
        parent_ok = act_parent_idx == exp_parent_idx

        if order_ok:
            result.order_matches += 1
        if level_ok:
            result.level_matches += 1
        if type_ok:
            result.row_type_matches += 1
        if parent_ok:
            result.parent_matches += 1
        if order_ok and level_ok and type_ok and parent_ok:
            result.full_matches += 1

    return result


# ---------------------------------------------------------------------------
# 3. Formula accuracy
# ---------------------------------------------------------------------------


@dataclass
class FormulaAccuracyResult:
    """Comparison of formula_of dependency lists against ground truth."""

    total_formulas: int = 0
    matched_formulas: int = 0
    mismatched_formulas: int = 0
    missing_formulas: int = 0

    @property
    def accuracy(self) -> float:
        return self.matched_formulas / self.total_formulas if self.total_formulas else 1.0


def formula_accuracy(
    actual: FinancialTable,
    expected: FinancialTable,
) -> FormulaAccuracyResult:
    """
    Compare formula_of lists for rows in *expected* against corresponding rows in *actual*.
    """
    result = FormulaAccuracyResult()
    act_rows_by_lbl: dict[str, list[FinancialRow]] = {}
    for r in actual.rows:
        act_rows_by_lbl.setdefault(r.label_normalized.strip(), []).append(r)

    for exp_row in expected.rows:
        if not exp_row.formula_of:
            continue
        result.total_formulas += 1
        key = exp_row.label_normalized.strip()
        queue = act_rows_by_lbl.get(key)
        act_row = queue.pop(0) if queue else None

        if act_row is None:
            result.missing_formulas += 1
            continue
        if act_row.formula_of == exp_row.formula_of:
            result.matched_formulas += 1
        else:
            result.mismatched_formulas += 1

    return result


# ---------------------------------------------------------------------------
# 3. Arithmetic consistency
# ---------------------------------------------------------------------------


@dataclass
class ArithmeticConsistencyResult:
    """Fraction of subtotal/total rows whose children sum correctly."""

    total_checks: int = 0
    passing_checks: int = 0
    tolerance: Decimal = DEFAULT_ARITHMETIC_TOLERANCE
    failures: list[dict] = field(default_factory=list)

    @property
    def consistency_rate(self) -> float:
        return self.passing_checks / self.total_checks if self.total_checks else 1.0


def _find_children(
    rows: list[FinancialRow], parent: FinancialRow
) -> list[FinancialRow]:
    """
    Return direct children of *parent* (rows whose parent_id == parent.id).
    Falls back to a positional heuristic if parent_ids are not set:
    children are contiguous rows between this subtotal row and the previous
    one at the same or higher level.
    """
    children = [r for r in rows if r.parent_id is not None and r.parent_id == parent.id]
    if children:
        return children

    # Positional fallback: rows between the previous subtotal-like row at
    # the same level and this row, with strictly greater indentation.
    parent_idx = next((i for i, r in enumerate(rows) if r.id == parent.id), None)
    if parent_idx is None:
        return []

    collected: list[FinancialRow] = []
    for r in reversed(rows[:parent_idx]):
        if r.level <= parent.level and r.is_subtotal_like():
            break
        if r.level > parent.level:
            collected.append(r)
    return collected


def arithmetic_consistency(
    actual: FinancialTable,
    tolerance: Decimal = DEFAULT_ARITHMETIC_TOLERANCE,
) -> ArithmeticConsistencyResult:
    """
    For every subtotal/total row in *actual*, sum the values of its children
    and compare against the row's stated value.

    Only periods where both the subtotal and at least one child have a
    non-None value are checked.
    """
    result = ArithmeticConsistencyResult(tolerance=tolerance)

    subtotals = [
        r for r in actual.rows
        if r.row_type in (RowRole.SUBTOTAL, RowRole.TOTAL)
    ]

    for sub in subtotals:
        children = _find_children(actual.rows, sub)
        if not children:
            continue  # can't check without children

        for period, sub_val in sub.values.items():
            if sub_val is None:
                continue
            child_vals = [
                c.values[period]
                for c in children
                if period in c.values and c.values[period] is not None
            ]
            if not child_vals:
                continue

            child_sum: Decimal = sum(child_vals, Decimal(0))  # type: ignore[arg-type]
            delta = abs(sub_val - child_sum)
            passes = delta <= tolerance

            result.total_checks += 1
            if passes:
                result.passing_checks += 1
            else:
                result.failures.append(
                    {
                        "row_label": sub.label_normalized,
                        "period": period,
                        "subtotal_value": float(sub_val),
                        "children_sum": float(child_sum),
                        "delta": float(delta),
                    }
                )

    return result


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------


@dataclass
class MetricsReport:
    """All metric results for one page."""

    page_number: int
    source_file: Optional[str]
    cell: CellAccuracyResult
    structure: StructureAccuracyResult
    arithmetic: ArithmeticConsistencyResult
    formula: Optional[FormulaAccuracyResult] = None


def evaluate_table(
    actual: FinancialTable,
    expected: FinancialTable,
    arithmetic_tolerance: Decimal = DEFAULT_ARITHMETIC_TOLERANCE,
) -> MetricsReport:
    """Compute all metrics and return a ``MetricsReport``."""
    return MetricsReport(
        page_number=actual.meta.page_number,
        source_file=actual.meta.source_file,
        cell=cell_accuracy(actual, expected),
        structure=structure_accuracy(actual, expected),
        arithmetic=arithmetic_consistency(actual, arithmetic_tolerance),
        formula=formula_accuracy(actual, expected),
    )


def report(
    reports: Sequence[MetricsReport],
    *,
    show_failures: bool = True,
) -> None:
    """
    Pretty-print all metric reports plus aggregate totals to stdout.

    Parameters
    ----------
    reports       – one MetricsReport per page evaluated
    show_failures – if True, list arithmetic failures in detail
    """
    _console.rule("[bold cyan]jp-fin-parser Evaluation Report")

    # Per-page table
    tbl = RichTable(
        "Page",
        "Source",
        "Cell Accuracy",
        "Structure (full)",
        "Arithmetic",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )

    total_cells = total_matched = 0
    total_struct = total_struct_full = 0
    total_arith = total_arith_pass = 0

    for r in reports:
        tbl.add_row(
            str(r.page_number),
            r.source_file or "—",
            f"{r.cell.value_accuracy:.1%}  ({r.cell.matched_values}/{r.cell.total_cells})",
            f"{r.structure.full_accuracy:.1%}  ({r.structure.full_matches}/{r.structure.total_rows})",
            f"{r.arithmetic.consistency_rate:.1%}  ({r.arithmetic.passing_checks}/{r.arithmetic.total_checks})",
        )
        total_cells += r.cell.total_cells
        total_matched += r.cell.matched_values
        total_struct += r.structure.total_rows
        total_struct_full += r.structure.full_matches
        total_arith += r.arithmetic.total_checks
        total_arith_pass += r.arithmetic.passing_checks

    _console.print(tbl)

    # Aggregates
    _console.print()
    _console.print("[bold]Aggregated across all pages:")
    _console.print(
        f"  Cell accuracy        : {total_matched/total_cells:.1%}  "
        f"({total_matched}/{total_cells})"
        if total_cells else "  Cell accuracy        : N/A"
    )
    _console.print(
        f"  Structure accuracy   : {total_struct_full/total_struct:.1%}  "
        f"({total_struct_full}/{total_struct})"
        if total_struct else "  Structure accuracy   : N/A"
    )
    _console.print(
        f"  Arithmetic consistency: {total_arith_pass/total_arith:.1%}  "
        f"({total_arith_pass}/{total_arith})"
        if total_arith else "  Arithmetic consistency: N/A"
    )

    # Arithmetic failure details
    if show_failures:
        all_failures = [f for r in reports for f in r.arithmetic.failures]
        if all_failures:
            _console.print()
            _console.print("[bold red]Arithmetic failures:")
            fail_tbl = RichTable(
                "Row label", "Period", "Subtotal", "Children sum", "Delta",
                show_header=True, header_style="bold red", border_style="dim",
            )
            for f in all_failures:
                fail_tbl.add_row(
                    f["row_label"],
                    f["period"],
                    str(f["subtotal_value"]),
                    str(f["children_sum"]),
                    str(f["delta"]),
                )
            _console.print(fail_tbl)
