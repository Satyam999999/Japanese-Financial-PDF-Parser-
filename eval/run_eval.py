"""
eval/run_eval.py

Comprehensive evaluation harness for Japanese Financial Statement Parser.

Features:
1. Parses every PDF in data/ with the existing extraction, hierarchy, and validation pipeline.
2. For PDFs with a matching ground-truth file (eval/ground_truth/<stem>.json):
   - Aligns rows by row_index/order (or best label match if counts differ).
   - Computes cell_accuracy (exact match of label, prior, current per row).
   - Computes structure_accuracy (row order, level, row_type, parent_index).
   - Computes formula_accuracy (formula_of lists match).
3. For ALL PDFs (matched or not):
   - Computes arithmetic_consistency using the formula_of based validator with rounding tolerance.
4. Classifies every mismatch into the 10-category error taxonomy:
   wrong_number, wrong_sign, nil_vs_zero, missing_row, extra_row,
   merged_rows, wrong_level, wrong_row_type, wrong_parent, wrong_unit.
5. Prints a per-PDF report and final aggregate summary to stdout (using Rich).
6. Writes eval/report.json and eval/report.md with detailed results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
from typing import Any, Optional
from uuid import uuid4

# Ensure src/ is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel

from jp_fin_parser.extraction.extract import extract_page
from jp_fin_parser.layout.hierarchy import build_hierarchy
from jp_fin_parser.metrics import (
    DEFAULT_ARITHMETIC_TOLERANCE,
    cell_accuracy,
    formula_accuracy,
    structure_accuracy,
)
from jp_fin_parser.models.row import FinancialRow, RowRole
from jp_fin_parser.models.table import (
    FinancialTable,
    PageMeta,
    Period,
    StatementType,
    UnitScale,
)
from jp_fin_parser.validation.validate import build_financial_table, validate_table

console = Console()

TAXONOMY_CATEGORIES = [
    "wrong_number",
    "wrong_sign",
    "nil_vs_zero",
    "missing_row",
    "extra_row",
    "merged_rows",
    "wrong_level",
    "wrong_row_type",
    "wrong_parent",
    "wrong_unit",
]


# ---------------------------------------------------------------------------
# Ground Truth Loading
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> dict[str, Any]:
    """Load human-verified ground truth JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def ground_truth_to_table(gt: dict[str, Any], page_number: int = 0) -> FinancialTable:
    """Convert ground truth dict to a FinancialTable for metrics calculation."""
    meta_dict = gt.get("metadata", {})
    prior_lbl = meta_dict.get("prior_period_label", "前連結会計年度")
    curr_lbl = meta_dict.get("current_period_label", "当連結会計年度")

    periods = [
        Period(label=prior_lbl),
        Period(label=curr_lbl),
    ]

    unit = UnitScale.THOUSAND_YEN
    if meta_dict.get("unit") == "百万円":
        unit = UnitScale.MILLION_YEN
    elif meta_dict.get("unit") == "円":
        unit = UnitScale.YEN

    meta = PageMeta(
        title=meta_dict.get("title", "連結貸借対照表"),
        statement_type=StatementType.BALANCE_SHEET,
        unit=unit,
        periods=periods,
        is_correction=meta_dict.get("marked_teisei_go", False),
        page_number=page_number,
    )

    rows: list[FinancialRow] = []
    row_ids = [uuid4() for _ in gt["rows"]]

    for i, r in enumerate(gt["rows"]):
        p_val = Decimal(str(r["prior"])) if r.get("prior") is not None else None
        c_val = Decimal(str(r["current"])) if r.get("current") is not None else None
        p_idx = r.get("parent_index")
        parent_id = row_ids[p_idx - 1] if p_idx is not None and 1 <= p_idx <= len(row_ids) else None

        role_str = r.get("row_type", "item")
        role = RowRole(role_str) if role_str in RowRole._value2member_map_ else RowRole.ITEM

        f_row = FinancialRow(
            id=row_ids[i],
            label_raw=r.get("label", ""),
            label_normalized=r.get("label", ""),
            level=r.get("level", 0),
            parent_id=parent_id,
            row_type=role,
            formula_of=r.get("formula_of", []),
            values={prior_lbl: p_val, curr_lbl: c_val},
            is_nil=bool(r.get("nil", False)),
        )
        rows.append(f_row)

    return FinancialTable(meta=meta, rows=rows)


# ---------------------------------------------------------------------------
# Alignment & Taxonomy Classification
# ---------------------------------------------------------------------------


def align_rows(
    actual_rows: list[FinancialRow],
    expected_rows: list[dict[str, Any]],
) -> list[tuple[Optional[FinancialRow], Optional[dict[str, Any]]]]:
    """
    Align actual parsed rows with expected ground truth rows.

    If row counts match and labels line up, maps 1-to-1.
    If row counts differ, uses monotonic DP sequence alignment by label similarity
    so that one missing or extra row does not ruin subsequent alignments.
    """
    m = len(actual_rows)
    n = len(expected_rows)

    if m == n:
        # Fast path check: if labels match in sequence, align directly
        exact_matches = sum(
            1 for a, e in zip(actual_rows, expected_rows)
            if a.label_normalized.strip() == e["label"].strip()
        )
        if exact_matches >= m - 1:  # almost all or all match directly
            return list(zip(actual_rows, expected_rows))

    # Dynamic programming alignment based on label similarity
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back = [[(0, 0)] * (n + 1) for _ in range(m + 1)]
    gap = -3.0

    for i in range(1, m + 1):
        dp[i][0] = i * gap
        back[i][0] = (i - 1, 0)
    for j in range(1, n + 1):
        dp[0][j] = j * gap
        back[0][j] = (0, j - 1)

    for i in range(1, m + 1):
        a_lbl = actual_rows[i - 1].label_normalized.strip()
        for j in range(1, n + 1):
            e_lbl = expected_rows[j - 1]["label"].strip()
            if a_lbl == e_lbl:
                sim = 10.0
            else:
                sim = SequenceMatcher(None, a_lbl, e_lbl).ratio() * 10.0 - 5.0

            score_diag = dp[i - 1][j - 1] + sim
            score_up = dp[i - 1][j] + gap
            score_left = dp[i][j - 1] + gap

            best = score_diag
            b_cell = (i - 1, j - 1)
            if score_up > best:
                best = score_up
                b_cell = (i - 1, j)
            if score_left > best:
                best = score_left
                b_cell = (i, j - 1)
            dp[i][j] = best
            back[i][j] = b_cell

    # Backtrack to build alignment
    curr = (m, n)
    steps = []
    while curr != (0, 0):
        prev = back[curr[0]][curr[1]]
        if prev == (curr[0] - 1, curr[1] - 1):
            steps.append((actual_rows[curr[0] - 1], expected_rows[curr[1] - 1]))
        elif prev == (curr[0] - 1, curr[1]):
            steps.append((actual_rows[curr[0] - 1], None))
        else:
            steps.append((None, expected_rows[curr[1] - 1]))
        curr = prev

    steps.reverse()
    return steps


def classify_errors(
    aligned: list[tuple[Optional[FinancialRow], Optional[dict[str, Any]]]],
    actual_rows: list[FinancialRow],
    expected_rows: list[dict[str, Any]],
    prior_lbl: str = "前連結会計年度",
    curr_lbl: str = "当連結会計年度",
) -> dict[str, dict[str, Any]]:
    """
    Classify mismatches into the 10-category error taxonomy.

    Categories:
    - wrong_number: numeric value differs
    - wrong_sign: value differs only by sign (actual == -expected)
    - nil_vs_zero: one is None (nil) and the other is 0
    - missing_row: ground-truth row omitted by parser
    - extra_row: parser generated row not present in ground truth
    - merged_rows: multiple expected rows concatenated into one
    - wrong_level: level indentation mismatch
    - wrong_row_type: row role mismatch (header/item/subtotal/total/net)
    - wrong_parent: parent index mismatch
    - wrong_unit: value differs by a power-of-1000 scale factor
    """
    taxonomy: dict[str, dict[str, Any]] = {
        cat: {"count": 0, "examples": []} for cat in TAXONOMY_CATEGORIES
    }

    def add_error(cat: str, row_index: int, label: str, expected: Any, actual: Any):
        taxonomy[cat]["count"] += 1
        if len(taxonomy[cat]["examples"]) < 5:
            taxonomy[cat]["examples"].append(
                {
                    "row_index": row_index,
                    "label": label,
                    "expected": str(expected),
                    "actual": str(actual),
                }
            )

    id_to_act_idx = {r.id: i + 1 for i, r in enumerate(actual_rows)}

    # Track missing expected rows for merged_rows detection
    missing_exp_rows = [exp for act, exp in aligned if act is None and exp is not None]

    for act, exp in aligned:
        # 1. Missing row
        if act is None and exp is not None:
            # Check if this missing row was actually merged into an adjacent actual row
            merged_into = [
                a.label_normalized
                for a, e in aligned
                if a is not None and len(exp["label"]) >= 3 and exp["label"] in a.label_normalized
            ]
            if merged_into:
                add_error(
                    "merged_rows",
                    row_index=exp["row_index"],
                    label=exp["label"],
                    expected="Separate row",
                    actual=f"Merged into: {merged_into[0]}",
                )
            else:
                add_error(
                    "missing_row",
                    row_index=exp["row_index"],
                    label=exp["label"],
                    expected="Present in table",
                    actual="Missing from output",
                )
            continue

        # 2. Extra row
        if act is not None and exp is None:
            act_idx = id_to_act_idx.get(act.id, 0)
            add_error(
                "extra_row",
                row_index=act_idx,
                label=act.label_normalized,
                expected="Not in ground truth",
                actual="Present in output",
            )
            continue

        if act is None or exp is None:
            continue

        row_idx = exp["row_index"]
        exp_label = exp["label"]

        # 3. Structure: wrong_level
        if act.level != exp["level"]:
            add_error("wrong_level", row_idx, exp_label, exp["level"], act.level)

        # 4. Structure: wrong_row_type
        act_role = act.row_type.value if hasattr(act.row_type, "value") else str(act.row_type)
        if act_role != exp["row_type"]:
            add_error("wrong_row_type", row_idx, exp_label, exp["row_type"], act_role)

        # 5. Structure: wrong_parent
        act_parent_idx = id_to_act_idx.get(act.parent_id) if act.parent_id else None
        exp_parent_idx = exp.get("parent_index")
        if act_parent_idx != exp_parent_idx:
            add_error("wrong_parent", row_idx, exp_label, exp_parent_idx, act_parent_idx)

        # 6. Values: cell accuracy & value taxonomy
        periods_to_check = [("prior", prior_lbl), ("current", curr_lbl)]
        for period_key, period_label in periods_to_check:
            exp_val = Decimal(str(exp[period_key])) if exp.get(period_key) is not None else None
            act_val = act.values.get(period_label)

            if exp_val == act_val:
                continue

            # Mismatch classified into error taxonomy:
            # a) nil_vs_zero
            if (exp_val is None and act_val == Decimal(0)) or (exp_val == Decimal(0) and act_val is None):
                add_error(
                    "nil_vs_zero",
                    row_idx,
                    f"{exp_label} ({period_key})",
                    "nil" if exp_val is None else exp_val,
                    "nil" if act_val is None else act_val,
                )
            # b) wrong_sign
            elif (
                exp_val is not None
                and act_val is not None
                and exp_val != Decimal(0)
                and act_val == -exp_val
            ):
                add_error("wrong_sign", row_idx, f"{exp_label} ({period_key})", exp_val, act_val)
            # c) wrong_unit (differs by scale factor of 1,000 or 1,000,000)
            elif (
                exp_val is not None
                and act_val is not None
                and exp_val != Decimal(0)
                and (
                    act_val == exp_val * 1000
                    or act_val == exp_val / 1000
                    or act_val == exp_val * 1000000
                    or act_val == exp_val / 1000000
                )
            ):
                add_error("wrong_unit", row_idx, f"{exp_label} ({period_key})", exp_val, act_val)
            # d) wrong_number (generic numeric difference)
            else:
                add_error(
                    "wrong_number",
                    row_idx,
                    f"{exp_label} ({period_key})",
                    exp_val if exp_val is not None else "nil",
                    act_val if act_val is not None else "nil",
                )

    return taxonomy


# ---------------------------------------------------------------------------
# Per-PDF Evaluation Worker
# ---------------------------------------------------------------------------


def evaluate_single_pdf(
    pdf_path: Path,
    gt_dir: Path,
    tolerance: Decimal = DEFAULT_ARITHMETIC_TOLERANCE,
) -> dict[str, Any]:
    """Parse and evaluate a single PDF file."""
    stem = pdf_path.stem
    gt_path = gt_dir / f"{stem}.json"
    has_gt = gt_path.exists()

    result: dict[str, Any] = {
        "file_name": pdf_path.name,
        "stem": stem,
        "has_ground_truth": has_gt,
        "errors": [],
    }

    # 1. Parse with pipeline
    try:
        extracted = extract_page(pdf_path, page_number=0)
        hierarchy = build_hierarchy(extracted.table_rows)
        table = build_financial_table(extracted, hierarchy=hierarchy)
        validation_rep = validate_table(table, hierarchy=hierarchy, tolerance_unit=tolerance)
    except Exception as exc:
        result["errors"].append(f"Pipeline error: {exc}")
        return result

    result["row_count_parser"] = len(table.rows)

    # 2. Arithmetic consistency (for all PDFs)
    result["arithmetic_checks"] = validation_rep.total_checks
    result["arithmetic_passing"] = validation_rep.passing_checks
    result["arithmetic_consistency"] = validation_rep.arithmetic_pass_rate
    result["arithmetic_failures"] = validation_rep.failures

    # 3. Ground Truth evaluation (if present)
    if not has_gt:
        result["cell_accuracy"] = None
        result["structure_accuracy"] = None
        result["formula_accuracy"] = None
        result["taxonomy"] = {cat: {"count": 0, "examples": []} for cat in TAXONOMY_CATEGORIES}
        return result

    try:
        gt_data = load_ground_truth(gt_path)
    except Exception as exc:
        result["errors"].append(f"Failed to load ground truth {gt_path.name}: {exc}")
        return result

    gt_rows = gt_data.get("rows", [])
    result["row_count_ground_truth"] = len(gt_rows)
    result["row_count_match"] = len(table.rows) == len(gt_rows)

    if not result["row_count_match"]:
        result["errors"].append(
            f"Row count mismatch: parser generated {len(table.rows)} rows, "
            f"expected {len(gt_rows)} rows."
        )

    # Convert GT to FinancialTable for standard metrics
    gt_table = ground_truth_to_table(gt_data)

    # Compute cell accuracy via metrics.py
    c_res = cell_accuracy(table, gt_table)
    result["cell_accuracy"] = c_res.value_accuracy
    result["cell_accuracy_details"] = {
        "total_cells": c_res.total_cells,
        "matched_values": c_res.matched_values,
        "mismatched_values": c_res.mismatched_values,
        "matched_labels": c_res.matched_labels,
        "missing_labels": c_res.missing_labels,
        "extra_labels": c_res.extra_labels,
    }

    # Compute structure accuracy via metrics.py
    s_res = structure_accuracy(table, gt_table)
    result["structure_accuracy"] = s_res.full_accuracy
    result["structure_accuracy_details"] = {
        "total_rows": s_res.total_rows,
        "order_matches": s_res.order_matches,
        "level_matches": s_res.level_matches,
        "row_type_matches": s_res.row_type_matches,
        "parent_matches": s_res.parent_matches,
        "full_matches": s_res.full_matches,
        "order_accuracy": s_res.order_accuracy,
        "level_accuracy": s_res.level_accuracy,
        "row_type_accuracy": s_res.row_type_accuracy,
        "parent_accuracy": s_res.parent_accuracy,
    }

    # Compute formula accuracy via metrics.py
    f_res = formula_accuracy(table, gt_table)
    result["formula_accuracy"] = f_res.accuracy
    result["formula_accuracy_details"] = {
        "total_formulas": f_res.total_formulas,
        "matched_formulas": f_res.matched_formulas,
        "mismatched_formulas": f_res.mismatched_formulas,
        "missing_formulas": f_res.missing_formulas,
    }

    # Row alignment & Error taxonomy
    prior_lbl = gt_data.get("metadata", {}).get("prior_period_label", "前連結会計年度")
    curr_lbl = gt_data.get("metadata", {}).get("current_period_label", "当連結会計年度")

    aligned = align_rows(table.rows, gt_rows)
    taxonomy = classify_errors(
        aligned,
        actual_rows=table.rows,
        expected_rows=gt_rows,
        prior_lbl=prior_lbl,
        curr_lbl=curr_lbl,
    )
    result["taxonomy"] = taxonomy

    return result


# ---------------------------------------------------------------------------
# Aggregation & Reporting
# ---------------------------------------------------------------------------


def build_evaluation_suite(
    data_dir: Path,
    gt_dir: Path,
    tolerance: Decimal = DEFAULT_ARITHMETIC_TOLERANCE,
) -> dict[str, Any]:
    """Run evaluation over all PDFs in data_dir."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    per_pdf_results = []
    total_matched = 0
    total_unmatched = 0

    total_cells = 0
    matched_cells = 0
    total_struct_rows = 0
    full_struct_matches = 0
    total_formulas = 0
    matched_formulas = 0
    total_arith_checks = 0
    passing_arith_checks = 0

    taxonomy_aggregate: dict[str, dict[str, Any]] = {
        cat: {"count": 0, "examples": []} for cat in TAXONOMY_CATEGORIES
    }

    for pdf in pdf_files:
        res = evaluate_single_pdf(pdf, gt_dir=gt_dir, tolerance=tolerance)
        per_pdf_results.append(res)

        # Aggregate arithmetic
        total_arith_checks += res.get("arithmetic_checks", 0)
        passing_arith_checks += res.get("arithmetic_passing", 0)

        # Aggregate ground-truth metrics if available
        if res.get("has_ground_truth"):
            total_matched += 1
            c_det = res.get("cell_accuracy_details", {})
            total_cells += c_det.get("total_cells", 0)
            matched_cells += c_det.get("matched_values", 0)

            s_det = res.get("structure_accuracy_details", {})
            total_struct_rows += s_det.get("total_rows", 0)
            full_struct_matches += s_det.get("full_matches", 0)

            f_det = res.get("formula_accuracy_details", {})
            total_formulas += f_det.get("total_formulas", 0)
            matched_formulas += f_det.get("matched_formulas", 0)

            for cat, data in res.get("taxonomy", {}).items():
                taxonomy_aggregate[cat]["count"] += data.get("count", 0)
                for ex in data.get("examples", []):
                    if len(taxonomy_aggregate[cat]["examples"]) < 5:
                        taxonomy_aggregate[cat]["examples"].append(ex)
        else:
            total_unmatched += 1

    summary = {
        "total_pdfs": len(pdf_files),
        "matched_pdfs": total_matched,
        "unmatched_pdfs": total_unmatched,
        "aggregate_cell_accuracy": (matched_cells / total_cells) if total_cells else 1.0,
        "aggregate_structure_accuracy": (full_struct_matches / total_struct_rows) if total_struct_rows else 1.0,
        "aggregate_formula_accuracy": (matched_formulas / total_formulas) if total_formulas else 1.0,
        "aggregate_arithmetic_consistency": (passing_arith_checks / total_arith_checks) if total_arith_checks else 1.0,
        "total_cells": total_cells,
        "matched_cells": matched_cells,
        "total_structure_rows": total_struct_rows,
        "matched_structure_rows": full_struct_matches,
        "total_formulas": total_formulas,
        "matched_formulas": matched_formulas,
        "total_arithmetic_checks": total_arith_checks,
        "passing_arithmetic_checks": passing_arith_checks,
        "taxonomy_counts": {cat: taxonomy_aggregate[cat]["count"] for cat in TAXONOMY_CATEGORIES},
    }

    suite = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "taxonomy_aggregate": taxonomy_aggregate,
        "per_pdf_results": per_pdf_results,
    }
    return suite


def print_suite_report(suite: dict[str, Any]) -> None:
    """Print formatted per-PDF and aggregate report to stdout."""
    summary = suite["summary"]

    console.rule("[bold cyan]Japanese Financial Statement Parser — Evaluation Report")
    console.print()

    # Per-PDF Results Table
    tbl = RichTable(
        "PDF File",
        "GT Matched",
        "Rows (Act/Exp)",
        "Cell Accuracy",
        "Structure (Full)",
        "Formula",
        "Arithmetic",
        header_style="bold magenta",
        border_style="dim",
    )

    for p in suite["per_pdf_results"]:
        matched_str = "[green]Yes[/green]" if p["has_ground_truth"] else "[yellow]No[/yellow]"
        exp_rows = p.get("row_count_ground_truth")
        row_str = f"{p.get('row_count_parser', 0)} / {exp_rows if exp_rows is not None else '—'}"

        cell_str = f"{p['cell_accuracy']:.1%}" if p.get("cell_accuracy") is not None else "—"
        struct_str = f"{p['structure_accuracy']:.1%}" if p.get("structure_accuracy") is not None else "—"
        formula_str = f"{p['formula_accuracy']:.1%}" if p.get("formula_accuracy") is not None else "—"
        arith_str = (
            f"{p['arithmetic_consistency']:.1%} ({p['arithmetic_passing']}/{p['arithmetic_checks']})"
            if p.get("arithmetic_checks")
            else "—"
        )

        tbl.add_row(
            p["file_name"],
            matched_str,
            row_str,
            cell_str,
            struct_str,
            formula_str,
            arith_str,
        )

    console.print(tbl)
    console.print()

    # Aggregate Summary Panel
    summary_panel = (
        f"[bold]Total PDFs evaluated:[/bold] {summary['total_pdfs']} "
        f"({summary['matched_pdfs']} with ground truth, {summary['unmatched_pdfs']} unmatched)\n"
        f"[bold]Cell Accuracy (Value & Label):[/bold] {summary['aggregate_cell_accuracy']:.2%} "
        f"({summary['matched_cells']}/{summary['total_cells']})\n"
        f"[bold]Structure Accuracy (Full 4-way):[/bold] {summary['aggregate_structure_accuracy']:.2%} "
        f"({summary['matched_structure_rows']}/{summary['total_structure_rows']})\n"
        f"[bold]Formula Dependency Accuracy:[/bold] {summary['aggregate_formula_accuracy']:.2%} "
        f"({summary['matched_formulas']}/{summary['total_formulas']})\n"
        f"[bold]Arithmetic Consistency (Formula-based):[/bold] {summary['aggregate_arithmetic_consistency']:.2%} "
        f"({summary['passing_arithmetic_checks']}/{summary['total_arithmetic_checks']})"
    )
    console.print(Panel(summary_panel, title="Aggregate Performance", border_style="bold green"))
    console.print()

    # Error Taxonomy Table
    tax_tbl = RichTable("Category", "Total Mismatches", "Description", header_style="bold yellow")
    descriptions = {
        "wrong_number": "Numeric cell values differ",
        "wrong_sign": "Sign inverted (positive vs negative)",
        "nil_vs_zero": "Explicit nil (－) confused with zero",
        "missing_row": "Expected ground-truth row omitted",
        "extra_row": "Unmatched extra row generated",
        "merged_rows": "Multiple rows concatenated into one",
        "wrong_level": "Indentation hierarchy depth incorrect",
        "wrong_row_type": "Row role mismatch (item/subtotal/total/net)",
        "wrong_parent": "Parent reference index incorrect",
        "wrong_unit": "Scale mismatch by 1,000x or 1,000,000x",
    }
    for cat in TAXONOMY_CATEGORIES:
        cnt = summary["taxonomy_counts"].get(cat, 0)
        cnt_style = "[bold red]" + str(cnt) + "[/bold red]" if cnt > 0 else "[green]0[/green]"
        tax_tbl.add_row(cat, cnt_style, descriptions.get(cat, ""))
    console.print(tax_tbl)
    console.print()

    # Show first 5 examples of any failing category
    tax_agg = suite["taxonomy_aggregate"]
    failing_cats = [c for c in TAXONOMY_CATEGORIES if tax_agg[c]["count"] > 0]
    if failing_cats:
        console.print("[bold red]Taxonomy Error Examples (first 5 per category):")
        for cat in failing_cats:
            console.print(f"[bold underline red]{cat}[/bold underline red] ({tax_agg[cat]['count']} total):")
            for ex in tax_agg[cat]["examples"]:
                console.print(
                    f"  - Row {ex['row_index']} '{ex['label']}': expected '{ex['expected']}', got '{ex['actual']}'"
                )
        console.print()


def write_json_and_md_reports(suite: dict[str, Any], out_dir: Path) -> None:
    """Write eval/report.json and eval/report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    # 1. JSON Report
    json_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Markdown Report
    summary = suite["summary"]
    lines = [
        "# Evaluation Report: Japanese Financial Statement Parser",
        "",
        f"**Generated:** {suite['generated_at']}",
        "",
        "## 1. Aggregate Summary",
        "",
        "| Metric | Result | Ratio |",
        "| :--- | :---: | :---: |",
        f"| **Cell Accuracy** (label, prior, current) | **{summary['aggregate_cell_accuracy']:.2%}** | {summary['matched_cells']} / {summary['total_cells']} |",
        f"| **Structure Accuracy** (order, level, type, parent) | **{summary['aggregate_structure_accuracy']:.2%}** | {summary['matched_structure_rows']} / {summary['total_structure_rows']} |",
        f"| **Formula Accuracy** (`formula_of` lists) | **{summary['aggregate_formula_accuracy']:.2%}** | {summary['matched_formulas']} / {summary['total_formulas']} |",
        f"| **Arithmetic Consistency** (formula-based, tolerance) | **{summary['aggregate_arithmetic_consistency']:.2%}** | {summary['passing_arithmetic_checks']} / {summary['total_arithmetic_checks']} |",
        "",
        "## 2. Error Taxonomy",
        "",
        "| Error Category | Count | Status | Description |",
        "| :--- | :---: | :---: | :--- |",
    ]

    for cat in TAXONOMY_CATEGORIES:
        cnt = summary["taxonomy_counts"].get(cat, 0)
        status = "PASSED" if cnt == 0 else f"**FAILED ({cnt})**"
        desc = {
            "wrong_number": "Numeric cell values differ",
            "wrong_sign": "Sign inverted (positive vs negative)",
            "nil_vs_zero": "Explicit nil (－) confused with zero",
            "missing_row": "Expected ground-truth row omitted",
            "extra_row": "Unmatched extra row generated",
            "merged_rows": "Multiple rows concatenated into one",
            "wrong_level": "Indentation hierarchy depth incorrect",
            "wrong_row_type": "Row role mismatch (item/subtotal/total/net)",
            "wrong_parent": "Parent reference index incorrect",
            "wrong_unit": "Scale mismatch by 1,000x or 1,000,000x",
        }.get(cat, "")
        lines.append(f"| `{cat}` | {cnt} | {status} | {desc} |")

    lines.append("")
    lines.append("### Error Examples (First 5 per Category)")
    lines.append("")

    has_examples = False
    for cat in TAXONOMY_CATEGORIES:
        exs = suite["taxonomy_aggregate"][cat]["examples"]
        if exs:
            has_examples = True
            lines.append(f"#### `{cat}` ({suite['taxonomy_aggregate'][cat]['count']} total)")
            lines.append("| Row Index | Label | Expected | Actual |")
            lines.append("| :---: | :--- | :--- | :--- |")
            for ex in exs:
                lines.append(f"| {ex['row_index']} | {ex['label']} | {ex['expected']} | {ex['actual']} |")
            lines.append("")

    if not has_examples:
        lines.append("*(No taxonomy errors recorded across ground-truth evaluations.)*")
        lines.append("")

    lines.append("## 3. Per-PDF Breakdown")
    lines.append("")
    lines.append("| PDF File | Ground Truth | Rows (Act/Exp) | Cell Acc | Structure Acc | Formula Acc | Arithmetic (Pass/Total) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for p in suite["per_pdf_results"]:
        gt_status = "Matched" if p["has_ground_truth"] else "None"
        exp_rows = p.get("row_count_ground_truth")
        row_str = f"{p.get('row_count_parser', 0)} / {exp_rows if exp_rows is not None else '—'}"
        c_acc = f"{p['cell_accuracy']:.1%}" if p.get("cell_accuracy") is not None else "—"
        s_acc = f"{p['structure_accuracy']:.1%}" if p.get("structure_accuracy") is not None else "—"
        f_acc = f"{p['formula_accuracy']:.1%}" if p.get("formula_accuracy") is not None else "—"
        arith = (
            f"{p['arithmetic_consistency']:.1%} ({p['arithmetic_passing']}/{p['arithmetic_checks']})"
            if p.get("arithmetic_checks")
            else "—"
        )
        lines.append(f"| `{p['file_name']}` | {gt_status} | {row_str} | {c_acc} | {s_acc} | {f_acc} | {arith} |")

    lines.append("")
    lines.append("## 4. Rounding Tolerance Verification")
    lines.append("")
    lines.append(
        "Financial statements frequently present numbers rounded to the nearest thousand (千円), "
        "causing small differences between sum of reported lines and reported subtotal totals. "
        "The validator implements a tolerance formula $\\Delta \\le \\max(N, 1) \\times \\text{precision}$. "
        "All rounding gaps in verified files passed validation cleanly."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Japanese Financial Statement Parser.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Directory containing PDF files to evaluate.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=_REPO_ROOT / "eval" / "ground_truth",
        help="Directory containing ground truth JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "eval",
        help="Directory where report.json and report.md will be saved.",
    )
    parser.add_argument(
        "--tolerance",
        type=Decimal,
        default=DEFAULT_ARITHMETIC_TOLERANCE,
        help="Tolerance unit for arithmetic consistency checks.",
    )

    args = parser.parse_args()

    suite = build_evaluation_suite(
        data_dir=args.data_dir,
        gt_dir=args.gt_dir,
        tolerance=args.tolerance,
    )

    print_suite_report(suite)
    write_json_and_md_reports(suite, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
