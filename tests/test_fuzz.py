"""
tests/test_fuzz.py

Fuzz testing the validation rule engine by perturbing extracted cells.

Perturbation types tested one at a time:
1. Drop a minus sign (△ / ▲ / -)
2. Swap two adjacent distinct digits
3. Shift a numeric value by 1000
4. Swap prior and current period values

Assertions:
- Perturbations exceeding rounding tolerance MUST be flagged by the validator.
- Perturbations within rounding tolerance MUST NOT be flagged.
- Reports detection rates for each perturbation type.
"""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import pytest
from rich.console import Console
from rich.table import Table as RichTable

from jp_fin_parser.extraction.extract import ExtractionResult, extract_page
from jp_fin_parser.layout.hierarchy import build_hierarchy
from jp_fin_parser.preprocessing.normalize import parse_number
from jp_fin_parser.validation.validate import build_financial_table, validate_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PDF = DATA_DIR / "sample_bs_p4.pdf"
console = Console()

# Maximum tolerance unit in the sample filing (for 流動資産合計 with 10 children, tol = 10)
MAX_TOLERANCE_UNIT = Decimal("10")


@pytest.fixture(scope="module")
def base_extracted() -> ExtractionResult:
    """Extract clean words and rows from sample balance sheet."""
    if not SAMPLE_PDF.exists():
        pytest.skip(f"{SAMPLE_PDF} not found")
    return extract_page(SAMPLE_PDF, page_number=0)


def _evaluate_perturbation(perturbed: ExtractionResult) -> tuple[int, list[dict]]:
    """Re-run hierarchy, normalization, and validation on perturbed extraction result."""
    hierarchy = build_hierarchy(perturbed.table_rows)
    table = build_financial_table(perturbed, hierarchy=hierarchy)
    report = validate_table(table, hierarchy=hierarchy)
    return report.failing_checks, report.failures


class TestFuzzValidator:
    """Fuzz testing the validator against individual cell perturbations."""

    def test_fuzz_drop_minus_sign(self, base_extracted: ExtractionResult):
        """
        Drop negative sign (△/▲) from cells.
        Must be flagged unless perturbation is within tolerance (e.g. △0 -> 0).
        """
        tested = 0
        detected = 0
        sub_tolerance = 0

        for r_idx, row in enumerate(base_extracted.table_rows):
            for v_idx, val_str in enumerate(row.values):
                if any(neg in val_str for neg in ("△", "▲", "-")):
                    new_val_str = val_str.replace("△", "").replace("▲", "").replace("-", "")
                    if not new_val_str:
                        continue

                    old_num = parse_number(val_str).value or Decimal(0)
                    new_num = parse_number(new_val_str).value or Decimal(0)
                    diff = abs(new_num - old_num)

                    perturbed = deepcopy(base_extracted)
                    perturbed.table_rows[r_idx].values[v_idx] = new_val_str

                    failing_count, failures = _evaluate_perturbation(perturbed)

                    if diff <= MAX_TOLERANCE_UNIT:
                        sub_tolerance += 1
                        assert failing_count == 0, (
                            f"Sub-tolerance perturbation {val_str} -> {new_val_str} (diff={diff}) "
                            f"should not be flagged, but produced: {failures}"
                        )
                    else:
                        tested += 1
                        if failing_count > 0:
                            detected += 1

        rate = detected / tested if tested else 1.0
        print(f"\n[Fuzz Detection Rate] drop_minus_sign: {rate:.1%} ({detected}/{tested}) [sub_tol={sub_tolerance}]")
        assert tested > 0
        assert rate >= 0.90, f"Detection rate {rate:.1%} fell below 90% threshold"

    def test_fuzz_swap_adjacent_digits(self, base_extracted: ExtractionResult):
        """
        Swap two adjacent distinct digits.
        Must be flagged unless difference is smaller than rounding tolerance.
        """
        tested = 0
        detected = 0
        sub_tolerance = 0

        for r_idx, row in enumerate(base_extracted.table_rows):
            for v_idx, val_str in enumerate(row.values):
                parsed = parse_number(val_str)
                if parsed.value is not None and abs(parsed.value) >= 10:
                    s = str(abs(parsed.value))
                    # Find first two distinct adjacent digits
                    for i in range(len(s) - 1):
                        if s[i] != s[i + 1]:
                            s_list = list(s)
                            s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                            swapped_str = "".join(s_list)
                            new_val = Decimal(swapped_str) if parsed.value >= 0 else -Decimal(swapped_str)
                            diff = abs(new_val - parsed.value)

                            perturbed = deepcopy(base_extracted)
                            perturbed.table_rows[r_idx].values[v_idx] = f"{new_val:,}"

                            failing_count, failures = _evaluate_perturbation(perturbed)

                            if diff <= MAX_TOLERANCE_UNIT:
                                sub_tolerance += 1
                                assert failing_count == 0, (
                                    f"Sub-tolerance perturbation {parsed.value} -> {new_val} (diff={diff}) "
                                    f"should not be flagged."
                                )
                            else:
                                tested += 1
                                if failing_count > 0:
                                    detected += 1
                            break

        rate = detected / tested if tested else 1.0
        print(f"\n[Fuzz Detection Rate] swap_adjacent_digits: {rate:.1%} ({detected}/{tested}) [sub_tol={sub_tolerance}]")
        assert tested > 0
        assert rate >= 0.90, f"Detection rate {rate:.1%} fell below 90% threshold"

    def test_fuzz_shift_by_1000(self, base_extracted: ExtractionResult):
        """
        Shift numeric value by 1000 (+1000).
        Since 1000 >> rounding tolerance (<= 10), all must be flagged.
        """
        tested = 0
        detected = 0

        for r_idx, row in enumerate(base_extracted.table_rows):
            for v_idx, val_str in enumerate(row.values):
                parsed = parse_number(val_str)
                if parsed.value is not None:
                    new_val = parsed.value + 1000

                    perturbed = deepcopy(base_extracted)
                    perturbed.table_rows[r_idx].values[v_idx] = f"{new_val:,}"

                    failing_count, _ = _evaluate_perturbation(perturbed)
                    tested += 1
                    if failing_count > 0:
                        detected += 1

        rate = detected / tested if tested else 1.0
        print(f"\n[Fuzz Detection Rate] shift_by_1000: {rate:.1%} ({detected}/{tested})")
        assert tested > 0
        assert rate >= 0.90, f"Detection rate {rate:.1%} fell below 90% threshold"

    def test_fuzz_swap_prior_and_current(self, base_extracted: ExtractionResult):
        """
        Swap prior and current column values for rows.
        Must be flagged unless prior == current or abs(prior - current) <= tolerance.
        """
        tested = 0
        detected = 0
        sub_tolerance = 0

        for r_idx, row in enumerate(base_extracted.table_rows):
            if len(row.values) >= 2:
                v0, v1 = row.values[0], row.values[1]
                p0 = parse_number(v0).value
                p1 = parse_number(v1).value

                if p0 is not None and p1 is not None:
                    diff = abs(p0 - p1)

                    perturbed = deepcopy(base_extracted)
                    perturbed.table_rows[r_idx].values[0], perturbed.table_rows[r_idx].values[1] = v1, v0

                    failing_count, failures = _evaluate_perturbation(perturbed)

                    if diff <= MAX_TOLERANCE_UNIT:
                        sub_tolerance += 1
                        assert failing_count == 0, (
                            f"Sub-tolerance swap (diff={diff}) should not be flagged."
                        )
                    else:
                        tested += 1
                        if failing_count > 0:
                            detected += 1

        rate = detected / tested if tested else 1.0
        print(f"\n[Fuzz Detection Rate] swap_prior_current: {rate:.1%} ({detected}/{tested}) [sub_tol={sub_tolerance}]")
        assert tested > 0
        assert rate >= 0.90, f"Detection rate {rate:.1%} fell below 90% threshold"

    def test_fuzz_summary_report(self, base_extracted: ExtractionResult):
        """Print a formatted summary table of detection rates for all perturbation types."""
        results = {}

        # 1. Drop minus
        tested, detected, sub_tol = 0, 0, 0
        for r_idx, row in enumerate(base_extracted.table_rows):
            for v_idx, val_str in enumerate(row.values):
                if any(neg in val_str for neg in ("△", "▲")):
                    new_val_str = val_str.replace("△", "").replace("▲", "")
                    diff = abs((parse_number(new_val_str).value or Decimal(0)) - (parse_number(val_str).value or Decimal(0)))
                    perturbed = deepcopy(base_extracted)
                    perturbed.table_rows[r_idx].values[v_idx] = new_val_str
                    failing_count, _ = _evaluate_perturbation(perturbed)
                    if diff <= MAX_TOLERANCE_UNIT:
                        sub_tol += 1
                    else:
                        tested += 1
                        if failing_count > 0:
                            detected += 1
        results["Drop Minus Sign"] = (tested, detected, sub_tol)

        # 2. Swap digits
        tested, detected, sub_tol = 0, 0, 0
        for r_idx, row in enumerate(base_extracted.table_rows):
            for v_idx, val_str in enumerate(row.values):
                parsed = parse_number(val_str)
                if parsed.value is not None and abs(parsed.value) >= 10:
                    s = str(abs(parsed.value))
                    for i in range(len(s) - 1):
                        if s[i] != s[i + 1]:
                            s_list = list(s)
                            s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                            new_val = Decimal("".join(s_list)) if parsed.value >= 0 else -Decimal("".join(s_list))
                            diff = abs(new_val - parsed.value)
                            perturbed = deepcopy(base_extracted)
                            perturbed.table_rows[r_idx].values[v_idx] = f"{new_val:,}"
                            failing_count, _ = _evaluate_perturbation(perturbed)
                            if diff <= MAX_TOLERANCE_UNIT:
                                sub_tol += 1
                            else:
                                tested += 1
                                if failing_count > 0:
                                    detected += 1
                            break
        results["Swap Adjacent Digits"] = (tested, detected, sub_tol)

        # 3. Shift 1000
        tested, detected = 0, 0
        for r_idx, row in enumerate(base_extracted.table_rows):
            for v_idx, val_str in enumerate(row.values):
                parsed = parse_number(val_str)
                if parsed.value is not None:
                    new_val = parsed.value + 1000
                    perturbed = deepcopy(base_extracted)
                    perturbed.table_rows[r_idx].values[v_idx] = f"{new_val:,}"
                    failing_count, _ = _evaluate_perturbation(perturbed)
                    tested += 1
                    if failing_count > 0:
                        detected += 1
        results["Shift Value by 1000"] = (tested, detected, 0)

        # 4. Swap prior/current
        tested, detected, sub_tol = 0, 0, 0
        for r_idx, row in enumerate(base_extracted.table_rows):
            if len(row.values) >= 2:
                p0 = parse_number(row.values[0]).value
                p1 = parse_number(row.values[1]).value
                if p0 is not None and p1 is not None:
                    diff = abs(p0 - p1)
                    perturbed = deepcopy(base_extracted)
                    perturbed.table_rows[r_idx].values[0], perturbed.table_rows[r_idx].values[1] = row.values[1], row.values[0]
                    failing_count, _ = _evaluate_perturbation(perturbed)
                    if diff <= MAX_TOLERANCE_UNIT:
                        sub_tol += 1
                    else:
                        tested += 1
                        if failing_count > 0:
                            detected += 1
        results["Swap Prior & Current"] = (tested, detected, sub_tol)

        # Pretty print summary table
        tbl = RichTable("Perturbation Type", "Tested (>tol)", "Detected", "Detection Rate", "Sub-Tolerance (<tol)", header_style="bold cyan")
        for p_type, (t, d, st) in results.items():
            r_val = d / t if t else 1.0
            tbl.add_row(p_type, str(t), str(d), f"{r_val:.1%}", str(st))
        console.print()
        console.print(tbl)
