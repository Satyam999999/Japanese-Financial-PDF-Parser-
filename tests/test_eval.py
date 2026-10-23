"""
tests/test_eval.py

Evaluation harness tests verifying:
1. Parsing sample_bs_p4.pdf achieves cell_accuracy == 1.0 (with error taxonomy printed on failure).
2. The six documented rounding gaps from the sample balance sheet all PASS validation.
"""

from decimal import Decimal
import json
from pathlib import Path
import pytest

from eval.run_eval import evaluate_single_pdf, load_ground_truth
from jp_fin_parser.extraction.extract import extract_page
from jp_fin_parser.layout.hierarchy import build_hierarchy
from jp_fin_parser.validation.validate import build_financial_table, validate_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GT_DIR = Path(__file__).resolve().parent.parent / "eval" / "ground_truth"
SAMPLE_PDF = DATA_DIR / "sample_bs_p4.pdf"
SAMPLE_GT = GT_DIR / "sample_bs_p4.json"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_bs_p4.pdf not found")
class TestEvaluationHarness:
    """Verifies that the full pipeline meets accuracy standards against ground truth."""

    @pytest.fixture(scope="class")
    def evaluation_result(self):
        """Run single PDF evaluation on sample_bs_p4.pdf."""
        return evaluate_single_pdf(SAMPLE_PDF, gt_dir=GT_DIR)

    def test_sample_pdf_cell_accuracy_is_1_0(self, evaluation_result):
        """
        Parsing the sample PDF must achieve cell_accuracy of 1.0.
        If it does not, the test fails and prints the taxonomy breakdown and examples.
        """
        cell_acc = evaluation_result.get("cell_accuracy")
        taxonomy = evaluation_result.get("taxonomy", {})

        # Build detailed taxonomy report string for failure diagnostics
        fail_msgs = []
        for cat, data in taxonomy.items():
            cnt = data.get("count", 0)
            if cnt > 0:
                fail_msgs.append(f"  [{cat}]: {cnt} errors")
                for ex in data.get("examples", []):
                    fail_msgs.append(
                        f"    - Row {ex['row_index']} '{ex['label']}': "
                        f"expected '{ex['expected']}', got '{ex['actual']}'"
                    )

        diag_report = "\n".join(fail_msgs) if fail_msgs else "No taxonomy errors."

        assert cell_acc == 1.0, (
            f"Expected cell_accuracy == 1.0, got {cell_acc}.\n"
            f"Taxonomy failure diagnostics:\n{diag_report}"
        )

        # Structure, formula, and taxonomy checks
        assert evaluation_result.get("structure_accuracy") == 1.0
        assert evaluation_result.get("formula_accuracy") == 1.0
        assert evaluation_result.get("arithmetic_consistency") == 1.0

        for cat, data in taxonomy.items():
            assert data["count"] == 0, f"Expected 0 errors in category '{cat}', got {data['count']}"

    def test_six_sample_rounding_gaps_all_pass(self):
        """
        Verify that the six rounding gaps from the sample all PASS validation:
        1. Row 13 流動資産合計 (current & prior)
        2. Row 18 建物及び構築物（純額） (current)
        3. Row 25 リース資産（純額） (current & prior)
        4. Row 30 有形固定資産合計 (current & prior)
        5. Row 34 無形固定資産合計 (current & prior)
        6. Row 41 投資その他の資産合計 (current & prior)
        """
        extracted = extract_page(SAMPLE_PDF, page_number=0)
        hierarchy = build_hierarchy(extracted.table_rows)
        table = build_financial_table(extracted, hierarchy=hierarchy)
        report = validate_table(table, hierarchy=hierarchy)

        assert report.failing_checks == 0
        checks_by_key = {
            (c.row_label, c.period): c for c in report.arithmetic_checks
        }

        # 1. Row 13 流動資産合計 current: expected 3,377,405 vs printed 3,377,407 (delta = 2)
        c1 = checks_by_key.get(("流動資産合計", "当連結会計年度"))
        assert c1 is not None
        assert c1.passes
        assert c1.expected == Decimal("3377405")
        assert c1.actual == Decimal("3377407")
        assert c1.delta == Decimal("2")
        assert c1.delta <= c1.tolerance

        # 2. Row 18 建物及び構築物（純額） current: expected 239,829 vs printed 239,828 (delta = 1)
        c2 = checks_by_key.get(("建物及び構築物（純額）", "当連結会計年度"))
        assert c2 is not None
        assert c2.passes
        assert c2.expected == Decimal("239829")
        assert c2.actual == Decimal("239828")
        assert c2.delta == Decimal("1")
        assert c2.delta <= c2.tolerance

        # 3. Row 25 リース資産（純額） prior: expected 63,147 vs printed 63,146 (delta = 1)
        c3_prior = checks_by_key.get(("リース資産（純額）", "前連結会計年度"))
        assert c3_prior is not None
        assert c3_prior.passes
        assert c3_prior.expected == Decimal("63147")
        assert c3_prior.actual == Decimal("63146")
        assert c3_prior.delta == Decimal("1")

        c3_curr = checks_by_key.get(("リース資産（純額）", "当連結会計年度"))
        assert c3_curr is not None
        assert c3_curr.passes
        assert c3_curr.expected == Decimal("41873")
        assert c3_curr.actual == Decimal("41872")
        assert c3_curr.delta == Decimal("1")

        # 4. Row 30 有形固定資産合計 current: expected 1,352,607 vs printed 1,352,610 (delta = 3)
        c4 = checks_by_key.get(("有形固定資産合計", "当連結会計年度"))
        assert c4 is not None
        assert c4.passes
        assert c4.expected == Decimal("1352607")
        assert c4.actual == Decimal("1352610")
        assert c4.delta == Decimal("3")
        assert c4.delta <= c4.tolerance

        # 5. Row 34 無形固定資産合計 current: expected 35,868 vs printed 35,869 (delta = 1)
        c5 = checks_by_key.get(("無形固定資産合計", "当連結会計年度"))
        assert c5 is not None
        assert c5.passes
        assert c5.expected == Decimal("35868")
        assert c5.actual == Decimal("35869")
        assert c5.delta == Decimal("1")
        assert c5.delta <= c5.tolerance

        # 6. Row 41 投資その他の資産合計 current: expected 369,361 vs printed 369,362 (delta = 1)
        c6 = checks_by_key.get(("投資その他の資産合計", "当連結会計年度"))
        assert c6 is not None
        assert c6.passes
        assert c6.expected == Decimal("369361")
        assert c6.actual == Decimal("369362")
        assert c6.delta == Decimal("1")
        assert c6.delta <= c6.tolerance
