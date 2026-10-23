# Evaluation Report: Japanese Financial Statement Parser

**Generated:** 2026-10-23T17:30:54.652962+00:00

## 1. Aggregate Summary

| Metric | Result | Ratio |
| :--- | :---: | :---: |
| **Cell Accuracy** (label, prior, current) | **100.00%** | 86 / 86 |
| **Structure Accuracy** (order, level, type, parent) | **100.00%** | 43 / 43 |
| **Formula Accuracy** (`formula_of` lists) | **100.00%** | 10 / 10 |
| **Arithmetic Consistency** (formula-based, tolerance) | **100.00%** | 40 / 40 |

## 2. Error Taxonomy

| Error Category | Count | Status | Description |
| :--- | :---: | :---: | :--- |
| `wrong_number` | 0 | PASSED | Numeric cell values differ |
| `wrong_sign` | 0 | PASSED | Sign inverted (positive vs negative) |
| `nil_vs_zero` | 0 | PASSED | Explicit nil (－) confused with zero |
| `missing_row` | 0 | PASSED | Expected ground-truth row omitted |
| `extra_row` | 0 | PASSED | Unmatched extra row generated |
| `merged_rows` | 0 | PASSED | Multiple rows concatenated into one |
| `wrong_level` | 0 | PASSED | Indentation hierarchy depth incorrect |
| `wrong_row_type` | 0 | PASSED | Row role mismatch (item/subtotal/total/net) |
| `wrong_parent` | 0 | PASSED | Parent reference index incorrect |
| `wrong_unit` | 0 | PASSED | Scale mismatch by 1,000x or 1,000,000x |

### Error Examples (First 5 per Category)

*(No taxonomy errors recorded across ground-truth evaluations.)*

## 3. Per-PDF Breakdown

| PDF File | Ground Truth | Rows (Act/Exp) | Cell Acc | Structure Acc | Formula Acc | Arithmetic (Pass/Total) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `balance_sheet_after_correction.pdf` | None | 43 / — | — | — | — | 100.0% (20/20) |
| `sample_bs_p4.pdf` | Matched | 43 / 43 | 100.0% | 100.0% | 100.0% | 100.0% (20/20) |

## 4. Rounding Tolerance Verification

Financial statements frequently present numbers rounded to the nearest thousand (千円), causing small differences between sum of reported lines and reported subtotal totals. The validator implements a tolerance formula $\Delta \le \max(N, 1) \times \text{precision}$. All rounding gaps in verified files passed validation cleanly.
