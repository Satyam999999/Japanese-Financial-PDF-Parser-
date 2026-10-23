# Japanese Financial PDF Parser: Core Development Prompts

This document contains the chronological record of the core development prompts used to build the **Japanese Financial PDF Parser** (`jp-fin-parser`), from repository architecture through extraction, normalization, hierarchy inference, arithmetic validation, evaluation harness, and documentation.

---

### Prompt 1: Phase 1: Project Framing, Conventions & Repository Architecture

```text
I'm building a parser that converts pages of Japanese public-company financial
filings (PDFs, e.g. 連結貸借対照表, 連結損益計算書, キャッシュ・フロー計算書, notes)
into correct structured tables. Input: one PDF page. Output: JSON + CSV.

Known conventions: units in header (千円/百万円), △ or ▲ = negative, － or - = nil,
two-period columns (前連結会計年度 / 当連結会計年度), 合計 rows are subtotals,
full-width digits and brackets may appear, corrected filings are marked (訂正後).
Indentation conveys hierarchy but is lost in plain text extraction.

Stack: Python 3.11, pdfplumber/PyMuPDF, pydantic, pytest. No paid APIs required;
an optional pluggable VLM fallback for scanned/complex pages.

Design principles: (1) never silently guess, every row carries a confidence and
validation flags; (2) layered pipeline with each stage independently testable;
(3) correctness measured by labeled ground truth AND arithmetic checks.

First, propose the repo layout and module boundaries. Don't write code yet.
```

---

### Prompt 2: Phase 2: Output Schema Design & Initial Metrics Specification

```text
Design the output schema with pydantic:
- PageMeta: title, statement_type, unit (and multiplier to yen), period labels,
  period end dates (parsed from 2025年3月31日 style), is_correction, page_number
- Row: label_raw, label_normalized, level (hierarchy depth), parent_id,
  row_type (header|item|subtotal|total), values: dict[period -> Decimal|None],
  is_nil (－ vs missing), source_bbox, confidence, flags
- Table: meta, rows, validation_report

Then write metrics.py implementing:
1. cell_accuracy vs ground-truth JSON (exact match on label + each value)
2. structure_accuracy (row order, level, row_type)
3. arithmetic_consistency (fraction of subtotals that reconcile within tolerance)
4. a single report function printing all three per page and aggregated.
Include pytest tests with tiny synthetic examples.
```

---

### Prompt 3: Phase 3: Extraction Engine Implementation (extract.py)

```text
Implement extract.py: given a PDF page, return positioned words/lines with bboxes
(pdfplumber). Then cluster into rows by y-coordinate (tolerance based on font
size) and columns by x-gaps. Handle:
- label column on the left, N numeric columns right-aligned
- wrapped labels spanning two lines
- header block above the table (title, unit line "（単位：千円）", period headers)
- page numbers like "- 4 -" and stray text excluded from the table
Record each row's left-edge x0 of the label for later hierarchy inference.
Detect whether the page has a usable text layer; if not, raise a clear
NeedsOCR signal. Test on the attached page and print the rows.
```

---

### Prompt 4: Phase 4: Extraction Engine with Pytest Verification

```text
Implement extract.py: given a PDF page, return positioned words/lines with bboxes
(pdfplumber). Then cluster into rows by y-coordinate (tolerance based on font
size) and columns by x-gaps. Handle:
- label column on the left, N numeric columns right-aligned
- wrapped labels spanning two lines
- header block above the table (title, unit line "（単位：千円）", period headers)
- page numbers like "- 4 -" and stray text excluded from the table
Record each row's left-edge x0 of the label for later hierarchy inference.
Detect whether the page has a usable text layer; if not, raise a clear
NeedsOCR signal. Test on the attached page and print the rows.
After each prompt, run `pytest`. Move on only when the current stage works.
 If something is wrong, fix it before the next stage.
```

---

### Prompt 5: Phase 5: Normalization & Preprocessing Implementation (normalize.py)

```text
Implement normalize.py:
- parse numbers: commas, full-width digits, △/▲/() = negative, －/―/- = nil (None
  with is_nil=True, distinct from missing), "△0" edge case
- apply the unit multiplier but keep the original value too
- NFKC-normalize labels, strip spaces inside Japanese labels, keep raw label
- parse Japanese dates (和暦 and 西暦) and fiscal-year labels
- detect row_type: labels ending in 合計/計 or in a known list
  (流動資産合計, 資産合計, 純資産合計...) are subtotal/total; section headers
  (資産の部, 流動資産) have no values
Full pytest coverage including ugly casesAfter each prompt, run `pytest`. Move on only when the current stage works.
 If something is wrong, fix it before the next stage..
```

---

### Prompt 6: Phase 6: Accounting Hierarchy & Monotonic Indentation Stack (hierarchy.py)

```text
Implement hierarchy.py to infer each row's level and parent. Use the label x0
indentation when available, and fall back to semantic rules: header rows (資産の部
→ 流動資産 → items → 流動資産合計). Notably, this page has patterns like
"建物及び構築物 / 減価償却累計額 / 建物及び構築物（純額）" where the (純額) row is
gross minus accumulated depreciation, not a sum of the previous rows. Model
these as a formula-type relation. Output a tree and a flat list, and test on the
attached balance sheet.After each prompt, run `pytest`. Move on only when the current stage works.
 If something is wrong, fix it before the next stage..
```

---

### Prompt 7: Phase 7: Validation Engine & Arithmetic Reconciliation (validate.py)

```text
Implement validate.py with a rule engine:
- subtotal rule: subtotal == sum(children) per period
- net rule: gross + accumulated depreciation (negative) == net
- balance sheet identity: 資産合計 == 負債純資産合計 when both pages are present
- tolerance: abs diff <= max(n_children, 1) units of the rounding precision,
  because filings round to thousands (e.g. the sample's current-year 流動資産
  components sum to 3,377,405 vs reported 3,377,407, which must PASS as rounding,
  not fail)
- each failure produces a flag with row ids, expected, actual, diff
- when exactly one child value is suspect, suggest the likely correction
  (a possible OCR/digit error) but don't apply it automatically
Return a validation_report and per-row confidence adjustments.
 After each prompt, run `pytest`. Move on only when the current stage works.
 If something is wrong, fix it before the next stage.
```

---

### Prompt 8: Phase 8: Evaluation Harness, Error Taxonomy & Fuzz Testing (run_eval.py)

```text
Create the evaluation harness. The hand-verified ground truth already exists at
eval/ground_truth/sample_bs_p4.json. Do NOT create, edit, or regenerate it, and
do NOT derive any expected values from my parser's output.

Build eval/run_eval.py that:
1. Parses every PDF in data/ with my existing pipeline.
2. For any PDF with a matching ground-truth file (same stem plus a
   ground_truth/<stem>.json), computes using the functions in metrics.py:
   - cell_accuracy: exact match of label, prior, current per row
   - structure_accuracy: row order, level, row_type, parent_index
   - formula_accuracy: formula_of lists match
3. For ALL PDFs, matched or not, computes arithmetic_consistency using my
   validator (formula_of based, with rounding tolerance).
4. Aligns rows by row_index/order. If the parser produced a different row count,
   report it as an error and align by best label match so one missing row
   doesn't wreck every later comparison.
5. Prints a per-PDF report and a final aggregate.
6. Writes eval/report.json and eval/report.md with the results.

Error taxonomy: classify every mismatch as one of: wrong_number, wrong_sign,
nil_vs_zero, missing_row, extra_row, merged_rows, wrong_level, wrong_row_type,
wrong_parent, wrong_unit. The report lists counts per category and the first
five examples of each (row_index, label, expected, actual).

Add tests/test_eval.py:
- parsing the sample PDF gets cell_accuracy of 1.0 (if it doesn't, the test
  should fail and print the taxonomy)
- the six rounding gaps from the sample (e.g. 流動資産 current: expected 3,377,405
  vs printed 3,377,407) all PASS validation

Add a fuzz test (tests/test_fuzz.py): take the parser's extracted words, apply
one perturbation at a time (drop a minus sign, swap two adjacent digits, shift a
value by 1000, swap prior and current), re-run normalize, hierarchy, and
validate, and assert the validator flags each perturbation, except
perturbations smaller than the rounding tolerance, which must not be flagged.
Report the detection rate for each perturbation type.
```

---

### Prompt 9: Phase 9: Comprehensive Submission Documentation (README.md)

```text
Write README.md for the submission: problem framing, my definition of "correct"
(three metrics and why), architecture diagram (mermaid), results on the sample,
known failure modes, what I'd do next with more time and more PDFs, and how the
arithmetic checks are used as a label-free confidence signal to prioritize human
review. Keep it concise and honest about limitations.
```
