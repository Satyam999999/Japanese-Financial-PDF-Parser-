"""
tests/unit/layout/test_hierarchy.py

Tests for jp_fin_parser.layout.hierarchy.

Structure
---------
1. _build_level_map – x0 clustering
2. _assign_parents  – monotone stack
3. _detect_formula_nodes – 純額 pattern
4. HierarchyResult helpers (to_tree, print_tree, flat_labels)
5. build_hierarchy end-to-end on synthetic rows
6. Integration test on the real balance sheet PDF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from jp_fin_parser.layout.hierarchy import (
    HierarchyResult,
    RelationType,
    _assign_parents,
    _build_level_map,
    _detect_formula_nodes,
    _nearest_breakpoint,
    build_hierarchy,
    HierarchyNode,
)

# ---------------------------------------------------------------------------
# Real PDF path
# ---------------------------------------------------------------------------
REAL_PDF = Path(__file__).parents[4] / "balance_sheet_after_correction.pdf"

# ---------------------------------------------------------------------------
# Minimal stub for ExtractedRow (avoids importing pdfplumber in unit tests)
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    """Minimal stand-in for extraction.ExtractedRow."""
    label: str
    label_x0: float
    values: list = field(default_factory=list)   # non-empty → has_values=True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    index: int,
    label: str,
    label_x0: float,
    level: int,
    has_values: bool = True,
    parent_idx: Optional[int] = None,
    children_idxs: Optional[list] = None,
    relation: RelationType = RelationType.SUM,
) -> HierarchyNode:
    return HierarchyNode(
        index=index,
        label=label,
        label_normalized=label,
        label_x0=label_x0,
        level=level,
        has_values=has_values,
        parent_idx=parent_idx,
        children_idxs=children_idxs or [],
        relation=relation,
    )


# ===========================================================================
# 1. _build_level_map
# ===========================================================================


class TestBuildLevelMap:
    def test_distinct_x0s_get_distinct_levels(self):
        rows = [_Row("A", 75.2), _Row("B", 88.2), _Row("C", 101.3), _Row("D", 114.2)]
        level_map, bps = _build_level_map(rows, tol=3.0)
        assert len(bps) == 4
        # Each unique x0 → different level
        assert level_map[bps[0]] == 0
        assert level_map[bps[1]] == 1
        assert level_map[bps[2]] == 2
        assert level_map[bps[3]] == 3

    def test_close_x0s_merged_into_same_level(self):
        """Two x0 values within tolerance → same level cluster."""
        rows = [_Row("A", 88.0), _Row("B", 89.5)]  # diff = 1.5 < tol 3.0
        _, bps = _build_level_map(rows, tol=3.0)
        assert len(bps) == 1   # merged into one cluster

    def test_breakpoints_sorted_ascending(self):
        rows = [_Row("A", 101.3), _Row("B", 75.2), _Row("C", 88.2)]
        _, bps = _build_level_map(rows, tol=3.0)
        assert bps == sorted(bps)

    def test_empty_rows(self):
        level_map, bps = _build_level_map([], tol=3.0)
        assert level_map == {}
        assert bps == []

    def test_single_row(self):
        rows = [_Row("A", 88.2)]
        level_map, bps = _build_level_map(rows, tol=3.0)
        assert len(bps) == 1
        assert level_map[bps[0]] == 0

    def test_real_bs_x0_values(self):
        """The four distinct x0 tiers in the real balance sheet."""
        rows = [
            _Row("固定資産",                75.2),
            _Row("有形固定資産",              88.2),
            _Row("建物及び構築物",           101.3),
            _Row("減価償却累計額",           114.2),
            _Row("建物及び構築物（純額）",     114.2),
            _Row("有形固定資産合計",         101.3),
            _Row("固定資産合計",             88.2),
            _Row("資産合計",                75.2),
        ]
        _, bps = _build_level_map(rows, tol=3.0)
        assert len(bps) == 4   # 75.2 / 88.2 / 101.3 / 114.2


# ===========================================================================
# 2. _nearest_breakpoint
# ===========================================================================


class TestNearestBreakpoint:
    def test_exact_match(self):
        assert _nearest_breakpoint(88.2, [75.2, 88.2, 101.3]) == 88.2

    def test_rounds_to_nearest(self):
        assert _nearest_breakpoint(89.0, [75.2, 88.2, 101.3]) == 88.2

    def test_empty_breakpoints(self):
        assert _nearest_breakpoint(50.0, []) == 0.0


# ===========================================================================
# 3. _assign_parents
# ===========================================================================


class TestAssignParents:
    def _make_nodes(self, levels: list[int]) -> list[HierarchyNode]:
        return [
            HierarchyNode(
                index=i, label=f"row{i}", label_normalized=f"row{i}",
                label_x0=float(lv * 13), level=lv, has_values=True,
            )
            for i, lv in enumerate(levels)
        ]

    def test_flat_all_roots(self):
        nodes = self._make_nodes([0, 0, 0])
        _assign_parents(nodes)
        for n in nodes:
            assert n.parent_idx is None
        # all are roots

    def test_one_level_deep(self):
        # parent(0), child(1), child(2)
        nodes = self._make_nodes([0, 1, 1])
        _assign_parents(nodes)
        assert nodes[0].parent_idx is None
        assert nodes[1].parent_idx == 0
        assert nodes[2].parent_idx == 0
        assert nodes[0].children_idxs == [1, 2]

    def test_two_levels_deep(self):
        # root(0), child(1), grandchild(2)
        nodes = self._make_nodes([0, 1, 2])
        _assign_parents(nodes)
        assert nodes[0].parent_idx is None
        assert nodes[1].parent_idx == 0
        assert nodes[2].parent_idx == 1
        assert nodes[1].children_idxs == [2]

    def test_sibling_after_deeper_subtree(self):
        # root(0), child_a(1), grandchild(2), child_b(1)
        nodes = self._make_nodes([0, 1, 2, 1])
        _assign_parents(nodes)
        assert nodes[3].parent_idx == 0   # child_b is child of root
        assert 3 in nodes[0].children_idxs

    def test_section_header_between_items(self):
        """
        Simulates: items(1) → section_header(0) → items(1)
        The second block of level-1 rows should be children of the header.
        """
        # item_a(1), section(0), item_b(1)
        nodes = self._make_nodes([1, 0, 1])
        _assign_parents(nodes)
        assert nodes[0].parent_idx is None  # item_a is a root (no prior level-0)
        assert nodes[1].parent_idx is None  # section is a root
        assert nodes[2].parent_idx == 1     # item_b is child of section

    def test_real_bs_structure(self):
        """
        Smoke-test the real balance-sheet indentation pattern:
        流動資産合計(1), 固定資産(0), 有形固定資産(1), 建物及び構築物(2),
        減価償却累計額(3), 建物及び構築物（純額）(3), 有形固定資産合計(2)
        """
        levels = [1, 0, 1, 2, 3, 3, 2]
        nodes = self._make_nodes(levels)
        _assign_parents(nodes)
        # 固定資産 (idx=1, level=0) is a root
        assert nodes[1].parent_idx is None
        # 有形固定資産 (idx=2, level=1) is child of 固定資産
        assert nodes[2].parent_idx == 1
        # 建物及び構築物 (idx=3, level=2) is child of 有形固定資産
        assert nodes[3].parent_idx == 2
        # 減価償却累計額 (idx=4, level=3) is child of 建物及び構築物
        assert nodes[4].parent_idx == 3
        # 建物及び構築物（純額）(idx=5, level=3) is also child of 建物及び構築物
        assert nodes[5].parent_idx == 3
        # 有形固定資産合計 (idx=6, level=2) is child of 有形固定資産 (popped back to lvl 1)
        assert nodes[6].parent_idx == 2


# ===========================================================================
# 4. _detect_formula_nodes
# ===========================================================================


class TestDetectFormulaNodes:
    def _build_junmei_nodes(self):
        """
        Simulate:
          idx=0  建物及び構築物       (level=2, parent=None)
          idx=1  減価償却累計額        (level=3, parent=0)
          idx=2  建物及び構築物（純額）(level=3, parent=0)
        """
        parent = HierarchyNode(
            index=0, label="建物及び構築物", label_normalized="建物及び構築物",
            label_x0=101.3, level=2, has_values=True,
            children_idxs=[1, 2],
        )
        dep = HierarchyNode(
            index=1, label="減価償却累計額", label_normalized="減価償却累計額",
            label_x0=114.2, level=3, has_values=True,
            parent_idx=0,
        )
        junmei = HierarchyNode(
            index=2, label="建物及び構築物（純額）", label_normalized="建物及び構築物（純額）",
            label_x0=114.2, level=3, has_values=True,
            parent_idx=0,
        )
        return [parent, dep, junmei]

    def test_junmei_marked_as_formula(self):
        nodes = self._build_junmei_nodes()
        _detect_formula_nodes(nodes)
        assert nodes[2].relation == RelationType.FORMULA

    def test_formula_components_populated(self):
        nodes = self._build_junmei_nodes()
        _detect_formula_nodes(nodes)
        assert 0 in nodes[2].formula_components   # gross parent
        assert 1 in nodes[2].formula_components   # depreciation sibling

    def test_non_junmei_not_marked(self):
        nodes = self._build_junmei_nodes()
        _detect_formula_nodes(nodes)
        assert nodes[0].relation == RelationType.SUM
        assert nodes[1].relation == RelationType.SUM

    def test_junmei_without_dep_sibling_not_formula(self):
        """若し減価償却累計額が無ければ FORMULA にしない。"""
        parent = HierarchyNode(
            index=0, label="土地", label_normalized="土地",
            label_x0=101.3, level=2, has_values=True, children_idxs=[1],
        )
        # No depreciation sibling — just a 純額 label (hypothetical)
        junmei = HierarchyNode(
            index=1, label="土地（純額）", label_normalized="土地（純額）",
            label_x0=114.2, level=3, has_values=True, parent_idx=0,
        )
        nodes = [parent, junmei]
        _detect_formula_nodes(nodes)
        assert nodes[1].relation == RelationType.SUM   # not changed

    def test_multiple_depreciation_sibling_all_captured(self):
        """
        Edge case: two 減価償却累計額 siblings before 純額 (unusual but handled).
        """
        parent = HierarchyNode(
            index=0, label="X", label_normalized="X",
            label_x0=101.3, level=2, has_values=True, children_idxs=[1, 2, 3],
        )
        dep1 = HierarchyNode(index=1, label="減価償却累計額", label_normalized="減価償却累計額",
                              label_x0=114.2, level=3, has_values=True, parent_idx=0)
        dep2 = HierarchyNode(index=2, label="減価償却累計額", label_normalized="減価償却累計額",
                              label_x0=114.2, level=3, has_values=True, parent_idx=0)
        junmei = HierarchyNode(index=3, label="X（純額）", label_normalized="X（純額）",
                                label_x0=114.2, level=3, has_values=True, parent_idx=0)
        nodes = [parent, dep1, dep2, junmei]
        _detect_formula_nodes(nodes)
        assert 1 in nodes[3].formula_components
        assert 2 in nodes[3].formula_components


# ===========================================================================
# 5. build_hierarchy end-to-end (synthetic)
# ===========================================================================


class TestBuildHierarchy:
    def _bs_rows(self):
        """Minimal synthetic balance sheet (assets side)."""
        return [
            # 流動資産 items (x0=88.2 before 固定資産 appears)
            _Row("現金及び預金",            88.2, ["2,002,429", "608,281"]),
            _Row("売掛金",                 88.2, ["2,019,476", "884,339"]),
            _Row("流動資産合計",             88.2, ["5,863,684", "3,377,407"]),
            # 固定資産 section (x0=75.2)
            _Row("固定資産",               75.2),
            _Row("有形固定資産",            88.2),
            _Row("建物及び構築物",          101.3, ["2,121,360", "1,880,082"]),
            _Row("減価償却累計額",          114.2, ["△1,923,585", "△1,640,253"]),
            _Row("建物及び構築物（純額）",    114.2, ["197,775",   "239,828"]),
            _Row("有形固定資産合計",        101.3, ["1,429,094", "1,352,610"]),
            _Row("固定資産合計",            88.2, ["1,670,426", "1,757,841"]),
            _Row("資産合計",               75.2, ["7,534,110", "5,135,249"]),
        ]

    def test_returns_hierarchy_result(self):
        result = build_hierarchy(self._bs_rows())
        assert isinstance(result, HierarchyResult)

    def test_four_level_breakpoints(self):
        result = build_hierarchy(self._bs_rows())
        assert len(result.level_breakpoints) == 4  # 75.2/88.2/101.3/114.2

    def test_root_nodes(self):
        result = build_hierarchy(self._bs_rows())
        root_labels = [result.nodes[i].label_normalized for i in result.roots]
        # 固定資産 (x0=75.2) and 資産合計 (x0=75.2) are roots;
        # the level-1 nodes before 固定資産 are also roots (no parent)
        assert "固定資産" in root_labels
        assert "資産合計" in root_labels

    def test_junmei_is_formula(self):
        result = build_hierarchy(self._bs_rows())
        node = result.node_for_label("建物及び構築物（純額）")
        assert node is not None
        assert node.relation == RelationType.FORMULA

    def test_junmei_formula_components_include_gross_and_dep(self):
        result = build_hierarchy(self._bs_rows())
        junmei = result.node_for_label("建物及び構築物（純額）")
        assert junmei is not None
        dep_idx = result.node_for_label("減価償却累計額").index  # first 減価償却
        gross_idx = result.node_for_label("建物及び構築物").index
        assert gross_idx in junmei.formula_components
        assert dep_idx in junmei.formula_components

    def test_subtotal_is_child_of_section(self):
        result = build_hierarchy(self._bs_rows())
        subtotal = result.node_for_label("有形固定資産合計")
        assert subtotal is not None
        # Its parent should be 有形固定資産
        parent = result.nodes[subtotal.parent_idx]
        assert parent.label_normalized in ("有形固定資産", "固定資産")

    def test_has_values_flag(self):
        result = build_hierarchy(self._bs_rows())
        fixed = result.node_for_label("固定資産")
        assert fixed is not None
        assert not fixed.has_values   # no values in _Row for 固定資産

    def test_flat_labels_order(self):
        rows = self._bs_rows()
        result = build_hierarchy(rows)
        flat = result.flat_labels()
        assert flat[0] == "現金及び預金"
        assert flat[-1] == "資産合計"
        assert len(flat) == len(rows)

    def test_to_tree_structure(self):
        result = build_hierarchy(self._bs_rows())
        tree = result.to_tree()
        assert isinstance(tree, list)
        assert len(tree) > 0
        # Every node in tree has required keys
        def check(node_dict):
            for key in ("label", "level", "relation", "has_values", "children"):
                assert key in node_dict, f"Missing key {key!r}"
            for child in node_dict["children"]:
                check(child)
        for root_dict in tree:
            check(root_dict)

    def test_print_tree_no_exception(self, capsys):
        result = build_hierarchy(self._bs_rows())
        result.print_tree()
        out = capsys.readouterr().out
        assert "建物及び構築物（純額）" in out
        assert "[formula]" in out

    def test_empty_rows(self):
        result = build_hierarchy([])
        assert result.nodes == []
        assert result.roots == []

    def test_single_row_is_root(self):
        result = build_hierarchy([_Row("現金及び預金", 88.2, ["100", "200"])])
        assert len(result.roots) == 1
        assert result.nodes[0].parent_idx is None


# ===========================================================================
# 6. Integration test on the real balance sheet PDF
# ===========================================================================

@pytest.mark.skipif(not REAL_PDF.exists(), reason="Real PDF not found")
class TestRealPDFHierarchy:
    @pytest.fixture(scope="class")
    def result(self):
        import sys
        sys.path.insert(0, str(REAL_PDF.parents[1] / "jp-fin-parser" / "src"))
        from jp_fin_parser.extraction.extract import extract_page
        extracted = extract_page(REAL_PDF, page_number=0)
        return build_hierarchy(extracted.table_rows)

    def test_returns_result(self, result):
        assert isinstance(result, HierarchyResult)

    def test_four_level_tiers(self, result):
        assert len(result.level_breakpoints) in (4, 5)

    def test_all_junmei_are_formula(self, result):
        """Every (純額) row in the real BS must be FORMULA."""
        junmei_nodes = [n for n in result.nodes if "（純額）" in n.label_normalized]
        assert len(junmei_nodes) >= 4   # BS has ≥4 pure-amount rows
        for n in junmei_nodes:
            assert n.relation == RelationType.FORMULA, (
                f"{n.label_normalized!r} should be FORMULA, got {n.relation}"
            )

    def test_depreciation_not_formula(self, result):
        """減価償却累計額 rows must NOT be marked FORMULA (they are SUM children)."""
        dep_nodes = [n for n in result.nodes if n.label_normalized == "減価償却累計額"]
        assert len(dep_nodes) >= 4
        for n in dep_nodes:
            assert n.relation != RelationType.FORMULA

    def test_固定資産_is_root_or_child_of_bu(self, result):
        node = result.node_for_label("固定資産")
        assert node is not None
        bu_node = result.node_for_label("資産の部")
        if bu_node is not None:
            assert node.parent_idx == bu_node.index
        else:
            assert node.parent_idx is None

    def test_資産合計_is_root(self, result):
        node = result.node_for_label("資産合計")
        assert node is not None
        assert node.parent_idx is None

    def test_有形固定資産合計_under_有形固定資産_or_固定資産(self, result):
        subtotal = result.node_for_label("有形固定資産合計")
        assert subtotal is not None
        parent = result.nodes[subtotal.parent_idx]
        assert parent.label_normalized in ("有形固定資産", "固定資産")

    def test_建物及び構築物_純額_components(self, result):
        junmei = result.node_for_label("建物及び構築物（純額）")
        assert junmei is not None
        assert len(junmei.formula_components) >= 2
        # The gross parent should be 建物及び構築物
        gross_node = result.nodes[junmei.formula_components[0]]
        assert "建物及び構築物" in gross_node.label_normalized
        assert "純額" not in gross_node.label_normalized

    def test_print_tree_contains_formula_tags(self, result, capsys):
        result.print_tree()
        out = capsys.readouterr().out
        assert "[formula]" in out

    def test_all_nodes_have_valid_parent_refs(self, result):
        """No dangling parent indices."""
        n = len(result.nodes)
        for node in result.nodes:
            if node.parent_idx is not None:
                assert 0 <= node.parent_idx < n
            for c in node.children_idxs:
                assert 0 <= c < n
