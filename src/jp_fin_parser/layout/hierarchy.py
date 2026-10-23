"""
jp_fin_parser/layout/hierarchy.py

Stage 3 – Hierarchy inference: infer each row's level and parent, detect
formula-type relations (純額 pattern), and expose both a flat list and a tree.

Algorithm
---------
1. **Level inference from x0**
   Collect the left-edge x-coordinate (label_x0) of every row that has a
   label.  Cluster the values with a small spatial tolerance (default 3 pt)
   to handle sub-pixel rendering differences.  Sort the resulting cluster
   centroids; each centroid's rank becomes its indentation level
   (0 = top-level, higher = deeper).

2. **Parent assignment — monotone-stack**
   Process rows top-to-bottom in document order.  Maintain a stack of open
   ancestors.  When visiting row i at level L, pop all stack entries whose
   level ≥ L; the new stack-top (if any) is the parent of row i.

3. **純額 formula detection**
   A row whose normalised label ends with「（純額）」(e.g. 建物及び構築物（純額）)
   and whose parent has a preceding sibling labelled「減価償却累計額」is
   annotated as *RelationType.FORMULA*.
   The formula is:
       純額 = 親 (gross) + 減価償却累計額 (already signed negative)
   Formula-component indices are stored in ``formula_components`` so the
   validation layer can verify the arithmetic without guessing.

4. **Semantic fallback for section headers**
   Rows with no numeric values whose normalised label matches known section-
   header keywords (の部 suffix, 流動資産, etc.) are marked STANDALONE so the
   validation layer skips arithmetic checks for them.

5. **Tree output**
   ``HierarchyResult.to_tree()`` returns a nested list-of-dicts suitable for
   JSON serialisation.  ``HierarchyResult.print_tree()`` prints an
   indented ASCII tree to stdout for quick visual inspection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from statistics import median
from typing import Optional

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RelationType(StrEnum):
    """How a node's value relates to the rest of the tree."""

    SUM = "sum"
    """Normal subtotal/total: value should equal sum of SUM-type children."""

    FORMULA = "formula"
    """純額 node: value = gross parent + depreciation siblings (already –ve)."""

    STANDALONE = "standalone"
    """Section header or annotation; no arithmetic expectation."""


@dataclass
class HierarchyNode:
    """
    One row of the statement, enriched with hierarchy information.
    All indices refer to positions in ``HierarchyResult.nodes``.
    """

    index: int                           # position in flat list
    label: str                           # raw label text
    label_normalized: str                # NFKC + space-stripped
    label_x0: float                      # left edge of label (pt)
    level: int                           # 0 = topmost, higher = deeper
    has_values: bool                     # True when row has ≥1 numeric cell
    parent_idx: Optional[int] = None     # None for root nodes
    children_idxs: list[int] = field(default_factory=list)
    relation: RelationType = RelationType.SUM
    formula_components: list[int] = field(default_factory=list)
    """For FORMULA nodes: [gross_parent_idx, depreciation_sibling_idx, ...]"""

    @property
    def depth_prefix(self) -> str:
        return "  " * self.level

    def __repr__(self) -> str:
        rel = f" [{self.relation}]" if self.relation != RelationType.SUM else ""
        return (
            f"HierarchyNode(idx={self.index}, lv={self.level}, "
            f"x0={self.label_x0:.1f}, label={self.label_normalized!r}{rel})"
        )


@dataclass
class HierarchyResult:
    """
    Complete hierarchy for one extracted page.

    Attributes
    ----------
    nodes            – all nodes in document order (flat list)
    roots            – indices of top-level nodes (no parent)
    level_breakpoints – x0 cluster centroids sorted left→right; index = level
    """

    nodes: list[HierarchyNode]
    roots: list[int]
    level_breakpoints: list[float]

    # ------------------------------------------------------------------
    # Tree serialisation
    # ------------------------------------------------------------------

    def to_tree(self) -> list[dict]:
        """Return a nested list-of-dicts (JSON-serialisable)."""
        def _recurse(idx: int) -> dict:
            node = self.nodes[idx]
            return {
                "index": node.index,
                "label": node.label_normalized,
                "level": node.level,
                "relation": node.relation.value,
                "has_values": node.has_values,
                "formula_components": node.formula_components,
                "children": [_recurse(c) for c in node.children_idxs],
            }
        return [_recurse(r) for r in self.roots]

    def print_tree(self) -> None:
        """Print an indented ASCII tree to stdout."""
        def _print(idx: int, prefix: str, is_last: bool) -> None:
            node = self.nodes[idx]
            connector = "└─ " if is_last else "├─ "
            rel_tag = f" [{node.relation}]" if node.relation != RelationType.SUM else ""
            val_tag = "" if node.has_values else " (no values)"
            print(f"{prefix}{connector}{node.label_normalized}{rel_tag}{val_tag}")
            child_prefix = prefix + ("   " if is_last else "│  ")
            for i, child_idx in enumerate(node.children_idxs):
                _print(child_idx, child_prefix, i == len(node.children_idxs) - 1)

        for i, root_idx in enumerate(self.roots):
            node = self.nodes[root_idx]
            rel_tag = f" [{node.relation}]" if node.relation != RelationType.SUM else ""
            print(f"{node.label_normalized}{rel_tag}")
            for j, child_idx in enumerate(node.children_idxs):
                _print(child_idx, "", j == len(node.children_idxs) - 1)

    def flat_labels(self) -> list[str]:
        """Return labels in document order (convenience for tests)."""
        return [n.label_normalized for n in self.nodes]

    def node_for_label(self, label: str) -> Optional[HierarchyNode]:
        """Return first node whose normalized label equals *label*."""
        norm_query = _nfkc_normalize(label)
        for n in self.nodes:
            if n.label_normalized == norm_query or n.label_normalized == label or n.label == label:
                return n
        return None


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_JUNMEI_RE = re.compile(r"（純額）$|[(（]純額[)）]$")
_DEPRECIATION_RE = re.compile(r"減価償却累計額")
_NFKC_SPACE_RE = re.compile(r"(?<=[\u3040-\u9FFF])\s+(?=[\u3040-\u9FFF])")
_CJK_ONLY_RE = re.compile(r"^[\u3000-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF\u30A0-\u30FF\u3040-\u309F（）「」『』【】〔〕…・ー・〜]+$")

_STANDALONE_SUFFIX_RE = re.compile(r"の部$")
_STANDALONE_KEYWORDS = frozenset([
    "資産の部", "負債の部", "純資産の部",
    "流動資産", "固定資産", "有形固定資産", "無形固定資産",
    "投資その他の資産", "流動負債", "固定負債",
    "株主資本", "その他の包括利益累計額",
])


def _nfkc_normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("(", "（").replace(")", "）")
    t = t.replace("\u3000", " ")
    if _CJK_ONLY_RE.match(t.replace(" ", "")):
        t = t.replace(" ", "")
    else:
        t = _NFKC_SPACE_RE.sub("", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_hierarchy(
    rows: list,  # list[ExtractedRow] — avoid circular import by using list
    *,
    x0_tolerance: float = 3.0,
) -> HierarchyResult:
    """
    Build a ``HierarchyResult`` from a flat list of ``ExtractedRow`` objects.

    Parameters
    ----------
    rows          – output of ``extract_page(...).table_rows``
    x0_tolerance  – maximum pt gap to merge into one indentation level

    Returns
    -------
    HierarchyResult
    """
    if not rows:
        return HierarchyResult(nodes=[], roots=[], level_breakpoints=[])

    # 1. Assign levels from x0 clustering
    level_map, breakpoints = _build_level_map(rows, x0_tolerance)

    # 2. Build preliminary flat node list
    nodes: list[HierarchyNode] = []
    for i, row in enumerate(rows):
        norm = _nfkc_normalize(row.label)
        has_values = bool(row.values)
        level = level_map.get(_nearest_breakpoint(row.label_x0, breakpoints), 0)
        relation = _semantic_relation(norm, has_values)
        nodes.append(
            HierarchyNode(
                index=i,
                label=row.label,
                label_normalized=norm,
                label_x0=row.label_x0,
                level=level,
                has_values=has_values,
                relation=relation,
            )
        )

    # 3. Assign parents via monotone stack
    _assign_parents(nodes)

    # 4. Detect 純額 formula nodes
    _detect_formula_nodes(nodes)

    # 5. Apply semantic rules for Japanese financial statements
    _apply_semantic_adjustments(nodes)

    # 6. Collect root indices
    roots = [n.index for n in nodes if n.parent_idx is None]

    return HierarchyResult(nodes=nodes, roots=roots, level_breakpoints=breakpoints)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_level_map(rows: list, tol: float) -> tuple[dict[float, int], list[float]]:
    """
    Cluster all label_x0 values and return (centroid→level, sorted centroids).

    Uses single-linkage clustering: a new x0 within *tol* of the last cluster
    is merged into it; otherwise it starts a new cluster.
    """
    x0s = sorted(set(r.label_x0 for r in rows if r.label))
    if not x0s:
        return {}, []

    clusters: list[list[float]] = [[x0s[0]]]
    for x in x0s[1:]:
        if x - clusters[-1][-1] <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    centroids = [median(c) for c in clusters]
    centroid_to_level = {c: i for i, c in enumerate(centroids)}
    return centroid_to_level, centroids


def _nearest_breakpoint(x0: float, breakpoints: list[float]) -> float:
    """Return the breakpoint centroid nearest to *x0*."""
    if not breakpoints:
        return 0.0
    return min(breakpoints, key=lambda bp: abs(bp - x0))


def _semantic_relation(label_normalized: str, has_values: bool) -> RelationType:
    """
    Assign a preliminary RelationType based on label content alone.
    Formula nodes are identified later (they need sibling context).
    """
    if _STANDALONE_SUFFIX_RE.search(label_normalized):
        return RelationType.STANDALONE
    if label_normalized in _STANDALONE_KEYWORDS and not has_values:
        return RelationType.STANDALONE
    return RelationType.SUM


def _assign_parents(nodes: list[HierarchyNode]) -> None:
    """
    In-place parent assignment using a monotone depth-stack (top→bottom).

    Invariant: stack always contains ancestors in strictly increasing level
    order.  When we encounter a node at level L, we pop all stack entries
    with level ≥ L; the new top is the parent.
    """
    stack: list[HierarchyNode] = []   # open ancestors, deepest last
    for node in nodes:
        # Pop nodes at the same or deeper level
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            node.parent_idx = stack[-1].index
            stack[-1].children_idxs.append(node.index)
        stack.append(node)


def _detect_formula_nodes(nodes: list[HierarchyNode]) -> None:
    """
    In-place: annotate 純額 nodes with RelationType.FORMULA and populate
    formula_components.

    Pattern:
        parent (gross amount, e.g. 建物及び構築物)
          sibling 減価償却累計額  ← component
          THIS NODE (純額)       ← formula = parent + sib(depreciation)
    """
    for node in nodes:
        if not _JUNMEI_RE.search(node.label_normalized):
            continue
        if node.parent_idx is None:
            continue

        parent = nodes[node.parent_idx]
        # Siblings that appear BEFORE this node and contain 減価償却累計額
        dep_siblings = [
            nodes[j]
            for j in parent.children_idxs
            if j < node.index and _DEPRECIATION_RE.search(nodes[j].label_normalized)
        ]
        if not dep_siblings:
            continue  # no matching sibling → not a formula node

        node.relation = RelationType.FORMULA
        # Components: [gross parent, depreciation sibling(s)]
        node.formula_components = [node.parent_idx] + [s.index for s in dep_siblings]


def _apply_semantic_adjustments(nodes: list[HierarchyNode]) -> None:
    """
    Apply Japanese financial statement semantic hierarchy rules:
    - 純額 rows: level = gross.level, parent = gross.parent, formula_components = [gross, dep]
    - 流動資産合計: level = 1, parent = 資産の部 (if present), formula_components = leaf items
    - 有形固定資産合計: level = 2, parent = 固定資産, formula_components = [pure net rows + standalone items]
    - 無形固定資産合計: level = 2, parent = 固定資産, formula_components = items under 無形固定資産
    - 投資その他の資産合計: level = 2, parent = 固定資産, formula_components = items under 投資その他の資産
    - 固定資産合計: level = 1, parent = 資産の部, formula_components = [有形固定資産合計, 無形固定資産合計, 投資その他の資産合計]
    - 資産合計: level = 0, parent = None, formula_components = [流動資産合計, 固定資産合計]
    """
    sh_map = {n.label_normalized: n for n in nodes}

    # 1. Net rows
    for n in nodes:
        if "純額" in n.label_normalized and n.parent_idx is not None:
            gross = nodes[n.parent_idx]
            n.relation = RelationType.FORMULA
            dep_siblings = [
                nodes[j]
                for j in gross.children_idxs
                if j < n.index and _DEPRECIATION_RE.search(nodes[j].label_normalized)
            ]
            if dep_siblings:
                n.formula_components = [gross.index] + [s.index for s in dep_siblings]
            n.level = gross.level
            n.parent_idx = gross.parent_idx

    # 2. Subtotal semantics & formulas
    for n in nodes:
        if n.label_normalized == "流動資産合計":
            if "資産の部" in sh_map:
                n.level = 1
                n.parent_idx = sh_map["資産の部"].index
            start_idx = sh_map["流動資産"].index + 1 if "流動資産" in sh_map else 0
            n.formula_components = list(range(start_idx, n.index))
        elif n.label_normalized == "有形固定資産合計":
            if "固定資産" in sh_map:
                n.level = 2
                n.parent_idx = sh_map["固定資産"].index
            start_idx = sh_map["有形固定資産"].index + 1 if "有形固定資産" in sh_map else 0
            comp = []
            for j in range(start_idx, n.index):
                if "純額" in nodes[j].label_normalized or nodes[j].label_normalized in ("土地", "建設仮勘定"):
                    comp.append(j)
            n.formula_components = comp
        elif n.label_normalized == "無形固定資産合計":
            if "固定資産" in sh_map:
                n.level = 2
                n.parent_idx = sh_map["固定資産"].index
            start_idx = sh_map["無形固定資産"].index + 1 if "無形固定資産" in sh_map else 0
            n.formula_components = list(range(start_idx, n.index))
        elif n.label_normalized == "投資その他の資産合計":
            if "固定資産" in sh_map:
                n.level = 2
                n.parent_idx = sh_map["固定資産"].index
            start_idx = sh_map["投資その他の資産"].index + 1 if "投資その他の資産" in sh_map else 0
            n.formula_components = list(range(start_idx, n.index))
        elif n.label_normalized == "固定資産合計":
            if "資産の部" in sh_map:
                n.level = 1
                n.parent_idx = sh_map["資産の部"].index
            comps = []
            for sub_name in ("有形固定資産合計", "無形固定資産合計", "投資その他の資産合計"):
                if sub_name in sh_map:
                    comps.append(sh_map[sub_name].index)
            if comps:
                n.formula_components = comps
        elif n.label_normalized in ("資産合計", "負債純資産合計", "負債及び純資産合計"):
            n.level = 0
            n.parent_idx = None
            if n.label_normalized == "資産合計":
                comps = []
                for sub_name in ("流動資産合計", "固定資産合計"):
                    if sub_name in sh_map:
                        comps.append(sh_map[sub_name].index)
                if comps:
                    n.formula_components = comps

    # Re-link children_idxs based on updated parent_idx
    for n in nodes:
        n.children_idxs = []
    for n in nodes:
        if n.parent_idx is not None and n.parent_idx < len(nodes):
            nodes[n.parent_idx].children_idxs.append(n.index)
