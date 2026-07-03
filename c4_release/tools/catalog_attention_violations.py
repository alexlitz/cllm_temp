"""Build the corpus-wide attention violation catalog.

Walks every ``AttentionHeadIR`` produced by ``all_core_ops()`` (materializing
``compiler_ir_factory``-only ops via a ``LayerCompiler``-provided
``dim_positions`` map), runs ``verify_attention_head`` solo first and then
with the full op list as cross-op competition, and emits a Markdown report.

Run from c4_release/:

    python tools/catalog_attention_violations.py

Writes to c4_release/.agent-logs/attention_violations_catalog_2026_06_01.md
(``.agent-logs/`` is gitignored via ``.git/info/exclude``).
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make c4_release importable regardless of where this is invoked from.
HERE = Path(__file__).resolve()
REPO = HERE.parents[1]  # c4_release/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neural_vm.dim_registry import build_default_registry  # noqa: E402
from neural_vm.verification.attention_verifier import (  # noqa: E402
    verify_attention_head,
)
from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops  # noqa: E402
from neural_vm.unified_compiler.ops.shared import (  # noqa: E402
    declare_setdim_compat_dims,
)
from neural_vm.unified_compiler.layer_compiler import LayerCompiler  # noqa: E402
from neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402


OUT_DIR = REPO / ".agent-logs"
OUT_PATH = OUT_DIR / "attention_violations_catalog_2026_06_01.md"


def build_dim_positions() -> Dict[str, int]:
    """Compile a default layout and return its ``dim_positions``.

    Matches the shape consumed by ``compiler_ir_factory(dim_positions, HD)``
    used throughout ``c4_release/neural_vm/unified_compiler/ops/``.
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    layout = compiler.compile()
    return dict(layout.dim_positions)


def materialize_ir(op, dim_positions, HD: int = 64):
    """Return the op's declarative IR (preferring `compiler_ir`, else factory)."""
    ir = getattr(op, "compiler_ir", None)
    if ir is not None:
        return ir
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is None:
        return None
    try:
        return factory(dim_positions, HD)
    except Exception:
        return None


def collect_heads(ops, dim_positions) -> List[Tuple[str, int, int, Any]]:
    """Return [(op_name, layer_idx, head_idx, head), ...]."""
    out: List[Tuple[str, int, int, Any]] = []
    for op in ops:
        ir = materialize_ir(op, dim_positions)
        if ir is None:
            continue
        if not hasattr(ir, "layers"):
            continue
        for layer_idx, layer in enumerate(ir.layers):
            attention = getattr(layer, "attention", None)
            if attention is None:
                continue
            rules = getattr(attention, "rules", None) or []
            for head_idx, head in enumerate(rules):
                out.append((op.name, layer_idx, head_idx, head))
    return out


class _IROnlyOp:
    """Tiny shim wrapping an already-materialized IR for cross-op competition.

    ``verify_attention_head`` walks ``op.compiler_ir`` directly via
    ``_collect_heads_from_op``, so we can supply a list of these and avoid
    re-materializing factories for every head we verify.
    """

    def __init__(self, name: str, ir):
        self.name = name
        self.compiler_ir = ir


def build_materialized_op_list(ops, dim_positions) -> List[_IROnlyOp]:
    out: List[_IROnlyOp] = []
    for op in ops:
        ir = materialize_ir(op, dim_positions)
        if ir is None:
            continue
        out.append(_IROnlyOp(op.name, ir))
    return out


def head_label(op_name: str, layer_idx: int, head_idx: int, head) -> str:
    spec = getattr(head, "spec", None)
    spec_idx = int(getattr(spec, "head_idx", -1)) if spec is not None else -1
    name = getattr(head, "name", None) or ""
    suffix = f" ({name})" if name else ""
    return f"{op_name} L{layer_idx} h{head_idx}/spec_idx={spec_idx}{suffix}"


def main() -> int:
    print("[catalog] building registry + default layout dim_positions...")
    registry = build_default_registry()
    dim_positions = build_dim_positions()
    print(f"[catalog] dim_positions: {len(dim_positions)} entries")

    ops = all_core_ops()
    print(f"[catalog] enumerating heads across {len(ops)} ops...")
    heads = collect_heads(ops, dim_positions)
    print(f"[catalog] total heads: {len(heads)}")

    # Pre-materialize IRs for cross-op competition pass.
    print("[catalog] materializing IRs for cross-op competition list...")
    cross_ops = build_materialized_op_list(ops, dim_positions)
    print(f"[catalog] cross-op list size: {len(cross_ops)} ops with IRs")

    # -- Solo pass (no competition) -----------------------------------------
    print("[catalog] solo pass (no cross-op competition)...")
    t0 = time.time()
    solo_results: List[Tuple[str, int, int, Any, List[Dict[str, Any]]]] = []
    for op_name, layer_idx, head_idx, head in heads:
        try:
            issues = verify_attention_head(head, registry)
        except Exception as e:
            issues = [{"kind": "exception", "reason": repr(e)}]
        solo_results.append((op_name, layer_idx, head_idx, head, issues))
    solo_runtime = time.time() - t0
    print(f"[catalog] solo pass took {solo_runtime:.1f}s")

    # -- Cross-op pass ------------------------------------------------------
    print("[catalog] cross-op pass with ops_for_competition...")
    t0 = time.time()
    cross_results: List[Tuple[str, int, int, Any, List[Dict[str, Any]]]] = []
    for op_name, layer_idx, head_idx, head in heads:
        try:
            issues = verify_attention_head(
                head, registry, ops_for_competition=cross_ops,
            )
        except Exception as e:
            issues = [{"kind": "exception", "reason": repr(e)}]
        cross_results.append((op_name, layer_idx, head_idx, head, issues))
    cross_runtime = time.time() - t0
    print(f"[catalog] cross-op pass took {cross_runtime:.1f}s")

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    kind_totals_solo: Counter = Counter()
    kind_totals_cross: Counter = Counter()
    per_op_issue_count_solo: Counter = Counter()
    per_op_issue_count_cross: Counter = Counter()
    per_op_head_count: Counter = Counter()
    per_head_solo: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    per_head_cross: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}

    for op_name, layer_idx, head_idx, head, issues in solo_results:
        per_op_head_count[op_name] += 1
        per_op_issue_count_solo[op_name] += len(issues)
        per_head_solo[(op_name, layer_idx, head_idx)] = issues
        for issue in issues:
            kind_totals_solo[issue.get("kind", "?")] += 1

    for op_name, layer_idx, head_idx, head, issues in cross_results:
        per_op_issue_count_cross[op_name] += len(issues)
        per_head_cross[(op_name, layer_idx, head_idx)] = issues
        for issue in issues:
            kind_totals_cross[issue.get("kind", "?")] += 1

    total_solo = sum(kind_totals_solo.values())
    total_cross = sum(kind_totals_cross.values())

    # Top-N problem heads (by cross-op count, since that's the strictest).
    heads_by_violation_count = sorted(
        [
            (
                (op_name, layer_idx, head_idx),
                len(issues),
                issues,
            )
            for (op_name, layer_idx, head_idx, head, issues) in cross_results
            if issues
        ],
        key=lambda x: -x[1],
    )

    top_problem = heads_by_violation_count[:20]

    # Per-op tally (cross + solo).
    all_op_names = sorted(per_op_head_count.keys())
    clean_ops: List[str] = []
    for name in all_op_names:
        if (
            per_op_issue_count_solo[name] == 0
            and per_op_issue_count_cross[name] == 0
        ):
            clean_ops.append(name)

    # ------------------------------------------------------------------
    # Write the catalog
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Attention Violation Catalog (2026-06-01)")
    lines.append("")
    lines.append(
        "Corpus-wide ``verify_attention_head`` run across every "
        "``AttentionHeadIR`` exposed by ``all_core_ops()``. Heads with "
        "factory-only IR are materialized through the default ``LayerCompiler`` "
        "layout's ``dim_positions``. ``compiler_ir`` ops are used as-is."
    )
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- ops in catalog: {len(ops)}")
    lines.append(f"- ops with IR (compiler_ir or factory): {len(cross_ops)}")
    lines.append(f"- total heads enumerated: {len(heads)}")
    lines.append(f"- solo pass runtime: {solo_runtime:.1f}s")
    lines.append(f"- cross-op pass runtime: {cross_runtime:.1f}s")
    lines.append(f"- total violations (solo): {total_solo}")
    lines.append(f"- total violations (cross-op): {total_cross}")
    lines.append("")
    lines.append("## Per-issue-kind totals")
    lines.append("")
    lines.append("| kind | solo | cross-op |")
    lines.append("|------|------|----------|")
    kinds_seen = sorted(
        set(kind_totals_solo.keys()) | set(kind_totals_cross.keys())
    )
    for k in kinds_seen:
        lines.append(
            f"| `{k}` | {kind_totals_solo.get(k, 0)} | "
            f"{kind_totals_cross.get(k, 0)} |"
        )
    lines.append("")

    lines.append("## Per-op summary")
    lines.append("")
    lines.append(
        "| op | heads | solo issues | cross-op issues |"
    )
    lines.append("|----|------:|------------:|----------------:|")
    op_rows = sorted(
        all_op_names,
        key=lambda n: -per_op_issue_count_cross[n],
    )
    for name in op_rows:
        lines.append(
            f"| {name} | {per_op_head_count[name]} | "
            f"{per_op_issue_count_solo[name]} | "
            f"{per_op_issue_count_cross[name]} |"
        )
    lines.append("")

    lines.append("## Clean ops (zero violations across all their heads)")
    lines.append("")
    if not clean_ops:
        lines.append("(none)")
    else:
        for name in clean_ops:
            lines.append(
                f"- {name} ({per_op_head_count[name]} heads)"
            )
    lines.append("")

    lines.append("## Top 20 problem heads (by cross-op violation count)")
    lines.append("")
    lines.append(
        "| rank | op | layer/head | total | dominant kind "
        "| top competitor |"
    )
    lines.append(
        "|-----:|----|------------|------:|---------------|----------------|"
    )
    for rank, ((op_name, layer_idx, head_idx), count, issues) in enumerate(
        top_problem, start=1
    ):
        kind_counter: Counter = Counter()
        top_comp = None
        for issue in issues:
            kind_counter[issue.get("kind", "?")] += 1
            if top_comp is None:
                tc = issue.get("top_competitor")
                if tc:
                    top_comp = tc
        dominant_kind = kind_counter.most_common(1)[0][0]
        lines.append(
            f"| {rank} | {op_name} | L{layer_idx} h{head_idx} "
            f"| {count} | {dominant_kind} | {top_comp or '-'} |"
        )
    lines.append("")

    lines.append("## Per-head detail (top 20)")
    lines.append("")
    for rank, ((op_name, layer_idx, head_idx), count, issues) in enumerate(
        top_problem, start=1
    ):
        # Pull the head for naming.
        head_obj = None
        for op_name_h, li, hi, head in heads:
            if op_name_h == op_name and li == layer_idx and hi == head_idx:
                head_obj = head
                break
        label = head_label(op_name, layer_idx, head_idx, head_obj)
        lines.append(f"### {rank}. {label}")
        lines.append("")
        lines.append(f"- total issues (cross-op): {count}")
        kind_counter = Counter(i.get("kind", "?") for i in issues)
        lines.append(
            "- kind breakdown: "
            + ", ".join(f"`{k}`={v}" for k, v in kind_counter.most_common())
        )
        # Sample up to 5 issues.
        lines.append("")
        lines.append("Sample issues:")
        lines.append("")
        for issue in issues[:5]:
            kind = issue.get("kind", "?")
            outdim = issue.get("output_dim", "?")
            mymag = issue.get("my_magnitude")
            comp = issue.get("competing_max")
            top_comp = issue.get("top_competitor")
            top_kind = issue.get("top_competitor_kind", "attn")
            scope = issue.get("my_effective_scope")
            comp_scope = issue.get("competitor_effective_scope")
            line = (
                f"- `{kind}` out=`{outdim}`"
            )
            if mymag is not None:
                line += f" my={mymag:.3g} vs competing={comp:.3g}"
            if top_comp:
                line += f" top=`{top_comp}` ({top_kind})"
            lines.append(line)
            if scope is not None:
                lines.append(f"  - my_scope={scope}")
            if comp_scope is not None:
                lines.append(f"  - competitor_scope={comp_scope}")
        lines.append("")

    # ------------------------------------------------------------------
    # Recommended next-targets (highest-impact heads to fix first).
    # ------------------------------------------------------------------
    lines.append("## Recommended next-target heads")
    lines.append("")
    lines.append(
        "Ordered by cross-op violation count. Heads that already fail "
        "solo are tagged ``[solo]`` — their bug isn't competition, it's "
        "a missing V path, unresolved output dim, etc."
    )
    lines.append("")
    for rank, ((op_name, layer_idx, head_idx), count, issues) in enumerate(
        top_problem[:10], start=1
    ):
        solo_n = len(per_head_solo.get((op_name, layer_idx, head_idx), []))
        solo_tag = " [solo]" if solo_n > 0 else ""
        head_obj = None
        for op_name_h, li, hi, head in heads:
            if op_name_h == op_name and li == layer_idx and hi == head_idx:
                head_obj = head
                break
        label = head_label(op_name, layer_idx, head_idx, head_obj)
        lines.append(
            f"{rank}. {label} — {count} cross-op issues (solo={solo_n}){solo_tag}"
        )
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"[catalog] wrote {OUT_PATH}")
    print(f"[catalog] heads={len(heads)} solo_violations={total_solo} "
          f"cross_violations={total_cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
