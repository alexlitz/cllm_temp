"""Phase 6 Wave 5B sweep: ``compare_symbolic_to_lowered_attn`` corpus run.

Walks every op in ``all_core_ops()`` whose declarative IR (``compiler_ir`` or
``compiler_ir_factory``) carries one or more ``AttentionHeadIR`` rules, runs
``compare_symbolic_to_lowered_attn`` per head, and tallies the result by the
failure categories defined in :mod:`neural_vm.unified_compiler.ir`:

* ``declaration_semantics`` — head spec references slots/dims that cannot be
  lowered into a ``PureAttention`` of the resolved shape (a real bug — the IR
  itself is malformed for the inferred shape).
* ``lowering`` — the lowered ``W_q``/``W_k``/``W_v``/``W_o`` matrices do not
  match what ``Primitives.generate_attention_head`` should produce (real bug).
* ``weight_output_mismatch`` — matrices match but ``PureAttention.forward``
  disagrees with the symbolic execution on at least one
  ``(query_pos, output_dim)``. Often surfaces shared-residual interference or
  signed-projection collapse rather than a head bug; tracked separately.

Run from c4_release/:

    python tools/sweep_compare_attn.py

Outputs (both gitignored via ``.agent-logs/``):

* ``c4_release/.agent-logs/sweep_compare_attn_phase6_wave5b.md``
* ``c4_release/.agent-logs/sweep_compare_attn_phase6_wave5b.json``

The tool is read-only: it never edits op files and never mutates the corpus.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make c4_release importable regardless of where this is invoked from.
# Two paths are needed:
#   * ``REPO`` (= ``.../c4_release``) for short ``from neural_vm....`` imports
#     used by this file and the rest of ``tools/``.
#   * ``REPO_PARENT`` (= ``.../misc/c4_release``) so that fully-qualified
#     ``from c4_release.neural_vm.base_layers import PureAttention`` lazy
#     imports inside ``compare_symbolic_to_lowered_attn`` resolve. Without
#     this entry every comparison raises ``ModuleNotFoundError`` and the
#     sweep degenerates into 79 exceptions.
HERE = Path(__file__).resolve()
REPO = HERE.parents[1]  # c4_release/
REPO_PARENT = HERE.parents[2]  # parent containing the c4_release package
for path in (REPO_PARENT, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_vm.unified_compiler.ir import (  # noqa: E402
    AttentionHeadIR,
    AttentionOp,
    CompilerIR,
    compare_symbolic_to_lowered_attn,
)
from neural_vm.unified_compiler.layer_compiler import LayerCompiler  # noqa: E402
from neural_vm.unified_compiler.ops.all_core_ops import (  # noqa: E402
    all_core_ops,
)
from neural_vm.unified_compiler.ops.shared import (  # noqa: E402
    declare_setdim_compat_dims,
)


OUT_DIR = REPO / ".agent-logs"
MD_PATH = OUT_DIR / "sweep_compare_attn_phase6_wave5b.md"
JSON_PATH = OUT_DIR / "sweep_compare_attn_phase6_wave5b.json"

HEAD_DIM = 64

# Bucket the (kind, ok?) tuple into the four categories the report tallies:
#   * ok                       — report.ok is True
#   * declaration_semantics    — primary failure is declaration_semantics
#   * lowering                 — primary failure is lowering
#   * weight_output_mismatch   — primary failure is weight_output_mismatch
#
# Categories follow the order in :class:`AttentionComparisonReport`. A head
# that hits declaration_semantics is automatically a "real bug" because the
# spec can't be lowered for the inferred shape. lowering failures are also
# real bugs (matrices don't match the IR). weight_output_mismatch is reported
# separately because it often reflects shared-residual interference rather
# than a head-local defect.
REAL_BUG_KINDS = {"declaration_semantics", "lowering"}


# Flag combinations to walk. Default-mode ops win on duplicate names; the
# flags-on pass picks up any additional attention heads that only register
# when conversational-I/O, tool-calling, or neural-I/O THINK are enabled.
# Mirrors ``tools/sweep_compare_ffn.py`` so the two reports cover the same
# corpus.
_FLAG_MODES: List[Tuple[str, Dict[str, bool]]] = [
    ("default", {}),
    ("all_flags_on", {
        "enable_conversational_io": True,
        "enable_tool_calling": True,
        "enable_neural_io_think_protocol": True,
    }),
]


def _op_has_ir(op) -> bool:
    return (
        getattr(op, "compiler_ir", None) is not None
        or getattr(op, "compiler_ir_factory", None) is not None
    )


def _enumerate_ops() -> List[Tuple[Any, List[str]]]:
    """Return ``[(op, enabled_flags), ...]`` deduped by ``op.name``.

    Default-mode op wins on duplicate names UNLESS it has no IR and the
    flags-on variant does. This matters for flag-gated ops like
    ``format_pointer_extraction``, which register in default mode with
    ``compiler_ir = compiler_ir_factory = None`` (the bake is a no-op when
    the flag is off) but expose a real IR once
    ``enable_conversational_io=True``. Without the IR-aware swap we would
    miss every flag-gated attention head.
    """
    seen: Dict[str, Tuple[Any, List[str]]] = {}
    default_names: set = set()
    for label, kwargs in _FLAG_MODES:
        ops = all_core_ops(**kwargs)
        if label == "default":
            default_names = {getattr(op, "name", repr(op)) for op in ops}
        for op in ops:
            name = getattr(op, "name", repr(op))
            flags = [k for k, v in kwargs.items() if v]
            new_has_ir = _op_has_ir(op)
            if name in seen:
                existing_op, existing_flags = seen[name]
                if _op_has_ir(existing_op) or not new_has_ir:
                    continue
                # Swap: the flag-on variant has IR; the default-mode placeholder
                # didn't. Preserve the ``requires_flag`` tag.
                tagged = (
                    sorted(set(flags) | {"requires_flag"})
                    if name not in default_names
                    else flags
                )
                seen[name] = (op, tagged)
                continue
            if name not in default_names:
                flags = sorted(set(flags) | {"requires_flag"})
            seen[name] = (op, flags)
    return list(seen.values())


def build_dim_positions() -> Dict[str, int]:
    """Compile a default layout and return its ``dim_positions``.

    Mirrors :func:`tools.catalog_attention_violations.build_dim_positions`:
    feed every op in ``all_core_ops()`` into a fresh ``LayerCompiler`` and
    use the resolved layout's ``dim_positions``. This is the same input the
    op's own ``compiler_ir_factory(dim_positions, head_dim)`` would receive
    at bake time. We use the default-mode op list to fix the layout; the
    flags-on heads still resolve their dim references through this map
    because the default layout pre-allocates the union of all dim names
    declared by every flag-gated op (the dims are declared at compiler-set
    time, not at bake time).
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    layout = compiler.compile()
    return dict(layout.dim_positions)


def materialize_ir(op, dim_positions: Dict[str, int]) -> Optional[CompilerIR]:
    """Return the op's declarative IR (preferring ``compiler_ir``, else factory)."""
    ir = getattr(op, "compiler_ir", None)
    if ir is not None:
        return ir
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is None:
        return None
    try:
        return factory(dim_positions, HEAD_DIM)
    except Exception:
        return None


def collect_attn_heads(
    ops_with_flags: List[Tuple[Any, List[str]]],
    dim_positions: Dict[str, int],
) -> List[Tuple[str, int, int, AttentionHeadIR, List[str]]]:
    """Return ``[(op_name, layer_idx, head_pos, head_ir, flags), ...]``.

    ``head_pos`` is the position in the layer's ``attention.rules`` tuple, NOT
    the head's ``spec.head_idx`` (which is reported separately in the per-head
    detail block). Ops without an IR, without any attention layer, or whose IR
    factory raises are silently skipped — the catalog tool follows the same
    convention.

    ``flags`` is the list of compiler flags the op requires (empty for
    default-mode ops). It's carried through into the per-head row so the
    final report can mark a head as flag-gated.
    """
    out: List[Tuple[str, int, int, AttentionHeadIR, List[str]]] = []
    for op, flags in ops_with_flags:
        ir = materialize_ir(op, dim_positions)
        if ir is None or not hasattr(ir, "layers"):
            continue
        for layer_idx, layer in enumerate(ir.layers):
            attention: Optional[AttentionOp] = getattr(layer, "attention", None)
            if attention is None:
                continue
            rules = getattr(attention, "rules", None) or []
            for head_pos, head in enumerate(rules):
                if isinstance(head, AttentionHeadIR):
                    out.append((op.name, layer_idx, head_pos, head, flags))
    return out


def _resolve_compatible_shape(head: AttentionHeadIR) -> Tuple[int, int, int]:
    """Return ``(head_dim, num_heads, model_dim)`` for one head.

    The IR's ``_resolve_attn_shape`` happily returns a ``model_dim`` that is
    not a multiple of ``HEAD_DIM`` when the head references residual dims
    beyond ``num_heads * HEAD_DIM`` (the spec is allowed to live in the
    layer's full residual stream, not just its own head slot). PureAttention
    then crashes inside ``forward`` because ``dim // num_heads != HEAD_DIM``.

    Two more wrinkles come up in production specs:

    * Some L10/L13 heads use slot indices > 64 (e.g.
      ``layer10_byte_passthrough_bake.head_1`` writes slot 81). In production
      these heads run inside a wider attention with ``HD >= max_slot + 1``;
      ``compare_symbolic_to_lowered_attn`` with the default ``HD=64`` would
      reject the spec at the declaration-semantics gate. We instead pick
      ``head_dim = max(HEAD_DIM, max_slot + 1)`` per head so the test
      matches the production-shape contract.
    * ``num_heads`` must be large enough to cover both the spec's own
      ``head_idx`` and ``ceil((max_residual_dim + 1) / head_dim)`` so every
      Q/K/V/O write lives inside the resolved matrices.

    The returned ``model_dim = num_heads * head_dim`` keeps PureAttention's
    invariant intact.
    """
    spec = head.spec
    max_slot = -1
    for write in spec.q + spec.k + spec.v + spec.o:
        max_slot = max(max_slot, int(write.slot))
    head_dim = max(HEAD_DIM, max_slot + 1) if max_slot >= 0 else HEAD_DIM

    max_residual_dim = -1
    for write in spec.q + spec.k + spec.v:
        max_residual_dim = max(max_residual_dim, int(write.dim))
    for write in spec.o:
        max_residual_dim = max(max_residual_dim, int(write.out_dim))

    head_count_from_dim = (
        (max_residual_dim + head_dim) // head_dim
        if max_residual_dim >= 0
        else 1
    )
    num_heads = max(int(spec.head_idx) + 1, head_count_from_dim, 1)
    return head_dim, num_heads, num_heads * head_dim


def _classify_report(report) -> str:
    """Bucket one ``AttentionComparisonReport`` into the sweep's category."""
    if report.ok:
        return "ok"
    kind = report.primary_failure_kind
    if kind in REAL_BUG_KINDS:
        return kind
    return "weight_output_mismatch"


def _first_failure_text(report) -> Optional[str]:
    """One-line triage hint for a failing report (kind + message)."""
    if report.ok or not report.issues:
        return None
    issue = report.issues[0]
    return f"[{issue.kind}] {issue.message}"


def run_sweep(
    heads: List[Tuple[str, int, int, AttentionHeadIR, List[str]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run ``compare_symbolic_to_lowered_attn`` per head; collect per-head rows."""

    rows: List[Dict[str, Any]] = []
    totals = Counter()
    t0 = time.time()
    for i, (op_name, layer_idx, head_pos, head, flags) in enumerate(heads):
        spec = head.spec
        head_idx = int(spec.head_idx)
        label = head.name or f"head_{head_pos}"
        row: Dict[str, Any] = {
            "op_name": op_name,
            "layer_idx": layer_idx,
            "head_pos": head_pos,
            "head_idx": head_idx,
            "name": label,
            "flags": list(flags),
            "n_q": len(spec.q),
            "n_k": len(spec.k),
            "n_v": len(spec.v),
            "n_o": len(spec.o),
        }
        head_dim, num_heads, model_dim = _resolve_compatible_shape(head)
        row["resolved_head_dim"] = head_dim
        row["resolved_num_heads_override"] = num_heads
        row["resolved_dim_override"] = model_dim
        t_head = time.time()
        try:
            # ``compare_symbolic_to_lowered_attn`` wraps a single
            # ``AttentionHeadIR`` into a fresh single-head CompilerIR at
            # layer_idx=0, so we never leak the original layer_idx into the
            # comparison; the head is validated in isolation. We pin
            # ``num_heads`` / ``dim`` because the IR's auto-shape resolver
            # returns a ``model_dim`` that PureAttention's strict
            # ``head_dim = dim // num_heads`` invariant rejects whenever the
            # spec touches residual dims past ``num_heads * HEAD_DIM`` (which
            # is most production heads). The per-head ``head_dim`` is bumped
            # past the default ``HEAD_DIM`` whenever the spec's max slot
            # demands it, matching the production attn module's HD.
            report = compare_symbolic_to_lowered_attn(
                head,
                head_dim,
                num_heads=num_heads,
                dim=model_dim,
            )
        except Exception as exc:
            row["category"] = "exception"
            row["ok"] = False
            row["primary_kind"] = "exception"
            row["first_failure"] = f"{type(exc).__name__}: {exc}"
            row["n_issues"] = 0
            row["failure_kinds"] = []
            row["runtime_s"] = time.time() - t_head
            totals["exception"] += 1
            rows.append(row)
            continue

        category = _classify_report(report)
        row["category"] = category
        row["ok"] = bool(report.ok)
        row["primary_kind"] = report.primary_failure_kind
        row["first_failure"] = _first_failure_text(report)
        row["n_issues"] = len(report.issues)
        row["failure_kinds"] = list(report.failure_kinds)
        row["resolved_num_heads"] = int(report.num_heads)
        row["runtime_s"] = time.time() - t_head
        totals[category] += 1
        rows.append(row)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"[sweep] {i + 1}/{len(heads)} heads done "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )
    return rows, dict(totals)


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-op tallies + the four headline counts the brief asks for."""
    per_op_total = Counter()
    per_op_ok = Counter()
    per_op_decl = Counter()
    per_op_low = Counter()
    per_op_mismatch = Counter()
    per_op_exception = Counter()

    real_bug_rows: List[Dict[str, Any]] = []
    mismatch_only_rows: List[Dict[str, Any]] = []

    for row in rows:
        op = row["op_name"]
        per_op_total[op] += 1
        cat = row["category"]
        if cat == "ok":
            per_op_ok[op] += 1
        elif cat == "declaration_semantics":
            per_op_decl[op] += 1
            real_bug_rows.append(row)
        elif cat == "lowering":
            per_op_low[op] += 1
            real_bug_rows.append(row)
        elif cat == "weight_output_mismatch":
            per_op_mismatch[op] += 1
            mismatch_only_rows.append(row)
        elif cat == "exception":
            per_op_exception[op] += 1
            # An exception during comparison is treated as a real bug for
            # the headline tally: the head's symbolic semantics cannot be
            # validated and that needs triage just like decl/lowering.
            real_bug_rows.append(row)

    return {
        "total_heads": len(rows),
        "fully_clean": sum(per_op_ok.values()),
        "mismatch_only": sum(per_op_mismatch.values()),
        "real_bugs": len(real_bug_rows),
        "per_op_total": dict(per_op_total),
        "per_op_ok": dict(per_op_ok),
        "per_op_declaration_semantics": dict(per_op_decl),
        "per_op_lowering": dict(per_op_low),
        "per_op_weight_output_mismatch": dict(per_op_mismatch),
        "per_op_exception": dict(per_op_exception),
        "real_bug_rows": real_bug_rows,
        "mismatch_only_rows": mismatch_only_rows,
    }


def _fmt_row_label(row: Dict[str, Any]) -> str:
    return (
        f"{row['op_name']} L{row['layer_idx']} "
        f"head_pos={row['head_pos']} spec_idx={row['head_idx']} "
        f"({row['name']})"
    )


def write_reports(
    rows: List[Dict[str, Any]],
    totals: Dict[str, int],
    agg: Dict[str, Any],
    *,
    n_ops_total: int,
    n_ops_with_attn_ir: int,
    sweep_runtime_s: float,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------- JSON (machine-readable) -------
    json_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "head_dim": HEAD_DIM,
        "n_ops_total": n_ops_total,
        "n_ops_with_attn_ir": n_ops_with_attn_ir,
        "sweep_runtime_s": round(sweep_runtime_s, 2),
        "totals": totals,
        "aggregate": {
            "total_heads": agg["total_heads"],
            "fully_clean": agg["fully_clean"],
            "mismatch_only": agg["mismatch_only"],
            "real_bugs": agg["real_bugs"],
        },
        "per_op": {
            "total": agg["per_op_total"],
            "ok": agg["per_op_ok"],
            "declaration_semantics": agg["per_op_declaration_semantics"],
            "lowering": agg["per_op_lowering"],
            "weight_output_mismatch": agg["per_op_weight_output_mismatch"],
            "exception": agg["per_op_exception"],
        },
        "per_head": rows,
    }
    JSON_PATH.write_text(json.dumps(json_payload, indent=2, sort_keys=False))

    # ------- Markdown (human-readable) -------
    lines: List[str] = []
    lines.append("# Phase 6 Wave 5B — sweep_compare_attn report")
    lines.append("")
    lines.append(
        "Corpus-wide ``compare_symbolic_to_lowered_attn`` run across every "
        "``AttentionHeadIR`` exposed by ``all_core_ops()``. Heads with "
        "factory-only IR are materialized through the default ``LayerCompiler`` "
        "layout's ``dim_positions``; ``compiler_ir`` ops are used as-is. The "
        "tool is read-only — see ``tools/sweep_compare_attn.py``."
    )
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- generated at: {json_payload['generated_at']}")
    lines.append(f"- head_dim: {HEAD_DIM}")
    lines.append(f"- ops in corpus: {n_ops_total}")
    lines.append(
        f"- ops with attention-bearing IR: {n_ops_with_attn_ir}"
    )
    lines.append(f"- sweep runtime: {sweep_runtime_s:.1f}s")
    lines.append("")
    lines.append("## Headline tallies")
    lines.append("")
    lines.append(f"- **total heads checked:** {agg['total_heads']}")
    lines.append(f"- **fully clean (ok):** {agg['fully_clean']}")
    lines.append(
        f"- **mismatch-only (weight_output_mismatch):** "
        f"{agg['mismatch_only']}"
    )
    lines.append(
        f"- **real bugs (declaration_semantics + lowering + exception):** "
        f"{agg['real_bugs']}"
    )
    lines.append("")
    lines.append("## Per-category totals")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|----------|------:|")
    for cat in (
        "ok",
        "declaration_semantics",
        "lowering",
        "weight_output_mismatch",
        "exception",
    ):
        lines.append(f"| `{cat}` | {totals.get(cat, 0)} |")
    lines.append("")

    lines.append("## Per-op summary")
    lines.append("")
    lines.append(
        "| op | heads | ok | decl_sem | lowering | weight_mismatch | exc |"
    )
    lines.append(
        "|----|------:|---:|---------:|---------:|----------------:|----:|"
    )
    op_rows = sorted(
        agg["per_op_total"].keys(),
        key=lambda n: (
            -(
                agg["per_op_declaration_semantics"].get(n, 0)
                + agg["per_op_lowering"].get(n, 0)
                + agg["per_op_exception"].get(n, 0)
            ),
            -agg["per_op_weight_output_mismatch"].get(n, 0),
            n,
        ),
    )
    for name in op_rows:
        lines.append(
            f"| {name} | {agg['per_op_total'][name]} | "
            f"{agg['per_op_ok'].get(name, 0)} | "
            f"{agg['per_op_declaration_semantics'].get(name, 0)} | "
            f"{agg['per_op_lowering'].get(name, 0)} | "
            f"{agg['per_op_weight_output_mismatch'].get(name, 0)} | "
            f"{agg['per_op_exception'].get(name, 0)} |"
        )
    lines.append("")

    lines.append("## Real-bug heads (declaration_semantics / lowering / exception)")
    lines.append("")
    if not agg["real_bug_rows"]:
        lines.append("_None — every head's spec lowers cleanly and matrix-matches its IR._")
    else:
        lines.append(
            "| op | layer | head_pos | spec_idx | name | flags | category | first failure |"
        )
        lines.append(
            "|----|------:|---------:|---------:|------|-------|----------|---------------|"
        )
        for row in agg["real_bug_rows"]:
            first = row["first_failure"] or ""
            # Trim very long failure messages for table readability; full
            # message is preserved in the JSON.
            if len(first) > 160:
                first = first[:157] + "..."
            # Escape pipe characters that would break the markdown table.
            first = first.replace("|", "\\|")
            flag_cell = (
                "flag-gated" if row.get("flags") else "-"
            )
            lines.append(
                f"| {row['op_name']} | {row['layer_idx']} | {row['head_pos']} "
                f"| {row['head_idx']} | {row['name']} | {flag_cell} "
                f"| `{row['category']}` | {first} |"
            )
    lines.append("")

    lines.append(
        "## Weight-output mismatch heads (matrices match IR, runtime diverges)"
    )
    lines.append("")
    if not agg["mismatch_only_rows"]:
        lines.append(
            "_None — every cleanly-lowered head also matches PureAttention.forward._"
        )
    else:
        lines.append(
            "Often reflects shared-residual interference (multi-head Q/K "
            "overlap at the synthetic state) rather than a head-local bug. "
            "Triage by re-running ``compare_symbolic_to_lowered_attn`` with a "
            "hand-crafted ``state=`` that isolates the head."
        )
        lines.append("")
        lines.append(
            "| op | layer | head_pos | spec_idx | name | flags | first failure |"
        )
        lines.append(
            "|----|------:|---------:|---------:|------|-------|---------------|"
        )
        for row in agg["mismatch_only_rows"]:
            first = row["first_failure"] or ""
            if len(first) > 160:
                first = first[:157] + "..."
            first = first.replace("|", "\\|")
            flag_cell = (
                "flag-gated" if row.get("flags") else "-"
            )
            lines.append(
                f"| {row['op_name']} | {row['layer_idx']} | {row['head_pos']} "
                f"| {row['head_idx']} | {row['name']} | {flag_cell} | {first} |"
            )
    lines.append("")

    MD_PATH.write_text("\n".join(lines))


def main() -> int:
    print("[sweep] building default LayerCompiler dim_positions...", flush=True)
    try:
        dim_positions = build_dim_positions()
    except Exception:
        print("[sweep] FAILED to build dim_positions:", flush=True)
        traceback.print_exc()
        return 2
    print(f"[sweep] dim_positions: {len(dim_positions)} entries", flush=True)

    ops_with_flags = _enumerate_ops()
    n_ops_total = len(ops_with_flags)
    print(
        f"[sweep] enumerating attention heads across {n_ops_total} ops "
        f"(default + flags-on, deduped)...",
        flush=True,
    )
    heads = collect_attn_heads(ops_with_flags, dim_positions)
    op_names_with_attn = {op_name for (op_name, _, _, _, _) in heads}
    n_ops_with_attn_ir = len(op_names_with_attn)
    print(
        f"[sweep] attention heads: {len(heads)} "
        f"(across {n_ops_with_attn_ir} ops)",
        flush=True,
    )

    print("[sweep] running compare_symbolic_to_lowered_attn per head...", flush=True)
    t0 = time.time()
    rows, totals = run_sweep(heads)
    sweep_runtime_s = time.time() - t0
    print(f"[sweep] sweep complete in {sweep_runtime_s:.1f}s", flush=True)

    agg = aggregate(rows)
    write_reports(
        rows,
        totals,
        agg,
        n_ops_total=n_ops_total,
        n_ops_with_attn_ir=n_ops_with_attn_ir,
        sweep_runtime_s=sweep_runtime_s,
    )

    print(f"[sweep] wrote {MD_PATH}", flush=True)
    print(f"[sweep] wrote {JSON_PATH}", flush=True)
    print(
        "[sweep] totals: "
        f"total={agg['total_heads']} "
        f"clean={agg['fully_clean']} "
        f"mismatch_only={agg['mismatch_only']} "
        f"real_bugs={agg['real_bugs']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
