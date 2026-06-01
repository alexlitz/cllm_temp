"""Identify cancel-pair generalization candidates from the L8 pattern.

Today's L8 SP-gather fix (commit ``6b661d5``) introduced a clever
semantically-neutral magnitude lift: two V slots reading ``CONST`` (always
1.0) with paired ``+N`` / ``-N`` O writes per output dim. Because CONST is
identically 1.0 at every key position and softmax weights normalize to 1,
the ``+N`` and ``-N`` deliveries cancel EXACTLY at the q-position output
regardless of attention sharpness. Net residual write delta: zero. But
the V1 ``attention_verifier``'s magnitude bound (``|O|*sum(|V|)``) sees
``+N + N = 2N`` extra magnitude. Free verifier honesty.

Question this tool answers: which OTHER heads in the corpus are candidates
for the same trick?

Decision rule per head with ``attention_strength_violation`` issues:

  * applicable     -- Q-side gating is disjoint from competitor's so the
                     two heads never fire at the same q-row at runtime.
                     Cancel-pair is structurally honest: it lifts the
                     magnitude bound without changing semantics.
  * v2_resolved    -- Heads with the SAME competitor op as self (top
                     competitor == self op) or the competitor is on a
                     different layer/marker scope. The Attention V2
                     scope-aware filter will recognize these as
                     bookkeeping competitors and drop the violation
                     without any lift.
  * semantically_wrong
                   -- Q-side gating is shared / overlapping AND the
                     output dim is actually contended at runtime
                     (would need a real magnitude bump, not a cancel
                     pair).
  * needs_investigation
                   -- Could not determine from the scope strings alone.

This is diagnostic only -- it does not modify any source. Reads the
existing catalog at ``.agent-logs/attention_violations_catalog_2026_06_01.md``
and writes a markdown file at
``.agent-logs/cancel_pair_generalization_candidates_2026_06_01.md``.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]  # c4_release/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neural_vm.dim_registry import build_default_registry  # noqa: E402
from neural_vm.unified_compiler.attention_verifier import (  # noqa: E402
    verify_attention_head,
)
from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops  # noqa: E402
from neural_vm.unified_compiler.ops.shared import (  # noqa: E402
    declare_setdim_compat_dims,
)
from neural_vm.unified_compiler.layer_compiler import LayerCompiler  # noqa: E402


CATALOG_PATH = REPO / ".agent-logs" / "attention_violations_catalog_2026_06_01.md"
OUT_PATH = REPO / ".agent-logs" / "cancel_pair_generalization_candidates_2026_06_01.md"


# ---------------------------------------------------------------------------
# Build dim_positions and materialize heads (mirror catalog_attention_violations).
# ---------------------------------------------------------------------------


def _build_dim_positions() -> Dict[str, int]:
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    layout = compiler.compile()
    return dict(layout.dim_positions)


def _materialize_ir(op, dim_positions, HD: int = 64):
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


def _collect_heads(ops, dim_positions):
    out: List[Tuple[str, int, int, Any]] = []
    for op in ops:
        ir = _materialize_ir(op, dim_positions)
        if ir is None or not hasattr(ir, "layers"):
            continue
        for layer_idx, layer in enumerate(ir.layers):
            attention = getattr(layer, "attention", None)
            if attention is None:
                continue
            for head_idx, head in enumerate(getattr(attention, "rules", None) or []):
                out.append((op.name, layer_idx, head_idx, head))
    return out


class _IROnlyOp:
    def __init__(self, name, ir):
        self.name = name
        self.compiler_ir = ir


def _build_cross_ops(ops, dim_positions):
    out = []
    for op in ops:
        ir = _materialize_ir(op, dim_positions)
        if ir is None:
            continue
        out.append(_IROnlyOp(op.name, ir))
    return out


# ---------------------------------------------------------------------------
# Q-side scope extraction: walk the head's q writes and pull MARK_* + gating
# dims with positive weight.
# ---------------------------------------------------------------------------


_Q_GATE_DIMS = {
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_STACK0",
    "IS_BYTE", "HAS_SE",
    "MEM_STORE", "MEM_LOAD",
    "OP_LEA", "OP_ADJ", "OP_ENT", "OP_LEV", "OP_JSR", "OP_PSH",
    "OP_LI", "OP_LC", "OP_SI", "OP_SC", "OP_ADD", "OP_SUB", "OP_MUL",
    "OP_DIV", "OP_MOD", "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_LT",
    "OP_SHL", "OP_SHR", "OP_IMM", "OP_JMP", "OP_EXIT",
}


def _resolve_dim(registry, dim_int):
    for name, slot in registry.slots.items():
        start = getattr(slot, "start", None)
        size = getattr(slot, "size", None)
        if start is None or size is None:
            continue
        if start <= dim_int < start + size:
            return name, dim_int - start
    return None


def _q_gate_signature(head, registry) -> Tuple[frozenset, frozenset]:
    """Return (positive_q_dims, negative_q_dims) Q-side gating dim names.

    Positive Q dims are firing conditions (head prefers q-rows where these
    are set); negative Q dims are exclusion conditions.
    """
    spec = getattr(head, "spec", None)
    if spec is None:
        return frozenset(), frozenset()
    pos, neg = set(), set()
    for q in getattr(spec, "q", ()):
        resolved = _resolve_dim(registry, int(getattr(q, "dim", 0)))
        if resolved is None:
            continue
        name, _ = resolved
        if name not in _Q_GATE_DIMS:
            continue
        w = float(getattr(q, "weight", 0.0))
        if w > 0:
            pos.add(name)
        elif w < 0:
            neg.add(name)
    return frozenset(pos), frozenset(neg)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify(
    head_pos: frozenset,
    head_neg: frozenset,
    comp_pos: frozenset,
    comp_neg: frozenset,
    own_op: str,
    competitor_op: Optional[str],
) -> Tuple[str, str]:
    """Return (verdict, reason)."""
    # If competitor is the same op, this is intra-op zero-sum.
    if competitor_op == own_op:
        return ("v2_resolved", "intra-op competitor (same op self-competition); V2 should de-self")

    # Disjoint by Q firing markers: head_pos vs comp_pos do not share a
    # marker AND each head positively requires at least one marker.
    head_markers = head_pos & _MARKER_DIMS
    comp_markers = comp_pos & _MARKER_DIMS
    if head_markers and comp_markers and not (head_markers & comp_markers):
        return (
            "applicable",
            f"Q markers disjoint: head={sorted(head_markers)} vs comp={sorted(comp_markers)}",
        )

    # Mutual exclusion via negative gating: head positively requires X,
    # competitor negatively excludes X.
    if head_pos & comp_neg:
        excl = head_pos & comp_neg
        return (
            "applicable",
            f"competitor excludes head's positive gate: {sorted(excl)}",
        )
    if comp_pos & head_neg:
        excl = comp_pos & head_neg
        return (
            "applicable",
            f"head excludes competitor's positive gate: {sorted(excl)}",
        )

    # Opcode gates disjoint: head positively requires opcode A, comp
    # positively requires opcode B, neither equal.
    head_ops = head_pos & _OPCODE_DIMS
    comp_ops = comp_pos & _OPCODE_DIMS
    if head_ops and comp_ops and not (head_ops & comp_ops):
        return (
            "applicable",
            f"Q opcodes disjoint: head={sorted(head_ops)} vs comp={sorted(comp_ops)}",
        )

    # Mem-store / mem-load mutex.
    if "MEM_STORE" in head_pos and "MEM_STORE" in comp_neg:
        return ("applicable", "competitor excludes MEM_STORE which head requires")
    if "MEM_STORE" in comp_pos and "MEM_STORE" in head_neg:
        return ("applicable", "head excludes MEM_STORE which competitor requires")

    # Overlapping Q markers + opcode gates + nothing distinguishes them.
    if head_markers and comp_markers and (head_markers & comp_markers):
        if head_ops or comp_ops:
            return (
                "needs_investigation",
                "shared markers but differing opcode gates -- runtime overlap unclear",
            )
        return (
            "semantically_wrong",
            f"shared markers {sorted(head_markers & comp_markers)} and no opcode/exclusion distinguisher",
        )

    if not head_markers and not comp_markers:
        return ("needs_investigation", "no MARK_* on either side -- can't compare q-side scopes")

    return ("needs_investigation", "no clear disjoint-firing signal in Q gates")


_MARKER_DIMS = frozenset({
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_STACK0",
})
_OPCODE_DIMS = frozenset({
    "OP_LEA", "OP_ADJ", "OP_ENT", "OP_LEV", "OP_JSR", "OP_PSH",
    "OP_LI", "OP_LC", "OP_SI", "OP_SC", "OP_ADD", "OP_SUB", "OP_MUL",
    "OP_DIV", "OP_MOD", "OP_OR", "OP_XOR", "OP_AND", "OP_EQ", "OP_LT",
    "OP_SHL", "OP_SHR", "OP_IMM", "OP_JMP", "OP_EXIT",
    "MEM_STORE", "MEM_LOAD",
})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("[candidates] building registry + dim_positions...")
    registry = build_default_registry()
    dim_positions = _build_dim_positions()
    ops = all_core_ops()
    print(f"[candidates] enumerating heads across {len(ops)} ops...")
    heads = _collect_heads(ops, dim_positions)
    print(f"[candidates] total heads: {len(heads)}")
    head_index: Dict[Tuple[str, int, int], Any] = {
        (op_name, li, hi): head for (op_name, li, hi, head) in heads
    }

    cross_ops = _build_cross_ops(ops, dim_positions)
    print(f"[candidates] cross-op list: {len(cross_ops)}")

    # Build lookup: op_name -> {(layer, head): head}.
    op_to_heads: Dict[str, List[Tuple[int, int, Any]]] = defaultdict(list)
    for op_name, li, hi, head in heads:
        op_to_heads[op_name].append((li, hi, head))

    # Per head: run verification with cross-op competition once, group
    # issues by (competitor_op, output_dim_band) so we count per-band
    # candidates not per-output-dim.
    print("[candidates] running per-head cross-op verification...")
    results: List[Dict[str, Any]] = []
    for op_name, li, hi, head in heads:
        try:
            issues = verify_attention_head(
                head, registry, ops_for_competition=cross_ops,
            )
        except Exception as e:
            issues = [{"kind": "exception", "reason": repr(e)}]

        # Only attention_strength_violation entries are cancel-pair-eligible.
        strength_issues = [
            i for i in issues
            if i.get("kind") == "attention_strength_violation"
        ]
        if not strength_issues:
            continue

        # Group strength_issues by top competitor.
        by_comp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for issue in strength_issues:
            comp = issue.get("top_competitor") or "<none>"
            by_comp[comp].append(issue)

        head_pos, head_neg = _q_gate_signature(head, registry)

        # Find competitor head's Q signature (look it up).
        for comp_label, comp_issues in by_comp.items():
            comp_head_obj = None
            comp_op_name = None
            # comp_label is head_name or op_name. Try match against
            # known heads.
            for (cn_op, cn_li, cn_hi), ch in head_index.items():
                ch_name = getattr(ch, "name", None)
                if ch_name == comp_label or cn_op == comp_label:
                    comp_head_obj = ch
                    comp_op_name = cn_op
                    break

            if comp_head_obj is not None:
                comp_pos, comp_neg = _q_gate_signature(comp_head_obj, registry)
            else:
                comp_pos, comp_neg = frozenset(), frozenset()

            verdict, reason = _classify(
                head_pos, head_neg, comp_pos, comp_neg,
                own_op=op_name, competitor_op=comp_op_name,
            )

            # Sample magnitudes / scopes from first issue.
            first = comp_issues[0]
            results.append({
                "op": op_name,
                "layer": li,
                "head": hi,
                "head_name": getattr(head, "name", None),
                "spec_head_idx": int(
                    getattr(getattr(head, "spec", None), "head_idx", -1)
                ),
                "competitor": comp_label,
                "competitor_op": comp_op_name,
                "n_issues": len(comp_issues),
                "my_magnitude": first.get("my_magnitude"),
                "competing_max": first.get("competing_max"),
                "head_q_pos": sorted(head_pos),
                "head_q_neg": sorted(head_neg),
                "comp_q_pos": sorted(comp_pos),
                "comp_q_neg": sorted(comp_neg),
                "verdict": verdict,
                "reason": reason,
                "sample_output": first.get("output_dim"),
            })

    print(f"[candidates] candidate (head, competitor) pairs: {len(results)}")

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    by_verdict: Counter = Counter()
    impact_by_verdict: Counter = Counter()
    for r in results:
        by_verdict[r["verdict"]] += 1
        impact_by_verdict[r["verdict"]] += r["n_issues"]

    # Per-head verdict (worst-case across competitors).
    _verdict_rank = {
        "semantically_wrong": 0,
        "needs_investigation": 1,
        "applicable": 2,
        "v2_resolved": 3,
    }
    per_head: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for r in results:
        key = (r["op"], r["layer"], r["head"])
        existing = per_head.get(key)
        if existing is None:
            per_head[key] = {
                "verdicts": Counter([r["verdict"]]),
                "total_issues": r["n_issues"],
                "competitors": [r["competitor"]],
                "spec_head_idx": r["spec_head_idx"],
                "head_name": r["head_name"],
            }
        else:
            existing["verdicts"][r["verdict"]] += 1
            existing["total_issues"] += r["n_issues"]
            existing["competitors"].append(r["competitor"])

    def head_verdict(verdict_counter: Counter) -> str:
        # If ALL verdicts agree, use that.
        if len(verdict_counter) == 1:
            return next(iter(verdict_counter))
        # If a "semantically_wrong" exists for any competitor, that wins
        # (cancel-pair can't fix the head -- it'd still violate that comp).
        if "semantically_wrong" in verdict_counter:
            return "semantically_wrong"
        if "needs_investigation" in verdict_counter:
            return "needs_investigation"
        if "applicable" in verdict_counter and "v2_resolved" in verdict_counter:
            # Mixed applicable + v2_resolved -- V2 cleans up some, cancel
            # pair would be needed for the rest. Tag as "applicable_partial".
            return "applicable_partial"
        return next(iter(verdict_counter))

    head_summary: List[Tuple[Tuple[str, int, int], Dict[str, Any], str]] = []
    for key, val in per_head.items():
        head_summary.append((key, val, head_verdict(val["verdicts"])))
    head_summary.sort(key=lambda x: -x[1]["total_issues"])

    head_verdict_counts: Counter = Counter()
    head_impact_counts: Counter = Counter()
    for _, val, hv in head_summary:
        head_verdict_counts[hv] += 1
        head_impact_counts[hv] += val["total_issues"]

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("# Cancel-Pair Generalization Candidates (2026-06-01)")
    lines.append("")
    lines.append(
        "Reads the per-(head, competitor) ``attention_strength_violation`` "
        "pairs from the live ``verify_attention_head`` cross-op pass and "
        "classifies each against the L8 SP-gather cancel-pair pattern "
        "(commit ``6b661d5``)."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "* L8 trick = two V slots reading ``CONST`` (=1.0) + paired ``+N/-N`` "
        "O writes per output dim. Cancels exactly at the q-position; lifts "
        "verifier magnitude bound from ``|O|*|V|`` to ``|O|*|V| + 2*|N|``."
    )
    lines.append(
        "* Cancel-pair is APPLICABLE only when the head and its competitor "
        "do not actually fire at the same q-row at runtime (otherwise the "
        "lift would be dishonest -- both heads really compete)."
    )
    lines.append(
        "* The today's-V2 scope-aware filter will recognize same-op "
        "self-competition and disjoint Q-side gating as bookkeeping and "
        "drop those violations automatically. Tagged as ``v2_resolved``."
    )
    lines.append("")
    lines.append("## Aggregate verdict counts")
    lines.append("")
    lines.append("### Per (head, competitor) pair")
    lines.append("")
    lines.append("| verdict | n_pairs | total issues |")
    lines.append("|---------|--------:|-------------:|")
    for v in ["applicable", "v2_resolved", "semantically_wrong", "needs_investigation"]:
        lines.append(f"| `{v}` | {by_verdict[v]} | {impact_by_verdict[v]} |")
    lines.append("")
    lines.append("### Per head (worst-case across competitors)")
    lines.append("")
    lines.append("| verdict | n_heads | total issues |")
    lines.append("|---------|--------:|-------------:|")
    for v in [
        "applicable", "applicable_partial", "v2_resolved",
        "semantically_wrong", "needs_investigation",
    ]:
        lines.append(
            f"| `{v}` | {head_verdict_counts[v]} | {head_impact_counts[v]} |"
        )
    lines.append("")

    # Top impactful candidates.
    lines.append("## Top 10 most-impactful candidate heads")
    lines.append("")
    lines.append(
        "| rank | op | head | total | verdict | top competitors |"
    )
    lines.append(
        "|-----:|----|------|------:|---------|-----------------|"
    )
    for rank, ((op_name, li, hi), val, hv) in enumerate(head_summary[:10], 1):
        comps = Counter(val["competitors"]).most_common(3)
        comp_str = ", ".join(f"`{c}`x{n}" for c, n in comps)
        lines.append(
            f"| {rank} | {op_name} | "
            f"L{li}h{hi}/spec={val['spec_head_idx']} "
            f"| {val['total_issues']} | `{hv}` | {comp_str} |"
        )
    lines.append("")

    # Full per-head detail.
    lines.append("## Per-head verdict detail")
    lines.append("")
    for (op_name, li, hi), val, hv in head_summary:
        lines.append(
            f"### {op_name} L{li}h{hi}/spec={val['spec_head_idx']} "
            f"-- verdict: `{hv}` ({val['total_issues']} issues)"
        )
        lines.append("")
        # Aggregate per-competitor verdict info.
        per_comp_rows = [r for r in results
                         if (r["op"], r["layer"], r["head"]) == (op_name, li, hi)]
        for r in per_comp_rows:
            lines.append(
                f"- vs `{r['competitor']}` "
                f"({r['n_issues']} issues, sample `{r['sample_output']}` "
                f"my={r['my_magnitude']} vs {r['competing_max']}): "
                f"**{r['verdict']}** -- {r['reason']}"
            )
            lines.append(
                f"  - head q_pos={r['head_q_pos']} q_neg={r['head_q_neg']}"
            )
            lines.append(
                f"  - comp q_pos={r['comp_q_pos']} q_neg={r['comp_q_neg']}"
            )
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"[candidates] wrote {OUT_PATH}")

    # Print compact summary for caller.
    print("\nSummary:")
    print(f"  heads with attention_strength_violation: {len(per_head)}")
    for v, n in head_verdict_counts.most_common():
        print(f"    {v:25s}: {n} heads ({head_impact_counts[v]} issues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
