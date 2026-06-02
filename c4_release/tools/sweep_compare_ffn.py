"""Phase 6 Wave 5A — sweep ``compare_symbolic_to_lowered_ffn`` across all ops.

For every op in ``all_core_ops()`` (default + flags-on flavours), pull the
op's ``CompilerIR`` (either via ``op.compiler_ir`` or
``op.compiler_ir_factory(dim_positions, head_dim)``). For each layer in the
IR whose ``FFNOp`` carries any rules, run

    compare_symbolic_to_lowered_ffn(ir, dim_positions, layer_idx=L, S=100.0)

and bucket the resulting issues into three coarse kinds (matching
``ir._validate_ffn_declarations`` / ``_validate_lowered_ffn`` /
``_run_lowered_ffn_comparison``):

  - ``declaration_semantics`` — IR references unresolved dims, or symbolic
    execution raised. Real bug: the IR cannot be lowered as written.
  - ``lowering`` — PureFFN weights after ``lower_ffn`` disagree with the
    IR-declared cell pattern. Real bug.
  - ``weight_output_mismatch`` — weights match the lowering contract but
    PureFFN.forward disagrees with the symbolic semantics on at least one
    output cell. Often a known synthetic-state limitation (the auto-built
    ``_synthetic_ffn_state`` may not satisfy every gate condition the rule
    expects, so the lowered SwiGLU column saturates a different way), but
    can also indicate a real semantic gap.

Read-only: no op file is touched and no weight bake hits the real model.
We allocate one fresh ``PureFFN`` per (op, layer) comparison from inside
``compare_symbolic_to_lowered_ffn``.

Outputs:
  - ``c4_release/.agent-logs/sweep_compare_ffn_phase6_wave5a.md`` (human)
  - ``c4_release/.agent-logs/sweep_compare_ffn_phase6_wave5a.json`` (structured)

Run:

    cd c4_release && python -m c4_release.tools.sweep_compare_ffn
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_PARENT = _HERE.parents[2]
if str(_PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_PARENT))


# ---------------------------------------------------------------------------
# Op enumeration
# ---------------------------------------------------------------------------

_FLAG_MODES: List[Tuple[str, Dict[str, bool]]] = [
    ("default", {}),
    ("all_flags_on", {
        "enable_conversational_io": True,
        "enable_tool_calling": True,
        "enable_neural_io_think_protocol": True,
    }),
]


def _enumerate_ops() -> List[Tuple[Any, List[str]]]:
    """Return [(op, enabled_flags_list)] deduped by op.name.

    Default-mode op wins on duplicate names; flags-on additions are tagged
    with the enabled flag list so the report can mark them as flag-gated.
    """
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    seen: Dict[str, Tuple[Any, List[str]]] = {}
    default_names: set = set()
    for label, kwargs in _FLAG_MODES:
        ops = all_core_ops(**kwargs)
        if label == "default":
            default_names = {getattr(op, "name", repr(op)) for op in ops}
        for op in ops:
            name = getattr(op, "name", repr(op))
            if name in seen:
                continue
            flags = [k for k, v in kwargs.items() if v]
            if name not in default_names:
                flags = sorted(set(flags) | {"requires_flag"})
            seen[name] = (op, flags)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Per-op sweep
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    layer_idx: int
    n_rules: int
    ok: bool
    runtime_s: float = 0.0
    declaration_semantics: int = 0
    lowering: int = 0
    weight_output_mismatch: int = 0
    synthetic_state_overflow: int = 0
    other: int = 0
    first_failure_kind: Optional[str] = None
    first_failure_message: Optional[str] = None
    exception: Optional[str] = None


@dataclass
class OpResult:
    name: str
    kind: str
    phase: Optional[float]
    layer_idx: Optional[int]
    enabled_flags: List[str] = field(default_factory=list)
    ir_source: str = "?"  # "compiler_ir", "compiler_ir_factory", or "none"
    has_ffn_rules: bool = False
    total_rules: int = 0
    layers_with_rules: int = 0
    layers_checked: List[LayerResult] = field(default_factory=list)
    ok: bool = False
    total_declaration_semantics: int = 0
    total_lowering: int = 0
    total_weight_output_mismatch: int = 0
    total_synthetic_state_overflow: int = 0
    total_other: int = 0
    first_failure_kind: Optional[str] = None
    first_failure_layer: Optional[int] = None
    first_failure_message: Optional[str] = None
    bucket: str = "no_ffn_rules"
    error: Optional[str] = None
    build_runtime_s: float = 0.0
    runtime_s: float = 0.0


def _get_ir(op, *, dim_positions, head_dim: int) -> Tuple[Any, str]:
    ir = getattr(op, "compiler_ir", None)
    if ir is not None:
        return ir, "compiler_ir"
    factory = getattr(op, "compiler_ir_factory", None)
    if factory is not None:
        try:
            ir = factory(dim_positions, head_dim)
        except TypeError:
            # Older signature variants — try with no args
            try:
                ir = factory()
            except Exception:
                return None, "factory_failed"
        except Exception:
            return None, "factory_failed"
        if ir is not None:
            return ir, "compiler_ir_factory"
    return None, "none"


def _layer_has_rules(layer) -> int:
    ffn = getattr(layer, "ffn", None)
    if ffn is None:
        return 0
    rules = getattr(ffn, "rules", ())
    return len(rules)


def _sweep_op(op, enabled_flags, dim_positions, head_dim: int,
              *, per_layer_budget_s: float = 60.0) -> OpResult:
    from neural_vm.unified_compiler.ir import (
        CompilerIR, compare_symbolic_to_lowered_ffn,
    )

    name = getattr(op, "name", repr(op))
    result = OpResult(
        name=name,
        kind=getattr(op, "kind", "?"),
        phase=getattr(op, "phase", None),
        layer_idx=getattr(op, "layer_idx", None),
        enabled_flags=list(enabled_flags),
    )

    t_build0 = time.time()
    try:
        ir, src = _get_ir(op, dim_positions=dim_positions, head_dim=head_dim)
    except Exception as exc:
        result.error = f"build: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        result.ir_source = "build_error"
        result.bucket = "build_error"
        result.build_runtime_s = time.time() - t_build0
        return result
    result.build_runtime_s = time.time() - t_build0
    result.ir_source = src

    if ir is None or not isinstance(ir, CompilerIR):
        result.bucket = "no_ir"
        return result

    # Identify layers with FFN rules.
    layers_with_rules: List[Tuple[int, int]] = []
    for li, layer in enumerate(ir.layers):
        n = _layer_has_rules(layer)
        if n > 0:
            layers_with_rules.append((li, n))
    result.layers_with_rules = len(layers_with_rules)
    result.total_rules = sum(n for _, n in layers_with_rules)
    result.has_ffn_rules = bool(layers_with_rules)

    if not result.has_ffn_rules:
        result.bucket = "no_ffn_rules"
        return result

    t0 = time.time()
    op_ok = True
    for layer_idx, n_rules in layers_with_rules:
        lr = LayerResult(layer_idx=layer_idx, n_rules=n_rules, ok=False)
        tL0 = time.time()
        try:
            report = compare_symbolic_to_lowered_ffn(
                ir,
                dim_positions,
                layer_idx=layer_idx,
                S=100.0,
            )
            lr.ok = bool(report.ok)
            for issue in report.issues:
                if issue.kind == "declaration_semantics":
                    lr.declaration_semantics += 1
                elif issue.kind == "lowering":
                    lr.lowering += 1
                elif issue.kind == "weight_output_mismatch":
                    lr.weight_output_mismatch += 1
                elif issue.kind == "_synthetic_state_overflow":
                    lr.synthetic_state_overflow += 1
                else:
                    lr.other += 1
                if lr.first_failure_kind is None:
                    lr.first_failure_kind = issue.kind
                    msg = issue.message or ""
                    lr.first_failure_message = msg.splitlines()[0][:300]
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc) else ""
            lr.exception = f"{type(exc).__name__}: {first[:300]}"
            lr.first_failure_kind = "exception"
            lr.first_failure_message = lr.exception
        lr.runtime_s = time.time() - tL0
        result.layers_checked.append(lr)

        result.total_declaration_semantics += lr.declaration_semantics
        result.total_lowering += lr.lowering
        result.total_weight_output_mismatch += lr.weight_output_mismatch
        result.total_synthetic_state_overflow += lr.synthetic_state_overflow
        result.total_other += lr.other
        if not lr.ok or lr.exception is not None:
            op_ok = False
            if result.first_failure_kind is None:
                result.first_failure_kind = (
                    lr.exception and "exception"
                ) or lr.first_failure_kind
                result.first_failure_layer = layer_idx
                result.first_failure_message = (
                    lr.exception or lr.first_failure_message
                )

        # Soft per-layer budget — avoid runaway on huge ops.
        if time.time() - tL0 > per_layer_budget_s:
            # Continue but tag for awareness in report.
            pass

    result.ok = op_ok
    result.runtime_s = time.time() - t0

    # Classify into final bucket.
    if op_ok:
        result.bucket = "clean"
    elif result.total_declaration_semantics > 0 or result.total_lowering > 0:
        result.bucket = "real_bug"
    elif any(lr.exception for lr in result.layers_checked):
        # Exception during symbolic/lowering counts as a real bug.
        result.bucket = "real_bug"
    elif (
        result.total_synthetic_state_overflow > 0
        and result.total_weight_output_mismatch == 0
    ):
        # Harness limitation: the auto-built synthetic state grew past fp32
        # range for a huge-rule layer. Not a semantic bug in the IR.
        result.bucket = "synthetic_state_overflow"
    elif result.total_weight_output_mismatch > 0:
        result.bucket = "mismatch_only"
    else:
        result.bucket = "other_failure"
    return result


# ---------------------------------------------------------------------------
# Aggregation + rendering
# ---------------------------------------------------------------------------

def _aggregate(results: List[OpResult]) -> Dict[str, Any]:
    buckets = Counter(r.bucket for r in results)
    with_rules = [r for r in results if r.has_ffn_rules]
    clean = [r for r in with_rules if r.bucket == "clean"]
    mismatch_only = [r for r in with_rules if r.bucket == "mismatch_only"]
    real_bug = [r for r in with_rules if r.bucket == "real_bug"]
    overflow = [r for r in with_rules
                if r.bucket == "synthetic_state_overflow"]
    other = [r for r in with_rules if r.bucket == "other_failure"]

    return {
        "total_ops_enumerated": len(results),
        "total_ops_with_ffn_rules": len(with_rules),
        "total_clean": len(clean),
        "total_mismatch_only": len(mismatch_only),
        "total_real_bug": len(real_bug),
        "total_synthetic_state_overflow": len(overflow),
        "total_other_failure": len(other),
        "bucket_counts": dict(buckets),
        "real_bug_ops": [r.name for r in real_bug],
        "mismatch_only_ops": [r.name for r in mismatch_only],
        "synthetic_state_overflow_ops": [r.name for r in overflow],
        "clean_ops": [r.name for r in clean],
        "other_failure_ops": [r.name for r in other],
    }


_MD_HEADER = """# Phase 6 Wave 5A — sweep of `compare_symbolic_to_lowered_ffn` across every FFN-rule-bearing op

Generated by `c4_release/tools/sweep_compare_ffn.py`.

For each op in `all_core_ops()` (default + all-flags-on), pulled the op's
`CompilerIR` (via `op.compiler_ir` or `op.compiler_ir_factory`), enumerated
layers whose `FFNOp` had any rules, and ran
`compare_symbolic_to_lowered_ffn` per layer. Buckets:

- **clean** — every layer's symbolic/lowering/forward all agree.
- **mismatch_only** — only `weight_output_mismatch` issues (often a
  synthetic-state limitation: `_synthetic_ffn_state` doesn't satisfy every
  gate, so the lowered SwiGLU saturates differently). Still worth a
  follow-up pass.
- **real_bug** — at least one `declaration_semantics` or `lowering` issue
  (or a thrown exception). The IR is broken or the lowering contract
  doesn't match what the verifier reads.
- **synthetic_state_overflow** — the auto-built `_synthetic_ffn_state`
  produced a value past fp32 range when packed into the lowered input
  tensor. Harness limitation on huge-rule layers (typical for
  fan-in-heavy ops with thousands of rules sharing a condition dim); not
  a semantic bug in the IR.
- **other_failure** — non-empty issues that don't fall into the above
  (currently unused; preserved for future failure kinds).
- **no_ffn_rules / no_ir / build_error** — informational; these ops carry
  no FFN rules to sweep.

`S=100.0`, `dim_positions` from `compile_compact_layout()`.

"""


def _render_md(results: List[OpResult], agg: Dict[str, Any]) -> str:
    lines: List[str] = [_MD_HEADER]
    lines.append("## Aggregates\n")
    lines.append(
        f"- Total ops enumerated (default + flags-on, deduped): "
        f"**{agg['total_ops_enumerated']}**"
    )
    lines.append(
        f"- Ops with FFN rules in their IR: "
        f"**{agg['total_ops_with_ffn_rules']}**"
    )
    lines.append(f"- **clean**: {agg['total_clean']}")
    lines.append(f"- **mismatch_only**: {agg['total_mismatch_only']}")
    lines.append(f"- **real_bug**: {agg['total_real_bug']}")
    lines.append(
        f"- **synthetic_state_overflow**: "
        f"{agg['total_synthetic_state_overflow']}"
    )
    lines.append(f"- **other_failure**: {agg['total_other_failure']}")
    lines.append("- Bucket distribution (incl. no-rule buckets):")
    for k, v in sorted(agg['bucket_counts'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: {v}")
    lines.append("")

    if agg["real_bug_ops"]:
        lines.append("## Real-bug ops (declaration_semantics / lowering / exception)\n")
        for name in agg["real_bug_ops"]:
            r = next(rr for rr in results if rr.name == name)
            lines.append(
                f"- `{name}` (kind={r.kind}, layer_idx={r.layer_idx}, "
                f"rules={r.total_rules}, decl={r.total_declaration_semantics}, "
                f"low={r.total_lowering}, wo={r.total_weight_output_mismatch}, "
                f"first={r.first_failure_kind} @ L{r.first_failure_layer}: "
                f"{r.first_failure_message})"
            )
        lines.append("")

    if agg["mismatch_only_ops"]:
        lines.append("## Mismatch-only ops (weight_output_mismatch)\n")
        for name in agg["mismatch_only_ops"]:
            r = next(rr for rr in results if rr.name == name)
            lines.append(
                f"- `{name}` (rules={r.total_rules}, mismatches="
                f"{r.total_weight_output_mismatch}, "
                f"first={r.first_failure_message})"
            )
        lines.append("")

    if agg.get("synthetic_state_overflow_ops"):
        lines.append(
            "## Synthetic-state-overflow ops "
            "(harness limitation, not real bug)\n"
        )
        for name in agg["synthetic_state_overflow_ops"]:
            r = next(rr for rr in results if rr.name == name)
            lines.append(
                f"- `{name}` (rules={r.total_rules}, "
                f"overflow={r.total_synthetic_state_overflow}, "
                f"first={r.first_failure_message})"
            )
        lines.append("")

    lines.append("## All swept ops (with FFN rules)\n")
    lines.append(
        "| Op | Kind | Phase | LayerIdx | Rules | LayersChecked | Bucket | "
        "Decl | Low | Wmismatch | OK | t(s) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|:-:|---:|")
    with_rules = [r for r in results if r.has_ffn_rules]
    with_rules.sort(key=lambda r: (
        # real_bug first, mismatch_only next, clean last
        {"real_bug": 0, "other_failure": 1, "mismatch_only": 2,
         "synthetic_state_overflow": 3, "clean": 4}.get(r.bucket, 5),
        r.phase if r.phase is not None else 1e9,
        r.name,
    ))
    for r in with_rules:
        ok = "Y" if r.ok else "n"
        phase = f"{r.phase:g}" if r.phase is not None else "-"
        li = r.layer_idx if r.layer_idx is not None else "-"
        flag = " (flag)" if "requires_flag" in r.enabled_flags else ""
        lines.append(
            f"| `{r.name}`{flag} | {r.kind} | {phase} | {li} "
            f"| {r.total_rules} | {len(r.layers_checked)} | {r.bucket} "
            f"| {r.total_declaration_semantics} | {r.total_lowering} "
            f"| {r.total_weight_output_mismatch} | {ok} | {r.runtime_s:.2f} |"
        )
    lines.append("")

    lines.append("## Ops with no FFN rules (informational)\n")
    no_rules = [r for r in results if not r.has_ffn_rules]
    by_bucket: Dict[str, List[OpResult]] = {}
    for r in no_rules:
        by_bucket.setdefault(r.bucket, []).append(r)
    for bucket, group in sorted(by_bucket.items()):
        lines.append(f"### bucket=`{bucket}` (n={len(group)})\n")
        for r in sorted(group, key=lambda x: x.name):
            extras = []
            if r.error:
                extras.append(r.error)
            if r.ir_source not in ("compiler_ir", "compiler_ir_factory"):
                extras.append(f"ir_source={r.ir_source}")
            extra = f" — {'; '.join(extras)}" if extras else ""
            lines.append(f"- `{r.name}` (kind={r.kind}){extra}")
        lines.append("")

    lines.append("## Per-op layer detail (FFN-rule-bearing ops only)\n")
    for r in with_rules:
        lines.append(f"### `{r.name}`\n")
        lines.append(
            f"- kind={r.kind}, phase={r.phase}, layer_idx={r.layer_idx}, "
            f"ir_source={r.ir_source}, bucket=**{r.bucket}**, ok={r.ok}, "
            f"total_rules={r.total_rules}, "
            f"runtime_s={r.runtime_s:.2f} (build={r.build_runtime_s:.2f})"
        )
        for lr in r.layers_checked:
            status = "ok" if lr.ok else (
                "exception" if lr.exception else "failed"
            )
            lines.append(
                f"  - layer {lr.layer_idx}: rules={lr.n_rules} "
                f"status={status} decl={lr.declaration_semantics} "
                f"low={lr.lowering} wo={lr.weight_output_mismatch} "
                f"other={lr.other} t={lr.runtime_s:.2f}s"
            )
            if lr.first_failure_kind:
                lines.append(
                    f"    first failure: [{lr.first_failure_kind}] "
                    f"{lr.first_failure_message}"
                )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from tests._per_op_audit import compile_compact_layout

    layout = compile_compact_layout()
    dim_positions = layout.dim_positions
    head_dim = getattr(layout, "head_dim", 64)

    ops_and_flags = _enumerate_ops()
    print(
        f"enumerated {len(ops_and_flags)} ops "
        f"(default + flags-on, deduped by name)",
        flush=True,
    )

    results: List[OpResult] = []
    t_all = time.time()
    for i, (op, flags) in enumerate(ops_and_flags):
        name = getattr(op, "name", repr(op))
        t_op = time.time()
        try:
            r = _sweep_op(op, flags, dim_positions, head_dim)
        except Exception as exc:
            traceback.print_exc()
            r = OpResult(
                name=name,
                kind=getattr(op, "kind", "?"),
                phase=getattr(op, "phase", None),
                layer_idx=getattr(op, "layer_idx", None),
                enabled_flags=list(flags),
                error=f"sweep_uncaught: {type(exc).__name__}: {exc}",
                bucket="build_error",
            )
        elapsed = time.time() - t_op
        print(
            f"[{i + 1:3d}/{len(ops_and_flags)}] {name:55s} "
            f"bucket={r.bucket} rules={r.total_rules} "
            f"layers={len(r.layers_checked)} t={elapsed:.2f}s",
            flush=True,
        )
        results.append(r)
    print(f"sweep total: {time.time() - t_all:.1f}s", flush=True)

    agg = _aggregate(results)

    out_dir = _PROJECT_PARENT / "c4_release" / ".agent-logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "sweep_compare_ffn_phase6_wave5a.md"
    json_path = out_dir / "sweep_compare_ffn_phase6_wave5a.json"

    md_path.write_text(_render_md(results, agg))
    json_path.write_text(json.dumps(
        {"aggregate": agg, "rows": [asdict(r) for r in results]},
        indent=2,
    ))
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(
        f"clean={agg['total_clean']} "
        f"mismatch_only={agg['total_mismatch_only']} "
        f"real_bug={agg['total_real_bug']} "
        f"synthetic_state_overflow={agg['total_synthetic_state_overflow']} "
        f"other_failure={agg['total_other_failure']} "
        f"with_ffn_rules={agg['total_ops_with_ffn_rules']}"
    )


if __name__ == "__main__":
    main()
