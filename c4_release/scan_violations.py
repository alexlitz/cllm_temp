"""Corpus-wide strength + scope violation scan.

Per-op reports saved to .agent-logs/violation_catalog_2026_06_01/per_op/<op>.txt.
A summary index is saved to .agent-logs/violation_catalog_2026_06_01/INDEX.md.

Set CUDA_VISIBLE_DEVICES="" before running.
"""
import os
import sys
import time
import traceback
from pathlib import Path

OUT_DIR = Path(".agent-logs/violation_catalog_2026_06_01")
PER_OP_DIR = OUT_DIR / "per_op"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_OP_DIR.mkdir(parents=True, exist_ok=True)

print(f"[t={time.time():.1f}] start", flush=True)

from neural_vm.dim_registry import build_default_registry
from neural_vm.verification.decl_verifier import (
    verify_rule_strength,
    verify_rule_scopes,
    _collect_ffn_rules_from_op,
)
print(f"[t={time.time():.1f}] imports done", flush=True)

try:
    from neural_vm.unified_compiler.backbone_bounds import load_default_bounds
    bounds_obj = load_default_bounds()
    bb = bounds_obj.as_strength_bound if bounds_obj is not None else None
except Exception as e:
    print(f"[warn] load_default_bounds failed: {e}", flush=True)
    bb = None

registry = build_default_registry()


def _factory_table():
    # Smaller ops first so per-op reports land quickly; L10 last (2059 rules — slowest).
    table = []
    from neural_vm.unified_compiler.ops.l0_ops import make_phase_a_ffn_op
    table.append(("phase_a_ffn", make_phase_a_ffn_op))
    from neural_vm.unified_compiler.ops.l8_ops import (
        make_layer8_multibyte_routing_op,
    )
    table.append(("layer8_multibyte_routing", make_layer8_multibyte_routing_op))
    from neural_vm.unified_compiler.ops.l15_ops import make_layer15_nibble_copy_op
    table.append(("layer15_nibble_copy", make_layer15_nibble_copy_op))
    from neural_vm.unified_compiler.ops.l16_ops import make_layer16_lev_routing_op
    table.append(("layer16_lev_routing", make_layer16_lev_routing_op))
    from neural_vm.unified_compiler.ops.l10_ops import (
        make_tail_bit32_result_correction_op,
    )
    table.append(("tail_bit32_result_correction", make_tail_bit32_result_correction_op))
    return table


factories = _factory_table()
ops = []
for name, f in factories:
    try:
        op = f()
        ops.append((name, op, len(_collect_ffn_rules_from_op(op))))
    except Exception:
        traceback.print_exc()

print(f"[t={time.time():.1f}] collected {len(ops)} ops:", flush=True)
for n, _, nr in ops:
    print(f"   {nr:>5} rules  {n}", flush=True)


def _format_strength(issues, limit=50):
    lines = []
    sv = [i for i in issues if i.get("kind") == "strength_violation"]
    nd = [i for i in issues if i.get("kind") == "no_dominates_at"]
    other = [i for i in issues if i.get("kind") not in ("strength_violation", "no_dominates_at")]
    lines.append(f"strength_violation: {len(sv)}")
    lines.append(f"no_dominates_at: {len(nd)}")
    lines.append(f"other: {len(other)}")
    lines.append("")
    if sv:
        lines.append("--- strength_violation top ---")
        for i in sv[:limit]:
            lines.append(
                f"  rule={i.get('rule')!r} dim={i.get('output_dim')!r} "
                f"my={i.get('my_contribution', 0):.1f} "
                f"comp={i.get('competing_max', 0):.1f} "
                f"bb={i.get('backbone_max', 0):.1f} "
                f"shortfall={i.get('shortfall', 0):.1f} "
                f"top={i.get('top_competitor')!r}"
            )
        if len(sv) > limit:
            lines.append(f"  ... and {len(sv)-limit} more")
        lines.append("")
    if nd:
        lines.append("--- no_dominates_at top ---")
        for i in nd[:20]:
            lines.append(
                f"  rule={i.get('rule')!r} output_dim={i.get('output_dim','?')!r}"
            )
        if len(nd) > 20:
            lines.append(f"  ... and {len(nd)-20} more")
        lines.append("")
    if other:
        lines.append("--- other kinds ---")
        for i in other[:20]:
            lines.append(f"  {i}")
        if len(other) > 20:
            lines.append(f"  ... and {len(other)-20} more")
    return "\n".join(lines)


def _format_scope(issues, limit=50):
    lines = []
    lines.append(f"scope_violation: {len(issues)}")
    lines.append("")
    if issues:
        lines.append("--- scope top ---")
        for i in issues[:limit]:
            reason = str(i.get("reason", ""))
            if len(reason) > 300:
                reason = reason[:300] + "..."
            lines.append(
                f"  rule={i.get('rule')!r} kind={i.get('kind')} reason={reason}"
            )
        if len(issues) > limit:
            lines.append(f"  ... and {len(issues)-limit} more")
    return "\n".join(lines)


index_rows = []

for name, op, n_rules in ops:
    t0 = time.time()
    print(f"[t={time.time():.1f}] {name}: solo strength on {n_rules} rules...", flush=True)
    try:
        s_issues = verify_rule_strength(op, registry, backbone_bounds=bb)
    except Exception:
        traceback.print_exc()
        s_issues = []
    t_solo = time.time() - t0

    print(f"[t={time.time():.1f}] {name}: solo done in {t_solo:.1f}s, {len(s_issues)} issues; cross-op...", flush=True)
    t0 = time.time()
    competition = [o for n, o, _ in ops if n != name]
    try:
        s_cross = verify_rule_strength(
            op, registry, backbone_bounds=bb, ops_for_competition=competition,
        )
    except Exception:
        traceback.print_exc()
        s_cross = s_issues
    t_cross = time.time() - t0

    print(f"[t={time.time():.1f}] {name}: cross-op done in {t_cross:.1f}s, {len(s_cross)} issues", flush=True)

    t0 = time.time()
    try:
        sc_issues = verify_rule_scopes(op, registry, require_scope=False)
    except Exception:
        traceback.print_exc()
        sc_issues = []
    t_scope = time.time() - t0
    print(f"[t={time.time():.1f}] {name}: scope done in {t_scope:.1f}s, {len(sc_issues)} issues", flush=True)

    sv_n = sum(1 for i in s_cross if i.get("kind") == "strength_violation")
    nd_n = sum(1 for i in s_cross if i.get("kind") == "no_dominates_at")
    sc_n = len(sc_issues)

    report_path = PER_OP_DIR / f"{name}.txt"
    body = [
        f"# Op: {name}",
        f"Total rules: {n_rules}",
        f"Solo strength runtime: {t_solo:.1f}s",
        f"Cross-op strength runtime: {t_cross:.1f}s",
        f"Scope runtime: {t_scope:.1f}s",
        "",
        "## STRENGTH (cross-op)",
        _format_strength(s_cross),
        "",
        "## SCOPE",
        _format_scope(sc_issues),
        "",
    ]
    report_path.write_text("\n".join(body))
    print(f"[t={time.time():.1f}] {name}: wrote {report_path}", flush=True)

    index_rows.append((name, n_rules, sv_n, nd_n, sc_n, t_solo + t_cross + t_scope))

# Write index
index_path = OUT_DIR / "INDEX.md"
lines = [
    "# Violation catalog — 2026-06-01",
    "",
    f"Generated at t={time.time():.0f}.",
    "Files: per_op/<op>.txt",
    "",
    "| Op | Rules | strength_violation | no_dominates_at | scope_violation | runtime |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, nr, sv_n, nd_n, sc_n, runtime in sorted(index_rows, key=lambda r: -r[2]):
    lines.append(
        f"| {name} | {nr} | {sv_n} | {nd_n} | {sc_n} | {runtime:.1f}s |"
    )
totals = (
    sum(r[2] for r in index_rows),
    sum(r[3] for r in index_rows),
    sum(r[4] for r in index_rows),
)
lines += [
    f"| **TOTAL** | {sum(r[1] for r in index_rows)} | {totals[0]} | {totals[1]} | {totals[2]} | — |",
    "",
]
index_path.write_text("\n".join(lines))
print(f"[t={time.time():.1f}] INDEX written to {index_path}", flush=True)
print(f"DONE", flush=True)
