#!/usr/bin/env python3
"""Per-op ``Operation.reads`` / ``writes`` derivation lint.

Companion to ``tools/undeclared_dim_audit.py`` (the structural-correctness
sweep) and ``neural_vm/unified_compiler/op_introspect.py`` (the derivation
plumbing). This lint walks the production layout, runs
``assert_declared_matches_derived`` on every op, and ratchets the per-op
mismatch counts against a baseline so:

  * **New op + missing dim** in declared reads/writes -> CI fails.
  * **Existing op's mismatch grows** -> CI fails.
  * **Op removed / migrated to derive_reads_writes_flag=True** -> baseline
    entry is dropped in the same commit; ratchet walks downward.

Two modes:

  * **Default (advisory)** — declared >= derived. Surfacing missing dims
    is informational; the ratchet trips on regressions only.
  * **--strict** — declared == derived. The migration target. Used by an
    op once its ``derive_reads_writes_flag`` is True (so the lint can
    confirm dropping the manual annotation produces no drift).

Usage::

    python c4_release/tools/lint_op_reads_writes.py
    python c4_release/tools/lint_op_reads_writes.py --json
    python c4_release/tools/lint_op_reads_writes.py --strict
    python c4_release/tools/lint_op_reads_writes.py --op layer2_mem_byte_flags

Exit codes:
    0  no regression vs baseline (in --strict, additionally no over-declarations)
    1  regression: new mismatch / growing mismatch
    2  invocation / IO error

The baseline below is a per-op ``(undeclared_reads_count, undeclared_writes_count)``
snapshot frozen at the audit-doc commit (2026-06-09, commit 92a99c4c).
After every fix that lands a reduction in the audit's counts, the
corresponding entry MUST be decremented (or deleted) in the SAME commit
so the ratchet only walks downward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple


# Make ``import c4_release`` work from c4_release/tools.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
_REPO_ROOT = os.path.dirname(_PKG)  # repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Per-op baseline of (undeclared_reads, undeclared_writes) counts captured
# 2026-06-09. Mirrors the table in
# docs/UNDECLARED_DIM_AUDIT_2026_06_09.md after alias canonicalization.
# Ops not listed are expected to have a clean (0, 0) mismatch -- any
# violation on a non-baselined op is a NEW regression.
#
# Migration discipline: when a fix lands, decrement (or delete) the
# corresponding row in the SAME commit. This guarantees the lint only
# walks downward.
_BASELINE: Dict[str, Tuple[int, int]] = {
    # (op_name, (undeclared_reads, undeclared_writes))
    "layer6_routing_ffn": (24, 10),
    "layer8_alu": (16, 8),
    "tail_bit32_result_correction": (24, 0),
    "function_call_weights": (17, 5),
    "layer10_alu": (21, 0),
    "binary_pop_sp_increment": (16, 4),
    "layer6_relay_heads_bake": (16, 0),
    "layer14_mem_generation": (15, 0),
    "layer16_lev_routing": (14, 1),
    "layer13_mem_addr_gather": (10, 1),
    "io_putchar_routing": (4, 3),
    "convo_io_pc_sp_latch": (5, 2),
    "layer6_ent_after_jsr_sp_byte0_fixup": (7, 0),
    "phase_a_ffn": (5, 0),
    "layer2_mem_byte_flags": (4, 0),
    "convo_io_state_machine": (2, 3),
    "layer1_ffn": (4, 0),
    "convo_io_step_resume": (1, 3),
    "layer10_psh_ax_broadcast_bake": (3, 0),
    "layer15_alu_high_byte_relay": (3, 0),
    "layer3_convo_io_state_init": (1, 1),
    "layer3_ffn": (2, 0),
    "layer4_ffn": (1, 1),
    "layer7_memory_heads": (2, 0),
    "layer3_carry_forward_attn": (1, 0),
    "layer5_fetch": (1, 0),
    "opcode_decode_ffn": (1, 0),
    "layer10_psh_stack0_passthrough_bake": (1, 0),
    "layer15_si_mem_addr0_from_stack0": (1, 0),
}


def _build_layout(args):
    from c4_release.neural_vm.unified_compiler.decl_verifier import (
        _build_layout_only,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_layout_only(
            alu_mode=args.alu_mode,
            enable_conversational_io=args.enable_conversational_io,
            enable_tool_calling=args.enable_tool_calling,
            n_heads=args.n_heads,
        )


def _build_alias_groups(dim_positions, dim_sizes) -> Dict[str, str]:
    """Mirror of ``undeclared_dim_audit._build_alias_groups``.

    Canonicalizes position-alias dim names (e.g.
    ``OUTPUT_HI_THIS_STEP`` -> ``OUTPUT_HI``) so the lint's mismatch
    counts line up with the audit doc.
    """
    groups: Dict[Tuple[int, int], List[str]] = {}
    for name, pos in dim_positions.items():
        size = dim_sizes.get(name, 1)
        groups.setdefault((pos, size), []).append(name)
    canon: Dict[str, str] = {}
    for (_pos, _size), names in groups.items():
        chosen = sorted(names, key=lambda n: (len(n), n))[0]
        for n in names:
            canon[n] = chosen
    return canon


def lint_layout(
    layout,
    *,
    strict: bool = False,
    op_filter: Optional[str] = None,
) -> Dict[str, object]:
    """Run derivation lint over every op in ``layout``.

    Returns a dict with:
      - ``mismatches``: list of per-op records with undeclared counts.
      - ``regressions``: per-op records that exceed baseline.
      - ``new_violations``: per-op records NOT in baseline with non-zero counts.
      - ``removed_baselined``: baselined op_names not seen this run.
      - ``clean_baselined``: baselined ops that now have (0, 0) counts.
    """
    from c4_release.neural_vm.unified_compiler.dim_flow import (
        _walk_ops_with_layers,
    )
    from c4_release.neural_vm.unified_compiler.op_introspect import (
        assert_declared_matches_derived,
    )

    dim_positions = layout.dim_positions
    dim_sizes = getattr(layout, "dim_sizes", {})
    alias_canon = _build_alias_groups(dim_positions, dim_sizes)

    seen: set = set()
    mismatches: List[Dict[str, object]] = []
    regressions: List[Dict[str, object]] = []
    new_violations: List[Dict[str, object]] = []
    clean_baselined: List[str] = []

    for _layer_idx, op in _walk_ops_with_layers(layout):
        name = getattr(op, "name", "<anonymous>")
        if name in seen:
            continue
        seen.add(name)
        if op_filter is not None and name != op_filter:
            continue

        m = assert_declared_matches_derived(
            op,
            dim_positions=dim_positions,
            dim_sizes=dim_sizes,
            strict=strict,
            alias_canon=alias_canon,
        )
        ur = len(m.undeclared_reads)
        uw = len(m.undeclared_writes)
        # Under strict mode, over-declarations count too.
        if strict:
            ur += len(m.over_declared_reads)
            uw += len(m.over_declared_writes)

        record = {
            "op_name": name,
            "undeclared_reads": list(m.undeclared_reads),
            "undeclared_writes": list(m.undeclared_writes),
            "over_declared_reads": list(m.over_declared_reads),
            "over_declared_writes": list(m.over_declared_writes),
            "ur_count": ur,
            "uw_count": uw,
        }

        if ur == 0 and uw == 0:
            if name in _BASELINE:
                clean_baselined.append(name)
            continue

        mismatches.append(record)

        base = _BASELINE.get(name)
        if base is None:
            new_violations.append(record)
        else:
            base_ur, base_uw = base
            if ur > base_ur or uw > base_uw:
                record["baseline_reads"] = base_ur
                record["baseline_writes"] = base_uw
                regressions.append(record)

    removed = [n for n in _BASELINE if n not in seen]

    return {
        "mismatches": mismatches,
        "regressions": regressions,
        "new_violations": new_violations,
        "removed_baselined": removed,
        "clean_baselined": clean_baselined,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--alu-mode", default="efficient",
        choices=("efficient", "lookup"),
    )
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--enable-conversational-io", action="store_true")
    p.add_argument("--enable-tool-calling", action="store_true")
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require declared == derived (no over-declaration). The "
            "migration target — used after an op flips "
            "derive_reads_writes_flag=True."
        ),
    )
    p.add_argument(
        "--op",
        default=None,
        help="Lint a single op by name (skip every other op).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit per-op mismatches as JSON.",
    )
    args = p.parse_args(argv)

    layout = _build_layout(args)
    result = lint_layout(layout, strict=args.strict, op_filter=args.op)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        if result["regressions"] or result["new_violations"]:
            return 1
        return 0

    print("=== lint_op_reads_writes ===")
    mode = "STRICT" if args.strict else "ADVISORY (declared >= derived)"
    print(f"mode: {mode}")
    print(
        f"baselined ops: {len(_BASELINE)} "
        f"(of which {len(result['clean_baselined'])} now clean)"
    )
    print(f"current ops with mismatches: {len(result['mismatches'])}")
    print()

    if result["new_violations"]:
        print(
            f"NEW VIOLATIONS ({len(result['new_violations'])}) — ops not in baseline:"
        )
        for r in result["new_violations"]:
            print(
                f"  {r['op_name']}: ur={r['ur_count']}, uw={r['uw_count']}"
            )
            if r["undeclared_reads"]:
                print(f"    undeclared reads:  {r['undeclared_reads']}")
            if r["undeclared_writes"]:
                print(f"    undeclared writes: {r['undeclared_writes']}")
            if args.strict and r["over_declared_reads"]:
                print(f"    over_declared reads:  {r['over_declared_reads']}")
            if args.strict and r["over_declared_writes"]:
                print(f"    over_declared writes: {r['over_declared_writes']}")
        print()

    if result["regressions"]:
        print(
            f"REGRESSIONS ({len(result['regressions'])}) — ops that grew beyond baseline:"
        )
        for r in result["regressions"]:
            print(
                f"  {r['op_name']}: ur={r['ur_count']} "
                f"(baseline {r['baseline_reads']}), "
                f"uw={r['uw_count']} (baseline {r['baseline_writes']})"
            )
        print()

    if result["clean_baselined"]:
        print(
            f"CLEAN BASELINED ({len(result['clean_baselined'])}) — drop these from _BASELINE:"
        )
        for name in result["clean_baselined"]:
            print(f"  {name}")
        print()

    if result["removed_baselined"]:
        print(
            f"REMOVED BASELINED ({len(result['removed_baselined'])}) — these ops no longer exist:"
        )
        for name in result["removed_baselined"]:
            print(f"  {name}")
        print()

    if not result["regressions"] and not result["new_violations"]:
        print("OK — no regressions vs baseline.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
