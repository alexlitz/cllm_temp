#!/usr/bin/env python3
"""Structural correctness sweep: undeclared reads/writes per op.

For every op in the compiled VM layout, this tool walks the op's
``compiler_ir`` (FFN rules + attention head specs) and compares the
*actual* dim names read/written against the op's *declared*
``Operation.reads`` / ``Operation.writes`` sets.

Mismatches surface two failure modes:

* **Undeclared reads** — IR consumes a dim that the op's contract does
  not list. Harms IR-as-contract guarantees; soft failure.
* **Undeclared writes** — IR produces a dim that the contract does not
  list. Hard failure: the compiler's dep graph uses ``writes`` to
  compute scheduling edges, so undeclared writes can silently sit
  alongside or upstream of a downstream reader without the schedule
  noticing.

Position aliases (e.g. ``OUTPUT_HI`` ↔ ``OUTPUT_HI_THIS_STEP`` at slot
85) are collapsed before comparison so that they don't show up as
false positives. Declared dim names with offset (``DIM+N``) or
step-alias (``DIM.*.-1``) suffixes are stripped to the base name.

Usage::

    python -m c4_release.tools.undeclared_dim_audit
    python -m c4_release.tools.undeclared_dim_audit --writes-only
    python -m c4_release.tools.undeclared_dim_audit --json > audit.json

Exit codes:
    0  always (read-only audit). Counts are printed to stdout.

This is the companion of ``dim_flow_audit`` — the latter is dim-
centric ("who reads/writes ``X``?"), this one is op-centric ("what
does op ``Y`` actually read/write that it didn't declare?").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple


# Make ``import c4_release`` work from c4_release/tools.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
_REPO_ROOT = os.path.dirname(_PKG)  # repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


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


def _normalize_declared(declared: Set[str]) -> Set[str]:
    """Strip alias / offset suffixes from declared names."""
    out: Set[str] = set()
    for n in declared:
        if not isinstance(n, str):
            continue
        base = n.split(".", 1)[0].split("+", 1)[0]
        out.add(base)
    return out


def _build_alias_groups(dim_positions, dim_sizes) -> Dict[str, str]:
    """Canonicalize position-alias groups to a single name per (pos, size)."""
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


def _ir_dim_sets(op, dim_positions, dim_sizes, head_dim: int = 64
                 ) -> Tuple[Set[str], Set[str]]:
    """Return (actual_reads, actual_writes) by walking the op's IR."""
    from c4_release.neural_vm.unified_compiler.dim_flow import (
        _materialize_op_ir,
        _ffn_rules_from_ir,
        _attention_heads_from_ir,
        _dim_int_to_name,
    )

    ir = _materialize_op_ir(op, dim_positions, head_dim)
    reads: Set[str] = set()
    writes: Set[str] = set()
    if ir is None:
        return reads, writes

    for rule in _ffn_rules_from_ir(ir):
        for ct in rule.conditions:
            reads.add(ct.dim.name)
        if rule.gate is not None:
            reads.add(rule.gate.name)
        for ct in rule.gate_terms:
            reads.add(ct.dim.name)
        for wt in rule.writes:
            if wt.weight != 0.0:
                writes.add(wt.dim.name)

    for head in _attention_heads_from_ir(ir):
        spec = getattr(head, "spec", None)
        if spec is None:
            continue
        for w in getattr(spec, "q", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
            if r is not None:
                reads.add(r[0])
        for w in getattr(spec, "k", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
            if r is not None:
                reads.add(r[0])
        for w in getattr(spec, "v", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.dim), dim_positions, dim_sizes)
            if r is not None:
                reads.add(r[0])
        for w in getattr(spec, "o", ()):
            if w.weight == 0.0:
                continue
            r = _dim_int_to_name(int(w.out_dim), dim_positions, dim_sizes)
            if r is not None:
                writes.add(r[0])

    return reads, writes


def audit_layout(layout, *, kinds: Optional[Set[str]] = None
                 ) -> Dict[str, Any]:
    """Walk all ops, return per-op findings + aggregate counts.

    ``kinds`` (optional): if set, only audit ops whose ``kind`` is in this
    set (e.g. ``{"ffn", "attn"}`` to focus on layer ops).
    """
    from c4_release.neural_vm.unified_compiler.dim_flow import (
        _walk_ops_with_layers,
    )

    dim_positions = layout.dim_positions
    dim_sizes = getattr(layout, "dim_sizes", {})
    alias_to_canon = _build_alias_groups(dim_positions, dim_sizes)

    def canon_set(s: Set[str]) -> Set[str]:
        return {alias_to_canon.get(n, n) for n in s}

    findings: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for layer_idx, op in _walk_ops_with_layers(layout):
        op_name = getattr(op, "name", "<anonymous>")
        if op_name in seen:
            continue
        seen.add(op_name)
        kind = getattr(op, "kind", "?")
        if kinds is not None and kind not in kinds:
            continue

        declared_reads = _normalize_declared(
            getattr(op, "reads", set()) or set()
        )
        declared_writes = _normalize_declared(
            getattr(op, "writes", set()) or set()
        )
        actual_reads, actual_writes = _ir_dim_sets(
            op, dim_positions, dim_sizes
        )

        ur = sorted(canon_set(actual_reads) - canon_set(declared_reads))
        uw = sorted(canon_set(actual_writes) - canon_set(declared_writes))

        if ur or uw:
            findings.append({
                "layer_idx": layer_idx,
                "op_name": op_name,
                "kind": kind,
                "declared_reads": sorted(declared_reads),
                "declared_writes": sorted(declared_writes),
                "actual_reads": sorted(actual_reads),
                "actual_writes": sorted(actual_writes),
                "undeclared_reads": ur,
                "undeclared_writes": uw,
            })

    return {
        "total_ops_audited": len(seen),
        "ops_with_undeclared_reads": sum(
            1 for f in findings if f["undeclared_reads"]
        ),
        "ops_with_undeclared_writes": sum(
            1 for f in findings if f["undeclared_writes"]
        ),
        "total_undeclared_reads": sum(
            len(f["undeclared_reads"]) for f in findings
        ),
        "total_undeclared_writes": sum(
            len(f["undeclared_writes"]) for f in findings
        ),
        "findings": findings,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Static structural-correctness sweep: enumerate ops whose IR "
            "reads/writes dims not in their declared reads/writes sets."
        ),
    )
    parser.add_argument(
        "--alu-mode", default="efficient",
        choices=("efficient", "lookup"),
        help="ALU mode for the layout build (default: efficient).",
    )
    parser.add_argument(
        "--n-heads", type=int, default=8,
        help="num_heads for the layout build (default: 8).",
    )
    parser.add_argument(
        "--enable-conversational-io", action="store_true",
        help="Build the layout with conversational IO ops enabled.",
    )
    parser.add_argument(
        "--enable-tool-calling", action="store_true",
        help="Build the layout with tool-calling ops enabled.",
    )
    parser.add_argument(
        "--writes-only", action="store_true",
        help="Only print ops with undeclared writes.",
    )
    parser.add_argument(
        "--layer-ops-only", action="store_true",
        help="Restrict to kind=ffn / kind=attn ops (skip block/model).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit full audit data as JSON (one object).",
    )
    args = parser.parse_args(argv)

    layout = _build_layout(args)
    kinds = {"ffn", "attn"} if args.layer_ops_only else None
    result = audit_layout(layout, kinds=kinds)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"=== undeclared-dim audit ===")
    print(f"total ops audited: {result['total_ops_audited']}")
    print(
        f"ops with undeclared reads:  "
        f"{result['ops_with_undeclared_reads']} "
        f"(total {result['total_undeclared_reads']} dim names)"
    )
    print(
        f"ops with undeclared writes: "
        f"{result['ops_with_undeclared_writes']} "
        f"(total {result['total_undeclared_writes']} dim names)"
    )
    print()
    print(f"{'Layer':>6} {'Op':<55} {'Kind':>6} {'UR':>3} {'UW':>3}")
    print("-" * 80)
    for f in result["findings"]:
        if args.writes_only and not f["undeclared_writes"]:
            continue
        print(
            f"{f['layer_idx']:>6} {f['op_name']:<55} "
            f"{f['kind']:>6} {len(f['undeclared_reads']):>3} "
            f"{len(f['undeclared_writes']):>3}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
