#!/usr/bin/env python3
"""CLI front-end for the dim-flow analyzer.

Given a residual dim name (e.g. ``STACK0_BYTE_VAL_1_LO``) this tool
compiles the unified-VM layout, walks every op's declared IR, and prints
a producer/consumer report:

  * Every op that writes the dim (layer, kind, op_name, rule, write
    value, gate conditions).
  * Every op that reads the dim (layer, kind, op_name, rule, read role,
    weight, gate conditions).
  * Zeroing writers (ops whose effective contribution to the dim is 0).
  * First-write / first-read ordering check.

Usage:

    python tools/dim_flow_audit.py STACK0_BYTE_VAL_1_LO

    python tools/dim_flow_audit.py STACK0_BYTE_VAL_1_LO --alu-mode lookup

    # Writers only:
    python tools/dim_flow_audit.py STACK0_BYTE_VAL_1_LO --writers-only

    # Readers only:
    python tools/dim_flow_audit.py STACK0_BYTE_VAL_1_LO --readers-only

    # Zeroing writers only:
    python tools/dim_flow_audit.py STACK0_BYTE_VAL_1_LO --zeroing-only

The output goes to stdout. The tool is read-only: it never writes
weights or modifies any source files.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings


# Make ``import neural_vm`` work from c4_release/tools.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


def _build_layout(args):
    from neural_vm.unified_compiler.decl_verifier import _build_layout_only

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_layout_only(
            alu_mode=args.alu_mode,
            enable_conversational_io=args.enable_conversational_io,
            enable_tool_calling=args.enable_tool_calling,
            n_heads=args.n_heads,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Static dim-flow analyzer: enumerate writers / readers of "
            "a residual dim slot across the compiled VM layout."
        ),
    )
    parser.add_argument(
        "dim_name",
        help="Residual dim slot name (e.g. STACK0_BYTE_VAL_1_LO).",
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
        "--head-dim", type=int, default=64,
        help=(
            "head_dim passed to compiler_ir_factory calls (default: 64). "
            "Used by attention head specs that key off head dim."
        ),
    )
    parser.add_argument(
        "--writers-only", action="store_true",
        help="Print only the writer list, skip readers and ordering check.",
    )
    parser.add_argument(
        "--readers-only", action="store_true",
        help="Print only the reader list, skip writers and ordering check.",
    )
    parser.add_argument(
        "--zeroing-only", action="store_true",
        help="Print only the zeroing-writer list.",
    )
    parser.add_argument(
        "--no-zeroing", action="store_true",
        help="Skip the zeroing-writer pass.",
    )
    args = parser.parse_args(argv)

    layout = _build_layout(args)

    from neural_vm.unified_compiler.dim_flow import (
        enumerate_dim_readers,
        enumerate_dim_writers,
        find_zeroing_writers,
        format_report,
        trace_dim_flow,
    )

    dim_name = args.dim_name
    if dim_name not in layout.dim_positions:
        print(
            f"ERROR: dim {dim_name!r} not found in layout. "
            f"Available dims: {len(layout.dim_positions)} declared. "
            f"(grep dim_registry.py / unified_compiler/ops/ for the "
            f"correct spelling.)",
            file=sys.stderr,
        )
        return 2

    if args.zeroing_only:
        zw = find_zeroing_writers(
            layout, dim_name, head_dim=args.head_dim,
        )
        print(f"=== zeroing writers: {dim_name} ===")
        if not zw:
            print("  <none>")
        else:
            for w in zw:
                print(
                    f"  L{w.layer_idx:02d} [{w.op_kind}] {w.op_name}"
                    f"::{w.rule_name or '-'} offset={w.offset} "
                    f"value={w.write_value_expr}"
                )
        return 0

    if args.writers_only:
        writers = enumerate_dim_writers(
            layout, dim_name, head_dim=args.head_dim,
        )
        print(f"=== writers of {dim_name} ({len(writers)}) ===")
        if not writers:
            print("  <no writers — dim is never set>")
        for w in writers:
            print(
                f"  L{w.layer_idx:02d} [{w.op_kind:4s}] {w.op_name}"
                f"::{w.rule_name or '-'} offset={w.offset} "
                f"value={w.write_value_expr}"
            )
            if w.gate_conditions:
                print(f"      gates: {', '.join(w.gate_conditions)}")
        return 0

    if args.readers_only:
        readers = enumerate_dim_readers(
            layout, dim_name, head_dim=args.head_dim,
        )
        print(f"=== readers of {dim_name} ({len(readers)}) ===")
        if not readers:
            print("  <no readers — dim is never consumed>")
        for r in readers:
            print(
                f"  L{r.layer_idx:02d} [{r.op_kind:4s}] {r.op_name}"
                f"::{r.rule_name or '-'} role={r.read_role} "
                f"offset={r.offset} w={r.weight:+g}"
            )
            if r.gate_conditions:
                print(f"      gates: {', '.join(r.gate_conditions)}")
        return 0

    # Default: full report.
    flow = trace_dim_flow(layout, dim_name, head_dim=args.head_dim)
    zw = None
    if not args.no_zeroing:
        zw = find_zeroing_writers(layout, dim_name, head_dim=args.head_dim)
    print(format_report(flow, zeroing_writers=zw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
