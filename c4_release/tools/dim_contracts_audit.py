#!/usr/bin/env python3
"""CI audit: run every registered ``DimContract`` against the live layout.

This is the fail-closed entry point for the producer-consumer contract
verifier defined in
``c4_release/neural_vm/unified_compiler/dim_contracts.py``.

Usage::

    python -m c4_release.tools.dim_contracts_audit
    python -m c4_release.tools.dim_contracts_audit --strict
    python -m c4_release.tools.dim_contracts_audit --alu-mode efficient

Exit codes:
    0  every registered contract passed
    1  at least one contract failed (errors printed to stdout)
    2  layout build failed (compile error surfaced)

The default behaviour mirrors ``decl_verifier.verify_claims_static``: it
builds a layout via the same ``_build_layout_only`` helper and walks
``ops_per_layer`` / ``block_ops`` / ``model_ops`` for op references.

The audit is intended to slot into CI alongside ``lint_raw_ffn_rule.py``
and ``audit_produces_consumes.py`` as the cross-op invariant check that
the per-op claim verifier cannot express.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from c4_release.neural_vm.unified_compiler.decl_verifier import (
    _build_layout_only,
)
from c4_release.neural_vm.unified_compiler.dim_contracts import (
    registered_dim_contracts,
    verify_all_registered_contracts,
)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--alu-mode",
        default="lookup",
        choices=("lookup", "efficient"),
        help="ALU mode passed through to the layout builder (default: lookup)",
    )
    parser.add_argument(
        "--enable-conversational-io",
        action="store_true",
        help="Enable conversational IO ops in the layout",
    )
    parser.add_argument(
        "--enable-tool-calling",
        action="store_true",
        help="Enable tool-calling ops in the layout",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Attention-head count for layout build (default: 8)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (exit 1) on any contract failure. Without --strict the "
            "audit still exits 1 when a contract fails -- this flag exists "
            "to allow future relaxation (e.g. --strict-notes to also escalate "
            "NOTE-level warnings) without changing the default."
        ),
    )
    parser.add_argument(
        "--strict-notes",
        action="store_true",
        help="Escalate NOTE-level warnings to exit-1 failures.",
    )
    args = parser.parse_args(argv)

    contracts = registered_dim_contracts()
    if not contracts:
        print(
            "No contracts registered. Import "
            "c4_release.neural_vm.unified_compiler.dim_contracts to "
            "trigger the starter set."
        )
        return 0

    print(f"Building layout (alu_mode={args.alu_mode!r}, n_heads={args.n_heads})...")
    try:
        layout = _build_layout_only(
            alu_mode=args.alu_mode,
            enable_conversational_io=args.enable_conversational_io,
            enable_tool_calling=args.enable_tool_calling,
            n_heads=args.n_heads,
        )
    except Exception as exc:
        print(f"layout build failed: {exc.__class__.__name__}: {exc}")
        return 2

    report = verify_all_registered_contracts(layout)
    print(report.format())

    n_failures = report.n_failures()
    if n_failures:
        print(
            f"\nFAILED: {n_failures}/{len(report.results)} contract(s) "
            f"violated. See errors above.",
            file=sys.stderr,
        )
        return 1
    if args.strict_notes:
        any_notes = any(r.notes for r in report.results)
        if any_notes:
            print(
                "\nFAILED (--strict-notes): one or more contracts produced "
                "NOTE-level diagnostics; rerun without --strict-notes for "
                "the default pass/fail.",
                file=sys.stderr,
            )
            return 1
    print(f"\nOK: {len(report.results)} contract(s) verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
