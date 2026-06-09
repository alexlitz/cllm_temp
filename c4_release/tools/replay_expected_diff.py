#!/usr/bin/env python3
"""CLI: replay a C4 program against the symbolic forward runner and diff
against the reference oracle. Prints the first divergent block for a
chosen dim family.

Usage:

    python c4_release/tools/replay_expected_diff.py \\
        --program "IMM 0x200; PSH; EXIT" \\
        --dim STACK0_BYTE_VAL_1_LO

Notes
-----

* The default compile path is the dynamic full-VM compile
  (``compile_full_vm_dynamic``). This is the load-bearing path used by
  the rest of the codebase; cold compile takes ~1-2 minutes the first
  time and is disk-cached after.
* Pass ``--declarations-only`` to use the faster declarations-only
  layout (skips weight-baking — sufficient for the diff since we only
  need ``ops_per_layer`` + ``dim_positions`` for ``SymbolicForwardRunner``).
* Pass ``--list-supported`` to dump the dim families the oracle can
  diff. Anything in :data:`dim_oracle.DEFERRED_DIM_FAMILIES` will raise
  a friendly error.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``c4_release`` importable when this script is run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from c4_release.neural_vm.unified_compiler.dim_diff import (
    diff_actual_vs_expected,
    find_first_divergent_block,
    format_divergence,
)
from c4_release.neural_vm.unified_compiler.dim_oracle import (
    DEFERRED_DIM_FAMILIES,
    SUPPORTED_DIM_FAMILIES,
    ReferenceOracle,
    is_supported_dim,
)
from c4_release.neural_vm.unified_compiler.symbolic_forward import (
    SymbolicForwardRunner,
    encode_instr,
    OP_ADD,
    OP_ADJ,
    OP_BNZ,
    OP_BZ,
    OP_ENT,
    OP_EXIT,
    OP_IMM,
    OP_JMP,
    OP_JSR,
    OP_LEA,
    OP_LEV,
    OP_LI,
    OP_MUL,
    OP_PSH,
    OP_SI,
    OP_SUB,
)


_OPCODE_TABLE = {
    "LEA": OP_LEA, "IMM": OP_IMM, "JMP": OP_JMP, "JSR": OP_JSR,
    "BZ": OP_BZ, "BNZ": OP_BNZ, "ENT": OP_ENT, "ADJ": OP_ADJ,
    "LEV": OP_LEV, "LI": OP_LI, "SI": OP_SI, "PSH": OP_PSH,
    "ADD": OP_ADD, "SUB": OP_SUB, "MUL": OP_MUL, "EXIT": OP_EXIT,
}


def parse_program(text: str):
    """Parse a ``"IMM 0x200; PSH; EXIT"`` style program into bytecode.

    Each instruction is ``OP [imm]``; instructions are separated by
    ``;``. ``imm`` is decimal or hex (``0x...``). Whitespace is ignored.
    """

    bytecode = []
    parts = [p.strip() for p in text.split(";") if p.strip()]
    for part in parts:
        tokens = part.split()
        if not tokens:
            continue
        op_name = tokens[0].upper()
        if op_name not in _OPCODE_TABLE:
            raise SystemExit(
                f"replay_expected_diff: unknown opcode {op_name!r}. "
                f"Known: {sorted(_OPCODE_TABLE)}"
            )
        op = _OPCODE_TABLE[op_name]
        imm = 0
        if len(tokens) >= 2:
            imm_str = tokens[1]
            imm = int(imm_str, 0)  # autodetect 0x prefix
        bytecode.append(encode_instr(op, imm))
    return bytecode


def _list_supported_and_exit() -> None:
    print("Supported dim families (oracle has a projection rule):")
    for name in SUPPORTED_DIM_FAMILIES:
        print(f"  {name}")
    print()
    print("Deferred dim families (cross-step / multi-pass; projection")
    print("not yet modeled):")
    for name in DEFERRED_DIM_FAMILIES:
        print(f"  {name}")
    sys.exit(0)


def _build_compiler(declarations_only: bool):
    """Return a compiled object exposing ``ops_per_layer`` + ``dim_positions``.

    Uses :func:`compile_full_vm_dynamic` with ``declarations_only=True``
    by default — the runner only needs the schedule + dim layout, not
    the lowered weights.
    """

    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    print(
        f"[replay_expected_diff] compiling VM "
        f"(declarations_only={declarations_only})... "
        "this can take 1-2 minutes on a cold cache.",
        file=sys.stderr,
    )
    result = compile_full_vm_dynamic(
        declarations_only=declarations_only,
        disk_cache=True,
    )
    # compile_full_vm_dynamic returns (model, layout).
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diff symbolic_forward actual against dim_oracle expected for "
            "a single dim family. Prints the first divergent block."
        )
    )
    parser.add_argument(
        "--program", "-p",
        default="IMM 0x200; PSH; EXIT",
        help=(
            "Semicolon-separated instruction list, e.g. "
            "'IMM 0x200; PSH; EXIT'. Default is the A3.5 demo program."
        ),
    )
    parser.add_argument(
        "--dim", "-d", default="STACK0_BYTE_VAL_1_LO",
        help=(
            "Dim family to diff (e.g. STACK0_BYTE_VAL_1_LO). Default is "
            "the A3.5 broadcast probe target."
        ),
    )
    parser.add_argument(
        "--declarations-only", action="store_true", default=True,
        help=(
            "Skip weight baking (fast path). Sufficient for the diff: "
            "the runner only needs ops_per_layer + dim_positions."
        ),
    )
    parser.add_argument(
        "--full-compile", action="store_true",
        help=(
            "Force the full bake (slow). Overrides --declarations-only. "
            "Only useful when downstream tooling needs the lowered weights."
        ),
    )
    parser.add_argument(
        "--atol", type=float, default=0.5,
        help=(
            "Absolute tolerance for actual-vs-expected comparison. Default "
            "0.5 flags both fully-zero divergences (writer never fired) and "
            "scaling mismatches; raise to ~1.0 if you only want fully-zero "
            "divergences (the symbolic_forward runner's S=100 scaling "
            "produces ~0.01 per firing vs the oracle's 1.0)."
        ),
    )
    parser.add_argument(
        "--list-supported", action="store_true",
        help="Print the supported / deferred dim families and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_supported:
        _list_supported_and_exit()

    if not is_supported_dim(args.dim):
        print(
            f"[replay_expected_diff] dim {args.dim!r} is not in "
            f"SUPPORTED_DIM_FAMILIES. Use --list-supported to see the "
            f"families the oracle covers.",
            file=sys.stderr,
        )
        return 2

    program = parse_program(args.program)
    print(
        f"[replay_expected_diff] program ({len(program)} instructions): "
        f"{args.program}",
        file=sys.stderr,
    )

    declarations_only = not args.full_compile
    compiler = _build_compiler(declarations_only)

    runner = SymbolicForwardRunner(compiler, program)
    runner.run_all()

    oracle = ReferenceOracle(program)

    diff = find_first_divergent_block(runner, oracle, args.dim, atol=args.atol)
    if diff is None:
        print(
            f"OK: no divergence found for dim {args.dim} across "
            f"{runner.n_blocks} blocks (atol={args.atol})."
        )
        return 0

    print(format_divergence(diff))
    print(f"  divergences in this block: {len(diff.divergences)}")
    for d in diff.divergences[:5]:
        print(
            f"    step={d.step_idx} pos={d.position} dim={d.dim_key} "
            f"actual={d.actual:g} expected={d.expected:g}"
        )
    if len(diff.divergences) > 5:
        print(f"    ... and {len(diff.divergences) - 5} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
