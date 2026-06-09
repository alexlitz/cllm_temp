#!/usr/bin/env python3
"""Symbolic forward audit — dump a slot's residual trace for a program.

Drives :class:`SymbolicForwardRunner` through a small canned program
(or one loaded from a file) and prints the per-(step, block) value of
a chosen residual dim. Useful for byte-identity diagnosis without
spending the ~1-2 minute cold-compile cost of the real bake.

Usage
-----

Demo: ``IMM 0x200; PSH; EXIT`` against the full compiled layout,
trace dim ``STACK0_BYTE_VAL_1_LO+2``::

    python -m c4_release.tools.symbolic_forward_audit \\
        --program imm_psh_exit \\
        --dim STACK0_BYTE_VAL_1_LO+2

With a custom hex bytecode string::

    python -m c4_release.tools.symbolic_forward_audit \\
        --bytecode-hex "0x000020001,0x0000000d,0x00000026" \\
        --dim OUTPUT_LO+5

Without a real compile (fast path — uses the declarations-only
layout)::

    python -m c4_release.tools.symbolic_forward_audit \\
        --program imm_psh_exit \\
        --dim STACK0_BYTE_VAL_1_LO+2 \\
        --declarations-only

Set ``--max-step N`` to cap the number of token steps; useful for
multi-instruction programs where you only want the first few rows.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


# ---------------------------------------------------------------------------
# Canned demo programs
# ---------------------------------------------------------------------------


def _demo_imm_psh_exit() -> List[int]:
    """The canonical 3-step demo: IMM 0x200; PSH; EXIT."""
    from c4_release.neural_vm.unified_compiler.symbolic_forward import (
        OP_EXIT,
        OP_IMM,
        OP_PSH,
        encode_instr,
    )

    return [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_EXIT),
    ]


_PROGRAMS = {
    "imm_psh_exit": _demo_imm_psh_exit,
}


def _parse_bytecode_hex(hex_str: str) -> List[int]:
    """Parse a comma-separated list of hex words into an int list."""
    out: List[int] = []
    for token in hex_str.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token, 16) if token.startswith(("0x", "0X")) else int(token))
    return out


# ---------------------------------------------------------------------------
# Compiler-loading
# ---------------------------------------------------------------------------


def _load_compiler(declarations_only: bool):
    """Return a layout-like object the runner can drive.

    By default, runs the real ``compile_full_vm_dynamic`` and returns
    the resulting ``ModelLayout``. With ``declarations_only=True``,
    sets the flag the compiler honours to skip the actual weight bakes
    (significantly faster for diagnosis-only use).
    """
    from c4_release.neural_vm.unified_compiler import compile_full_vm_dynamic

    _, layout = compile_full_vm_dynamic(
        declarations_only=declarations_only,
    )
    return layout


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_trace(entries, dim_name: str) -> None:
    if not entries:
        print(f"(no recorded writes to {dim_name})")
        return
    print(f"## Trace for `{dim_name}`\n")
    print("| step | position | block | value |")
    print("| ---: | ---: | ---: | ---: |")
    for e in entries:
        print(
            f"| {e.step_idx} | {e.position} | {e.block_idx} | "
            f"{e.value:.6f} |"
        )


def _print_program_summary(program: List[int]) -> None:
    from c4_release.neural_vm.unified_compiler.symbolic_forward import (
        decode_instr,
    )

    print("## Program\n")
    print("| pc | op | imm |")
    print("| ---: | ---: | ---: |")
    for pc, word in enumerate(program):
        op, imm = decode_instr(word)
        print(f"| {pc} | {op} | 0x{imm:06x} |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="symbolic_forward_audit",
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--program", choices=sorted(_PROGRAMS),
        help="Named canned demo program",
    )
    grp.add_argument(
        "--bytecode-hex",
        help='Comma-separated hex words, e.g. "0x00000201,0x0000000d,0x00000026"',
    )
    p.add_argument(
        "--dim", required=True,
        help="Residual dim to trace (e.g. STACK0_BYTE_VAL_1_LO+2)",
    )
    p.add_argument(
        "--max-step", type=int, default=None,
        help="Cap the number of token steps to run",
    )
    p.add_argument(
        "--declarations-only", action="store_true",
        help="Use the declarations-only compile path (skips weight bakes)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from c4_release.neural_vm.unified_compiler.symbolic_forward import (
        SymbolicForwardRunner,
    )

    if args.program is not None:
        program = _PROGRAMS[args.program]()
    else:
        program = _parse_bytecode_hex(args.bytecode_hex)
    if args.max_step is not None:
        program = program[: args.max_step]
    _print_program_summary(program)

    print("## Compiling layout (this may take ~30-60s for the full path)\n")
    layout = _load_compiler(declarations_only=args.declarations_only)
    runner = SymbolicForwardRunner(layout, program)
    runner.run_all()

    entries = runner.get_dim_trace(args.dim)
    _print_trace(entries, args.dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
