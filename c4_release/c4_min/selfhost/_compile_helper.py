"""_compile_helper.py — compile a paged fixed-point dot to c4 ISA once.

Split out of ``_matmul_paged_src._compile`` because that helper asserts no
``IMM>255`` leaked and runs the BYTE VM with a wrapped-negative guard — both are
fine for the byte-window proof but the true-self-emulation grounding runs the
32-bit VM where large accumulators are EXPECTED (not an error).  This helper just
compiles; the caller runs whichever VM it wants."""
from __future__ import annotations

from typing import List

from c4_min import isa
from c4_min.selfhost._matmul_paged_src import paged_dot_c


def compile_paged_dot(w: List[int], x: List[int]):
    """Compile a paged fixed-point dot to a list of ``isa.Instr`` (c4 toolchain).
    Asserts no ``IMM>255`` leaked (the paging invariant) but does NOT run any VM."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(paged_dot_c(w, x))
    code = bytecode_to_isa(bc)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked: {over}"
    return code
