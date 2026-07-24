"""GENERAL fixed-point matmul self-emulation checks — an arbitrary ``M x K @ K x N``
kernel (``_matmul_general_src``) compiled through the REAL c4 toolchain and run
byte-exact through the deterministic DRAFT VM
(``nibble_pure_forward_complete.ref_interpret``).

This kernel exists to MEASURE the honest draft-VM ``steps/MAC`` rate at real
scale (see ``measure_matmul_steps_per_mac.py``): the documented ``4.75 steps/MAC``
is the rate on the UNMASKED ``src.compiler`` reference VM, NOT the byte-masking
draft VM.  These checks pin that the general kernel is byte-exact and that its
per-MAC / per-output step increments are CONSTANT (so the rate extrapolation is
sound).

All checks are CPU-only (compile + draft VM); NO neural model build, NO GPU.
    OMP_NUM_THREADS=4 python -m pytest \
        c4_min/selfhost/test_matmul_general_self_emulation.py -x -s
"""
from __future__ import annotations

from c4_min import isa
from c4_min.selfhost._matmul_general_src import (
    matmul_general_c, matmul_general_reference, SCALE)

_ALLOWED_OPS = {
    isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
    isa.LI, isa.LC, isa.SI, isa.SC,
    isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
    isa.JMP, isa.BZ, isa.BNZ, isa.JSR, isa.ENT, isa.ADJ, isa.LEV,
    isa.PRTF, isa.NOP, isa.HALT,
}


def _compile(src):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from draft VM): {over}"
    return code, data


def _run(code):
    from c4_min.nibble_pure_forward_complete import ref_interpret

    out = []
    trace = ref_interpret(code, max_steps=500000, mask=0xFFFFFFFF, out=out)
    return out, trace


def test_general_matmul_ops_are_verified_subset():
    """The compiled general matmul uses only the verified op set; every IMM is
    byte-sized (locals-only, no malloc/heap global address leak)."""
    A = [1, 2, 3, 1]
    B = [2, 1, 1, 3]
    code, _ = _compile(matmul_general_c(A, B, 2, 2, 2, SCALE))
    used = {i.op for i in code}
    assert used <= _ALLOWED_OPS, f"unexpected ops: {sorted(used - _ALLOWED_OPS)}"
    assert isa.PRTF in used and isa.MUL in used and isa.DIV in used


def test_general_matmul_byte_exact_2x2x2():
    """A full 2x2x2 matmul is byte-exact vs the fixed-point reference and never
    wraps AX negative (the byte-exactness condition)."""
    A = [1, 2, 3, 1]
    B = [2, 1, 1, 3]
    code, _ = _compile(matmul_general_c(A, B, 2, 2, 2, SCALE))
    out, trace = _run(code)
    ref = matmul_general_reference(A, B, 2, 2, 2, SCALE)
    assert out == ref, f"draft {out} != ref {ref}"
    assert not [v for v in trace if v >= 2 ** 31], "wrapped-negative AX"


def test_general_matmul_rectangular_byte_exact():
    """A rectangular 2x4x2 (unequal M,K,N) is byte-exact — the general kernel is not
    limited to square shapes."""
    A = [1, 0, 2, 1, 0, 1, 1, 0]   # 2x4
    B = [1, 1, 0, 1, 1, 0, 0, 1]   # 4x2
    code, _ = _compile(matmul_general_c(A, B, 2, 4, 2, SCALE))
    out, _ = _run(code)
    assert out == matmul_general_reference(A, B, 2, 4, 2, SCALE)


def test_inline_and_call_forms_agree():
    """The fpmul-CALL and INLINE forms produce identical result bytes (both
    byte-exact) — they differ only in per-MAC step cost, not correctness."""
    A = [2, 1, 1, 2]
    B = [1, 1, 2, 1]
    ref = matmul_general_reference(A, B, 2, 2, 2, SCALE)
    for call in (True, False):
        code, _ = _compile(matmul_general_c(A, B, 2, 2, 2, SCALE, call_fpmul=call))
        out, _ = _run(code)
        assert out == ref, f"call_fpmul={call}: {out} != {ref}"


def test_per_mac_increment_is_constant():
    """The DRAFT-VM step count grows by a CONSTANT increment per inner-loop MAC
    (K-sweep at M=N=1) — the property that makes the measured steps/MAC a sound
    rate, not a size-dependent artifact.  Grounds the ~101 steps/MAC headline."""
    steps = []
    for K in range(1, 7):
        A = [1] * K
        B = [1] * K
        code, _ = _compile(matmul_general_c(A, B, 1, K, 1, SCALE))
        out, trace = _run(code)
        assert out == matmul_general_reference(A, B, 1, K, 1, SCALE)
        steps.append(len(trace))
    incr = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    assert len(set(incr)) == 1, f"per-MAC increment not constant: {incr}"
    assert incr[0] == 101, f"expected 101 steps/MAC (fpmul-call), got {incr[0]}"


def test_per_output_increment_is_constant():
    """The step count grows by a CONSTANT increment per output element (M-sweep at
    K=N=1) — so the per-matmul cost model steps = base + b*MACs + c*outputs is a
    valid two-parameter fit."""
    steps = []
    for M in range(1, 7):
        A = [1] * M
        B = [1]
        code, _ = _compile(matmul_general_c(A, B, M, 1, 1, SCALE))
        out, trace = _run(code)
        assert out == matmul_general_reference(A, B, M, 1, 1, SCALE)
        steps.append(len(trace))
    incr = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    assert len(set(incr)) == 1, f"per-output increment not constant: {incr}"
    assert incr[0] == 205, f"expected 205 steps/output (1 MAC + overhead), got {incr[0]}"


if __name__ == "__main__":
    import sys

    test_general_matmul_ops_are_verified_subset()
    test_general_matmul_byte_exact_2x2x2()
    test_general_matmul_rectangular_byte_exact()
    test_inline_and_call_forms_agree()
    test_per_mac_increment_is_constant()
    test_per_output_increment_is_constant()
    print("all general matmul self-emulation checks passed")
    sys.exit(0)
