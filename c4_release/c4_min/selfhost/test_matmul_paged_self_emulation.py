"""PAGED fixed-point matmul self-emulation checks — a dot / matmul of ARBITRARY
contraction ``K`` (``_matmul_paged_src``) compiled through the REAL c4 toolchain
and run byte-exact through the deterministic DRAFT VM
(``nibble_pure_forward_complete.ref_interpret``).

The paged kernel removes the ``N <= 15`` window that capped ``_matmul_general_src``
/ ``_matmul_coo_src`` (the draft VM masks BOTH the ``LEA`` frame address AND the
``JSR`` return PC to a byte).  These checks pin that the paged kernel is byte-exact
at K = 15, 16, 64, 104 (the tiny model's real K) — i.e. past the old N<=15 cliff —
so the grounded step counts (``ground_paged_matmul_steps`` / ``ground_full_model_steps``)
run on a byte-faithful kernel.

CPU-only (compile + draft VM); NO neural model build, NO GPU.
    OMP_NUM_THREADS=4 python -m pytest \
        c4_min/selfhost/test_matmul_paged_self_emulation.py -x -s
"""
from __future__ import annotations

import random

import pytest

from c4_min import isa
from c4_min.selfhost._matmul_paged_src import (
    paged_dot_c, paged_dot_reference, paged_matmul_c, paged_matmul_reference,
    paged_coo_dot_c, paged_coo_dot_reference, SCALE)

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

    bytecode, _data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from draft VM): {over}"
    bad = {isa.NAMES.get(i.op, i.op) for i in code if i.op not in _ALLOWED_OPS}
    assert not bad, f"unverified ops leaked: {bad}"
    return code


def _run(code):
    from c4_min.nibble_pure_forward_complete import ref_interpret

    out = []
    tr = ref_interpret(code, max_steps=5_000_000, mask=0xFFFFFFFF, out=out)
    assert len(tr) < 5_000_000, "hit max_steps (byte-window wrap -> non-termination)"
    wrapped = [v for v in tr if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values"
    return out, len(tr)


# --------------------------------------------------------------------------- #
# byte-exactness PAST the old N<=15 window                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [15, 16, 64, 104])
def test_paged_dot_byte_exact(K):
    """The paged dot is byte-exact vs the reference at K = 15, 16, 64, 104 —
    exactly the sizes the old (general/COO) kernels could NOT reach (they broke at
    N=16)."""
    rng = random.Random(100 + K)
    w = [rng.randint(0, 1) for _ in range(K)]
    x = [rng.randint(0, 3) for _ in range(K)]
    out, _steps = _run(_compile(paged_dot_c(w, x)))
    assert out == paged_dot_reference(w, x)


def test_paged_dot_step_count_data_independent():
    """The paged dot's step count depends ONLY on K, not the operand values — the
    property the grounded step counts rely on (run once per length, scale by count)."""
    K = 64
    counts = set()
    for seed in range(4):
        rng = random.Random(seed)
        w = [rng.randint(0, 255) for _ in range(K)]
        x = [rng.randint(0, 255) for _ in range(K)]
        _out, steps = _run(_compile(paged_dot_c(w, x)))
        counts.add(steps)
    assert len(counts) == 1, f"step count is data-dependent: {counts}"


def test_paged_matmul_byte_exact_bigK():
    """A full ``M x K @ K x N`` matmul at the tiny model's real K=104 is byte-exact
    (the general kernel could only reach a 2x2x2 before diverging)."""
    rng = random.Random(7)
    M, K, N = 2, 104, 2
    A = [rng.randint(0, 1) for _ in range(M * K)]
    B = [rng.randint(0, 1) for _ in range(K * N)]
    out, _steps = _run(_compile(paged_matmul_c(A, B, M, K, N)))
    assert out == paged_matmul_reference(A, B, M, K, N)


def test_paged_coo_dot_byte_exact():
    """The paged COO dot (over pre-gathered nonzeros) is byte-exact at a large nnz —
    the sparse kernel whose step count scales with NNZ, past the old window."""
    rng = random.Random(3)
    nnz = 80
    vals = [rng.randint(0, 1) for _ in range(nnz)]
    xs = [rng.randint(0, 3) for _ in range(nnz)]
    out, _steps = _run(_compile(paged_coo_dot_c(vals, xs)))
    assert out == paged_coo_dot_reference(vals, xs)
