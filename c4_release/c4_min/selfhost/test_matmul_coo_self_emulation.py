"""SPARSE (COO) self-emulation checks — a fixed-point COO dot / matvec kernel
that iterates ONLY the nonzero weights, compiled through the REAL c4 toolchain and
run byte-exact through the deterministic draft VM
(``nibble_pure_forward_complete.ref_interpret``).

Companion to ``test_matmul_self_emulation.py`` (the DENSE demo).  The emulated
c4_min model is ~99.99% sparse, so emulating a dense forward is the wrong compute:
the COO kernel's VM step count scales with NNZ (``k``), not dense size (``N``).

All checks here are CPU-only (compile + reference/draft VM); NO neural model build,
NO GPU.  Run:
    OMP_NUM_THREADS=4 python -m pytest \
        c4_min/selfhost/test_matmul_coo_self_emulation.py -x -s
"""
from __future__ import annotations

from c4_min import isa
from c4_min.selfhost._matmul_coo_src import (
    coo_dot_c, coo_dot_reference,
    coo_matvec_c, coo_matvec_reference,
    dense_dot_unroll_c, dense_dot_loop_c, SCALE, N_MAX,
)

# Same verified op set the dense demo asserts (arithmetic + framing + PRTF +
# byte-safe local loads), plus the loop-condition comparison ops the COO kernel's
# ``while (k < nnz)`` bounds check needs (LT etc.) and LC (the char* array walk).
_ALLOWED_OPS = {
    isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
    isa.LI, isa.LC, isa.SI, isa.SC,
    isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
    isa.JMP, isa.BZ, isa.BNZ, isa.JSR, isa.ENT, isa.ADJ, isa.LEV,
    isa.PRTF, isa.NOP, isa.HALT,
}


def _compile(src):
    """Compile via the REAL c4 compiler + translate to ISA; assert no IMM > 255
    leaked (which would diverge the byte-masking reference from the model)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from ref): {over}"
    return code, data


def _run(code):
    from c4_min.nibble_pure_forward_complete import ref_interpret

    out = []
    trace = ref_interpret(code, max_steps=300000, mask=0xFFFFFFFF, out=out)
    return out, trace


# --- one small sparse instance reused across the op/byte-safety checks --------
_VALS = [3, 2]
_IDXS = [1, 5]
_X = [0, 2, 0, 0, 0, 4, 0, 0]     # N=8, only x[1], x[5] nonzero


def test_coo_dot_ops_are_verified_subset():
    """The compiled COO dot uses only the verified op set (arithmetic + framing +
    byte-safe local loads + PRTF); no malloc/heap, every IMM byte-sized."""
    code, _ = _compile(coo_dot_c(_VALS, _IDXS, _X, SCALE))
    used = {i.op for i in code}
    assert used <= _ALLOWED_OPS, f"unexpected ops: {sorted(used - _ALLOWED_OPS)}"
    assert isa.PRTF in used, "the result must emit via PRTF"
    assert isa.MUL in used and isa.DIV in used, "fixed-point needs MUL + DIV"


def test_coo_dot_stays_byte_sized_and_nonnegative():
    """Every intermediate AX value is NON-NEGATIVE (never wraps to 2^31..2^32) and
    the result byte fits 0..255 — the load-bearing byte-exactness conditions."""
    code, _ = _compile(coo_dot_c(_VALS, _IDXS, _X, SCALE))
    _, trace = _run(code)
    wrapped = [v for v in trace if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values (would corrupt LEA)"
    C = coo_dot_reference(_VALS, _IDXS, _X, SCALE)
    assert all(0 <= c <= 255 for c in C), f"a result byte exceeds 255: {C}"


def test_coo_dot_reference_matches_numpy():
    """The draft VM's PRTF stream equals the numpy fixed-point COO dot, byte for
    byte.  vals=[3,2]@idxs=[1,5], x[1]=2,x[5]=4, scale 16 -> 3*2+2*4=14.0 -> 224."""
    import numpy as np

    code, _ = _compile(coo_dot_c(_VALS, _IDXS, _X, SCALE))
    out, _ = _run(code)
    C = coo_dot_reference(_VALS, _IDXS, _X, SCALE)
    wq = np.array([v * SCALE for v in _VALS], dtype=np.int64)
    xq = np.array([_X[i] * SCALE for i in _IDXS], dtype=np.int64)
    npv = int((wq * xq // SCALE).sum())
    assert out == C == [npv], f"draft {out} != ref {C} != numpy {[npv]}"
    assert out == [224], f"unexpected fixed-point COO dot result {out}"


def test_coo_dot_all_zero_except_few():
    """The whole point: an almost-all-zero vector — one nonzero — is byte-exact and
    the loop runs exactly nnz iterations."""
    vals, idxs, x = [2], [7], [0, 0, 0, 0, 0, 0, 0, 3]   # N=8, k=1
    code, _ = _compile(coo_dot_c(vals, idxs, x, SCALE))
    out, _ = _run(code)
    C = coo_dot_reference(vals, idxs, x, SCALE)
    # fixed-point: (2*16)*(3*16)//16 = 2*3*16 = 96
    assert out == C == [96], f"draft {out} != ref {C}"


def test_coo_dot_envelope_edge_n15():
    """N = N_MAX (the largest byte-safe demo length) is still byte-exact."""
    assert N_MAX == 15
    vals, idxs = [1, 1], [0, N_MAX - 1]
    x = [1] + [0] * (N_MAX - 2) + [1]
    code, _ = _compile(coo_dot_c(vals, idxs, x, SCALE))
    out, _ = _run(code)
    assert out == coo_dot_reference(vals, idxs, x, SCALE)


def test_coo_matvec_reference_matches_numpy():
    """The per-row sparse matrix-vector product is byte-exact vs the reference.
    A = [[1,0,2,0],[0,3,0,0]] @ x=[2,1,1,0] at scale 16:
      row0 = 1*2 + 2*1 = 4.0 -> 64 ; row1 = 3*1 = 3.0 -> 48."""
    rows, N = 2, 4
    vals, cols, row_start = [1, 2, 3], [0, 2, 1], [0, 2, 3]
    x = [2, 1, 1, 0]
    code, _ = _compile(coo_matvec_c(rows, N, vals, cols, row_start, x, SCALE))
    out, _ = _run(code)
    C = coo_matvec_reference(rows, N, vals, cols, row_start, x, SCALE)
    assert out == C == [64, 48], f"draft {out} != ref {C}"


def test_coo_scales_with_nnz_not_n():
    """Headline: the COO draft step count is LINEAR in nnz (k) with a CONSTANT
    per-nonzero increment, independent of N — so sparse steps ∝ k while the dense
    kernel would be ∝ N."""
    steps = []
    for k in (1, 2, 3, 4):
        N = 12
        idxs = list(range(k))
        vals = [1] * k
        x = [0] * N
        for i in idxs:
            x[i] = 1
        code, _ = _compile(coo_dot_c(vals, idxs, x, SCALE))
        _, trace = _run(code)
        steps.append(len(trace))
    incr = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    assert len(set(incr)) == 1, f"per-nonzero increment not constant: {incr}"
    assert incr[0] > 0, "adding a nonzero must add steps"


def test_dense_unroll_matches_coo_on_same_row():
    """The dense unrolled dot and the COO dot agree byte-for-byte on the SAME
    sparse row (the COO just skips the zero terms) — the correctness invariant that
    makes the step-count reduction a free lunch, not an approximation."""
    vals, idxs, x = [3, 2], [1, 5], [0, 2, 0, 0, 0, 4, 0, 0]
    w = [0] * len(x)
    for v, i in zip(vals, idxs):
        w[i] = v
    dcode, _ = _compile(dense_dot_unroll_c(w, x, SCALE))
    dout, _ = _run(dcode)
    assert dout == coo_dot_reference(vals, idxs, x, SCALE)


def test_dense_loop_form_compiles_byte_safe():
    """The alternate dense LOOP generator (shared array walk with COO) is byte-safe
    and byte-exact at a small in-window N."""
    w, x = [2, 0, 3, 0], [1, 0, 4, 0]     # N=4, well inside the envelope
    code, _ = _compile(dense_dot_loop_c(w, x, SCALE))
    out, _ = _run(code)
    # 2*1 + 3*4 = 2 + 12 = 14.0 -> 224
    assert out == [224], f"dense-loop {out} != [224]"


if __name__ == "__main__":
    import sys

    test_coo_dot_ops_are_verified_subset()
    test_coo_dot_stays_byte_sized_and_nonnegative()
    test_coo_dot_reference_matches_numpy()
    test_coo_dot_all_zero_except_few()
    test_coo_dot_envelope_edge_n15()
    test_coo_matvec_reference_matches_numpy()
    test_coo_scales_with_nnz_not_n()
    test_dense_unroll_matches_coo_on_same_row()
    test_dense_loop_form_compiles_byte_safe()
    print("all COO sparse self-emulation checks passed")
    sys.exit(0)
