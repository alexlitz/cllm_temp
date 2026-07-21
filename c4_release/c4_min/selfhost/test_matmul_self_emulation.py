"""ONE-LAYER SELF-EMULATION — a fixed-point dot product / MatMul (the load-bearing
inner op of a transformer layer, and of the ONNX runtime that would self-host the
model) computed BYTE-EXACT through the ACTUAL c4_min ``model.forward``.

The weight operand is a slice of the c4_min model's OWN embedding weight matrix
(quantized to fixed-point) — so a piece of the transformer computes a matmul over
that same transformer's weights.  This is the honest, minimal instance of
self-hosting Rel-3 running through the neural forward (the matmul analogue of the
bounded Mandelbrot in ``c4_min/test_mandelbrot_neural.py``).

BOUNDED SCOPE (honest): a few-hundred-VM-step compute, NOT full self-hosting.  The
full self-forward is the ~2.4M-step / ~88-day wall (see
``docs/SELFHOST_3LAYER_FEASIBILITY.md`` and
``docs/ONE_LAYER_SELF_EMULATION_2026_07_20.md``).  The byte-exact demonstrated
instance is the **2-element dot product** (96 VM steps); the full 2x2 matmul
(296 steps) exceeds the model's memory-CAM fidelity window and diverges partway.

Cheap checks (compile + reference) always run.  The heavy neural forward is gated
behind ``C4_RUN_SELF_EMULATION=1`` (bakes the streaming model + a neural run;
``C4_SELF_EMULATION_DEVICE=cuda:0`` for GPU).

Run:
    OMP_NUM_THREADS=4 C4_RUN_SELF_EMULATION=1 C4_SELF_EMULATION_DEVICE=cuda:0 \
        python -m pytest c4_min/selfhost/test_matmul_self_emulation.py -x -s
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min.selfhost._matmul_src import (
    dot_c, dot_reference, matvec_c, matvec_reference,
    matmul_c, matmul_reference, SCALE)

# A fixed weight row / block for the cheap (non-neural) checks.  The heavy neural
# test uses the model's OWN weight slice (extracted from the built model).
_W_FIXED = [1, 2]
_X = [2, 3]
_A_FIXED = [[1, 1], [2, 1]]
_B = [[1, 2], [3, 1]]

_ALLOWED_OPS = {
    isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
    isa.LI, isa.LC, isa.SI, isa.SC,
    isa.JMP, isa.BZ, isa.BNZ, isa.JSR, isa.ENT, isa.ADJ, isa.LEV,
    isa.PRTF, isa.NOP, isa.HALT,
}


def _compile(src):
    """Compile via the REAL c4_min C compiler and translate to the model's ISA.
    Asserts no IMM > 255 leaked (which would diverge the byte-masking reference
    from the model)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from ref): {over}"
    return code, data


def test_dot_ops_are_verified_subset():
    """The compiled dot product uses only the op set the neural model verifies
    (arithmetic + framing + PRTF): no MALC/FREE/MSET/MCMP/OPEN/READ, every IMM
    byte-sized.  Cheap (compile-only) — always runs."""
    code, _ = _compile(dot_c(_W_FIXED, _X, SCALE))
    used = {i.op for i in code}
    assert used <= _ALLOWED_OPS, f"unexpected ops: {sorted(used - _ALLOWED_OPS)}"
    assert isa.PRTF in used, "the result must emit via PRTF"
    assert isa.MUL in used and isa.DIV in used, "fixed-point needs MUL + DIV"


def test_dot_stays_byte_sized_and_nonnegative():
    """Every intermediate AX value is NON-NEGATIVE (never wraps to 2^31..2^32) and
    the result byte fits in 0..255 — both load-bearing for byte-exactness (a wrapped
    AX corrupts the next LEA; a stored local is one byte wide).  Cheap (reference
    only) — always runs."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    code, _ = _compile(dot_c(_W_FIXED, _X, SCALE))
    trace = ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF)
    wrapped = [v for v in trace if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values (would corrupt LEA)"
    C = dot_reference(_W_FIXED, _X, SCALE)
    assert all(0 <= c <= 255 for c in C), f"a result byte exceeds 255: {C}"


def test_dot_reference_matches_numpy():
    """The reference VM's PRTF stream equals the numpy fixed-point dot product, byte
    for byte.  Documents the exact byte the neural run must reproduce.  Cheap
    (reference only) — always runs."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    code, _ = _compile(dot_c(_W_FIXED, _X, SCALE))
    out = []
    ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=out)
    C = dot_reference(_W_FIXED, _X, SCALE)
    assert out == C, f"reference VM {out} != numpy {C}"
    # w=[1,2] . x=[2,3] at scale 16 -> 1*2 + 2*3 = 8.0 -> 128
    assert out == [128], f"unexpected fixed-point dot result {out}"


def test_matmul_reference_matches_numpy():
    """The full 2x2 matmul reference VM PRTF equals numpy fixed-point, byte for byte
    (the reference/native path is correct even where the neural forward later
    diverges).  Cheap (reference only) — always runs."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    code, _ = _compile(matmul_c(_A_FIXED, _B, SCALE))
    out = []
    ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=out)
    C = matmul_reference(_A_FIXED, _B, SCALE)
    assert out == C == [64, 48, 80, 80], f"reference VM {out} != numpy {C}"


def test_matvec_reference_matches_numpy():
    """The 2x2 @ 2x1 matrix-vector reference VM PRTF equals numpy fixed-point, byte
    for byte.  Cheap (reference only) — always runs; documents the bytes the
    largest byte-exact neural instance reproduces."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    x = [2, 3]
    code, _ = _compile(matvec_c(_A_FIXED, x, SCALE))
    out = []
    ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=out)
    C = matvec_reference(_A_FIXED, x, SCALE)
    # A=[[1,1],[2,1]] @ x=[2,3] at scale 16 -> [1*2+1*3, 2*2+1*3] = [5,7] -> [80,112]
    assert out == C == [80, 112], f"reference VM {out} != numpy {C}"


@pytest.mark.skipif(
    not os.environ.get("C4_RUN_SELF_EMULATION"),
    reason="heavy: bakes the streaming model + a neural run "
           "(set C4_RUN_SELF_EMULATION=1; C4_SELF_EMULATION_DEVICE=cuda:0 for GPU)",
)
def test_dot_model_forward_byte_exact():
    """The fixed-point 2-element dot product over the model's OWN embedding weight
    row, computed by the ACTUAL streaming ``model.forward`` (KV-cached driver,
    eviction ON), equals the numpy fixed-point reference byte-for-byte — a real,
    COMPLETE "a transformer neuron's dot product, on the neural VM" demonstration.

    Bounded scope: 96 VM steps (one ``model.forward`` per step).  This is NOT full
    self-hosting; the full self-forward is the ~2.4M-step wall (see the docs note).
    Memory: streaming build (~6 GB peak) + eviction holds the run flat.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from c4_min.selfhost._matmul_run import run

    device = os.environ.get("C4_SELF_EMULATION_DEVICE", "cpu")
    ok, out, C_ref, info = run(kind="dot", device=device, use_own_weights=True)
    assert ok, f"neural PRTF {out} != numpy fixed-point {C_ref}"
    assert out == C_ref, f"neural {out} != reference {C_ref}"
    assert info["neural_steps"] >= info["ref_steps"], "model under-stepped"


@pytest.mark.skipif(
    not os.environ.get("C4_RUN_SELF_EMULATION"),
    reason="heavy: bakes the streaming model + a neural run "
           "(set C4_RUN_SELF_EMULATION=1; C4_SELF_EMULATION_DEVICE=cuda:0 for GPU)",
)
def test_matvec_model_forward_byte_exact():
    """The fixed-point 2x2 @ 2x1 MATRIX-VECTOR product over the model's OWN
    embedding weight block (a weight matrix applied to an input vector = a layer's
    forward on one token), computed by the ACTUAL streaming ``model.forward``,
    equals the numpy fixed-point reference byte-for-byte.  This is the LARGEST
    byte-exact self-emulation instance (168 VM steps, 2 outputs).  Bounded scope —
    NOT full self-hosting (see the docs note)."""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from c4_min.selfhost._matmul_run import run

    device = os.environ.get("C4_SELF_EMULATION_DEVICE", "cpu")
    ok, out, C_ref, info = run(kind="matvec", device=device, use_own_weights=True)
    assert ok, f"neural PRTF {out} != numpy fixed-point {C_ref}"
    assert out == C_ref, f"neural {out} != reference {C_ref}"
    assert info["neural_steps"] >= info["ref_steps"], "model under-stepped"


if __name__ == "__main__":
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ["C4_RUN_SELF_EMULATION"] = "1"
    test_dot_ops_are_verified_subset()
    test_dot_stays_byte_sized_and_nonnegative()
    test_dot_reference_matches_numpy()
    test_matmul_reference_matches_numpy()
    test_matvec_reference_matches_numpy()
    test_dot_model_forward_byte_exact()
    test_matvec_model_forward_byte_exact()
    print("all self-emulation checks passed")
    sys.exit(0)
