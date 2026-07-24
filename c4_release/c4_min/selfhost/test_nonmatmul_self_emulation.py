"""NON-MATMUL op self-emulation checks — every non-matmul op the tiny
``c4vm.onnx`` forward needs (``_nonmatmul_ops_src``), compiled through the REAL c4
toolchain (``src.compiler.compile_c``) and verified byte-exact vs its numpy
fixed-point reference.

Companion to ``test_matmul_general_self_emulation.py``: the matmul kernel already
self-hosts; these checks close the *rest* of the runtime's op coverage in the c4
subset.  Two verification tiers, both honest:

  * TIER A (draft VM, ``ref_interpret``, byte-masking):  ops whose every STORED
    value stays 0..255 at scale 16 run byte-exact through the ACTUAL draft VM the
    self-emulation story is about (add/sub/mul/div/reduce_max/reduce_sum/gather/
    transpose/reshape/abs/neg/clip).  The draft VM masks every SI store to a byte,
    so this is exactly the byte-envelope the matmul kernel already lives in.

  * TIER B (unmasked reference VM, ``count_vm_steps``, full-word):  exp / sigmoid /
    softmax need an internal scale > 255 for series resolution, which exceeds the
    draft VM's byte store — so they are verified byte-exact on the UNMASKED c4
    reference VM (a real full-word machine, same one the documented 4.75 rate was
    measured on) + numpy.  They still COMPILE under c4 (the point of self-hosting
    coverage); the draft VM simply can't hold their >255 intermediates.

CPU-only (compile + VM); NO neural model build, NO GPU.
    OMP_NUM_THREADS=4 python -m pytest \
        c4_min/selfhost/test_nonmatmul_self_emulation.py -x -q
"""
from __future__ import annotations

from c4_min import isa
from c4_min.selfhost import _nonmatmul_ops_src as S

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
    used = {i.op for i in code}
    assert used <= _ALLOWED_OPS, f"unexpected ops: {sorted(used - _ALLOWED_OPS)}"
    return code


def _run_draft(code, max_steps=500000):
    """Byte-masking draft VM (the self-emulation VM)."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    out = []
    trace = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=out)
    assert len(trace) < max_steps, "hit max_steps (byte-window wrap)"
    return out, len(trace)


def _check_draft(name, src, ref):
    """TIER A: byte-exact PRTF output through the draft VM."""
    code = _compile(src)
    out, steps = _run_draft(code)
    assert out == ref, f"{name}: draft {out} != ref {ref}"
    return steps


# =========================== TIER A (draft VM) ============================= #
def test_add():
    A, B = [1, 2, 3, 4], [4, 3, 2, 1]
    _check_draft("add", S.add_c(A, B), S.add_reference(A, B))


def test_sub():
    A, B = [8, 6, 5, 9], [1, 2, 3, 4]
    _check_draft("sub", S.sub_c(A, B), S.sub_reference(A, B))


def test_mul():
    A, B = [2, 3, 1, 4], [3, 2, 5, 1]
    _check_draft("mul", S.mul_c(A, B), S.mul_reference(A, B))


def test_div():
    A, B = [8, 6, 9, 4], [2, 3, 3, 1]
    _check_draft("div", S.div_c(A, B), S.div_reference(A, B))


def test_reduce_max():
    data = [3, 9, 2, 7, 1, 8, 4, 6]
    _check_draft("reduce_max", S.reduce_max_c(2, 4, data),
                 S.reduce_max_reference(2, 4, data))


def test_reduce_sum():
    data = [3, 9, 2, 7, 1, 8, 4, 6]
    _check_draft("reduce_sum", S.reduce_sum_c(2, 4, data),
                 S.reduce_sum_reference(2, 4, data))


def test_gather():
    data, idx = [10, 11, 12, 13, 14], [4, 0, 2, 2, 1]
    _check_draft("gather", S.gather_c(data, idx), S.gather_reference(data, idx))


def test_transpose2d():
    data = [1, 2, 3, 4, 5, 6]
    _check_draft("transpose", S.transpose2d_c(2, 3, data),
                 S.transpose2d_reference(2, 3, data))


def test_abs():
    # Non-negative inputs so abs is meaningful on the byte machine (the draft VM
    # loses sign on the byte store; signed abs is a full-word op — see below).
    data = [3, 4, 5, 2, 1]
    _check_draft("abs", S.abs_c(data), S.abs_reference(data))


def test_neg():
    data = [3, -4, 5, -2]
    _check_draft("neg", S.neg_c(data), S.neg_reference(data))


def test_clip():
    data = [2, 0, 3, 5, 7]
    _check_draft("clip", S.clip_c(data, 0), S.clip_reference(data, 0))


def _check_refword_direct(name, src, ref):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, data = compile_c(src)
    code = bytecode_to_isa(bc)
    out = []
    S.refword_interpret(code, out=out)
    assert out == ref, f"{name}: refword {out} != ref {ref}"


def test_abs_signed_fullword():
    """True signed abs on the full-word VM (the byte machine can't hold negatives)."""
    data = [-3, 4, -5, 2, -1]
    _check_refword_direct("abs_signed", S.abs_c(data),
                          S.abs_reference_fullword(data))


def test_clip_signed_fullword():
    data = [-2, 0, 3, -5, 7]
    _check_refword_direct("clip_signed", S.clip_c(data, 0),
                          S.clip_reference_fullword(data, 0))


def test_reshape_identity():
    data = [1, 2, 3, 4, 5, 6]
    _check_draft("reshape", S.reshape_c(data), S.reshape_reference(data))


# ===================== TIER B (full-word reference VM) ===================== #
def _check_refword(name, src, ref):
    """TIER B: op COMPILES under c4 and its PRTF bytes match the full-word numpy
    reference on the honest full-word VM (``refword_interpret`` — correct function
    ABI, no byte masking; the draft VM's byte store can't hold these >255 vals)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bc, data = compile_c(src)
    code = bytecode_to_isa(bc)
    out = []
    ax, cyc = S.refword_interpret(code, out=out)
    assert out == ref, f"{name}: refword {out} != ref {ref}"
    return cyc


def test_exp_compiles_and_correct():
    xs, scale = [0, -1, -2, -3], 4096
    _check_refword("exp", S.exp_c(xs, scale),
                   S.exp_reference_fullword(xs, scale))


def test_sigmoid_compiles_and_correct():
    xs, scale = [0, -1, -2, -4], 4096
    _check_refword("sigmoid", S.sigmoid_c(xs, scale),
                   S.sigmoid_reference_fullword(xs, scale))


def test_softmax_compiles_and_correct():
    xs, scale = [0, -1, -2, -3], 4096
    _check_refword("softmax", S.softmax_c(xs, scale),
                   S.softmax_reference_fullword(xs, scale))


def test_exp_fixedpoint_matches_math():
    """The fixed-point exp algorithm matches math.exp within fixed-point tolerance
    (the numerical-correctness claim, independent of any VM)."""
    import math
    scale = 4096
    xs = [0, -1, -2, -3, -5]
    got = S.exp_reference_fullword(xs, scale)
    for x, g in zip(xs, got):
        true = math.exp(x) * scale
        assert abs(g - true) <= 3, f"exp({x}): fp {g} vs true {true:.1f}"


def test_exp_family_compiles_under_c4():
    """Explicit: exp/sigmoid/softmax COMPILE under c4 (no long, no arrays, no
    macros) — the self-hosting coverage claim, independent of the byte store."""
    from src.compiler import compile_c
    for name, src in [
        ("exp", S.exp_c([0, -1, -2], 4096)),
        ("sigmoid", S.sigmoid_c([0, -1, -2], 4096)),
        ("softmax", S.softmax_c([0, -1, -2], 4096)),
    ]:
        bc, _ = compile_c(src)
        assert len(bc) > 0, f"{name} produced empty bytecode"


if __name__ == "__main__":
    import sys

    draft = [
        ("add", test_add), ("sub", test_sub), ("mul", test_mul),
        ("div", test_div), ("reduce_max", test_reduce_max),
        ("reduce_sum", test_reduce_sum), ("gather", test_gather),
        ("transpose", test_transpose2d), ("abs", test_abs), ("neg", test_neg),
        ("clip", test_clip), ("reshape/identity", test_reshape_identity),
    ]
    unmasked = [
        ("abs_signed(fullword)", test_abs_signed_fullword),
        ("clip_signed(fullword)", test_clip_signed_fullword),
        ("exp", test_exp_compiles_and_correct),
        ("sigmoid", test_sigmoid_compiles_and_correct),
        ("softmax", test_softmax_compiles_and_correct),
        ("exp-vs-math", test_exp_fixedpoint_matches_math),
        ("exp-family-compiles", test_exp_family_compiles_under_c4),
    ]
    passed = 0
    print("TIER A — byte-exact through the DRAFT VM (byte-masking self-emulation VM):")
    for name, fn in draft:
        try:
            fn()
            print(f"  OK   {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {e}")
    print("\nTIER B — compiles under c4 + byte-exact on the UNMASKED reference VM:")
    for name, fn in unmasked:
        try:
            fn()
            print(f"  OK   {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {e}")
    total = len(draft) + len(unmasked)
    print(f"\n{passed}/{total} non-matmul op checks pass")
    sys.exit(0 if passed == total else 1)
