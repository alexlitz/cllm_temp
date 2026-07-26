"""Strength-reduced matmul c4-source checks — the flat-linear + running-pointer
matmul kernels (``measure_strength_reduced_matmul``) compiled through the REAL c4
toolchain and run BYTE-EXACT through the deterministic DRAFT VM
(``nibble_pure_forward_complete.ref_interpret``).

These pin the hand-optimisation the task asked for:
  * every variant (original 2-D index / flat-linear / strength-reduced, CALL and
    INLINE) is byte-exact vs the shared fixed-point reference AND vs each other;
  * the per-MAC step increment is CONSTANT within the window (so the K-sweep slope
    is a sound marginal rate);
  * the reduction is monotone: strength-reduced < flat-linear < original;
  * the looped linear operand is NOT ADDM-foldable (runtime pointer deref), while a
    fully-unrolled global-array MAC IS (the honest peephole caveat).

CPU-only (compile + draft VM); NO neural model build, NO GPU.
    OMP_NUM_THREADS=4 python -m pytest \
        c4_min/selfhost/test_strength_reduced_matmul.py -x -s
"""
from __future__ import annotations

from c4_min.selfhost.measure_strength_reduced_matmul import (
    _compile, _run, _dot_ref, kernel_original, kernel_flat_linear,
    kernel_strength_reduced, _slope, main)


def _out(gen):
    out, _ = _run(_compile(gen))
    return out


def test_all_variants_byte_exact_and_agree():
    """Every variant computes the identical result byte-for-byte, vs the reference
    and vs each other, across a range of K."""
    for K in range(2, 8):
        ref = _dot_ref(K)
        outs = {
            "orig-CALL": _out(kernel_original(K, True)),
            "orig-INLINE": _out(kernel_original(K, False)),
            "flat-linear": _out(kernel_flat_linear(K)),
            "sr-CALL": _out(kernel_strength_reduced(K, True)),
            "sr-INLINE": _out(kernel_strength_reduced(K, False)),
        }
        for name, o in outs.items():
            assert o == ref, f"K={K} {name}: {o} != ref {ref}"


def test_per_mac_increment_is_constant():
    """The K-sweep step increment is a single constant for each variant (so the
    slope is an honest marginal steps/MAC).  _slope asserts constancy internally."""
    sizes = range(2, 8)
    orig_call, _, _ = _slope(lambda K: kernel_original(K, True), sizes, "orig-CALL")
    flat, _, _ = _slope(kernel_flat_linear, sizes, "flat-linear")
    sr, _, _ = _slope(lambda K: kernel_strength_reduced(K, False), sizes, "sr-INLINE")
    # documented anchors (draft VM is deterministic → exact)
    assert orig_call == 101
    assert flat == 92
    assert sr == 66


def test_reduction_is_monotone():
    """Strength-reduced < flat-linear < original: the optimisation actually cuts
    steps, and by the expected margins."""
    sizes = range(2, 8)
    orig_call, _, _ = _slope(lambda K: kernel_original(K, True), sizes, "orig-CALL")
    orig_inline, _, _ = _slope(lambda K: kernel_original(K, False), sizes, "orig-INLINE")
    flat, _, _ = _slope(kernel_flat_linear, sizes, "flat-linear")
    sr_call, _, _ = _slope(lambda K: kernel_strength_reduced(K, True), sizes, "sr-CALL")
    sr_inline, _, _ = _slope(lambda K: kernel_strength_reduced(K, False), sizes, "sr-INLINE")
    assert sr_inline < sr_call < orig_call
    assert sr_inline < orig_inline < orig_call
    assert sr_inline < flat < orig_call
    # total reduction is real but modest (loads/mul/loop-control are the floor)
    assert 1.4 < orig_call / sr_inline < 1.7


def test_addm_foldability_caveat():
    """The LOOPED linear operand is a runtime pointer deref → 0 ADDM folds; only a
    fully-unrolled GLOBAL-array MAC (immediate absolute address) folds."""
    from c4_min import codegen_fuse as CF
    from src.compiler import compile_c

    w_flat, _ = compile_c(kernel_flat_linear(6))
    w_sr, _ = compile_c(kernel_strength_reduced(6, False))
    _, nf_flat = CF.fuse_bytecode(w_flat)
    _, nf_sr = CF.fuse_bytecode(w_sr)
    assert nf_flat == 0 and nf_sr == 0

    unrolled = ("int a0; int a1; int a2; int b0; int b1; int b2; int acc; int s;"
                "int main(){int i; s=16; acc=0;"
                "a0=1*s;a1=2*s;a2=3*s;b0=1*s;b1=2*s;b2=1*s;"
                "acc=acc+a0*b0/s; acc=acc+a1*b1/s; acc=acc+a2*b2/s;"
                "printf(acc); return 0;}")
    w_unr, _ = compile_c(unrolled)
    _, nf_unr = CF.fuse_bytecode(w_unr)
    assert nf_unr == 6


def test_measure_main_runs():
    """The end-to-end measurement runs and returns the documented rates."""
    r = main(verbose=False)
    assert r["orig_call"] == 101 and r["sr_inline"] == 66
