"""Tests for native-fp32 FADD / FMUL / FLI / FSI BAKED into the real vanilla
``blogspec_model.Transformer`` (``c4_min.native_fp32_baked``).

The load-bearing proof: an fp32 MAC (``FLI a; FLI b; FMUL; FADD acc``) runs
BYTE-through the genuine ``model.forward`` (softmax1 + ALiBi + SwiGLU + residual)
and is VALUE-FAITHFUL vs numpy fp32 — native fp32 is vanilla, not simulated.

Also asserts the ``C4_FP32_ALU`` gate is default-OFF and that this module has NO
side effect on the byte-exact integer VM build (importing / not-instantiating it
touches nothing on the integer path).

Run:  python -m pytest c4_min/test_native_fp32_baked.py -x -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from c4_min.native_fp32_baked import (
    DEFAULT_S,
    build_fp32_mac_model,
    f32,
    fmul_faithful_range,
    fp32_alu_enabled,
    run_fp32_dot,
    run_fp32_mac,
    run_fp32_mac_attn_gather,
    signed_silu_mul,
)

# --------------------------------------------------------------------------- #
# 1. single fp32 MAC through the REAL model.forward (co-located operands)      #
# --------------------------------------------------------------------------- #
MAC_CASES = [
    (3.0, 4.0, 0.0), (3.0, 4.0, 5.0), (-2.5, 4.0, 1.0), (1.5, -3.0, 0.0),
    (0.1, 0.1, 0.0), (7.0, 13.0, -10.0), (-6.0, -7.0, 0.0), (0.0, 5.0, 2.0),
]


@pytest.mark.parametrize("a,b,acc0", MAC_CASES)
def test_fp32_mac_value_faithful(a, b, acc0):
    got = run_fp32_mac(a, b, acc0=acc0, force=True)
    ref = f32(f32(acc0) + f32(f32(a) * f32(b)))
    assert got == ref, (a, b, acc0, got, ref)


# --------------------------------------------------------------------------- #
# 2. attention-GATHER MAC: operands on SEPARATE tokens, softmax1 gathers them  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,acc0", MAC_CASES)
def test_fp32_mac_attn_gather_value_faithful(a, b, acc0):
    got = run_fp32_mac_attn_gather(a, b, acc0=acc0, force=True)
    ref = f32(f32(acc0) + f32(f32(a) * f32(b)))
    # attention gather + fp32 arithmetic: value-faithful within fp32 tol.
    assert abs(got - ref) <= 1e-5 * (1 + abs(ref)), (a, b, acc0, got, ref)


# --------------------------------------------------------------------------- #
# 3. length-K fp32 DOT through the REAL forward (a full unrolled MAC loop)      #
# --------------------------------------------------------------------------- #
def _seq_dot_fp32(a, b) -> float:
    acc = np.float32(0.0)
    for i in range(len(a)):
        acc = np.float32(acc + np.float32(np.float32(a[i]) * np.float32(b[i])))
    return float(acc)


@pytest.mark.parametrize("K", [1, 2, 3, 5, 8, 12])
def test_fp32_dot_value_faithful(K):
    rng = np.random.default_rng(K)
    a = rng.standard_normal(K).astype(np.float32)
    b = rng.standard_normal(K).astype(np.float32)
    got = run_fp32_dot(a.tolist(), b.tolist(), force=True)
    ref = _seq_dot_fp32(a, b)          # the VM's exact fp32 accumulate order
    assert abs(got - ref) <= 1e-6 * (1 + abs(ref)), (K, got, ref)
    # and within tol of numpy's dot (differs only by summation order)
    assert abs(got - float(np.dot(a, b))) <= 1e-4 * (1 + abs(float(np.dot(a, b))))


def test_fp32_dot_small_known():
    got = run_fp32_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], force=True)
    assert got == 32.0             # 4 + 10 + 18


# --------------------------------------------------------------------------- #
# 4. the signed silu FMUL gadget is exact vs numpy fp32                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("S", [256.0, 4096.0, 65536.0])
def test_signed_silu_mul_exact(S):
    for a in (-3.0, -1.0, 0.0, 1.0, 2.5, 3.0, 13.0):
        for b in (-2.0, 0.0, 1.0, 4.0, -7.5):
            got = float(signed_silu_mul(a, b, S))
            assert got == f32(f32(a) * f32(b)), (S, a, b, got)


def test_fmul_faithful_range_is_fp32_eps():
    """The gadget's worst-case error floor is fp32 epsilon, and the worst case is
    the small-|S*a| region (not the 2^24 ceiling) — the grounded envelope."""
    r = fmul_faithful_range(S=DEFAULT_S, n_samples=30_000)
    assert r["power_of_two_S"]
    # worst relative error is at the fp32 epsilon floor
    assert r["worst_rel_err"] < 3.0 * r["fp32_eps"]
    # the worst case sits in the small-|S*a| region
    assert r["worst_rel_err_small_Sa_region"] == r["worst_rel_err"]


# --------------------------------------------------------------------------- #
# 5. FADD is a residual add; FMUL is the silu gadget (weight-cost report)       #
# --------------------------------------------------------------------------- #
def test_weight_cost_report():
    fm = build_fp32_mac_model(3.0, 4.0, force=True)
    c = fm.weight_cost
    assert c["FMUL"]["blocks"] == 1 and c["FMUL"]["ffn_units"] == 2
    assert c["FMUL"]["weights"] == 6            # the 6-weight silu gadget
    assert c["FADD"]["blocks"] == 1 and c["FADD"]["new_residual_dims"] == 0
    assert c["FLI"]["ffn_units"] == 0           # embedding-literal load, 0 units


def test_it_is_the_real_vanilla_transformer():
    """The model IS a blogspec_model.Transformer (softmax1 + ALiBi + SwiGLU) run
    through its genuine forward — not a bespoke interpreter."""
    from c4_min.blogspec_model import FFN, Attn, Transformer
    fm = build_fp32_mac_model(2.0, 3.0, force=True)
    assert isinstance(fm.model, Transformer)
    assert fm.model.sink == "softmax1" and fm.model.positional == "alibi"
    for blk in fm.model.blocks:
        assert isinstance(blk.ffn, FFN) and isinstance(blk.attn, Attn)


# --------------------------------------------------------------------------- #
# 6. C4_FP32_ALU gate: default OFF; integer VM build unaffected                 #
# --------------------------------------------------------------------------- #
def test_fp32_alu_gate_default_off():
    # default env: the fp32 ALU MODE is off (integer golden untouched).
    prev = os.environ.pop("C4_FP32_ALU", None)
    try:
        assert fp32_alu_enabled() is False
        # building without force must refuse (guards the default/integer path).
        with pytest.raises(RuntimeError):
            build_fp32_mac_model(1.0, 2.0, force=False)
    finally:
        if prev is not None:
            os.environ["C4_FP32_ALU"] = prev


def test_fp32_alu_gate_on_enables():
    prev = os.environ.get("C4_FP32_ALU")
    os.environ["C4_FP32_ALU"] = "1"
    try:
        assert fp32_alu_enabled() is True
        fm = build_fp32_mac_model(1.0, 2.0, force=False)   # allowed when gate on
        assert fm is not None
    finally:
        if prev is None:
            os.environ.pop("C4_FP32_ALU", None)
        else:
            os.environ["C4_FP32_ALU"] = prev


def test_no_side_effect_on_integer_build():
    """Importing native_fp32_baked must not build/mutate the integer VM. We assert
    the two-limb integer ADD still runs byte-exact after import (a proxy that the
    integer build path is untouched by the fp32 module)."""
    import c4_min.native_fp32_baked  # noqa: F401  (import already at top; explicit)
    from c4_min import nibble_vm as N
    vm = N.NibbleVM(code_size=12)
    prog = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]
    _, frames = vm.run(prog, max_steps=50)
    assert N.decode_trace(frames)[-1] == 13


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))
