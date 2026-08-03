"""Tests for the native fused fixed-point ops FIXEDMUL / FIXEDDIV for Doom
(``doom_fixedpoint`` + ``measure_doom_fixedpoint``).

Covers: the gate (default OFF, golden-neutral), byte-exactness vs the C
functions as they run on the VM (native ref, SiLU-gadget megablock core, on-VM
c4vm32 bytecode, compiled C golden), the fused megablock block-counts, the
intrinsic recognition byte-identity, and the measured step reduction.
"""
import random

import pytest

from c4_min import doom_fixedpoint as FP
from c4_min import measure_doom_fixedpoint as M


# --------------------------------------------------------------------------- #
# gate                                                                        #
# --------------------------------------------------------------------------- #
def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("C4_DOOM_FIXEDPOINT", raising=False)
    assert not FP.fixedpoint_enabled()


def test_gate_on(monkeypatch):
    monkeypatch.setenv("C4_DOOM_FIXEDPOINT", "1")
    assert FP.fixedpoint_enabled()


def test_register_opcodes_idempotent():
    FP.register_opcodes()
    FP.register_opcodes()
    from c4_min import isa
    assert isa.BY_NAME["FIXEDMUL"] == FP.FIXEDMUL
    assert isa.BY_NAME["FIXEDDIV"] == FP.FIXEDDIV
    # opcodes are past the neural one-hot band -> registering them is byte-neutral
    assert FP.FIXEDMUL >= isa.NUM_OPS and FP.FIXEDDIV >= isa.NUM_OPS


# --------------------------------------------------------------------------- #
# reference: byte-exact vs the C functions                                    #
# --------------------------------------------------------------------------- #
def test_fixed_mul_matches_int64_shift():
    rng = random.Random(0)
    for _ in range(2000):
        a = rng.randint(-(1 << 31), (1 << 31) - 1)
        b = rng.randint(-(1 << 31), (1 << 31) - 1)
        want = ((a * b) >> 16) & 0xFFFFFFFF        # ((int64)a*b)>>16, low 32
        assert FP.fixed_mul(a, b) == want, (a, b)


def test_fixed_mul_fracunit_identity():
    # FixedMul(x, FRACUNIT) == x  (multiplying by 1.0)
    for x in (0, 1, 100, -100, 0x1234, -0x1234, 0x00030000, -0x00030000):
        assert FP.fixed_mul(x, FP.FRACUNIT) == x & 0xFFFFFFFF


def test_fixed_div_guard_returns_min_max():
    # overflow guard: (abs(a)>>14) >= abs(b) -> MININT / MAXINT by sign
    assert FP.fixed_div(0x10000000, 1) == FP.MAXINT          # both positive
    assert FP.fixed_div(-0x10000000, 1) == FP.MININT & 0xFFFFFFFF
    assert FP.fixed_div(0x10000000, -1) == FP.MININT & 0xFFFFFFFF


def test_fixed_div_normal_regime():
    # 3.0 / 2.0 == 1.5  in fixed point
    assert FP.fixed_div(3 * FP.FRACUNIT, 2 * FP.FRACUNIT) == (FP.FRACUNIT * 3) // 2


def test_reference_and_gadget_agree_full_battery():
    """The SiLU-gadget megablock core is byte-exact to the native reference on
    the full id_port battery (incl. negatives + overflow)."""
    r = FP.verify_byte_exact(against_c4vm32=False)
    assert r["gadget_mul_fail"] == 0
    assert r["gadget_div_fail"] == 0


def test_reference_matches_on_vm_c4vm32():
    """Native reference is byte-exact vs the ACTUAL c4vm32 32-bit-word execution
    (the substrate the transformer's Doom image runs).  Skipped if id_port is
    not present."""
    vm = FP._load_c4vm32()
    if vm is None:
        pytest.skip("id_port/c4vm32.py not present")
    cases = FP.battery_cases()
    for a, b in cases:
        assert vm["mul"](a, b) == FP.fixed_mul(a, b), ("mul", a, b)
        assert vm["div"](a, b) == FP.fixed_div(a, b), ("div", a, b)


# --------------------------------------------------------------------------- #
# fused megablocks                                                            #
# --------------------------------------------------------------------------- #
def test_megablock_block_counts():
    mul_mb = FP.fixedmul_megablock()
    div_mb = FP.fixeddiv_megablock()
    # FEWER blocks than the current native DIV (179 blocks); one decoded VM step.
    assert mul_mb.n_blocks == 7
    assert div_mb.n_blocks == 5
    # reuse the shared expand + ax-mux blocks
    assert "alu-expand" in mul_mb.blocks and "ax-mux" in mul_mb.blocks
    assert "alu-expand" in div_mb.blocks and "ax-mux" in div_mb.blocks
    # FixedMul-specific >>16 select; FixedDiv-specific guard + sign
    assert "fmul-shr16" in mul_mb.blocks
    assert "fdiv-abs-guard" in div_mb.blocks and "fdiv-neg" in div_mb.blocks


# --------------------------------------------------------------------------- #
# intrinsic recognition byte-identity                                         #
# --------------------------------------------------------------------------- #
def test_intrinsic_substitution_rewrites_jsr():
    from c4_min import isa
    img, fn_pc = M._build_call_program("mul", 3, 4)
    imap = FP.IntrinsicMap({fn_pc: FP.FIXEDMUL})
    sub, n = FP.substitute_intrinsics(img, imap)
    assert n == 1
    assert sub[4].op == FP.FIXEDMUL          # the JSR became the native op
    assert sub[5].op == isa.NOP              # the ADJ became a NOP
    assert len(sub) == len(img)              # length preserved (branch targets stable)


def test_function_bodies_are_byte_exact():
    """The faithful FixedMul/FixedDiv subroutine BODIES compute the reference
    values (so the byte-identity proof compares the right algorithm)."""
    cases = FP.battery_cases()
    for which, ref in (("mul", FP.fixed_mul), ("div", FP.fixed_div)):
        for a, b in cases:
            img, _ = M._build_call_program(which, a, b)
            vm = M.MiniVM32(list(img))
            vm.run()
            assert (vm.ax & 0xFFFFFFFF) == ref(a, b), (which, a, b)


def test_intrinsic_byte_identical_battery():
    r = M.prove_intrinsic_byte_identical(n=300)
    assert r["substitutions"] == 2 * r["n"]     # one per (mul + div) case
    assert r["mul_mismatch"] == 0
    assert r["div_mismatch"] == 0
    assert r["call_body_wrong"] == 0
    assert r["mul_vs_ref"] == 0 and r["div_vs_ref"] == 0


# --------------------------------------------------------------------------- #
# step reduction                                                             #
# --------------------------------------------------------------------------- #
def test_native_op_is_one_step():
    from c4_min import isa
    # a bare native op program: PSH a ; PSH b ; FIXEDMUL ; HALT  -> 1 op does the work
    img = [isa.Instr(isa.IMM, 3 * FP.FRACUNIT), isa.Instr(isa.PSH, 0),
           isa.Instr(isa.IMM, 2 * FP.FRACUNIT), isa.Instr(isa.PSH, 0),
           isa.Instr(FP.FIXEDMUL, 0), isa.Instr(isa.HALT, 0)]
    vm = M.MiniVM32(img)
    vm.run()
    assert vm.ax == FP.fixed_mul(3 * FP.FRACUNIT, 2 * FP.FRACUNIT)


def test_step_reduction_is_large():
    mul_steps = M.fixedmul_call_steps()
    div_steps = M.fixeddiv_call_steps()
    # the native op is ONE VM step; the function call is many.
    assert mul_steps > 50        # hi/lo split multiply body
    assert div_steps > 500       # 48-iteration long division body
    slice_ = M.render_slice_steps(n_fixedmul=1000, n_fixeddiv=200)
    assert slice_["native_steps"] == 1200
    # a large reduction (function path >> native path)
    assert slice_["func_steps"] / slice_["native_steps"] > 100
