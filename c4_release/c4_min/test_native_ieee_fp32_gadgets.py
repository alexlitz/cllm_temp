"""The fp unit REDUCES to gadgets the repo ALREADY bakes byte-exact.

These tests exercise the ACTUAL ``nibble_alu32`` low-level primitives through a
REAL SwiGLU forward pass and confirm each computes the EXACT integer operation
the IEEE-fp32 bit-surgery algorithm needs — so the vanilla bake of FMUL / FADD is
a WIRING job over existing byte-exact gadgets, not new math.

Primitives proven exact here (the fp unit's whole gadget vocabulary):
  * ``_mul_gate``            integer multiply  a*b        (silu-gated)
  * ``_step_ge``             integer >= compare  [x>=thr]
  * ``_floor_div_pow``       shift floor(x/2^k)          (staircase)
  * ``_guard``               select  cond ? val : 0
  * ``_mul_gate`` + ``_floor_div_pow2`` + ``_nibble_carry_round``
                             the full 24x24 -> 48-bit schoolbook multiply
                             (the FMUL core, sig_a * sig_b, EXACT)
"""
import random

import torch

from c4_min import nibble_alu32 as A
from c4_min.nibble_vm import _empty_spec

_DIM_SMALL = 24


def _eval_spec(spec, x):
    """One SwiGLU FFN block forward: out = W_down @ (silu(W_up x + b_up) *
    (W_gate x + b_gate)) + b_down.  The exact forward the model runs."""
    up = spec["W_up"] @ x + spec["b_up"]
    gate = spec["W_gate"] @ x + spec["b_gate"]
    h = torch.nn.functional.silu(up) * gate
    return spec["W_down"] @ h + spec["b_down"]


def _trunc(spec, u):
    out = {}
    for k, v in spec.items():
        if k == "W_down":
            out[k] = v[:, :u].contiguous()
        elif k == "b_down":
            out[k] = v
        else:
            out[k] = v[:u].contiguous()
    return out


def test_mul_gate_is_exact_integer_multiply():
    A._ONE = 0
    spec = _empty_spec(_DIM_SMALL, 4)
    u = A._mul_gate(spec, 0, 1, 2, 3, 1.0)      # dst(3) = x1 * x2
    spec = _trunc(spec, u)
    for a in range(16):
        for b in range(16):
            x = torch.zeros(_DIM_SMALL)
            x[0], x[1], x[2] = 1.0, float(a), float(b)
            assert abs(_eval_spec(spec, x)[3].item() - a * b) < 1e-3


def test_step_ge_is_exact_compare():
    A._ONE = 0
    spec = _empty_spec(_DIM_SMALL, 6)
    u = A._step_ge(spec, 0, {1: 1.0}, 0.0, 8, 3, 1.0)   # dst = [x1 >= 8]
    spec = _trunc(spec, u)
    for a in range(16):
        x = torch.zeros(_DIM_SMALL)
        x[0], x[1] = 1.0, float(a)
        assert abs(_eval_spec(spec, x)[3].item() - (1.0 if a >= 8 else 0.0)) < 1e-3


def test_floor_div_pow_is_exact_shift():
    A._ONE = 0
    spec = _empty_spec(_DIM_SMALL, 40)
    u = A._floor_div_pow(spec, 0, {1: 1.0}, 0.0, 16, 15, 3, 1.0)  # floor(x/16), x<256
    spec = _trunc(spec, u)
    for a in range(256):
        x = torch.zeros(_DIM_SMALL)
        x[0], x[1] = 1.0, float(a)
        assert abs(_eval_spec(spec, x)[3].item() - (a // 16)) < 1e-3


def test_guard_is_exact_select():
    A._ONE = 0
    spec = _empty_spec(_DIM_SMALL, 4)
    u = A._guard(spec, 0, [(1, 1.0, 0.0)], {2: 1.0}, 0.0, 3, 1.0)  # dst = cond ? val : 0
    spec = _trunc(spec, u)
    for cond in (0, 1):
        for val in range(16):
            x = torch.zeros(_DIM_SMALL)
            x[0], x[1], x[2] = 1.0, float(cond), float(val)
            exp = val if cond else 0
            assert abs(_eval_spec(spec, x)[3].item() - exp) < 1e-3


def test_full_24x24_multiply_exact_through_real_gadgets():
    """The FMUL core (sig_a * sig_b, 24x24 -> up-to-48-bit) computed by the REAL
    ``_mul_gate`` partial products + ``_floor_div_pow2`` split + ``_nibble_carry_round``
    ripple — EXACT for the full 48-bit product."""
    NIB, NCOL = 6, 12               # 24-bit operands = 6 nibbles; 48-bit product = 12
    ONE = 0
    Aoff = 1
    Boff = Aoff + NIB
    PPoff = Boff + NIB
    COLoff = PPoff + NIB * NIB
    dim = COLoff + NCOL
    A._ONE = ONE

    def build_products():
        spec = _empty_spec(dim, NIB * NIB * 3)
        u = 0
        for i in range(NIB):
            for j in range(NIB):
                idx = i * NIB + j
                u = A._clear(spec, u, PPoff + idx)
                u = A._mul_gate(spec, u, Aoff + i, Boff + j, PPoff + idx, 1.0)
        return _trunc(spec, u)

    def build_split():
        spec = _empty_spec(dim, NCOL + NIB * NIB * (1 + 15 * 2))
        u = 0
        for c in range(NCOL):
            u = A._clear(spec, u, COLoff + c)
        for i in range(NIB):
            for j in range(NIB):
                c = i + j
                pp = PPoff + i * NIB + j
                u = A._ident(spec, u, {pp: 1.0}, 0.0, COLoff + c, 1.0)
                if c + 1 < NCOL:
                    u = A._floor_div_pow2(spec, u, {pp: 1.0}, 0.0, 16, 15,
                                          COLoff + c, -16.0, COLoff + c + 1, 1.0)
                else:
                    u = A._floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, 15, COLoff + c, -16.0)
        return _trunc(spec, u)

    def build_carry():
        spec = _empty_spec(dim, NCOL * (2 + 15 * 2 + 15 * 2 + 2))
        u = A._nibble_carry_round(spec, 0, COLoff, COLoff, NCOL)
        return _trunc(spec, u)

    prod_spec, split_spec, carry_spec = build_products(), build_split(), build_carry()
    rng = random.Random(0)
    for _ in range(200):
        a, b = rng.getrandbits(24), rng.getrandbits(24)
        x = torch.zeros(dim)
        x[ONE] = 1.0
        for i in range(NIB):
            x[Aoff + i] = float((a >> (4 * i)) & 0xF)
        for j in range(NIB):
            x[Boff + j] = float((b >> (4 * j)) & 0xF)
        x = x + _eval_spec(prod_spec, x)
        x = x + _eval_spec(split_spec, x)
        for _r in range(NCOL):
            x = x + _eval_spec(carry_spec, x)
        got = 0
        for c in range(NCOL):
            got |= (int(round(x[COLoff + c].item())) & 0xF) << (4 * c)
        assert got == a * b, (a, b, got, a * b)
