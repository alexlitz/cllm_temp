"""Guard the F_DIV transformer MEGABLOCK (``compile_fp_div_blocks``) against the
IEEE-754 single oracle (``isa.f32_op_bits``, itself gcc/SoftFloat-exact).

Regression guard for task #800: the restoring-division remainder is kept as an EXACT
bit-vector (0/1 per slot, half-snapped each iteration) so the per-iteration doubling
cannot amplify the ~value*2^-24 gadget residue into the quotient — the bug that made
3.0/2.0 -> 1.0000004 and 1.0/3.0 last-2-bits off.  These cases are byte-exact; the
subnormal-operand / gradual-underflow-to-subnormal cases are the documented shared
GAP (F_MUL/F_ADD flush to zero too) and are NOT asserted here.

All FP work is behind ``C4_FLOAT_OPS`` (default OFF) so it is golden-neutral.
"""
import os
import struct

import pytest

os.environ.setdefault("C4_FLOAT_OPS", "1")

import torch

from c4_min import isa
from c4_min import nibble_fp32 as fp
from c4_min import _fp32_blockrun as BR


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


@pytest.fixture(scope="module")
def div_setup():
    torch.set_grad_enabled(False)
    L = BR.make_layout()
    fp.extend_layout_for_fp32(L)
    dim = L.D
    blocks = fp.compile_fp_div_blocks(L, dim)
    return L, dim, blocks


def _run(div_setup, a_bits, b_bits):
    L, dim, blocks = div_setup
    x = BR.new_residual(L, dim)
    BR.seed_bits(x, L.STACK0, a_bits)
    BR.seed_bits(x, L.AX, b_bits)
    x = BR.apply_blocks(x, blocks)
    return BR.read_nibbles(x, L.FP32.F_RES, 8)


def _nan(u):
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF) != 0


@pytest.mark.parametrize("av,bv", [
    (3.0, 2.0),      # the named bug #1 -> 1.5 (quotient >= 1 leading-bit case)
    (1.0, 3.0),      # the named bug #2 -> 0.33333334 (round/sticky)
    (6.0, 2.0), (1.0, 2.0), (5.0, 4.0), (7.0, 8.0), (1.0, 7.0),
    (10.0, 3.0), (1.0, 1.0), (100.0, 7.0), (2.0, 3.0), (355.0, 113.0),
    (0.5, 0.25), (2.0, 7.0), (9.0, 4.0), (1.0, 10.0), (123.0, 456.0),
    (-3.0, 2.0), (3.0, -2.0), (-1.0, -4.0),
])
def test_fdiv_known_bit_exact(div_setup, av, bv):
    a, b = _bits(av), _bits(bv)
    got = _run(div_setup, a, b)
    want = isa.f32_op_bits(isa.F_DIV, a, b)
    assert got == want, f"{av}/{bv}: got {got:08x} want {want:08x}"


def test_fdiv_named_bugs(div_setup):
    # explicit re-statement of the two bugs the fix closed.
    assert _run(div_setup, _bits(3.0), _bits(2.0)) == _bits(1.5)
    assert _run(div_setup, _bits(1.0), _bits(3.0)) == 0x3EAAAAAB   # 0.33333334


def test_fdiv_random_normal_bit_exact(div_setup):
    """Random normal/normal pairs with a NORMAL result are byte-exact.  Skip subnormal
    operands (decode gap) and subnormal results (gradual-underflow gap)."""
    import random
    rng = random.Random(2024)
    tested = 0
    tries = 0
    while tested < 120 and tries < 4000:
        tries += 1
        a = rng.getrandbits(32)
        b = rng.getrandbits(32)
        # both operands strictly-normal, finite, nonzero
        if not (1 <= (a >> 23) & 0xFF <= 254):
            continue
        if not (1 <= (b >> 23) & 0xFF <= 254):
            continue
        want = isa.f32_op_bits(isa.F_DIV, a, b)
        we = (want >> 23) & 0xFF
        wm = want & 0x7FFFFF
        if we == 0 and wm != 0:
            continue                       # subnormal result -> documented gap
        got = _run(div_setup, a, b)
        assert got == want or (_nan(got) and _nan(want)), (hex(a), hex(b))
        tested += 1
    assert tested >= 100


def test_fdiv_specials(div_setup):
    for av, bv, wantf in [
        (float("inf"), 2.0, float("inf")),
        (1.0, float("inf"), 0.0),
        (-1.0, 0.0, float("-inf")),        # finite / -0? here 0.0 is +0 -> -inf
        (0.0, 5.0, 0.0),
    ]:
        a, b = _bits(av), _bits(bv)
        got = _run(div_setup, a, b)
        want = isa.f32_op_bits(isa.F_DIV, a, b)
        assert got == want or (_nan(got) and _nan(want)), f"{av}/{bv}: {got:08x} vs {want:08x}"
