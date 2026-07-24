"""Tests for byte-exact IEEE-754 fp32 FMUL + FADD via bit-surgery
(``native_ieee_fp32``), built from the repo's byte-exact integer/nibble gadgets.

These GUARD the central claim: running the ACTUAL IEEE-754 algorithm with exact
integer ops is BIT-IDENTICAL to numpy.float32 across the covered operand classes
(normal x normal, both signs, half-ULP round-to-even ties, overflow/underflow,
and the +/-0 / inf / NaN specials), and that it reduces to the exact-integer
gadget primitives (so the nibble-gadget path matches the reference bit-for-bit).
"""
import random
import struct

import numpy as np
import pytest

from c4_min import native_ieee_fp32 as FP


# --------------------------------------------------------------------------- #
# numpy fp32 oracle + helpers                                                 #
# --------------------------------------------------------------------------- #
def np_mul(ua, ub):
    a = np.uint32(ua).view(np.float32)
    b = np.uint32(ub).view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.float32(a * b).view(np.uint32).item()


def np_add(ua, ub):
    a = np.uint32(ua).view(np.float32)
    b = np.uint32(ub).view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.float32(a + b).view(np.uint32).item()


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def is_subnormal(u):
    return ((u >> 23) & 0xFF) == 0 and (u & 0x7FFFFF) != 0


def is_nan(u):
    return ((u >> 23) & 0xFF) == 0xFF and (u & 0x7FFFFF) != 0


def rand_normal(rng):
    while True:
        u = rng.getrandbits(32)
        if 1 <= (u >> 23) & 0xFF <= 254:
            return u


# --------------------------------------------------------------------------- #
# 1. field surgery                                                            #
# --------------------------------------------------------------------------- #
def test_field_roundtrip_exact():
    rng = random.Random(0)
    for _ in range(20000):
        u = rng.getrandbits(32)
        s, e, m = FP.unpack_fields(u)
        assert FP.pack_fields(s, e, m) == u


def test_field_values():
    # 1.5 = 0x3FC00000: sign 0, exp 127, mantissa 0x400000.
    assert FP.unpack_fields(0x3FC00000) == (0, 127, 0x400000)
    assert FP.pack_fields(0, 127, 0x400000) == 0x3FC00000


# --------------------------------------------------------------------------- #
# 2. FMUL bit-exactness                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b", [
    (2.5, 4.0), (1.5, 1.5), (-2.0, 3.0), (0.1, 0.1), (3.0, 7.0),
    (-2.5, -4.0), (1.0, -1.0), (123.456, 0.789), (1e10, 1e-10),
])
def test_fmul_known(a, b):
    ua, ub = bits(a), bits(b)
    assert FP.fmul_bits(ua, ub) == np_mul(ua, ub)


def test_fmul_random_normal_bit_exact():
    rng = random.Random(1)
    for _ in range(50000):
        ua, ub = rand_normal(rng), rand_normal(rng)
        e = np_mul(ua, ub)
        if is_subnormal(e):
            continue                        # subnormal RESULT = documented deferral
        assert FP.fmul_bits(ua, ub) == e, (hex(ua), hex(ub))


def test_fmul_half_ulp_round_to_even_ties():
    rng = random.Random(3)
    tested = 0
    while tested < 5000:
        sig_a = (1 << 23) | (rng.getrandbits(23) | 1)
        for t in (22, 23):
            hb = 24 - t
            odd = ((1 << (hb - 1)) | (rng.getrandbits(max(hb - 1, 1)) & ((1 << (hb - 1)) - 1))) | 1
            sig_b = odd << t
            if not (1 << 23) <= sig_b < (1 << 24):
                continue
            prod = sig_a * sig_b
            shift = 24 if (prod >> 47) else 23
            if (prod & ((1 << shift) - 1)) != (1 << (shift - 1)):
                continue
            ua = FP.pack_fields(0, 127, sig_a & 0x7FFFFF)
            ub = FP.pack_fields(0, 127, sig_b & 0x7FFFFF)
            assert FP.fmul_bits(ua, ub) == np_mul(ua, ub)
            tested += 1
    assert tested >= 5000


# --------------------------------------------------------------------------- #
# 3. FADD bit-exactness                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b", [
    (2.5, 4.0), (1.5, 1.5), (-2.0, 3.0), (0.1, 0.2), (1e30, 1e30),
    (1.0, -1.0), (1.0, 1e-10), (1e10, -1e10 + 1.0), (-5.5, 2.25),
])
def test_fadd_known(a, b):
    ua, ub = bits(a), bits(b)
    assert FP.fadd_bits(ua, ub) == np_add(ua, ub)


def test_fadd_random_normal_bit_exact():
    rng = random.Random(2)
    for _ in range(50000):
        ua, ub = rand_normal(rng), rand_normal(rng)
        e = np_add(ua, ub)
        if is_subnormal(e):
            continue
        assert FP.fadd_bits(ua, ub) == e, (hex(ua), hex(ub))


def test_fadd_cancellation_bit_exact():
    rng = random.Random(9)
    for _ in range(20000):
        ua = rand_normal(rng)
        ub = (ua ^ 0x80000000) ^ rng.getrandbits(20)   # near-negation
        e = np_add(ua, ub)
        if is_subnormal(e):
            continue
        assert FP.fadd_bits(ua, ub) == e


def test_fadd_exact_cancellation_is_plus_zero():
    ua = bits(3.14159)
    assert FP.fadd_bits(ua, ua ^ 0x80000000) == 0     # +0.0


def test_fadd_half_ulp_round_to_even_ties():
    rng = random.Random(7)
    tested = 0
    while tested < 5000:
        ma = (1 << 23) | rng.getrandbits(23)
        mb = (1 << 23) | rng.getrandbits(23)
        diff = rng.randint(1, 3)
        ea, eb, G = 130, 130 - rng.randint(1, 3), 3
        eb = 130 - diff
        big = ma << G
        small = (mb << G) >> diff
        if (mb << G) & ((1 << diff) - 1):
            small |= 1
        summ = big + small
        target = 23 + G
        hi = summ.bit_length() - 1
        if hi > target:
            sh = hi - target
            summ = (summ >> sh) | (1 if summ & ((1 << sh) - 1) else 0)
        if (summ & ((1 << G) - 1)) != (1 << (G - 1)):
            continue
        ua = FP.pack_fields(0, ea, ma & 0x7FFFFF)
        ub = FP.pack_fields(0, eb, mb & 0x7FFFFF)
        assert FP.fadd_bits(ua, ub) == np_add(ua, ub)
        tested += 1
    assert tested >= 5000


# --------------------------------------------------------------------------- #
# 4. special cases                                                            #
# --------------------------------------------------------------------------- #
_SPECIALS = [0.0, -0.0, 1.0, -1.0, 2.5, -3.5, float("inf"), float("-inf"),
             float("nan"), 3.0e38, -3.0e38, 1e20, 1e30, -1e30]


@pytest.mark.parametrize("a", _SPECIALS)
@pytest.mark.parametrize("b", _SPECIALS)
def test_specials_grid(a, b):
    ua, ub = bits(a), bits(b)
    if is_subnormal(ua) or is_subnormal(ub):
        pytest.skip("subnormal operand (deferred)")
    gm, em = FP.fmul_bits(ua, ub), np_mul(ua, ub)
    ga, ea = FP.fadd_bits(ua, ub), np_add(ua, ub)
    if not is_subnormal(em):
        assert gm == em or (is_nan(gm) and is_nan(em)), f"MUL {a}*{b}"
    if not is_subnormal(ea):
        assert ga == ea or (is_nan(ga) and is_nan(ea)), f"ADD {a}+{b}"


def test_overflow_to_inf():
    big = bits(1e30)
    assert FP.fmul_bits(big, big) == 0x7F800000        # +inf
    assert FP.fmul_bits(big, bits(-1e30)) == 0xFF800000  # -inf


def test_zero_signs():
    assert FP.fmul_bits(bits(-2.0), bits(0.0)) == 0x80000000    # -0
    assert FP.fmul_bits(bits(2.0), bits(0.0)) == 0x00000000     # +0
    assert FP.fadd_bits(bits(-0.0), bits(-0.0)) == 0x80000000   # -0
    assert FP.fadd_bits(bits(-0.0), bits(0.0)) == 0x00000000    # +0


def test_inf_times_zero_is_nan():
    assert is_nan(FP.fmul_bits(0x7F800000, bits(0.0)))
    assert is_nan(FP.fadd_bits(0x7F800000, 0xFF800000))         # inf + -inf


# --------------------------------------------------------------------------- #
# 5. the nibble-gadget realization equals the reference (=> bit-exact)        #
# --------------------------------------------------------------------------- #
def test_gadget_path_matches_reference():
    rng = random.Random(4)
    for _ in range(30000):
        ua, ub = rand_normal(rng), rand_normal(rng)
        assert FP.fmul_gadgets(ua, ub)[0] == FP.fmul_bits(ua, ub)
        assert FP.fadd_gadgets(ua, ub)[0] == FP.fadd_bits(ua, ub)


def test_gadget_path_matches_reference_on_specials():
    for a in _SPECIALS:
        for b in _SPECIALS:
            ua, ub = bits(a), bits(b)
            assert FP.fmul_gadgets(ua, ub)[0] == FP.fmul_bits(ua, ub)
            assert FP.fadd_gadgets(ua, ub)[0] == FP.fadd_bits(ua, ub)


def test_gadget_op_count_is_bounded():
    # the cost claim: an fp op is a few dozen exact-integer gadget ops, and it
    # reduces to mul/add/shift/cmp/select only (no other primitive appears).
    rng = random.Random(5)
    for _ in range(5000):
        ua, ub = rand_normal(rng), rand_normal(rng)
        _, ocm = FP.fmul_gadgets(ua, ub)
        _, oca = FP.fadd_gadgets(ua, ub)
        assert ocm.mul == 1                    # exactly ONE 24x24 integer multiply
        assert oca.mul == 0                    # FADD needs no multiply
        assert ocm.total() <= 40
        assert oca.total() <= 60
        # only the five gadget families are used (the dict has exactly these keys).
        assert set(ocm.as_dict()) == {"mul", "add", "shift", "cmp", "sel", "total"}
