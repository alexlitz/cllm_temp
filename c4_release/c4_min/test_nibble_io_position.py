"""Tests for the shared I/O position substrate (BLOG_SPEC §"Printing and Reading
Input" 696-717, §"Position Offset Calculation" 718-739).

Proves the spec's claims for the GETCHAR / PUTCHAR / argv substrate:

  1. multi-slope BOS position signature ``(exp(−m_k·d))`` is injective and the
     tuple recovers the absolute distance ``d`` exactly (§710 "uniquely
     identifies position d");
  2. the base-16 nibble cascade reconstructs any 32-bit offset byte-exactly in 8
     layers, matching a bit-shift reference; the digit-emit / residual-subtract
     share one threshold bank (§726-730);
  3. an I/O buffer addressed by absolute position reads back byte-for-byte vs a
     plain-list reference, via both argmax AND the real ``softmax1`` attention
     (§491 ZFOD), with out-of-range -> 0 (zero-fill-on-demand);
  4. the end-to-end scalar-offset -> cascade -> byte path (the GETCHAR index,
     §851) is exact;
  5. it retrieves the byte at position N through the vanilla
     ``blogspec_model.Attn`` softmax1 + ALiBi head (§704-712), for real.

Run:  PYTHONPATH=<repo> python -m pytest c4_min/test_nibble_io_position.py
 or:  python c4_min/test_nibble_io_position.py
"""
from __future__ import annotations

import math
import random

import torch

from c4_min import nibble_io_position as P


# ===========================================================================
# 1.  Multi-slope BOS position signature  (§704-712)
# ===========================================================================
def test_alibi_slopes_match_foundation():
    """The signature heads ARE the foundation ALiBi heads (§307-311)."""
    from c4_min.blogspec_model import Attn
    for nh in (4, 8, 11):
        head = Attn(dim=nh * 4, n_heads=nh, max_seq_len=16)
        assert torch.allclose(
            torch.tensor(P.alibi_slopes(nh), dtype=torch.float64),
            head.alibi_slopes.double(), atol=1e-9), nh


def test_signature_components_are_exp_neg_slope_d():
    """Each component is exactly ``exp(−m_k·d)`` (§710)."""
    nh = 8
    sl = P.alibi_slopes(nh)
    for d in (0, 1, 7, 42, 1000):
        sig = P.position_signature(d, nh)
        for k, m in enumerate(sl):
            assert abs(float(sig[k]) - math.exp(-m * d)) < 1e-12, (d, k)


def test_signature_is_injective():
    """Different positions -> different signatures (the uniqueness claim, §710).
    Checked on the slowest-decay component (raw float64, no rounding), which is
    the last to lose resolution."""
    nh = 8
    sl = P.alibi_slopes(nh)
    k = min(range(nh), key=lambda i: sl[i])
    seen = {}
    for d in range(0, 20000):
        key = float(P.position_signature(d, nh)[k])   # exact float64 value
        assert key not in seen, (d, seen.get(key))
        seen[key] = d


def test_signature_inversion_exact():
    """The tuple recovers the integer distance ``d`` (§710)."""
    nh = 8
    for d in [0, 1, 2, 5, 13, 42, 255, 1000, 4095, 65535, 100000, 180000]:
        sig = P.position_signature(d, nh)
        assert P.position_from_signature(sig, nh) == d, d


def test_signature_inversion_random_full_range():
    """Inversion is exact across the whole invertible range (§710)."""
    rng = random.Random(0xC4)
    nh = 8
    dmax = P.invertible_range(nh)
    for _ in range(500):
        d = rng.randint(0, dmax)
        sig = P.position_signature(d, nh)
        assert P.position_from_signature(sig, nh) == d, d


def test_invertible_range_is_smallest_slope_bound():
    """The invertible range is ``708 / m_min`` (§734 float64 normal-range floor);
    a gentler smallest slope widens it (§710 'enough heads at different slopes')."""
    nh = 8
    dmax = P.invertible_range(nh)
    assert abs(dmax - 708.0 / min(P.alibi_slopes(nh))) < 1.0
    # exact at the boundary, lost just past it (all-zero tuple -> clamps to dmax)
    assert P.position_from_signature(P.position_signature(dmax, nh), nh) == dmax
    # a custom, gentler smallest slope extends the invertible range.
    wide = P.alibi_slopes(nh)[:-1] + [1e-4]
    assert P.invertible_range(nh, wide) > dmax * 10
    for d in (dmax, dmax * 5, dmax * 20):
        sig = P.position_signature(d, nh, slopes=wide)
        assert P.position_from_signature(sig, nh, slopes=wide) == d, d


def test_signature_monotone_decay():
    """Every component strictly decreases with distance (positive slopes)."""
    nh = 8
    prev = P.position_signature(0, nh)
    for d in range(1, 500):
        cur = P.position_signature(d, nh)
        assert all(float(cur[k]) < float(prev[k]) for k in range(nh)), d
        prev = cur


# ===========================================================================
# 2.  Base-16 nibble-cascade offset extractor  (§726-730)
# ===========================================================================
def _ref_nibbles(off: int, n=8):
    return [(off >> (4 * j)) & 0xF for j in range(n)]


def test_cascade_matches_bitshift_reference():
    """Cascade digits == the plain bit-shift nibble decomposition, exactly."""
    for off in [0, 1, 15, 16, 255, 256, 0xABCD, 0xF0F0, 0xDEADBEEF,
                0xFFFFFFFF, 0x80000000, 0x12345678]:
        digits, _ = P.nibble_cascade_offset(off)
        assert digits == _ref_nibbles(off), (hex(off), digits)


def test_cascade_reassembles_exactly():
    """``offset_from_digits(cascade(off)) == off`` over the full 32-bit range."""
    rng = random.Random(1)
    for _ in range(2000):
        off = rng.randint(0, 0xFFFFFFFF)
        digits, _ = P.nibble_cascade_offset(off)
        assert P.offset_from_digits(digits) == off, hex(off)


def test_cascade_residual_terminates_at_zero():
    """A well-formed 8-nibble offset drives the residual to 0 (§726)."""
    for off in [0, 255, 0xABCD, 0xDEADBEEF, 0xFFFFFFFF]:
        _, residuals = P.nibble_cascade_offset(off)
        assert residuals[-1] == 0, hex(off)


def test_cascade_digit_count_is_eight_layers():
    """8 layers for a 32-bit offset (§726 'in just 8 layers instead of 32')."""
    digits, residuals = P.nibble_cascade_offset(0x12345678)
    assert len(digits) == 8 and len(residuals) == 8


def test_cascade_low_nibble_sharpness():
    """The hardest case is the unit-spaced low nibble (§736): all 16 low digits
    extract exactly."""
    for d0 in range(16):
        digits, _ = P.nibble_cascade_offset(0xAB00 + d0)
        assert digits[0] == d0, d0


# ===========================================================================
# 3.  Byte@position retrieval — buffer vs plain-list reference  (§710, §715)
# ===========================================================================
def _mk_buffer(data, marker_pos=0, nh=8):
    buf = P.IOPositionBuffer(marker_pos=marker_pos, n_heads=nh)
    buf.extend(data)
    return buf


def test_read_offset_argmax_exact():
    data = [0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00, 0xFF, 0x7A]
    buf = _mk_buffer(data)
    for i, want in enumerate(data):
        assert buf.read_offset(i) == want, i


def test_read_offset_softmax1_exact():
    """Retrieval survives the REAL softmax1 ZFOD attention (§491)."""
    data = [0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x2C, 0x20, 0x57, 0x6F]
    buf = _mk_buffer(data)
    for i, want in enumerate(data):
        assert buf.read_offset_softmax1(i) == want, i


def test_read_out_of_range_is_zfod_zero():
    """Unwritten / out-of-range offset -> 0 (softmax1 sink dominates, §491)."""
    buf = _mk_buffer([0x41, 0x42, 0x43])
    for bad in (3, 4, 10, 999, -1):
        assert buf.read_offset(bad) == 0, bad
        assert buf.read_offset_softmax1(bad) == 0, bad


def test_read_empty_buffer_is_zero():
    buf = P.IOPositionBuffer()
    assert buf.read_offset(0) == 0
    assert buf.read_offset_softmax1(0) == 0


def test_marker_offset_is_position_relative_not_slot():
    """Positions are ALiBi distance from the marker, not KV slot (§712): the
    same buffer at different marker positions reads back identically."""
    data = [0x11, 0x22, 0x33, 0x44]
    b0 = _mk_buffer(data, marker_pos=0)
    b7 = _mk_buffer(data, marker_pos=7)
    for i, want in enumerate(data):
        assert b0.read_offset(i) == want == b7.read_offset(i), i
        assert b0.read_offset_softmax1(i) == want == b7.read_offset_softmax1(i)


def test_read_random_buffers_vs_reference():
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(1, 40)
        data = [rng.randint(0, 255) for _ in range(n)]
        buf = _mk_buffer(data, marker_pos=rng.randint(0, 50))
        for i in range(n):
            assert buf.read_offset(i) == data[i], (i, data)
            assert buf.read_offset_softmax1(i) == data[i], (i, data)


# ===========================================================================
# 4.  End-to-end scalar-offset -> cascade -> byte  (the GETCHAR index, §851)
# ===========================================================================
def test_scalar_offset_end_to_end():
    data = [ord(c) for c in "position-substrate!"]
    buf = _mk_buffer(data)
    for i, want in enumerate(data):
        # index arrives as a scalar, cascade extracts nibbles, then read.
        assert buf.read_scalar_offset(i) == want, i


# ===========================================================================
# 5.  Real ``blogspec_model.Attn`` softmax1 + ALiBi head byte@N retrieval
# ===========================================================================
def test_real_attn_head_retrieves_byte():
    d = P.demo_position_head_forward(n_heads=8)
    assert d["read"] == d["want"], d


def test_real_attn_head_various_offsets():
    """Retrieve every byte of a buffer through the real Attn forward."""
    from c4_min.blogspec_model import Attn
    nh = 8
    slopes = P.alibi_slopes(nh)
    buf = [0x43, 0x34, 0x5F, 0x6D, 0x69, 0x6E]     # "C4_min"
    d_key, d_val = 1, 256
    D = d_key + d_val
    head_dim = D
    dim = head_dim * nh
    attn = Attn(dim=dim, n_heads=nh, max_seq_len=len(buf) + 8)
    with torch.no_grad():
        attn.alibi_slopes.copy_(torch.tensor(slopes))
        attn.W_q.zero_(); attn.W_k.zero_(); attn.W_v.zero_(); attn.W_o.zero_()
        for h in range(nh):
            base = h * head_dim
            attn.W_q[base + 0, 0] = P.BOS_KEY_SCALE
            attn.W_k[base + 0, 0] = 1.0
            for v in range(d_val):
                attn.W_v[base + d_key + v, d_key + v] = 1.0
            if h == 0:
                for v in range(d_val):
                    attn.W_o[d_key + v, base + d_key + v] = 1.0
    seq = [("MARKER", None)] + [("BYTE", b) for b in buf]
    S = len(seq)
    x = torch.zeros(1, S, dim)
    for i, (kind, val) in enumerate(seq):
        x[0, i, 0] = 1.0
        if kind == "BYTE":
            x[0, i, d_key + val] = 1.0
    out = attn(x)
    for off, want in enumerate(buf):
        read_pos = 1 + off                          # marker at 0
        read = int(out[0, read_pos, d_key:d_key + d_val].argmax().item())
        assert read == want, (off, read, want)


if __name__ == "__main__":
    import sys
    fns = [f for name, f in sorted(globals().items())
           if name.startswith("test_") and callable(f)]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {f.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
