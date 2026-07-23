"""Gate for the ATTENTION-CAM DIVMOD AUTOMATON (``div_automaton_attn``).

Confirms the divisibility-DFA divmod transition table realised as a softmax1
attention CAM (one attention lookup per step) is byte-exact for BOTH quotient and
remainder over the battery, through the REAL softmax1-attention + SwiGLU forward:

* byte-exact q AND r for the battery {2,3,7,9,10,16,60,100,255,256,1000, small
  primes 11/13/17/97/251, powers of two} over edges + >=2000 random 32-bit
  dividends;
* ``b == 0 -> (0, 0)`` (ISA_SPEC §4.2);
* exactly ONE attention lookup per step (the CAM block replaces the FFN automaton's
  two FFN transition blocks EQ+RD);
* the CAM is a sharp hardmax: the worst-case exact-match softmax1 weight rounds to
  1.0 (mismatching rows collapse to ~0), so the retrieval is byte-exact;
* the key/query magnitudes stay in the sharp-CAM regime (per-bit ±G, G=16, ~50 max)
  — the SAME regime ``_bake_memory_cam`` runs in.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_div_automaton_attn.py -v
"""
from __future__ import annotations

import random

import pytest

from c4_min import div_automaton_attn as daa

MASK32 = 0xFFFFFFFF

# battery: {2,3,7,9,10,16,60,100,255,256,1000, small primes 11/13/17/97/251, pow2}.
_BATTERY = sorted({2, 3, 7, 9, 10, 16, 60, 100, 255, 256, 1000,
                   11, 13, 17, 97, 251,
                   1, 4, 8, 32, 64, 128, 256, 512, 1024})

# the FFN one-hot automaton (const_divmod_automaton) baseline: 25 blocks, 3/step,
# with TWO FFN transition blocks (EQ + RD) per step.
_FFN_AUTOMATON_BLOCKS = 25
_FFN_TRANSITION_BLOCKS_PER_STEP = 2


def _edges(b: int):
    return [e & MASK32 for e in (0, 1, b - 1, b, b + 1, MASK32)]


@pytest.mark.parametrize("b", _BATTERY)
def test_qr_byte_exact(b):
    """Both q and r byte-exact vs Python divmod over edges + >=2000 random 32-bit
    dividends, through the REAL softmax1 attention CAM + SwiGLU residual forward."""
    blocks, L = daa.build_div_automaton_attn(b, in_width=32)
    rng = random.Random(1000 + b)
    tests = _edges(b) + [rng.randint(0, MASK32) for _ in range(2000)]
    for a in tests:
        gq, gr = daa.run_blocks(blocks, L, a)
        assert (gq, gr) == (a // b, a % b), (b, hex(a), gq, gr, a // b, a % b)


def test_divisor_zero_is_zero_zero():
    """``b == 0 -> (q, r) = (0, 0)`` (ISA_SPEC §4.2)."""
    blocks, L = daa.build_div_automaton_attn(0, in_width=32)
    rng = random.Random(7)
    for a in [0, 1, 7, 100, MASK32] + [rng.randint(0, MASK32) for _ in range(50)]:
        assert daa.run_blocks(blocks, L, a) == (0, 0), hex(a)


@pytest.mark.parametrize("b", [7, 100, 256, 1000])
def test_one_attention_lookup_per_step(b):
    """Exactly ONE attention lookup per step — the CAM block replaces the FFN
    automaton's TWO FFN transition blocks (EQ + RD).  Total depth = 3·in_nibs + 1
    = 25 blocks for a 32-bit dividend (same block count, but the heavy b·16 table
    is now a shared KV memory, not b·16 FFN guard units replicated per step)."""
    m = daa.measure(b, in_width=32)
    assert m["attn_blocks_per_step"] == 1 < _FFN_TRANSITION_BLOCKS_PER_STEP
    assert m["total_blocks"] == 3 * m["in_nibs"] + 1 == _FFN_AUTOMATON_BLOCKS
    # exactly in_nibs attention lookups total (one per step).
    assert m["in_nibs"] == 8


@pytest.mark.parametrize("b", _BATTERY)
def test_cam_is_sharp_hardmax(b):
    """The CAM is a sharp hardmax: the worst-case exact-match softmax1 weight rounds
    to 1.0 (every mismatching (R,d) row collapses to ~0), which is WHY the retrieval
    is byte-exact.  Same sharp-CAM regime as _bake_memory_cam."""
    m = daa.measure(b, in_width=32)
    assert m["worst_exact_weight"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("b", _BATTERY)
def test_key_query_magnitude_in_cam_regime(b):
    """The key/query magnitudes stay small (per-bit ±G with G=16, bias ~sqrt(nbits)·G
    <~60) — the sharp-CAM regime _bake_memory_cam runs in, well within fp32 and with
    NO 16·R amplification anywhere (the QENC staircase reads a nibble bit only)."""
    m = daa.measure(b, in_width=32)
    assert m["kq_magnitude"] < 100.0            # per-bit ±G / bias regime, not amplified.


def test_feasible_b_range():
    """The b·16-row KV table is the dominant cost (as the FFN automaton's b·16 guard
    table was); feasible up to b·16 <= 16384 -> b <= 1024, past which the
    b-independent digit-recurrence wins.  The whole battery (max b=1024) is inside."""
    assert daa.feasible_b_max() == 1024
    for b in _BATTERY:
        assert b * 16 <= daa.FEASIBLE_TABLE_WIDTH


def test_kv_rows_scale_with_b():
    """Lock the b·16 KV-row scaling + the state-nibble growth boundary (b<=16 -> 1
    nibble, b<=256 -> 2, b<=4096 -> 3)."""
    assert daa.measure(7)["state_nibs"] == 1
    assert daa.measure(16)["state_nibs"] == 1
    assert daa.measure(17)["state_nibs"] == 2
    assert daa.measure(256)["state_nibs"] == 2
    assert daa.measure(257)["state_nibs"] == 3
    for b in (7, 100, 1000):
        assert daa.measure(b)["n_kv_rows"] == b * 16
        assert daa.measure(b)["table_width"] == b * 16
