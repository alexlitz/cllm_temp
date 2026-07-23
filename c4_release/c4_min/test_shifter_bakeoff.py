"""Gate for the SHIFTER DESIGN BAKEOFF (``shifter_bakeoff``).

Confirms the granularity sweep's headline claims WITHOUT a DIM-8192 forward — the
gadgets are run on their small residual planes (seconds):

* the finest fp32-exact SMALLEST shifter is the NIBBLE-granular one (fewer coarse
  stages than the landed bit-granular, fine shift still tiny), byte-exact vs the
  32-bit reference for SHL and SHR;
* the byte-granular shifter is byte-exact + fp32-exact but LARGER (its fine-shift
  carry staircase explodes the weights) — coarser is not automatically smaller;
* the 16-bit-granular shifter is NOT fp32-exact (its fine product exceeds 2**24)
  and NOT byte-exact — the coarse savings are eaten by the fine shift;
* the landed bit-granular gadget is confirmed at ~5.3K nz / 8 blocks.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_shifter_bakeoff.py -v
"""
from __future__ import annotations

import random

import pytest

from c4_min import isa
from c4_min import shifter_bakeoff as sb


_XS = [0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0x1, 0x0, 0xF0F0F0F0]
_NS = [0, 1, 3, 7, 8, 15, 16, 24, 31, 32, 40]


def _byte_exact_over_grid(run, left) -> None:
    rng = random.Random(2024)
    cases = [(x, n) for x in _XS for n in _NS]
    cases += [(rng.randint(0, 0xFFFFFFFF), rng.randint(0, 0x3F)) for _ in range(200)]
    for x, n in cases:
        got = run(x, n)
        want = sb.ref_shift32(x, n, left)
        assert got == want, (hex(x), n, hex(got), hex(want))


@pytest.mark.parametrize("w", [4, 8])
@pytest.mark.parametrize("left", [True, False])
def test_nibble_and_byte_granular_byte_exact(w, left):
    """Nibble (w=4) and byte (w=8) granular shifters are byte-exact vs the 32-bit
    reference for BOTH SHL and SHR, through the real SwiGLU plane forward."""
    blocks, L = (sb.build_chunk(w, left=True) if left else sb.build_chunk_shr(w))
    _byte_exact_over_grid(lambda x, n: sb.run_blocks(blocks, L, x, n), left)


@pytest.mark.parametrize("op", [isa.SHL, isa.SHR])
def test_bit_granular_is_landed_size(op):
    """The landed bit-granular gadget confirms at 8 blocks / ~5.3K nz, symmetric."""
    depth, nz, marg, _run = sb.measure_bit_granular(op)
    assert depth == 8
    assert 5000 <= nz <= 5600
    assert marg < sb.FP32_INT_LIMIT


@pytest.mark.parametrize("left", [True, False])
def test_nibble_is_smaller_than_bit(left):
    """The headline: the NIBBLE-granular shifter is fp32-exact, byte-exact AND
    strictly SMALLER (fewer nz) than the landed bit-granular one."""
    op = isa.SHL if left else isa.SHR
    _bd, bit_nz, _bm, _br = sb.measure_bit_granular(op)
    blocks, L = (sb.build_chunk(4, left=True) if left else sb.build_chunk_shr(4))
    nib_nz = sb._blocks_nz(blocks)
    marg = sb._max_relu_arg(blocks)
    assert marg < sb.FP32_INT_LIMIT               # fp32-exact
    assert nib_nz < bit_nz                        # SMALLER than bit-granular
    _byte_exact_over_grid(lambda x, n: sb.run_blocks(blocks, L, x, n), left)


@pytest.mark.parametrize("left", [True, False])
def test_byte_granular_is_shallower_but_larger(left):
    """Byte-granular: shallower (fewer blocks) than bit-granular but LARGER (more
    nz) — the coarse-savings-vs-wider-fine-shift crossover has flipped by w=8."""
    op = isa.SHL if left else isa.SHR
    bit_depth, bit_nz, _bm, _br = sb.measure_bit_granular(op)
    blocks, _L = (sb.build_chunk(8, left=True) if left else sb.build_chunk_shr(8))
    assert len(blocks) < bit_depth                # shallower
    assert sb._blocks_nz(blocks) > bit_nz         # but heavier


@pytest.mark.parametrize("left", [True, False])
def test_16bit_granular_breaks_fp32(left):
    """16-bit-granular: the fine product exceeds 2**24, so it is NOT fp32-exact
    (flagged by max_arg) and NOT byte-exact — the coarse win is eaten."""
    blocks, L = (sb.build_chunk(16, left=True) if left else sb.build_chunk_shr(16))
    marg = sb._max_relu_arg(blocks)
    assert marg >= sb.FP32_INT_LIMIT              # fp32-BROKEN (the honest failure)
    # and not byte-exact across the grid.
    bad = 0
    for x in _XS:
        for n in _NS:
            if sb.run_blocks(blocks, L, x, n) != sb.ref_shift32(x, n, left):
                bad += 1
    assert bad > 0


def test_table_builds():
    """The full bakeoff driver + table formatter run end to end."""
    rows = sb.run_bakeoff()
    table = sb.format_table(rows)
    assert "nibble granular" in table and "16-bit granular" in table
    # smallest bit-exact fp32 variant is the nibble one for both ops.
    for op in ("SHL", "SHR"):
        exact_fp32 = [r for r in rows if r["op"] == op and r["fp32_ok"]
                      and r["exact"] == r["total"]]
        smallest = min(exact_fp32, key=lambda r: r["nz"])
        assert smallest["name"] == "nibble granular", (op, smallest["name"])
