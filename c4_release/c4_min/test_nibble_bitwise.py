"""Byte-exact gate for the NIBBLE bitwise / shift dispatch (``nibble_bitwise``).

Covers the BARREL SHIFTER (``perbit_shift_select_rules``): the per-bit SHL/SHR
that replaced the dense per-(shift-amount × nibble-position × nibble-value) select
(SHL 2732 + SHR 8044 rules) — 96% of the old 11k-unit ``bw-select`` block — with a
barrel shift over the SHARED source bit planes (``A_BIT``), shrinking the +bitwise
FFN intermediate from ~11.3k to ~1.6k so it fits stock 0.5B.  Every case runs
through the REAL SwiGLU forward (``run_compiled``), not the standalone gadget math,
and is compared to the 32-bit reference gadgets.

Memory-safe: builds only the tiny per-op dispatch (a handful of small FFN blocks),
CPU, no model bake.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_nibble_bitwise.py -v
"""
from __future__ import annotations

import random

import pytest

from c4_min import isa
from c4_min import nibble_bitwise as bw
from c4_min.blogspec_layout import NibbleLayout


# Every masked shift count 0..31 for each op, plus a spread of 32-bit sources.
_SHIFT_EDGE_SRCS = [0, 1, 0xFFFFFFFF, 0x80000000, 0x0F0F0F0F, 0xDEADBEEF,
                    0x00000001, 0x7FFFFFFF]
_BITWISE_EDGES = [(0, 0), (0xFFFFFFFF, 0), (0, 0xFFFFFFFF),
                  (0xFFFFFFFF, 0xFFFFFFFF), (0x0F0F0F0F, 0xF0F0F0F0),
                  (0x80000000, 0x80000000), (0xDEADBEEF, 0xCAFEBABE)]

_REF = {
    isa.OR: bw.or_gadget, isa.XOR: bw.xor_gadget, isa.AND: bw.and_gadget,
    isa.SHL: bw.shl_gadget, isa.SHR: bw.shr_gadget,
}


def _run_op_through_forward(op, cases):
    """Compile ``op``'s dispatch ONCE and run every ``(pop, operand2)`` case through
    the real SwiGLU forward; return the list of decoded 32-bit results."""
    L = NibbleLayout()
    weights = bw.compile_dispatch(L, bw.append_bitwise_shift_to_dispatch(L, op))
    return [bw.run_compiled(L, weights, pop, op2) for pop, op2 in cases]


@pytest.mark.parametrize("op", [isa.SHL, isa.SHR])
def test_barrel_shift_byte_exact_edges(op):
    """The barrel shifter is byte-exact for EVERY masked shift count 0..31 (and the
    over-mask 32) across a spread of 32-bit sources, through the real SwiGLU forward."""
    cases = [(src, s) for src in _SHIFT_EDGE_SRCS for s in range(33)]
    got = _run_op_through_forward(op, cases)
    for (src, s), g in zip(cases, got):
        want = _REF[op](src, s)
        assert g == want, (isa.NAMES[op], hex(src), s, hex(g), hex(want))


@pytest.mark.parametrize("op", [isa.SHL, isa.SHR])
def test_barrel_shift_byte_exact_random(op):
    """400 random 32-bit source / shift-count pairs (shift counts include values
    past 0x1F to exercise the AX & 0x1F mask) — byte-exact vs the reference."""
    rng = random.Random(1234 + op)
    cases = [(rng.randint(0, 0xFFFFFFFF), rng.randint(0, 0x3F)) for _ in range(400)]
    got = _run_op_through_forward(op, cases)
    for (src, s), g in zip(cases, got):
        assert g == _REF[op](src, s), (isa.NAMES[op], hex(src), s)


@pytest.mark.parametrize("op", [isa.OR, isa.XOR, isa.AND])
def test_bitwise_byte_exact(op):
    """OR/XOR/AND (the shared per-bit combine) stay byte-exact through the forward —
    400 random pairs + the edges — so the barrel-shift refactor did not disturb the
    bitwise ops that share the A_BIT/B_BIT bit planes."""
    rng = random.Random(99 + op)
    cases = [(rng.randint(0, 0xFFFFFFFF), rng.randint(0, 0xFFFFFFFF))
             for _ in range(400)] + _BITWISE_EDGES
    got = _run_op_through_forward(op, cases)
    for (a, b), g in zip(cases, got):
        assert g == _REF[op](a, b), (isa.NAMES[op], hex(a), hex(b))


def test_perbit_shift_rule_count_is_lean():
    """The barrel shifter is the ~7x LEAN select: SHL/SHR are a few hundred rules
    each (32 bits × 32 shift-counts + 16 self-clears) instead of the old 2732/8044,
    which is what shrinks the +bitwise intermediate to fit stock 0.5B."""
    L = NibbleLayout()
    bw.extend_layout_for_bitwise(L)
    for op in (isa.SHL, isa.SHR):
        n = len(bw.perbit_shift_select_rules(L, op))
        assert n < 700, (isa.NAMES[op], n)          # was 2732 / 8044


def test_perbit_path_drops_dead_operand_onehots():
    """In the per-bit path the dense A_OH/B_OH operand one-hots (512 dims) are not
    allocated — the barrel shifter and the bitwise combine both read A_BIT/B_BIT."""
    L = NibbleLayout()
    bw.extend_layout_for_bitwise(L)
    # default env -> per-bit path.
    assert L.A_OH is None and L.B_OH is None
    assert getattr(L, "A_BIT", None) is not None    # the shared bit planes exist.
