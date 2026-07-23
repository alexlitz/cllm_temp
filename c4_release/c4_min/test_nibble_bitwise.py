"""Byte-exact gate for the NIBBLE bitwise / shift dispatch (``nibble_bitwise``).

Covers the bitwise LOG-SHIFTER (``shift_stage_blocks`` / ``perbit_shift_select_rules``):
the SHL/SHR that replaced BOTH the dense per-value select AND the intermediate
barrel select (the per-(source-bit × shift-amount) cross-product, ~17K nz) with 5
conditional 2**k stages (``shift by 2**k`` gated on AX bit k) over the SHARED source
bit planes (``A_BIT``), plus an ``n >= 32 -> 0`` fold.  Each stage is a per-bit 2:1
mux built from the boolean AND/OR machinery — no MUL/DIV, fp32-exact.  Every case
runs through the REAL SwiGLU forward (``run_compiled``), not the standalone gadget
math, and is compared to the 32-bit reference gadgets (byte-identical to
``nibble_pure_forward_complete.ref_interpret(mask=0xFFFFFFFF)``: ``n >= 32 -> 0``).

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
    """The log-shifter is byte-exact for EVERY shift count 0..31 (and the n>=32->0
    counts 32..40) across a spread of 32-bit sources, through the real SwiGLU
    forward — including 0x80000000 / 0xFFFFFFFF / 0xDEADBEEF / 0x1."""
    cases = [(src, s) for src in _SHIFT_EDGE_SRCS for s in list(range(33)) + [40]]
    got = _run_op_through_forward(op, cases)
    for (src, s), g in zip(cases, got):
        want = _REF[op](src, s)
        assert g == want, (isa.NAMES[op], hex(src), s, hex(g), hex(want))


@pytest.mark.parametrize("op", [isa.SHL, isa.SHR])
def test_barrel_shift_byte_exact_random(op):
    """400 random 32-bit source / shift-count pairs (shift counts include values
    >= 32 to exercise the n>=32->0 fold) — byte-exact vs the reference."""
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
    """The log-shifter is the LEAN shift: SHL/SHR are ~557 rules each across the 5
    conditional 2**k stages (+ keep + recompose) instead of the old barrel select's
    thousands, and the two ops are SYMMETRIC (identical rule/weight cost) — SHR is
    no longer 3x SHL.  Depth is LOG_STAGES + 2 = 7 shift blocks."""
    L = NibbleLayout()
    bw.extend_layout_for_bitwise(L)
    for op in (isa.SHL, isa.SHR):
        n = len(bw.perbit_shift_select_rules(L, op))
        assert n < 700, (isa.NAMES[op], n)          # was 2732 / 8044
        assert len(bw.shift_stage_blocks(L, op)) == bw.LOG_STAGES + 2
    # SHL and SHR cost the SAME (only the mux direction differs).
    assert (len(bw.perbit_shift_select_rules(L, isa.SHL))
            == len(bw.perbit_shift_select_rules(L, isa.SHR)))


def test_perbit_path_drops_dead_operand_onehots():
    """The log-shifter drops the dense A_OH/B_OH operand one-hots (512 dims) AND the
    shift-amount one-hots — it reads the shared A_BIT source planes and takes the
    shift amount straight from the AX planes B_BIT[0..4]."""
    L = NibbleLayout()
    bw.extend_layout_for_bitwise(L)
    assert L.A_OH is None and L.B_OH is None
    assert getattr(L, "A_BIT", None) is not None    # the shared bit planes exist.
    # no shift-amount one-hot bands allocated any more.
    assert getattr(L, "SHIFT_LO_OH", None) is None
    assert getattr(L, "SHIFT_BIT4", None) is None
    assert L.SH_STAGE is not None and len(L.SH_STAGE) == bw.LOG_STAGES


@pytest.mark.parametrize("op", [isa.SHL, isa.SHR])
def test_log_shift_matches_ref_interpret_32bit(op):
    """The log-shifter (through the real SwiGLU forward) is byte-identical to
    ``nibble_pure_forward_complete.ref_interpret(mask=0xFFFFFFFF)`` — the full-width
    unsigned shift ``(pop <</>> ax) & 0xFFFFFFFF`` with ``ax`` UNMASKED, so n>=32->0
    — over the prompt's edge grid."""
    def ref32(pop, ax):
        return ((pop << ax) if op == isa.SHL else (pop >> ax)) & 0xFFFFFFFF
    xs = [0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0x1]
    ns = [0, 1, 7, 15, 16, 31, 32, 40]
    cases = [(x, n) for x in xs for n in ns]
    got = _run_op_through_forward(op, cases)
    for (x, n), g in zip(cases, got):
        assert g == ref32(x, n), (isa.NAMES[op], hex(x), n, hex(g), hex(ref32(x, n)))
