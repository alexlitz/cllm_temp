"""Family-2 WIDTH-32 verification (#667): comparisons / branches / loop counters
that exceed the 8-bit fold must match ideal (full 32-bit) C, not the 8-bit-masked
folded corpus.

The 8-bit substrate (default) folds every value-lane result mod 256, so a
``while (i < n)`` with ``n > 255`` and any ordering compare on a >255 operand
diverges from ideal C (the operand aliases to ``v & 0xFF`` BEFORE the — already
32-bit-capable — compare gadget sees it).  ``C4_VM_WIDTH32=1`` widens the whole
value substrate to full 32-bit (fp64 exec, per-byte / cascade requant), fixing the
compare + branch + loop-counter narrow path.

These tests are memory-lean: one small unified model (code_size=8, no mdm table,
no bitwise) shared across cases; each program is a handful of steps.

Run: OMP_NUM_THREADS=4 PYTHONPATH=<repo> C4_VM_WIDTH32=1 \
        python -m pytest c4_min/test_width32_family2.py -q
"""
from __future__ import annotations

import os

import pytest
import torch

from . import isa
from .nibble_vm import (
    load_program, _emit_and_reembed, _snap_lane, _snap_lane_bytes, vm_width32,
)

WIDTH32 = os.environ.get("C4_VM_WIDTH32", "0") == "1"
_requires_w32 = pytest.mark.skipif(
    not WIDTH32, reason="Family-2 32-bit tests require C4_VM_WIDTH32=1")


# --- the per-byte / cascade requant snap is exact to the full 2^32 -----------
@pytest.mark.skipif(not WIDTH32, reason="width-32 snap")
def test_snap_lane_bytes_exact_to_2_32():
    cases = [0, 1, 42, 255, 256, 300, 1000, 1024, 65535, 65536, 100000,
             16777215, 16777216, (1 << 24) + 7, (1 << 31), (1 << 32) - 1,
             -1, -995, (1 << 32), (1 << 32) + 2, 3_000_000_000, 5_000_000_000]
    for v in cases:
        got = _snap_lane_bytes(torch.tensor(float(v), dtype=torch.float64))
        assert got == (v % (1 << 32)), (v, got)


# --- shared small model ------------------------------------------------------
@pytest.fixture(scope="module")
def unified():
    from .nibble_unified import build_unified_model
    model, L, _meta = build_unified_model(
        code_size=8, include_mdm_table=False, include_bitwise=False)
    return model, L


def _run_final_ax(model, L, prog, max_steps=4200):
    code = isa.assemble(prog)
    state = load_program(model, L, code)
    ax = None
    for _ in range(max_steps):
        x = state.view(1, 1, -1)
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        out = x[0, 0]
        ax = _snap_lane(out[L.AX_VAL])
        state = _emit_and_reembed(out, L)
        if float(out[L.HALTED]) > 0.5:
            break
    return ax


def _steps(model, L, prog, max_steps=4200):
    code = isa.assemble(prog)
    state = load_program(model, L, code)
    n = 0
    for _ in range(max_steps):
        x = state.view(1, 1, -1)
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        out = x[0, 0]
        n += 1
        state = _emit_and_reembed(out, L)
        if float(out[L.HALTED]) > 0.5:
            break
    return n


def _countdown(n):
    # AX=n; head: PSH; IMM 1; SUB; BNZ head; HALT.  Reaches 0 in exactly n passes.
    return [("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]


# --- loop counter survives past 255 (the loop_countdown corpus program) ------
@_requires_w32
@pytest.mark.parametrize("n", [3, 200, 255, 256, 300, 1000])
def test_loop_countdown_reaches_zero(unified, n):
    model, L = unified
    final = _run_final_ax(model, L, _countdown(n), max_steps=n * 6 + 50)
    assert final == 0, f"countdown from {n} ended at {final}, not 0"


@_requires_w32
@pytest.mark.parametrize("n", [3, 200, 255, 256, 300])
def test_loop_countdown_iteration_count(unified, n):
    # exactly n iterations of 4 steps (PSH,IMM,SUB,BNZ) + IMM head + HALT = 4n+2.
    model, L = unified
    steps = _steps(model, L, _countdown(n), max_steps=n * 6 + 50)
    assert steps == 4 * n + 2, f"n={n}: {steps} steps != {4*n+2} (expected {n} iters)"


# --- compare / branch is full 32-bit, matching ideal C -----------------------
def _ideal(op, a, b):
    return {"EQ": a == b, "NE": a != b, "LT": a < b, "GT": a > b,
            "LE": a <= b, "GE": a >= b}[op]


CMP_CASES = [(3, 5), (5, 3), (5, 5), (200, 100), (100, 200), (255, 255),
             (300, 100), (100, 300), (300, 300), (1000, 5), (5, 1000),
             (1000, 1000), (65535, 1), (1, 65535)]


@_requires_w32
@pytest.mark.parametrize("op", ["EQ", "NE", "LT", "GT", "LE", "GE"])
def test_compare_full_32bit(unified, op):
    model, L = unified
    for a, b in CMP_CASES:
        prog = [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]
        got = _run_final_ax(model, L, prog, max_steps=50)
        exp = 1 if _ideal(op, a, b) else 0
        assert got == exp, f"{op} {a} {b}: model {got} != ideal {exp}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
