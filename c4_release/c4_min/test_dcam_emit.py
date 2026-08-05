"""HIGH-ADDRESS in-transformer JIT on the DIRECT-CAM code path (``C4_CFM_EMIT``).

The 8e6946e7 RoPE CFM path keys the code CAM on the SLOW rotary lanes, capping the
reliable code address at ~12-16 bits at program-length Δpos, so a runtime EMIT to a
HIGH address decays past usable.  This ports EMIT onto the POSITION-INVARIANT
direct-CAM code fetch (``nibble_pure_forward_complete`` / ``_bake_code_cam_head``,
``CODE_ADDR_BITS=20``, ALiBi slope 0 — the same fetch Doom's ~440K PCs use), so a
runtime-emitted frame keyed bits(addr) resolves fetch@PC at ANY address (up to
2^CODE_ADDR_BITS) with NO positional decay.

Two tiers:
  * REFERENCE tier (no model, always runs): the ``isa.interpret`` EMIT semantics for
    append / re-emit / high-address / larger multi-statement programs.
  * MODEL tier (GPU-guarded): the REAL lean streaming-sparse CFM model, byte-exact vs
    ``isa.interpret``, incl the high-address sweep + a 13-instruction expr+branch
    loaded program.

Flag DEFAULT OFF; the golden 069cc32f build (flag-OFF AND flag-ON) is byte-identical
(EMIT adds no baked weights — the wide address key rides the existing direct-CAM
lanes; high-address branch TARGETS are driver bookkeeping, no weight change).
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa

IMM, PSH, HALT, JMP, EMIT, ADD, NOP, LI, BZ = (
    isa.IMM, isa.PSH, isa.HALT, isa.JMP, isa.EMIT, isa.ADD, isa.NOP, isa.LI, isa.BZ)


def _emit_seq(target, op, immval):
    return [isa.Instr(IMM, immval), isa.Instr(PSH, 0),
            isa.Instr(IMM, op), isa.Instr(EMIT, target)]


def _pad(prog, n):
    while len(prog) < n:
        prog.append(isa.Instr(NOP, 0))
    return prog


def _prog_high_addr(target):
    prog = []
    prog += _emit_seq(target, IMM, 14)
    prog += _emit_seq(target + 1, HALT, 0)
    prog += [isa.Instr(JMP, target)]
    return _pad(prog, 10), 14


# ===========================================================================
# REFERENCE TIER — high-address EMIT semantics via the golden c4 interpreter.
# ===========================================================================
@pytest.mark.parametrize("target", [1 << 12, 1 << 16, 1 << 18, 1 << 19, (1 << 20) - 2])
def test_reference_high_addr_emit(target):
    """EMIT IMM 14; HALT to a HIGH code address, JMP there, run -> AX=14 byte-exact
    (the c4 oracle grows its code array and fetches the emitted frame at any addr)."""
    prog, expect = _prog_high_addr(target)
    tr = isa.interpret(prog, max_steps=64)
    assert tr[-1] == expect, (target, tr[-3:])


def test_reference_larger_expr_branch():
    """A 13-instruction multi-statement expr + conditional branch loaded program."""
    for (d0, d1, d2, s, want) in [(2, 3, 4, 1, 9), (5, 6, 7, 0, 111)]:
        base = 200
        loader = []

        def eifm(k, addr):
            return [isa.Instr(IMM, k), isa.Instr(LI, 0), isa.Instr(PSH, 0),
                    isa.Instr(IMM, IMM), isa.Instr(EMIT, addr)]
        loader += eifm(0, base + 0)
        loader += _emit_seq(base + 1, PSH, 0)
        loader += eifm(1, base + 2)
        loader += _emit_seq(base + 3, ADD, 0)
        loader += _emit_seq(base + 4, PSH, 0)
        loader += eifm(2, base + 5)
        loader += _emit_seq(base + 6, ADD, 0)
        loader += eifm(3, base + 7)
        loader += _emit_seq(base + 8, BZ, base + 11)
        loader += _emit_seq(base + 9, IMM, d0 + d1 + d2)
        loader += _emit_seq(base + 10, JMP, base + 12)
        loader += _emit_seq(base + 11, IMM, 111)
        loader += _emit_seq(base + 12, HALT, 0)
        loader += [isa.Instr(JMP, base)]
        loader = _pad(loader, 90)
        tr = isa.interpret(loader, max_steps=200,
                           mem_init={0: d0, 1: d1, 2: d2, 3: s})
        assert tr[-1] == want, (d0, d1, d2, s, tr[-3:])


# ===========================================================================
# MODEL TIER — the REAL lean streaming-sparse direct-CAM CFM model (GPU-guarded).
# ===========================================================================
import torch  # noqa: E402

DEVICE = ("cuda:0" if torch.cuda.is_available() else None)
requires_cuda = pytest.mark.skipif(DEVICE is None, reason="direct-CAM model build needs a GPU")


@pytest.fixture(scope="module")
def dcam_cfm():
    if DEVICE is None:
        pytest.skip("no GPU")
    os.environ["C4_PF_CFM"] = "1"
    os.environ["C4_CODE_ADDR_BITS"] = "20"
    os.environ["C4_CFM_EMIT"] = "1"
    os.environ["C4_PC_WIDE"] = "1"
    from c4_min.compact_alloc import build_compact_sparse_streaming
    model, L, _ = build_compact_sparse_streaming(code_size=48)
    model.to(DEVICE)
    return model, L


@requires_cuda
def test_model_append_and_run(dcam_cfm):
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete_emit
    model, L = dcam_cfm
    prog = []
    prog += _emit_seq(10, IMM, 14)
    prog += _emit_seq(11, HALT, 0)
    prog += [isa.Instr(JMP, 10)]
    prog = _pad(prog, 12)
    res = run_pure_forward_complete_emit(model, L, prog, max_steps=64)
    assert res["exact"] and res["ax_trace"][-1] == 14, res["ax_trace"]


@requires_cuda
@pytest.mark.parametrize("target", [1 << 16, 1 << 18, 1 << 19])
def test_model_high_addr_emit(dcam_cfm, target):
    """THE ceiling lift: EMIT past 2^16 (and 2^18, toward Doom's ~440K PC range),
    JMP there, run byte-exact — the RoPE ~12-16-bit ceiling is gone."""
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete_emit
    model, L = dcam_cfm
    prog, expect = _prog_high_addr(target)
    res = run_pure_forward_complete_emit(model, L, prog, max_steps=64)
    assert res["exact"] and res["ax_trace"][-1] == expect, (target, res["ax_trace"])


@requires_cuda
def test_model_reemit_high_addr(dcam_cfm):
    """Self-modifying: EMIT IMM 3 at 2^18, run it, RE-EMIT the SAME addr to IMM 9,
    run the new instruction -> AX=9 (latest-write-wins, one frame per address)."""
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete_emit
    model, L = dcam_cfm
    T = 1 << 18
    prog = []
    prog += _emit_seq(T, IMM, 3)
    prog += _emit_seq(T + 1, JMP, 12)
    prog += [isa.Instr(JMP, T)]
    prog += [isa.Instr(NOP, 0)] * 3
    prog += _emit_seq(T, IMM, 9)
    prog += _emit_seq(T + 1, HALT, 0)
    prog += [isa.Instr(JMP, T)]
    prog = _pad(prog, 24)
    res = run_pure_forward_complete_emit(model, L, prog, max_steps=64)
    assert res["exact"] and res["ax_trace"][-1] == 9, res["ax_trace"]


@requires_cuda
@pytest.mark.parametrize("d0,d1,d2,s,want", [(2, 3, 4, 1, 9), (5, 6, 7, 0, 111)])
def test_model_larger_expr_branch(dcam_cfm, d0, d1, d2, s, want):
    """A 13-instruction multi-statement expr + conditional branch LOADED compiler
    (reads d0,d1,d2,s from the data band, EMITs the program, runs it) byte-exact."""
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete_emit
    model, L = dcam_cfm
    base = 200
    loader = []

    def eifm(k, addr):
        return [isa.Instr(IMM, k), isa.Instr(LI, 0), isa.Instr(PSH, 0),
                isa.Instr(IMM, IMM), isa.Instr(EMIT, addr)]
    loader += eifm(0, base + 0)
    loader += _emit_seq(base + 1, PSH, 0)
    loader += eifm(1, base + 2)
    loader += _emit_seq(base + 3, ADD, 0)
    loader += _emit_seq(base + 4, PSH, 0)
    loader += eifm(2, base + 5)
    loader += _emit_seq(base + 6, ADD, 0)
    loader += eifm(3, base + 7)
    loader += _emit_seq(base + 8, BZ, base + 11)
    loader += _emit_seq(base + 9, IMM, d0 + d1 + d2)
    loader += _emit_seq(base + 10, JMP, base + 12)
    loader += _emit_seq(base + 11, IMM, 111)
    loader += _emit_seq(base + 12, HALT, 0)
    loader += [isa.Instr(JMP, base)]
    loader = _pad(loader, 90)
    res = run_pure_forward_complete_emit(model, L, loader, max_steps=200,
                                         seed_mem={0: d0, 1: d1, 2: d2, 3: s},
                                         n_rt_pool=16)
    assert res["exact"] and res["ax_trace"][-1] == want, (res["ax_trace"][-4:], want)


@requires_cuda
def test_model_read_only_unregressed(dcam_cfm):
    """A no-EMIT program still decodes byte-exact on the direct-CAM path (the
    read-only Doom fetch is untouched)."""
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
    model, L = dcam_cfm
    for prog in (
        isa.assemble([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]),
        isa.assemble([("IMM", 1), ("JMP", 3), ("IMM", 9), ("IMM", 7), ("HALT", 0)]),
    ):
        tr = run_pure_forward_complete(model, L, prog, max_steps=32)
        assert tr == isa.interpret(prog, max_steps=32), (prog, tr)
