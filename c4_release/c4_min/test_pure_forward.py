"""Pure-forward VM tests — one VM step = one ``model.forward``, compute in weights.

Proves the mission's make-or-break: ``IMM 6; PSH; IMM 7; ADD; EXIT`` runs entirely
through the standard autoregressive generation loop (argmax + append), the op
result computed by the FFN weights inside ``model.forward``, the register state
reconstructed from the emitted 30-token frames by the block-0 attention — with a
trace guard asserting NO call to ``blogspec_run._apply_op`` / ``DictMemStack`` / a
per-call ALU gadget.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_pure_forward.py
"""
from __future__ import annotations

import os

import pytest
import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import blogspec_run as BR
from c4_min.nibble_pure_forward import (
    build_pure_forward_model, run_pure_forward, assert_no_python_compute,
    _NoPythonComputeGuard, build_frame_tokens, make_overlay, N_ROLES,
)


PROOF_PROG = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]


def _build(code_size=16):
    return build_pure_forward_model(code_size=code_size, include_memory=False)


# --- the frame-ingest attention reconstructs the register state from the stream --
def test_frame_ingest_reconstructs_registers_from_token_stream():
    """The block-0 softmax1+ALiBi attention reads all five registers out of the
    prior frame in the token stream — state lives in the SEQUENCE, gathered by a
    real attention head, not a python variable."""
    model, L = _build()
    for regs in [(2, 13, 0x10000, 0x10000, 6), (0, 0, 0x10000, 0x10000, 0),
                 (5, 255, 0xFFFC, 0x10000, 128)]:
        pc, ax, sp, bp, stk = regs
        frame = build_frame_tokens(pc, ax, sp, bp, stk)
        toks = torch.tensor([[V.BOS] + frame])
        overlay = make_overlay(isa.assemble([("IMM", 0), ("HALT", 0)]), L)
        x = model.embed[toks].clone()
        overlay(x)
        # Stock ingest = block-0 attention.  Under C4_INGEST_WIDE the ingest is the
        # wide-preroute | wide-gather | wide-snap chain up to (and incl.) the
        # ingest+recompose block, so run through that chain instead of block-0 attn.
        names = getattr(L, "_block_names", None)
        if names and "wide-gather" in names:
            stop = names.index("ingest+recompose")
            for bi in range(stop + 1):
                x = model.blocks[bi](x)
            state = x[0, -1]
        else:
            state = model.blocks[0].attn(x)[0, -1]
        got = {}
        for name, base in (("pc", L.PC), ("ax", L.AX), ("sp", L.SP),
                           ("bp", L.BP), ("stk", L.STACK0)):
            got[name] = sum(round(float(state[base + j])) << (4 * j) for j in range(8))
        assert (got["pc"], got["ax"], got["sp"], got["bp"], got["stk"]) == regs, got


# --- the make-or-break: IMM;PSH;ADD;EXIT -> [6,6,7,13,13], pure-forward -----------
def test_proof_program_pure_forward_13():
    model, L = _build()
    code = isa.assemble(PROOF_PROG)
    trace = run_pure_forward(model, L, code)
    assert trace == isa.interpret(code) == [6, 6, 7, 13, 13], trace


def test_proof_program_is_pure_no_python_compute():
    """The trace guard asserts the pure-forward run enters NONE of _apply_op /
    DictMemStack / the per-call gadgets — the op ran in model.forward."""
    model, L = _build()
    code = isa.assemble(PROOF_PROG)
    trace = assert_no_python_compute(run_pure_forward, model, L, code)
    assert trace == [6, 6, 7, 13, 13], trace


def test_trace_guard_is_live_catches_old_path():
    """Positive control: the guard MUST flag _apply_op + DictMemStack when the old
    hybrid path runs (otherwise the purity proof is vacuous)."""
    code = isa.assemble(PROOF_PROG)
    g = _NoPythonComputeGuard()
    with g:
        mem = BR.DictMemStack()
        BR._apply_op(code[0], 0, 0, 0x100000, 0x100000, 0, mem)
    caught = set(g.violations)
    assert "_apply_op" in caught and "DictMemStack.__init__" in caught, caught


# --- state is in the token stream: exactly one 30-token frame per step ------------
def test_token_stream_is_the_state():
    model, L = _build()
    code = isa.assemble(PROOF_PROG)
    trace, stream = run_pure_forward(model, L, code, collect_tokens=True)
    # BOS + init frame + one 30-token frame per executed step.
    assert stream[0] == V.BOS
    assert (len(stream) - 1) % V.FRAME_LEN == 0
    n_frames = (len(stream) - 1) // V.FRAME_LEN
    assert n_frames == len(trace) + 1, (n_frames, len(trace))   # +1 init frame


def test_one_head_per_register_byte():
    """The ingest uses exactly one gather head per (register, byte) = 20 heads;
    the memory build adds one §Memory KV head (21 total).  Under C4_INGEST_WIDE the
    ingest instead uses ONE query + ONE KV head (the wide-gather block)."""
    import os
    if os.environ.get("C4_INGEST_WIDE", "0") not in ("0", "", "false", "False"):
        model_off, Loff = build_pure_forward_model(code_size=16, include_memory=False)
        gi = Loff._block_names.index("wide-gather")
        assert model_off.blocks[gi].attn.n_heads == 1     # 20 gather heads -> 1
        return
    model_off, _ = build_pure_forward_model(code_size=16, include_memory=False)
    assert model_off.blocks[0].attn.n_heads == N_ROLES == 20
    model_on, _ = build_pure_forward_model(code_size=16, include_memory=True)
    assert model_on.blocks[0].attn.n_heads == N_ROLES + 1 == 21


# --- M3: memory in KV — LI/SI via softmax1-KV attention over the emitted MEM tokens
def _build_mem(code_size=20):
    # memory-only lean build: no cmp/bitwise/muldiv tables (the 8-bit muldiv table
    # is ~181k hidden units and would make this build minutes-slow for no reason —
    # the KV-memory path is independent of the arithmetic experts).
    return build_pure_forward_model(code_size=code_size, include_memory=True,
                                    include_cmp=False, include_bitwise=False,
                                    include_muldiv=False)


def test_store_then_load_pure_forward():
    """A store-then-load runs pure-forward: the store DATA rides in the emitted MEM
    token in the stream, the load VALUE is retrieved by the model's KV attention."""
    model, L = _build_mem()
    prog = [("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
            ("IMM", 0x40), ("LI", 0), ("HALT", 0)]
    code = isa.assemble(prog)
    trace = assert_no_python_compute(run_pure_forward, model, L, code)
    assert trace == isa.interpret(code), (trace, isa.interpret(code))
    assert trace[-1] == 42


def test_memory_zfod_latest_wins_two_addr():
    """ZFOD (unwritten reads 0), latest-write-wins, and cross-address, all pure."""
    model, L = _build_mem()
    cases = {
        "zfod": ([("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
                  ("IMM", 0x80), ("LI", 0), ("HALT", 0)], 0),
        "latest": ([("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
                    ("IMM", 0x40), ("PSH", 0), ("IMM", 99), ("SI", 0),
                    ("IMM", 0x40), ("LI", 0), ("HALT", 0)], 99),
        "two_addr": ([("IMM", 0x40), ("PSH", 0), ("IMM", 11), ("SI", 0),
                      ("IMM", 0x44), ("PSH", 0), ("IMM", 22), ("SI", 0),
                      ("IMM", 0x44), ("LI", 0), ("HALT", 0)], 22),
    }
    for name, (prog, exp) in cases.items():
        code = isa.assemble(prog)
        trace = assert_no_python_compute(run_pure_forward, model, L, code)
        assert trace == isa.interpret(code), (name, trace, isa.interpret(code))
        assert trace[-1] == exp, (name, trace[-1], exp)


# --- EXTENDED op families: CMP / BITWISE / 8-bit MULDIV, all pure-forward ---------
# Each is an OP_IS[op]-gated FFN expert computed inside model.forward; the operand
# is popped from STACK0 (the ingested stack-top) and combined with AX. Lean builds
# (only the family under test) keep these fast — the 8-bit muldiv table is the one
# slow build (~181k hidden units), so it gets a small case set.
def _push_op(a, b, op):
    """IMM a ; PSH ; IMM b ; <op> ; HALT  →  AX = a <op> b."""
    return [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]


def test_cmp_family_pure_forward():
    """EQ/NE/LT/GT/LE/GE run pure-forward: the §Comparisons zero-detector +
    sign-of-diff computed ungated each step, the boolean written by the opcode-gated
    expert — all in model.forward, no python compare."""
    model, L = build_pure_forward_model(code_size=20, include_memory=False,
                                        include_cmp=True, include_bitwise=False,
                                        include_muldiv=False)
    cases = [("EQ", 5, 5), ("EQ", 5, 6), ("NE", 5, 6), ("LT", 3, 7),
             ("GT", 7, 3), ("LE", 3, 3), ("GE", 3, 7)]
    for op, a, b in cases:
        code = isa.assemble(_push_op(a, b, op))
        trace = assert_no_python_compute(run_pure_forward, model, L, code)
        assert trace == isa.interpret(code), (op, a, b, trace, isa.interpret(code))


def test_cmp_signed_gadget_two_complement():
    """The signed-compare gadget (``compile_cmp_compute`` + the clamp/sign-correct
    ``compile_cmp_signed_finalize``) yields the SIGNED (two's-complement, bit 31)
    LT/GT/LE/GE verdict — the #673 fix.  Feeds the two FFN blocks the 32-bit value
    lanes (magnitudes) + the register NIBBLE bands (which carry the sign bit), incl.
    the cross-sign case where the recompose leaves the scalar noisy near 2^32, and
    checks every op against Python's SIGNED order.  Runs in fp64 (the width-32
    substrate) — pure gadget algebra, no full-model round-trip (the lite pure-forward
    model's single-slot frame is 8-bit and cannot carry a 32-bit stack value; this
    isolates the CMP weights the production model shares byte-for-byte)."""
    from .nibble_pure_forward import (
        PureForwardLayout, compile_cmp_compute, compile_cmp_signed_finalize)
    from .blogspec_model import FFN
    from .blogspec_layout import NIB_PER_REG
    from c4_min import blogspec_vocab as _V

    L = PureForwardLayout(code_size=12, n_heads=21)
    dim = L.D

    def _mk(spec):
        f = FFN(dim, spec["W_up"].shape[0]).double()
        with torch.no_grad():
            for kk in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
                getattr(f, kk).copy_(spec[kk].double())
        return f

    f1, f2 = _mk(compile_cmp_compute(L, dim)), _mk(compile_cmp_signed_finalize(L, dim))

    def verdict(a, b):
        """Run the two-block gadget on signed 32-bit a (STK) vs b (AX)."""
        stk, ax = a & 0xFFFFFFFF, b & 0xFFFFFFFF
        x = torch.zeros(1, 1, dim, dtype=torch.float64)
        x[0, 0, L.ONE] = 1.0
        x[0, 0, L.STK_VAL] = float(stk); x[0, 0, L.AX_VAL] = float(ax)
        for j, nv in enumerate(_V.nibbles_of_value(stk, NIB_PER_REG)):
            x[0, 0, L.STACK0 + j] = float(nv)
        for j, nv in enumerate(_V.nibbles_of_value(ax, NIB_PER_REG)):
            x[0, 0, L.AX + j] = float(nv)
        y = f2(f1(x))[0, 0].detach()
        return round(float(y[L.CMP_LT])), round(float(y[L.CMP_GT])), round(float(y[L.CMP_EQ]))

    pairs = [(-10, 5), (5, -10), (-3, 0), (0, -3), (-3, -5), (-5, -3),
             (-1, -2), (7, -3), (-3, 7), (100, -100), (3, 7), (7, 3), (5, 5),
             (-2147483648, 0), (0, -2147483648), (2147483647, -1)]
    for a, b in pairs:
        lt, gt, eq = verdict(a, b)
        assert (lt, gt, eq) == (int(a < b), int(a > b), int(a == b)), \
            (a, b, "got LT/GT/EQ", lt, gt, eq)


def test_bitwise_family_pure_forward():
    """OR/XOR/AND/SHL/SHR run pure-forward via the folded per-nibble table FFN
    blocks; the result recomposes to AX inside model.forward — no python bitops."""
    model, L = build_pure_forward_model(code_size=20, include_memory=False,
                                        include_cmp=False, include_bitwise=True,
                                        include_muldiv=False)
    cases = [("OR", 0x0C, 0x03), ("XOR", 0xFF, 0x0F), ("AND", 0xF0, 0x3C),
             ("SHL", 0x03, 2), ("SHR", 0xF0, 3)]
    for op, a, b in cases:
        code = isa.assemble(_push_op(a, b, op))
        trace = assert_no_python_compute(run_pure_forward, model, L, code)
        assert trace == isa.interpret(code), (op, a, b, trace, isa.interpret(code))


# NOTE: the 8-bit MUL/DIV/MOD LOOKUP-TABLE test (``test_muldiv_family_pure_forward``)
# has been REMOVED — the dense 256x256x3 table is gone from the LEAN pure-forward
# model.  MUL/DIV/MOD now run ONLY through the efficient nibble_alu32 ALU in the Qwen
# build; they are covered byte-exact + 32-bit-exact by
# ``test_qwen_full_vm::test_muldiv_through_qwen``.


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} pure-forward tests passed")
