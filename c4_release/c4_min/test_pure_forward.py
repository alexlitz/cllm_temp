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
    the memory build adds one §Memory KV head (21 total)."""
    model_off, _ = build_pure_forward_model(code_size=16, include_memory=False)
    assert model_off.blocks[0].attn.n_heads == N_ROLES == 20
    model_on, _ = build_pure_forward_model(code_size=16, include_memory=True)
    assert model_on.blocks[0].attn.n_heads == N_ROLES + 1 == 21


# --- M3: memory in KV — LI/SI via softmax1-KV attention over the emitted MEM tokens
def _build_mem(code_size=20):
    return build_pure_forward_model(code_size=code_size, include_memory=True)


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
