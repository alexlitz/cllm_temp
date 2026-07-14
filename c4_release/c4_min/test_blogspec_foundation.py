"""Foundation tests for the BLOG_SPEC-faithful re-founding of ``c4_min``.

Proves the four spec pillars the mission asks for, on the vanilla transformer:

  1. NIBBLE representation — register values live as 16 4-bit nibbles; the byte
     tokens are decoded straight from those nibble dims by the LM head.
  2. 30-token register emission — every VM step appends a full 30-token frame;
     the emit->re-embed of those exact-integer byte tokens IS the requantization
     (no ``torch.round`` anywhere in the exec path).
  3. softmax1 + ALiBi — the model that ingests/emits is the ``blogspec_model``
     vanilla decode-only transformer, and its own forward pass reconstructs a
     register value from the emitted frame.
  4. the proof program ``IMM 6; PSH; IMM 7; ADD; EXIT`` -> 13 runs end to end.

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_blogspec_foundation.py
(or directly: python c4_min/test_blogspec_foundation.py).
"""
from __future__ import annotations

import inspect

import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import blogspec_run as R
from c4_min import blogspec_compiler as C
from c4_min.blogspec_model import softmax1, Attn


PROOF_PROG = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]


# --- 1. softmax1 + ALiBi (§Vanillaness, §The Attention Layer) --------------
def test_softmax1_is_zfod():
    # matches the naive spec form when max>=0
    x = torch.tensor([0.0, 0.0, 0.0])
    assert abs(float(softmax1(x).sum()) - 0.75) < 1e-6
    # a query that matches NOTHING attends to nothing (the +1 sink dominates)
    empty = softmax1(torch.tensor([-50.0, -50.0, -50.0]))
    assert float(empty.sum()) < 1e-6           # ZFOD


def test_attention_uses_alibi_slopes():
    a = Attn(dim=8, n_heads=4)
    # geometric slopes 2^(-8/n*(i+1)) (§307-311)
    exp = [2.0 ** (-8.0 / 4 * (i + 1)) for i in range(4)]
    assert torch.allclose(a.alibi_slopes, torch.tensor(exp))


# --- 2. NIBBLE representation (§Internal Representation) -------------------
def test_registers_are_16_nibbles():
    from c4_min.blogspec_layout import NibbleLayout, NIB_PER_REG
    L = NibbleLayout()
    assert NIB_PER_REG == 16
    # each register band is 16 dims wide
    for name in ("PC", "AX", "SP", "BP", "STACK0"):
        base, size = L.band(name)
        assert size == 16, (name, size)


def test_lm_head_decodes_byte_from_nibbles():
    """The LM byte-head reads two nibble dims and argmaxes to the byte."""
    from c4_min.blogspec_layout import NibbleLayout
    L = NibbleLayout()
    for v in range(256):
        s = R.state_from_regs(L, pc=0, ax=v, sp=0, bp=0, stack0=0)
        got = R._decode_byte_from_nibbles(s, L, L.AX, byte_index=0)
        assert got == v, (v, got)


def test_32bit_value_spans_high_nibbles():
    """0x10000 (SP init) needs nibble index 4 — proves >8-bit nibble range."""
    from c4_min.blogspec_layout import NibbleLayout
    L = NibbleLayout()
    s = R.state_from_regs(L, pc=0, ax=0, sp=0x10000, bp=0, stack0=0)
    val = 0
    for bi in range(4):
        val |= R._decode_byte_from_nibbles(s, L, L.SP, bi) << (8 * bi)
    assert val == 0x10000


# --- 3. 30-token emission + the model's own ingest ------------------------
def test_step_frame_is_30_tokens():
    f = V.build_step_frame(pc=4, ax=13, sp=0x10000, bp=0x10000)
    assert len(f) == 30 and f[0] == V.REG_PC and f[-1] == V.STEP_END
    assert V.parse_step_frame(f)["ax"] == 13


def test_model_forward_ingests_register_from_frame():
    """The REAL softmax1+ALiBi forward reconstructs an AX byte from the frame."""
    model, L, _ = C.build_step_model(PROOF_PROG)
    for v in [0, 6, 13, 42, 128, 255]:
        assert R.ingest_ax_lowbyte(model, L, v) == v, v


# --- 4. the proof program end to end --------------------------------------
def test_proof_program_add_13():
    model, L, code = C.build_step_model(PROOF_PROG)
    tokens, frames = R.run_program(model, L, code)
    trace = R.decode_trace(frames)
    assert trace == isa.interpret(code), (trace, isa.interpret(code))
    assert trace[-1] == 13                       # IMM 6; PSH; IMM 7; ADD -> 13
    # exactly one 30-token frame per VM step, + BOS + HALT terminator
    assert len(tokens) == len(frames) * 30 + 2
    assert tokens[0] == V.BOS and tokens[-1] == V.HALT


def test_no_torch_round_in_exec_path():
    """The exec path (run + compiler + model) contains no ``round`` call — the
    30-token emit IS the requantization (replacing torch.round). We check the
    AST for actual ``round``/``torch.round`` call nodes, so docstrings and
    comments that merely mention the word don't trip the guard."""
    import ast
    for mod in (R, C, __import__("c4_min.blogspec_model", fromlist=["x"])):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                assert name != "round", (mod.__name__, ast.dump(node))


# --- ALU gadgets are the SwiGLU primitives, exact ------------------------
def test_nibble_alu_gadgets_exact():
    import random
    random.seed(1)
    for _ in range(500):
        a, b = random.randint(0, 255), random.randint(0, 255)
        assert C.nibble_add_gadget(a, b) == ((a + b) & 0xFF)
        assert C.nibble_sub_gadget(a, b) == ((a - b) & 0xFF)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} foundation tests passed")
