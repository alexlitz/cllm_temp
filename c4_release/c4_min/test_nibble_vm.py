"""Tests for the UNIVERSAL NIBBLE VM base (``nibble_vm``).

Proves the mission's checklist on ONE fixed-weight model:

  1. RUNTIME-PC DISPATCH — the executed instruction is chosen by the live PC
     (branches skip instructions); ported from ``gf-assembled:control.py``.
  2. UNIVERSAL FETCH-FROM-DATA — the program lives in CODE_OP/CODE_IMM data bands
     (INPUT, not baked); ONE weight-hash-invariant model runs MANY programs;
     ported from ``greenfield-universal:universal.py``.
  3. VANILLA REQUANT — the 30-token frame round-trip (LM-head argmax -> re-embed
     nibbles) requantises; NO ``torch.round`` on the exec path; ported from
     ``greenfield-vanilla-requant:recurrent_vanilla.py``.
  4. NIBBLE + softmax1 + ALiBi — the step-block is the blogspec vanilla
     transformer; registers are 16-nibble bands.
  5. DEEP LOOP — a countdown runs 100s of steps, byte-exact.

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_nibble_vm.py
(or directly: python c4_min/test_nibble_vm.py).
"""
from __future__ import annotations

import ast
import inspect

import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import nibble_vm as N
from c4_min.blogspec_model import softmax1


# A representative program set exercising every base op + wrap + branch taken/not.
PROGRAMS = {
    "add_13":      [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)],
    "sub_5":       [("IMM", 12), ("PSH", 0), ("IMM", 7), ("SUB", 0), ("HALT", 0)],
    "add_wrap":    [("IMM", 200), ("PSH", 0), ("IMM", 100), ("ADD", 0), ("HALT", 0)],
    "sub_wrap":    [("IMM", 3), ("PSH", 0), ("IMM", 10), ("SUB", 0), ("HALT", 0)],
    "lea":         [("LEA", 3), ("HALT", 0)],
    "lea_200":     [("LEA", 200), ("HALT", 0)],
    "jmp_skip":    [("IMM", 1), ("JMP", 3), ("IMM", 99), ("IMM", 42), ("HALT", 0)],
    "bz_taken":    [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
    "bz_nottaken": [("IMM", 5), ("BZ", 4), ("IMM", 9), ("HALT", 0)],
    "bnz_taken":   [("IMM", 5), ("BNZ", 3), ("IMM", 99), ("IMM", 8), ("HALT", 0)],
    "bnz_nottkn":  [("IMM", 0), ("BNZ", 4), ("IMM", 7), ("HALT", 0)],
}

CODE_SIZE = 12  # one model, sized for the largest program, runs them all.


# --- 4. NIBBLE + softmax1 + ALiBi -----------------------------------------
def test_step_model_is_softmax1_alibi():
    vm = N.NibbleVM(code_size=6)
    # ZFOD: a query matching nothing attends to nothing (the +1 sink).
    assert float(softmax1(torch.tensor([[-50.0, -50.0, -50.0]])).sum()) < 1e-6
    # the step-block attention carries ALiBi slopes (geometric, §307-311).
    slopes = vm.model.blocks[0].attn.alibi_slopes
    exp = [2.0 ** (-8.0 / 4 * (i + 1)) for i in range(4)]
    assert torch.allclose(slopes, torch.tensor(exp))


def test_registers_are_nibble_bands():
    L = N.NibbleVMLayout(code_size=6)
    for name in ("PC", "AX", "SP", "BP", "STACK0"):
        base, size = L.band(name)
        assert size == 16, (name, size)   # 16 4-bit nibble dims per register


# --- 2. UNIVERSAL: ONE fixed model runs MANY programs ----------------------
def test_one_model_runs_many_programs():
    vm = N.NibbleVM(code_size=CODE_SIZE)
    h0 = vm.weight_hash()
    for name, prog in PROGRAMS.items():
        code = isa.assemble(prog)
        _, frames = vm.run(prog)
        trace = N.decode_trace(frames)
        assert trace == isa.interpret(code), (name, trace, isa.interpret(code))
        # the WEIGHTS never change — universality is the program-in-DATA swap.
        assert vm.weight_hash() == h0, name


def test_program_lives_in_data_not_weights():
    """Two different programs on the SAME model instance produce different traces
    with an INVARIANT weight hash — the program is DATA, the weights are the fixed
    interpreter."""
    vm = N.NibbleVM(code_size=CODE_SIZE)
    h0 = vm.weight_hash()
    _, f1 = vm.run(PROGRAMS["add_13"])
    _, f2 = vm.run(PROGRAMS["sub_5"])
    assert N.decode_trace(f1)[-1] == 13
    assert N.decode_trace(f2)[-1] == 5
    assert vm.weight_hash() == h0        # ONE model, two programs


# --- 1. RUNTIME-PC DISPATCH: forward branches skip instructions ------------
def test_branch_skips_instruction_via_runtime_pc():
    vm = N.NibbleVM(code_size=CODE_SIZE)
    # JMP over an IMM 99 -> the skipped instruction never executes.
    _, frames = vm.run(PROGRAMS["jmp_skip"])
    assert 99 not in N.decode_trace(frames)
    assert N.decode_trace(frames)[-1] == 42


# --- 3. VANILLA REQUANT: 30-token frames, no torch.round -------------------
def test_thirty_token_frames():
    vm = N.NibbleVM(code_size=CODE_SIZE)
    tokens, frames = vm.run(PROGRAMS["add_13"])
    # exactly one 30-token register frame per VM step + BOS + HALT terminator.
    assert len(tokens) == len(frames) * V.FRAME_LEN + 2
    assert tokens[0] == V.BOS and tokens[-1] == V.HALT
    # each frame parses back to the register integers.
    for k in range(len(frames)):
        frame = tokens[1 + k * V.FRAME_LEN: 1 + (k + 1) * V.FRAME_LEN]
        assert V.parse_step_frame(frame)["ax"] == frames[k]["ax"]


def test_no_torch_round_in_exec_path():
    """The exec path contains no ``round`` / ``torch.round`` CALL node — the frame
    round-trip (LM-head argmax -> re-embed) IS the requantiser (docstrings that
    merely mention the word don't trip the AST guard)."""
    import c4_min.blogspec_model as M
    for mod in (N, M):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                assert name != "round", (mod.__name__, ast.dump(node))


# --- 5. DEEP LOOP: hundreds of steps, byte-exact ---------------------------
def test_deep_loop_countdown_exact():
    """A countdown from 100 runs ~400 VM steps on ONE recurrently-applied
    step-block, byte-exact vs the reference — unbounded, decoupled from the bake."""
    prog = [("IMM", 100), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]
    code = isa.assemble(prog)
    vm = N.NibbleVM(code_size=len(code))
    h0 = vm.weight_hash()
    tokens, frames = vm.run(prog, max_steps=5000)
    trace = N.decode_trace(frames)
    assert trace == isa.interpret(code, max_steps=5000)
    assert len(frames) > 300                       # genuinely a DEEP loop
    assert trace[-1] == 0                           # counted all the way down
    assert len(tokens) == len(frames) * V.FRAME_LEN + 2
    assert vm.weight_hash() == h0                   # same step-block every iteration


# --- the dispatch interface is extensible (the fan-out contract) -----------
def test_dispatch_interface_is_op_gated_rules():
    """The base dispatch is a list of ``FFNRule`` each gated on ``OP_IS[op]`` — the
    contract a fan-out gadget appends to. Confirm each base op has a gate on its
    own decoded-opcode one-hot band."""
    L = N.NibbleVMLayout(code_size=6)
    rules = N.base_dispatch_rules(L)
    gated_ops = set()
    for r in rules:
        for (band, lo, hi) in r.when:
            if L.OP_IS <= band < L.OP_IS + isa.NUM_OPS:
                gated_ops.add(band - L.OP_IS)
    for op in (isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB,
               isa.JMP, isa.BZ, isa.BNZ, isa.HALT):
        assert op in gated_ops, isa.NAMES[op]


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} nibble-VM tests passed")
