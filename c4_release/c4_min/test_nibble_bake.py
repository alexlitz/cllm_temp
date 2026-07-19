"""nibble_bake tests: a C program COMPILED INTO THE TRANSFORMER WEIGHTS.

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_nibble_bake.py
(or directly: python c4_min/test_nibble_bake.py).

The load-bearing claims proved here (BLOG_SPEC.md §"Baking Programs into the
Transformer Weights"):

  1. A BAKED program runs byte-exact vs the reference interpreter — WITH NO
     BYTECODE ANYWHERE IN THE INPUT STATE (the baked run starts from the bare
     embedding: ONE=1, all registers/PC = 0). The program lives entirely in the
     WEIGHTS (the eq-check -> AND -> one-hot -> MoE-value read-only code segment).

  2. The CONTRAST with the universal interpreter: the universal model needs the
     program LOADED INTO DATA to run it (zero it out and it no longer runs the
     program); the baked model needs no such load.

  3. The baking mechanism is exactly eq-check -> AND -> one-hot -> MoE-value, and
     the only thing that changed vs universal is the FETCH — the downstream
     decode/dispatch/branch/fold/emit blocks are byte-identical (imported from
     universal.py), so the two produce the same trace on the same program.
"""
from __future__ import annotations

import torch

from c4_min import isa
from c4_min.nibble_bake import (BakedProgram, build_baked_step,
                                initial_state_no_program, run_baked,
                                compile_pc_split, compile_nibble_eq,
                                compile_addr_and, compile_baked_value, NIBBLE)


# ------------------------------------------------------------- helpers --------

def _check_baked(prog, max_steps=100000):
    """Bake ``prog`` into weights, run with NO bytecode in input, assert == ref."""
    code = isa.assemble(prog)
    ref = isa.interpret(code, max_steps=max_steps)
    bp = BakedProgram(prog)
    got = bp.run(max_steps=max_steps)
    assert got == ref, f"{prog}: baked {got} != ref {ref}"
    return bp, got, ref


# ============================================================================
# 1. BAKED programs run byte-exact — with the program in WEIGHTS, not data.
# ============================================================================

def test_straightline_add_baked():
    _, got, _ = _check_baked([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0),
                              ("HALT", 0)])
    assert got[-1] == 8


def test_sub_underflow_baked():
    _, got, _ = _check_baked([("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0),
                              ("HALT", 0)])
    assert got[-1] == 254           # 7 - 9 = -2 -> 254 (mod 256)


def test_jmp_and_branch_baked():
    _check_baked([("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)])
    _check_baked([("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])
    _check_baked([("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])


def test_computed_if_then_else_baked():
    def ite(a, b, X, Y):
        return [("IMM", a), ("PSH", 0), ("IMM", b), ("SUB", 0),
                ("BZ", 7), ("IMM", Y), ("JMP", 8), ("IMM", X), ("HALT", 0)]
    for a, b, X, Y in [(5, 5, 111, 222), (5, 7, 111, 222)]:
        _, got, _ = _check_baked(ite(a, b, X, Y))
        assert got[-1] == (X if a == b else Y)


def test_deep_countdown_baked():
    """A loop far past any small unroll — the baked step-block, applied
    recurrently, runs it exactly (802 VM steps)."""
    prog = [("IMM", 200), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("HALT", 0)]
    _, got, ref = _check_baked(prog)
    assert len(ref) == 802 and got[-1] == 0


def test_pc_crosses_nibble_boundary():
    """A program with >16 instructions forces PC_HI != 0, exercising BOTH nibble
    equality fields (the eq-check must AND lo AND hi, not just lo)."""
    prog = [("IMM", i % 7) for i in range(18)] + [("HALT", 0)]   # 19 slots
    _, got, ref = _check_baked(prog)
    assert len(prog) > NIBBLE                     # PC reaches >= 16
    assert got == ref


# ============================================================================
# 2. THE PROOF: the baked program runs with NO bytecode in the input state.
# ============================================================================

def test_no_bytecode_in_input():
    """The baked run's initial state is the BARE embedding: ONE=1, everything
    else 0. There is no CODE band at all — the bytecode is entirely in weights."""
    prog = [("IMM", 42), ("HALT", 0)]
    bp = BakedProgram(prog)
    state = initial_state_no_program(bp.model, bp.L)

    # (a) the input is exactly the baked embedding row (nothing loaded on top).
    assert torch.equal(state, bp.model.embed[0])
    # (b) exactly ONE band is nonzero, and it is the constant ONE lane.
    nz = (state.abs() > 1e-9).nonzero().flatten().tolist()
    assert nz == [bp.L.ONE], f"unexpected nonzero input bands: {nz}"
    assert float(state[bp.L.ONE]) == 1.0
    # (c) the layout allocates NO CODE_OP/CODE_IMM/CODE_WORD data band at all.
    assert not hasattr(bp.L, "CODE_OP")
    assert not hasattr(bp.L, "CODE_IMM")
    assert not hasattr(bp.L, "CODE_WORD")
    # (d) and yet it runs the program byte-exact.
    got = bp.run()
    assert got == isa.interpret(isa.assemble(prog))
    assert got[-1] == 42


def test_program_lives_in_weights():
    """Changing the program changes the WEIGHTS (the MoE-value biases), not the
    input. Two different bakes with the SAME (all-zero) input produce different
    traces — proof the program is carried by the weights."""
    p1 = [("IMM", 11), ("HALT", 0)]
    p2 = [("IMM", 99), ("HALT", 0)]
    b1, b2 = BakedProgram(p1), BakedProgram(p2)
    # identical (bytecode-free) initial states...
    s1 = initial_state_no_program(b1.model, b1.L)
    s2 = initial_state_no_program(b2.model, b2.L)
    assert torch.equal(s1, s2)
    # ...but different weights -> different output.
    assert b1.run()[-1] == 11
    assert b2.run()[-1] == 99
    # the difference is in the baked MoE-value block (block index 3), the code seg.
    w1 = b1.model.blocks[3].ffn.b_gate
    w2 = b2.model.blocks[3].ffn.b_gate
    assert not torch.equal(w1, w2)               # the program IS these biases


# ============================================================================
# 3. CONTRAST with the universal interpreter (program in DATA vs in WEIGHTS).
# ============================================================================

def test_contrast_universal_needs_program_in_data():
    """The universal interpreter runs the program from DATA: blank the data and it
    no longer runs THAT program. The baked model has no data to blank."""
    try:
        from c4_min.universal import UniversalInterpreter, load_program, run_universal
    except Exception:
        import pytest
        pytest.skip("universal.py not present on this branch")

    prog = [("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]
    ref = isa.interpret(isa.assemble(prog))

    ui = UniversalInterpreter(code_size=8)
    # WITH the program loaded into data: correct.
    assert ui.run(prog) == ref

    # WITHOUT loading the program (bare embedding, no data): the universal model
    # does NOT reproduce the program (it fetches all-zero code == LEA 0 forever).
    import torch.nn.functional as F
    from c4_min.compiler import head_matrix
    from c4_min.universal import _step_once, _requantize
    W, b = head_matrix(ui.L, ui.model.dim, ui.L.OUT_SLOTS[0], halt_band=None)
    state = ui.model.embed[0].clone()            # bare embedding, NO load_program
    blank_trace = []
    for _ in range(len(ref) + 2):
        state = _step_once(ui.model, state)
        state = _requantize(state, ui.L)
        blank_trace.append(int(F.linear(state, W, b).argmax().item()))
        if float(state[ui.L.HALT_SEEN[0]]) > 0.5:
            break
    assert blank_trace != ref                    # universal needs the data


def test_contrast_baked_needs_no_data():
    """The mirror image: the baked model reproduces the SAME program from the SAME
    bytecode-free bare embedding that made the universal model fail above."""
    prog = [("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]
    ref = isa.interpret(isa.assemble(prog))
    bp = BakedProgram(prog)
    # no load step exists; run() starts from the bare embedding.
    assert bp.run() == ref


def test_baked_matches_universal_on_same_program():
    """eq-check baked fetch and data-band universal fetch yield the SAME trace for
    the same program across a battery — the fetch is the ONLY thing that changed."""
    try:
        from c4_min.universal import UniversalInterpreter
    except Exception:
        import pytest
        pytest.skip("universal.py not present on this branch")

    progs = [
        [("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)],
        [("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0), ("HALT", 0)],
        [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
        [("IMM", 200), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)],
    ]
    ui = UniversalInterpreter(code_size=8)
    for prog in progs:
        assert BakedProgram(prog).run() == ui.run(prog), prog


# ============================================================================
# 4. The baking MECHANISM, unit-tested at each stage (eq-check/AND/one-hot/MoE).
# ============================================================================

def _fwd(spec, state, dim):
    """Run one baked FFN spec on a [D] state, return the new [D]."""
    import torch.nn.functional as F
    W_up, b_up = spec["W_up"], spec["b_up"]
    W_gate, b_gate = spec["W_gate"], spec["b_gate"]
    W_down, b_down = spec["W_down"], spec["b_down"]
    up = F.linear(state, W_up) + b_up
    gate = F.linear(state, W_gate) + b_gate
    hidden = F.silu(up) * gate
    return state + F.linear(hidden, W_down, b_down)


def test_mechanism_stages_produce_onehot_and_value():
    """Walk the four fetch stages on a hand-built state and assert the eq-check,
    the AND one-hot, and the MoE value each do exactly what the spec says."""
    code = isa.assemble([("IMM", 42), ("PSH", 0), ("ADD", 0), ("SUB", 0),
                         ("IMM", 9)] + [("HALT", 0)] * 14)  # 19 slots -> PC_HI used
    bp = build_baked_step(code)
    model, L = bp
    dim = L.D

    s_split = compile_pc_split(L, dim)
    s_eq = compile_nibble_eq(L, dim)
    s_and = compile_addr_and(L, len(code), dim)
    s_val = compile_baked_value(L, code, dim)

    for pc in (0, 2, 3, 17):                       # incl. pc>=16 (two-nibble key)
        st = torch.zeros(dim)
        st[L.ONE] = 1.0
        st[L.PC] = float(pc)
        st = _fwd(s_split, st, dim)               # PC -> PC_LO/PC_HI
        assert round(float(st[L.PC_LO])) == pc & 0xF
        assert round(float(st[L.PC_HI])) == pc >> 4
        st = _fwd(s_eq, st, dim)                   # eq-check: nibble one-hots
        assert round(float(st[L.EQ_LO + (pc & 0xF)])) == 1
        assert round(float(st[L.EQ_HI + (pc >> 4)])) == 1
        st = _fwd(s_and, st, dim)                  # AND -> ADDR_IS one-hot
        hot = [i for i in range(len(code))
               if round(float(st[L.ADDR_IS[i]])) == 1]
        assert hot == [pc], f"pc={pc} addr one-hot = {hot}"
        st = _fwd(s_val, st, dim)                  # MoE value -> OP_VAL/IMM
        assert round(float(st[L.OP_VAL])) == code[pc].op
        assert round(float(st[L.IMM])) == code[pc].imm


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} nibble_bake tests passed")
    raise SystemExit(0 if passed == len(tests) else 1)
