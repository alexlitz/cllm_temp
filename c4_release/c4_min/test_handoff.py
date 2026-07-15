"""COMPILE-THEN-EXECUTE HANDOFF tests: a transformer that produces bytecode in
memory and then runs it — the mechanism behind "Model that Directly Runs C Code".

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_handoff.py
(or directly: python c4_min/test_handoff.py).

The load-bearing claims (BLOG_SPEC.md §"Model that Directly Runs C Code"):

  1. EMIT writes a produced instruction word into CODE MEMORY at a runtime slot,
     and the universal fetch then reads THAT freshly-produced word and runs it —
     the compile (produce bytecode) -> execute (jump to it) handoff.

  2. The produced code slots are EMPTY (== 0) in the initial state: the program
     that ultimately runs is NOT present at load; it is produced at runtime.

  3. The whole thing is ONE fixed-weight interpreter (universal fetch + dispatch +
     EMIT store); only the CODE data changes. Adding EMIT did not disturb the
     universal ops (a normal program runs byte-identically to the reference).
"""
from __future__ import annotations

import torch

from c4_min import isa
from c4_min import nibble_handoff as H


# ---------------------------------------------------------------- helpers -----

def _ref(gen, code_size):
    code = H._assemble_emit(gen)
    return H.interpret_with_emit(code, code_size)


# ============================================================================
# 1. EMIT is a clean store into code memory (unaffected slots untouched).
# ============================================================================

def test_normal_program_unchanged_by_emit_support():
    """Adding EMIT to the ISA must not perturb the universal ops: a normal program
    runs byte-identically to the reference interpreter."""
    hm = H.HandoffMachine(code_size=16)
    for prog in (
        [("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)],
        [("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0), ("HALT", 0)],
        [("IMM", 0), ("BZ", 4), ("IMM", 99), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
    ):
        got = hm.run(prog)
        ref = isa.interpret(isa.assemble(prog))
        assert got == ref, f"{prog}: {got} != {ref}"


def test_single_emit_writes_code_memory():
    """EMIT imm writes CODE_WORD[imm] := AX + 256*STACK0 (the assembled word),
    leaving every other slot untouched."""
    # AX=1 (op IMM), STACK0=42 (imm) -> word 0x2A01 into slot 7.
    gen = [("IMM", 42), ("PSH", 0), ("IMM", 1), ("EMIT", 7), ("HALT", 0)]
    hm = H.HandoffMachine(code_size=16)
    trace, words = hm.run(gen, return_code=True)
    assert words[7] == (isa.IMM | (42 << 8)) == 0x2A01
    # only slot 7 among the empty region changed
    for i in range(8, 16):
        assert words[i] == 0, f"slot {i} unexpectedly written: {words[i]:#x}"


# ============================================================================
# 2. THE HANDOFF: produce one instruction, jump to it, execute it.
# ============================================================================

def test_handoff_single_instruction():
    """Emit 'IMM 42' into an empty slot, JMP there; the universal fetch reads the
    freshly-produced word and runs it -> AX == 42."""
    OUT = 5
    gen = [("IMM", 42), ("PSH", 0), ("IMM", isa.IMM), ("EMIT", OUT),
           ("JMP", OUT), ("IMM", 0), ("HALT", 0)]  # slot5 placeholder, slot6 HALT
    hm = H.HandoffMachine(code_size=16)

    # slot OUT is EMPTY (placeholder IMM 0) at load — a placeholder, then produced.
    code = H._assemble_emit(gen)
    st = H.load_program(hm.model, hm.L, code)
    assert int(round(float(st[hm.L.CODE_WORD[OUT]]))) == (isa.IMM | (0 << 8))

    trace, words = hm.run(gen, return_code=True)
    ref_trace, ref_words = _ref(gen, 16)
    assert words[OUT] == (isa.IMM | (42 << 8)), f"produced {words[OUT]:#x}"
    assert trace == ref_trace
    assert trace[-1] == 42


# ============================================================================
# 3. COMPILE-THEN-EXECUTE: produce a MULTI-instruction program, run it.
# ============================================================================

def test_handoff_compile_and_run_expression():
    """The headline: a generator that COMPILES '2+3' into a 5-instruction bytecode
    program (IMM 2; PSH; IMM 3; ADD; HALT) in EMPTY code memory at runtime, then
    JMPs to it. The universal fetch reads the produced bytecode and evaluates it
    -> AX == 5. The produced slots are all 0 at load (the program is NOT there)."""
    produced = [("IMM", 2), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]
    CODE_SIZE, OUT_BASE = 32, 24
    gen = []
    for k, (name, pim) in enumerate(produced):
        gen += [("IMM", pim), ("PSH", 0), ("IMM", isa.BY_NAME[name]),
                ("EMIT", OUT_BASE + k)]
    gen += [("JMP", OUT_BASE)]

    hm = H.HandoffMachine(code_size=CODE_SIZE)
    code = H._assemble_emit(gen)

    # (a) the produced program's slots are EMPTY (0) at load — NOT present.
    st = H.load_program(hm.model, hm.L, code)
    for k in range(len(produced)):
        assert int(round(float(st[hm.L.CODE_WORD[OUT_BASE + k]]))) == 0

    trace, words = hm.run(gen, return_code=True, max_steps=4096)
    ref_trace, ref_words = _ref(gen, CODE_SIZE)

    # (b) the runtime produced EXACTLY the c4-encoded bytecode for 2+3.
    expected = [(isa.BY_NAME[n] | (i << 8)) for n, i in produced]
    produced_words = [words[OUT_BASE + k] for k in range(len(produced))]
    assert produced_words == expected, f"{produced_words} != {expected}"

    # (c) jumping into the freshly-produced code evaluated it: 2+3 == 5.
    assert trace == ref_trace
    assert trace[-1] == 5


def test_handoff_compile_and_run_subtraction():
    """A second compiled program, 10-3=7, to show the generator is programmable
    (the produced opcodes/immediates are data the generator computes)."""
    produced = [("IMM", 10), ("PSH", 0), ("IMM", 3), ("SUB", 0), ("HALT", 0)]
    CODE_SIZE, OUT_BASE = 32, 24
    gen = []
    for k, (name, pim) in enumerate(produced):
        gen += [("IMM", pim), ("PSH", 0), ("IMM", isa.BY_NAME[name]),
                ("EMIT", OUT_BASE + k)]
    gen += [("JMP", OUT_BASE)]
    hm = H.HandoffMachine(code_size=CODE_SIZE)
    trace, words = hm.run(gen, return_code=True, max_steps=4096)
    ref_trace, _ = _ref(gen, CODE_SIZE)
    assert trace == ref_trace
    assert trace[-1] == 7


def test_handoff_produce_and_run_loop():
    """The fetch re-reads the produced code memory EVERY step: produce a countdown
    LOOP (with a backward BNZ) into empty memory, JMP to it, and run it to 0. If the
    fetch only replayed the produced code once, the loop would not iterate."""
    CODE_SIZE, OUT_BASE = 48, 24
    produced = [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                ("BNZ", OUT_BASE + 1), ("HALT", 0)]
    gen = []
    for k, (name, pim) in enumerate(produced):
        gen += [("IMM", pim), ("PSH", 0), ("IMM", isa.BY_NAME[name]),
                ("EMIT", OUT_BASE + k)]
    gen += [("JMP", OUT_BASE)]
    hm = H.HandoffMachine(code_size=CODE_SIZE)
    trace, words = hm.run(gen, return_code=True, max_steps=4096)
    ref_trace, _ = _ref(gen, CODE_SIZE)
    expected = [(isa.BY_NAME[n] | (i << 8)) for n, i in produced]
    assert [words[OUT_BASE + k] for k in range(len(produced))] == expected
    assert trace == ref_trace
    assert trace[-1] == 0                     # countdown reached zero via the loop


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} handoff tests passed")
    raise SystemExit(0 if passed == len(tests) else 1)
