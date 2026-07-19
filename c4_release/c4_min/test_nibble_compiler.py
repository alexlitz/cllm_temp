"""MODEL-RUNS-C, FULL PATH tests: a COMPILER (in bytecode, and in the WEIGHTS)
reads C SOURCE, compiles it to bytecode, and runs it — C source in -> result out.

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_nibble_compiler.py
(or directly: python c4_min/test_nibble_compiler.py).

The load-bearing claims (BLOG_SPEC.md §"Model that Directly Runs C Code"):

  1. The new ISA ops are byte-exact bilinear read/writes over data bands: LC reads
     the SOURCE at a runtime address, LI/SI read/write scratch memory, MUL folds
     the SP stack, EMIT writes code memory (and pops). The SP-indexed stack makes
     nested expressions (depth > 1) evaluate correctly (2+3*4 == 14, not 15).

  2. THE COMPILER: a program written in c4_min bytecode reads a C expression out of
     the SOURCE band, parses it with correct operator precedence (* over +), EMITs
     the compiled bytecode, and JMPs to it — all on ONE fixed-weight interpreter.
     The produced bytecode is the SAME op|imm<<8 packed encoding real c4 emits.

  3. THE COMPILER IN THE WEIGHTS: the same compiler baked into the FFN biases (the
     hybrid baked/memory word-select), so the input state carries ONLY the C source
     — no compiler bytecode anywhere in the data — and it still emits+runs 2+3*4.
"""
from __future__ import annotations

import torch

from c4_min import isa
from c4_min import nibble_compiler as C


# small shared machines (built once)
_CM = None
_BCM = None


def _cm():
    global _CM
    if _CM is None:
        _CM = C.CompilerMachine(code_size=176, src_size=8, mem_size=8, stack_depth=8)
    return _CM


def _bcm():
    global _BCM
    if _BCM is None:
        _BCM = C.BakedCompilerMachine(C.expr_compiler_bytecode(), code_size=176,
                                      src_size=8, mem_size=8, stack_depth=8)
    return _BCM


# ============================================================================
# 1. The new ISA ops (LC / LI / SI / MUL / EMIT) are byte-exact vs the reference.
# ============================================================================

def test_mul_folds_the_stack():
    prog = [("IMM", 3), ("PSH", 0), ("IMM", 4), ("MUL", 0), ("HALT", 0)]
    got = _cm().run(prog, source="")
    ref, _ = C.interpret_full(C._assemble(prog), [], 176, mem_size=8)
    assert got == ref and got[-1] == 12


def test_lc_reads_source_at_runtime_address():
    for addr, ch in [(0, "2"), (1, "+"), (2, "3"), (3, "*"), (4, "4")]:
        prog = [("IMM", addr), ("LC", 0), ("HALT", 0)]
        got = _cm().run(prog, source="2+3*4")
        assert got[-1] == ord(ch), f"src[{addr}] -> {got[-1]} != {ord(ch)}"


def test_li_si_roundtrip():
    # store 42 at mem[5], read it back
    prog = [("IMM", 5), ("PSH", 0), ("IMM", 42), ("SI", 0),
            ("IMM", 5), ("LI", 0), ("HALT", 0)]
    got = _cm().run(prog, source="")
    ref, _ = C.interpret_full(C._assemble(prog), [], 176, mem_size=8)
    assert got == ref and got[-1] == 42


def test_sp_indexed_stack_depth_2_nested_expr():
    # 2+3*4 as a produced program needs depth 2 — the single-mirror stack gave 15.
    prog = [("IMM", 2), ("PSH", 0), ("IMM", 3), ("PSH", 0), ("IMM", 4),
            ("MUL", 0), ("ADD", 0), ("HALT", 0)]
    got = _cm().run(prog, source="")
    assert got[-1] == 14, f"depth-2 nested expr gave {got[-1]} != 14"


# ============================================================================
# 2. THE COMPILER (bytecode): C source in -> compiled bytecode -> runs it.
# ============================================================================

_EXPRS = {"2+3*4": 14, "2*3+4": 10, "1+2+3": 6, "2*3*4": 24,
          "4+5*6": 34, "3*4+5": 17, "9+8+7": 24, "5*6*7": 210}


def test_compiler_compiles_and_runs_precedence():
    comp = C.expr_compiler_bytecode()
    cm = _cm()
    for src, want in _EXPRS.items():
        got = cm.run(comp, source=src, max_steps=4000)
        assert got[-1] == want, f"{src}: model -> {got[-1]} != {want}"


def test_compiler_produces_the_c4_bytecode_for_2_plus_3_times_4():
    """The produced bytecode is byte-identical to what the real c4 compiler emits
    for 2+3*4: IMM 2; PSH; IMM 3; PSH; IMM 4; MUL; ADD; HALT (op|imm<<8 packed)."""
    comp = C.expr_compiler_bytecode()
    trace, words = _cm().run(comp, source="2+3*4", return_code=True, max_steps=4000)
    O = C.COMPILER_OUTBASE
    produced = [words[O + k] for k in range(8)]
    expected = [C.make_word(n, i) for n, i in
                [("IMM", 2), ("PSH", 0), ("IMM", 3), ("PSH", 0),
                 ("IMM", 4), ("MUL", 0), ("ADD", 0), ("HALT", 0)]]
    assert produced == expected, f"{[hex(w) for w in produced]}"
    assert trace[-1] == 14


def test_compiler_produced_slots_empty_at_load():
    """The produced program's code slots are 0 at load — the program is NOT present;
    it is produced at runtime by the compiler reading the source."""
    comp = C.expr_compiler_bytecode()
    cm = _cm()
    code = C._assemble(comp)
    st = C.load_program(cm.model, cm.L, code, C._source_bytes("2+3*4"))
    O = C.COMPILER_OUTBASE
    for k in range(8):
        assert int(round(float(st[cm.L.CODE_WORD[O + k]]))) == 0


def test_compiler_matches_reference_exactly():
    comp = C.expr_compiler_bytecode()
    cm = _cm()
    for src in _EXPRS:
        got = cm.run(comp, source=src, max_steps=4000)
        ref, _ = C.interpret_full(C._assemble(comp), C._source_bytes(src), 176,
                                  mem_size=8, max_steps=4000)
        assert got == ref, f"{src}: model != reference"


# ============================================================================
# 3. THE COMPILER IN THE WEIGHTS: only the C source is input, no bytecode.
# ============================================================================

def test_baked_compiler_no_bytecode_in_input():
    bcm = _bcm()
    st = C.initial_state_baked(bcm.model, bcm.L, C._source_bytes("2+3*4"))
    # NO compiler bytecode and NO produced program in the input — CODE_WORD is 0.
    for i in range(bcm.L.CODE_SIZE):
        assert int(round(float(st[bcm.L.CODE_WORD[i]]))) == 0
    # only the source is present
    assert [int(st[bcm.L.SRC[i]]) for i in range(5)] == [ord(c) for c in "2+3*4"]


def test_baked_compiler_compiles_and_runs():
    bcm = _bcm()
    for src, want in [("2+3*4", 14), ("2*3+4", 10), ("1+2+3", 6)]:
        got = bcm.run(src, max_steps=4000)
        assert got[-1] == want, f"baked {src} -> {got[-1]} != {want}"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} compiler tests passed")
    raise SystemExit(0 if passed == len(tests) else 1)
