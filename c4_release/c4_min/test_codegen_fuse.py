"""test_codegen_fuse.py — the memory-operand-ALU CODEGEN peephole (C4_CODEGEN_FUSE).

Proves:
  1. flag-OFF ``compile_c_fused`` == ``compile_c`` (default codegen untouched -> the
     neural golden is byte-identical, this is a compiler-only change);
  2. the fold is BYTE-EXACT of RESULT — a folded program computes the same visible
     output (PRTF stream + final AX) as the original, over a battery of programs;
  3. the fold FIRES on absolute-address (global) operands and is INERT on
     frame-relative (pointer / local) operands and on the c4 matmul;
  4. the folded bytecode runs byte-exact on the ACTUAL neural exact-steps model
     (``nibble_exact_steps``) — the fold's ``<OP>M`` is executable, not just a
     paper opcode.

Run:  python -m pytest c4_min/test_codegen_fuse.py -q
  or: python c4_min/test_codegen_fuse.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min import codegen_fuse as CF
from c4_min.run_1096_pure_forward import bytecode_to_isa
from c4_min.measure_codegen_fuse import _ref_interpret_fused
from src.compiler import compile_c


def _observable(words):
    """(final AX, PRTF output bytes) of a c4 word list under the census semantics."""
    code = bytecode_to_isa(words)
    out = []
    trace, _, _ = _ref_interpret_fused(code, out=out)
    return (trace[-1] if trace else None), out


def _assert_byte_exact(src, expect_folds=None, link_stdlib=True):
    words, _ = compile_c(src, link_stdlib=link_stdlib)
    folded, nf = CF.fuse_bytecode(words)
    base = _observable(words)
    fold = _observable(folded)
    assert base == fold, f"fold changed semantics: base {base} != folded {fold}\n{src}"
    if expect_folds is not None:
        assert nf == expect_folds, f"expected {expect_folds} folds, got {nf}"
    return nf, words, folded


# ---------------------------------------------------------------------------
# 1. flag-OFF: compile_c_fused == compile_c (default codegen untouched).
# ---------------------------------------------------------------------------
def test_flag_off_codegen_unchanged():
    src = "int g; int main(){ int r; g=10; r=g+g; return r; }"
    os.environ.pop("C4_CODEGEN_FUSE", None)
    base, _ = compile_c(src)
    fused, _ = CF.compile_c_fused(src)
    assert base == fused, "flag-OFF must leave compile_c output byte-identical"


def test_flag_on_codegen_folds():
    src = "int g; int main(){ int r; g=10; r=g+g; return r; }"
    os.environ["C4_CODEGEN_FUSE"] = "1"
    try:
        base, _ = compile_c(src)
        fused, _ = CF.compile_c_fused(src)
        assert len(fused) < len(base), "flag-ON should shorten a foldable program"
    finally:
        os.environ.pop("C4_CODEGEN_FUSE", None)


# ---------------------------------------------------------------------------
# 2 + 3. byte-exact of result over a battery; fold fires / is inert as expected.
# ---------------------------------------------------------------------------
def test_global_operand_folds_byte_exact():
    src = ("int g; int h; int main(){ int r;"
           " g=10; h=3; r=g+h; r=g*h; r=g-h; r=g/h; r=g%h; return r; }")
    nf, _, _ = _assert_byte_exact(src)
    assert nf >= 5, f"expected the 5 global-operand ALU ops to fold, got {nf}"


def test_all_five_ops_fold_correctly():
    # each op in isolation, checking the FINAL result matches base-ISA semantics.
    for c_op, py in (("+", lambda a, b: (a + b) & 0xFF),
                     ("-", lambda a, b: (a - b) & 0xFF),
                     ("*", lambda a, b: (a * b) & 0xFF),
                     ("/", lambda a, b: (a // b) & 0xFF),
                     ("%", lambda a, b: (a % b) & 0xFF)):
        src = f"int g; int h; int main(){{ int r; g=200; h=7; r=g{c_op}h; return r; }}"
        nf, words, folded = _assert_byte_exact(src)
        assert nf >= 1, f"op {c_op} did not fold"
        final, _ = _observable(folded)
        assert final == py(200, 7), f"op {c_op}: {final} != {py(200,7)}"


def test_nested_expression_folds_to_fixpoint():
    # (g + h) * k : the outer '*' left operand is (g+h) (not a pure load) so only the
    # inner g+h folds; the fixpoint pass must still be correct.
    src = "int g; int h; int k; int main(){ int r; g=5; h=6; k=7; r=(g+h)*k; return r; }"
    nf, _, folded = _assert_byte_exact(src)
    final, _ = _observable(folded)
    assert final == ((5 + 6) * 7) & 0xFF


def test_frame_local_operands_do_not_fold():
    # locals are LEA-relative (bp+off) -> NOT absolute-address -> must NOT fold.
    src = "int main(){ int a; int b; int r; a=10; b=3; r=a+b; r=a*b; return r; }"
    words, _ = compile_c(src)
    _, nf = CF.fuse_bytecode(words)
    assert nf == 0, f"frame-local operands must not fold, got {nf}"
    _assert_byte_exact(src, expect_folds=0)


def test_matmul_is_inert_and_byte_exact():
    import random
    from c4_min.selfhost._matmul_general_src import matmul_general_c
    rng = random.Random(7)
    M, K, N = 2, 3, 2
    A = [rng.randint(0, 2) for _ in range(M * K)]
    B = [rng.randint(0, 2) for _ in range(K * N)]
    for call_fpmul in (True, False):
        src = matmul_general_c(A, B, M, K, N, call_fpmul=call_fpmul)
        words, _ = compile_c(src)
        folded, nf = CF.fuse_bytecode(words)
        assert nf == 0, f"matmul (fpmul={call_fpmul}) should have 0 folds, got {nf}"
        assert _observable(words) == _observable(folded)


def test_store_in_span_aborts_fold():
    # r = g + (g = 2): the left g is snapshotted (old value 10), then the b-code
    # stores g:=2 BEFORE the ADD.  A naive <OP>M would re-read mem[g]=2 -> 4; the
    # store guard must abort the fold so the result stays 10+2=12.
    src = "int g; int main(){ int r; g=10; r = g + (g = 2); return r; }"
    words, _ = compile_c(src)
    _, nf = CF.fuse_bytecode(words)
    assert nf == 0, f"a store aliasing the operand must abort the fold, got {nf}"
    _assert_byte_exact(src, expect_folds=0)
    final, _ = _observable(CF.fuse_bytecode(words)[0])
    assert final == 12, final


def test_control_flow_relocation_preserved():
    # a fold BEFORE a while-loop shifts the loop's PC targets; the relocation must
    # keep the loop correct.
    src = ("int g; int main(){ int i; int r; g=4; r=g+g; i=0;"
           " while (i < 3) { r = r + g; i = i + 1; } return r; }")
    nf, _, folded = _assert_byte_exact(src)
    assert nf >= 1
    final, _ = _observable(folded)
    # r = g+g = 8; then loop adds g(=4) three times -> 8 + 12 = 20.
    assert final == 20, final


# ---------------------------------------------------------------------------
# 4. the folded bytecode is EXECUTABLE on the real neural exact-steps model.
# ---------------------------------------------------------------------------
def test_folded_runs_on_neural_model():
    os.environ["C4_EXACT_STEPS"] = "1"
    os.environ["C4_MEM_OPERAND"] = "1"
    from c4_min import nibble_exact_steps as ES

    def W(op, imm=0):
        return int(op) + (int(imm) << 8)

    A = 0x40
    # IMM A; LI; PSH; IMM 7; ADD; HALT   ==   mem[A] + 7
    words = [W(CF.IMM, A), W(CF.LI), W(CF.PSH), W(CF.IMM, 7), W(CF.ADD), W(38)]
    folded, nf = CF.fuse_bytecode(words)
    assert nf == 1
    fcode = bytecode_to_isa(folded)
    assert [i.op for i in fcode] == [isa.IMM, ES.ADDM, isa.HALT]
    seed = {A: 10}
    model, L = ES.build_exact_steps_model(code_size=len(fcode))
    tr = ES.run_exact_steps(model, L, fcode, max_steps=8, seed_mem=seed, mask=0xFF)
    ref = ES.ref_interpret_exact(fcode, seed_mem=seed, mask=0xFF)
    assert tr == ref and tr[-1] == (10 + 7) & 0xFF, (tr, ref)


if __name__ == "__main__":
    test_flag_off_codegen_unchanged();            print("  flag-OFF codegen unchanged: OK")
    test_flag_on_codegen_folds();                 print("  flag-ON folds: OK")
    test_global_operand_folds_byte_exact();        print("  global-operand fold byte-exact: OK")
    test_all_five_ops_fold_correctly();            print("  all 5 <OP>M fold correctly: OK")
    test_nested_expression_folds_to_fixpoint();    print("  nested fixpoint: OK")
    test_frame_local_operands_do_not_fold();       print("  frame-local inert: OK")
    test_matmul_is_inert_and_byte_exact();         print("  matmul inert + byte-exact: OK")
    test_store_in_span_aborts_fold();              print("  store-in-span aborts fold: OK")
    test_control_flow_relocation_preserved();      print("  control-flow relocation: OK")
    test_folded_runs_on_neural_model();            print("  folded runs on neural model: OK")
    print("ALL CODEGEN-FUSE TESTS PASS")
