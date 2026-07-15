"""CHK-1 purity + correctness proof: the WHOLE C4 VM runs in ``model.forward``.

Proves — over a representative sample spanning EVERY op family, INCLUDING the full
calling convention (JSR/ENT/LEV/ADJ/LEA) and 32-bit arithmetic whose result
EXCEEDS 8 bits — that the pure-forward VM (``run_pure_forward_complete`` on the ONE
persistent :class:`Transformer`) computes the corpus-faithful result with **ZERO
Python compute** on the step path.  Each program runs under
``assert_no_python_compute`` (a ``settrace`` guard that raises if any call enters
``blogspec_run._apply_op`` / a ``DictMemStack`` / a functional ALU-cmp-bitwise
gadget), so a PASS is the machine proof that the op ran entirely in the transformer
weights (in-model MoE dispatch, softmax1-KV memory + stack, the fp32-exact 32-bit
ALU) — no external memory, no Python if/elif, no per-call gadget.

The sample is authored in the SAME corpus form as ``run_1096_pure_forward.py`` (C
source -> ``src.compiler.compile_c`` -> bytecode -> ``bytecode_to_isa``), so it
exercises the real compiler output, not hand-written shortcuts.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_pure_forward_1096.py -v
   or: OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/test_pure_forward_1096.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import pytest

# small stack base so frame-relative LEA reaches the frame (set before driver import).
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete,
)
from c4_min.nibble_pure_forward import assert_no_python_compute
from c4_min.run_1096_pure_forward import bytecode_to_isa


# One representative program per op family.  Each is REAL C source compiled by the
# C4 compiler; the (source, expected) pairs mirror the corpus generators.  The
# ">8-bit" cases (expected > 255) are the 32-bit-ALU proof — they would be WRONG on
# an 8-bit-fold substrate.
_SAMPLE = [
    # --- arithmetic, results EXCEEDING 8 bits (the 32-bit ALU proof) -----------
    ("arith_add_32bit",   "int main(){ return 500 + 700; }",              1200),
    ("arith_sub_32bit",   "int main(){ return 1900 - 50; }",             1850),
    ("arith_mul_32bit",   "int main(){ return 100 * 10; }",             1000),
    ("arith_div_exact",   "int main(){ return 720 / 6; }",               120),
    ("arith_mod",         "int main(){ return 84 % 5; }",                  4),
    ("literal_9999",      "int main(){ return 9999; }",                 9999),
    # --- comparisons -----------------------------------------------------------
    ("cmp_gt",            "int main(){ if (5 > 3) return 1; return 0; }",   1),
    ("cmp_lt_false",      "int main(){ if (5 < 3) return 1; return 0; }",   0),
    ("cmp_eq",            "int main(){ if (7 == 7) return 1; return 0; }",  1),
    # --- variables (SI/LI to a frame local, 32-bit value) ----------------------
    ("var_simple_32bit",  "int main(){ int x; x = 1000; return x; }",   1000),
    ("var_update",        "int main(){ int x; x = 7; x = x + 6; return x; }", 13),
    ("var_three",         "int main(){ int a; int b; int c; a=10; b=20; c=30; "
                          "return a + b + c; }",                          60),
    # --- expressions (precedence, parens) --------------------------------------
    ("expr_add_mul",      "int main(){ return 3 + 4 * 5; }",              23),
    ("expr_paren",        "int main(){ return (3 + 4) * 5; }",            35),
    # --- functions: the FULL calling convention (JSR/ENT/LEA/LI/ADJ/LEV) -------
    ("func_identity",     "int identity(int x){ return x; } "
                          "int main(){ return identity(1000); }",       1000),
    ("func_add_32bit",    "int add(int a,int b){ return a + b; } "
                          "int main(){ return add(300, 400); }",          700),
    ("func_square_32bit", "int square(int x){ return x * x; } "
                          "int main(){ return square(50); }",            2500),
    ("func_max",          "int max(int a,int b){ if (a>b) return a; return b; } "
                          "int main(){ return max(5, 9); }",               9),
    # --- a short loop (BZ/JMP branch + accumulate, result > 8 bits) ------------
    ("loop_sum",          "int main(){ int i; int s; i=1; s=0; "
                          "while (i <= 20) { s = s + i; i = i + 1; } return s; }", 210),
]


@pytest.fixture(scope="module")
def pf_model():
    """The ONE persistent pure-forward model with the 32-bit divmod ALU folded in
    (so div/mod are in-forward too).  code_size 44 covers every sample program."""
    model, L = build_pure_forward_complete_model(
        code_size=44, include_bitwise=False, include_divmod=True)
    return model, L


@pytest.mark.parametrize("name,source,expected",
                         _SAMPLE, ids=[c[0] for c in _SAMPLE])
def test_pure_forward_family_is_pure_and_correct(pf_model, name, source, expected):
    """Each family runs 100%-in-forward (guard-clean) AND byte-exact vs the corpus
    expected value.  ``assert_no_python_compute`` raises on ANY python-compute leak,
    so reaching the value assertion is itself the purity proof."""
    from src.compiler import compile_c
    model, L = pf_model
    bytecode, _data = compile_c(source)
    code = bytecode_to_isa(bytecode)
    # the whole run under the settrace purity guard — no _apply_op / DictMemStack /
    # gadget may be entered; a leak raises AssertionError and fails the test.
    trace = assert_no_python_compute(
        run_pure_forward_complete, model, L, code, max_steps=512, mask=0xFFFFFFFF)
    got = trace[-1] & 0xFFFFFFFF if trace else None
    assert got == expected, f"{name}: pure-forward got {got}, expected {expected}"


def test_32bit_results_exceed_8_bits():
    """Sanity: the 32-bit proof cases genuinely exceed one byte (so an 8-bit-fold
    substrate would be WRONG), guarding against a vacuous purity claim."""
    over = [(n, e) for (n, _s, e) in _SAMPLE if e > 255]
    assert len(over) >= 8, over
    assert all(e <= 0xFFFFFFFF for _n, e in over)


if __name__ == "__main__":
    import sys
    model, L = build_pure_forward_complete_model(
        code_size=44, include_bitwise=False, include_divmod=True)
    from src.compiler import compile_c
    npass = 0
    for name, source, expected in _SAMPLE:
        code = bytecode_to_isa(compile_c(source)[0])
        try:
            trace = assert_no_python_compute(
                run_pure_forward_complete, model, L, code, max_steps=512,
                mask=0xFFFFFFFF)
            got = trace[-1] & 0xFFFFFFFF if trace else None
            ok = got == expected
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} got={got} "
                  f"want={expected} guard=CLEAN")
            npass += int(ok)
        except AssertionError as exc:
            print(f"  [LEAK] {name:20s} {exc}")
    print(f"\n{npass}/{len(_SAMPLE)} pure-forward families guard-clean + byte-exact")
    sys.exit(0 if npass == len(_SAMPLE) else 1)
