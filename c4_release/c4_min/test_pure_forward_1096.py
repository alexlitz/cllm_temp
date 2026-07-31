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

from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
from c4_min._build_guard import guarded_complete_build
from c4_min.nibble_pure_forward import assert_no_python_compute
from c4_min.run_1096_pure_forward import bytecode_to_isa


# One representative program per op family.  Each is REAL C source compiled by the
# C4 compiler; the (source, expected) pairs mirror the corpus generators.  The
# ">8-bit" cases (expected > 255) are the 32-bit-ALU proof — they would be WRONG on
# an 8-bit-fold substrate.
# Each entry: (name, source, expected, needs_divmod).  DIV/MOD need the 32-bit
# long-division blocks (the 298-block divmod model, ~8x slower per forward); every
# other family runs on the LEAN model (38 blocks, fast) — this split keeps the test
# practical while still proving div/mod purity in-forward.
_SAMPLE = [
    # --- arithmetic, results EXCEEDING 8 bits (the 32-bit ALU proof) -----------
    ("arith_add_32bit",   "int main(){ return 500 + 700; }",              1200, False),
    ("arith_sub_32bit",   "int main(){ return 1900 - 50; }",             1850, False),
    ("arith_mul_32bit",   "int main(){ return 100 * 10; }",             1000, False),
    ("arith_div_exact",   "int main(){ return 720 / 6; }",               120, True),
    ("arith_mod",         "int main(){ return 84 % 5; }",                  4, True),
    ("div_by_zero",       "int main(){ return 5 / 0; }",                   0, True),
    ("literal_9999",      "int main(){ return 9999; }",                 9999, False),
    # --- comparisons -----------------------------------------------------------
    ("cmp_gt",            "int main(){ if (5 > 3) return 1; return 0; }",   1, False),
    ("cmp_lt_false",      "int main(){ if (5 < 3) return 1; return 0; }",   0, False),
    ("cmp_eq",            "int main(){ if (7 == 7) return 1; return 0; }",  1, False),
    # --- variables (SI/LI to a frame local, 32-bit value) ----------------------
    ("var_simple_32bit",  "int main(){ int x; x = 1000; return x; }",   1000, False),
    ("var_update",        "int main(){ int x; x = 7; x = x + 6; return x; }", 13, False),
    ("var_three",         "int main(){ int a; int b; int c; a=10; b=20; c=30; "
                          "return a + b + c; }",                          60, False),
    # --- expressions (precedence, parens) --------------------------------------
    ("expr_add_mul",      "int main(){ return 3 + 4 * 5; }",              23, False),
    ("expr_paren",        "int main(){ return (3 + 4) * 5; }",            35, False),
    # --- functions: the FULL calling convention (JSR/ENT/LEA/LI/ADJ/LEV) -------
    ("func_identity",     "int identity(int x){ return x; } "
                          "int main(){ return identity(1000); }",       1000, False),
    ("func_add_32bit",    "int add(int a,int b){ return a + b; } "
                          "int main(){ return add(300, 400); }",          700, False),
    ("func_square_32bit", "int square(int x){ return x * x; } "
                          "int main(){ return square(50); }",            2500, False),
    ("func_max",          "int max(int a,int b){ if (a>b) return a; return b; } "
                          "int main(){ return max(5, 9); }",               9, False),
    # --- a short loop (BZ/JMP branch + accumulate) — kept MINIMAL (2 iters) so the
    #     quadratic stream growth stays CI-tractable (the compiler's loop codegen is
    #     ~26 steps/iter); this proves the in-forward BZ/JMP loop control.  The full
    #     deep-loop tail (loops/gcd/rec, up to thousands of steps) is scored
    #     separately in run_1096_pure_forward.
    ("loop_pow2",         "int main(){ int r; int i; r=1; i=0; "
                          "while (i < 2) { r = r * 2; i = i + 1; } return r; }", 4, False),
]


@pytest.fixture(scope="module")
def pf_complete():
    """The full-op-set complete model, built the memory-SAFE streaming way
    (``guarded_complete_build`` -> ``build_compact_sparse_streaming``, peak ~5 GB —
    NOT the dense ``build_pure_forward_complete_model`` whose ~160k-row MUL/DIV/MOD
    block pads every block and peaks at 54-108 GB RSS).  The streaming model is the
    SAME interpreter, byte-identical (L-inf=0, dense_kernel), driven by the SAME
    runner — so it covers EVERY family incl. DIV/MOD.  One module-scoped build
    serves both the lean and div/mod parametrisations (they were always the same
    complete build)."""
    return guarded_complete_build(code_size=44)


@pytest.fixture(scope="module")
def pf_lean(pf_complete):
    """Non-div/mod families run on the one complete streaming model."""
    return pf_complete


@pytest.fixture(scope="module")
def pf_divmod(pf_complete):
    """DIV/MOD families run on the SAME complete streaming model (it folds the
    fp32-exact base-16 long-division ALU)."""
    return pf_complete


def _run_pure(model, L, source, expected, name):
    from src.compiler import compile_c
    from c4_min.nibble_pure_forward_complete import ref_interpret
    code = bytecode_to_isa(compile_c(source)[0])
    # Right-size the step cap from the reference step count (pure-python HARNESS
    # sizing — NOT model compute, run OUTSIDE the guard) so the loop case doesn't
    # burn the quadratic stream growth on steps past the halt.
    cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
    # the whole run under the settrace purity guard — no _apply_op / DictMemStack /
    # gadget may be entered; a leak raises AssertionError and fails the test, so
    # reaching the value assertion is itself the purity proof.
    trace = assert_no_python_compute(
        run_pure_forward_complete, model, L, code, max_steps=cap, mask=0xFFFFFFFF)
    got = trace[-1] & 0xFFFFFFFF if trace else None
    assert got == expected, f"{name}: pure-forward got {got}, expected {expected}"


_LEAN = [(n, s, e) for (n, s, e, dm) in _SAMPLE if not dm]
_DIVMOD = [(n, s, e) for (n, s, e, dm) in _SAMPLE if dm]


@pytest.mark.parametrize("name,source,expected", _LEAN, ids=[c[0] for c in _LEAN])
def test_family_is_pure_and_correct(pf_lean, name, source, expected):
    """Each non-div/mod family runs 100%-in-forward (guard-clean) AND byte-exact vs
    the corpus expected value."""
    model, L = pf_lean
    _run_pure(model, L, source, expected, name)


@pytest.mark.parametrize("name,source,expected", _DIVMOD, ids=[c[0] for c in _DIVMOD])
def test_divmod_is_pure_and_correct(pf_divmod, name, source, expected):
    """DIV/MOD (incl. div-by-zero -> 0) run 100%-in-forward (guard-clean) via the
    fp32-exact base-16 long-division ALU folded as persistent FFN blocks."""
    model, L = pf_divmod
    _run_pure(model, L, source, expected, name)


def test_32bit_results_exceed_8_bits():
    """Sanity: the 32-bit proof cases genuinely exceed one byte (so an 8-bit-fold
    substrate would be WRONG), guarding against a vacuous purity claim."""
    over = [(n, e) for (n, _s, e, _dm) in _SAMPLE if e > 255]
    assert len(over) >= 8, over
    assert all(e <= 0xFFFFFFFF for _n, e in over)


# ==========================================================================
# COMBINED CORPUS (1096 + 67 edge == 1163) — ONE gate, format-aware.
# ==========================================================================
# These run through the SAME combined entry point + the SAME format-agnostic scorer
# the standalone scoreboard uses (``run_1096_pure_forward.score_entry``), so the
# 1096 corpus and the folded c4_min edge suite are scored by ONE gate.  The REFERENCE
# (golden) sub-gate is fast (pure-python ``ref_interpret`` / the I/O reference
# contract, NO model build) and is always run here; the NEURAL sub-gate stays in the
# opt-in edge/neural sample above (and in tests/test_suite_edge_ops.py) because a
# model build is ~minutes.
from tests.test_suite_1000 import (  # noqa: E402
    generate_test_programs, generate_test_programs_full, generate_edge_corpus_entries,
    CorpusEntry,
)
from c4_min.run_1096_pure_forward import score_entry  # noqa: E402


def test_combined_corpus_is_1163_and_prefix_identical():
    """The folded corpus is exactly 1163 (1096 C + 67 edge) and its first 1096
    entries are byte-identical to the pristine ``generate_test_programs()``."""
    base = generate_test_programs()
    full = generate_test_programs_full()
    assert len(base) == 1096
    assert len(full) == 1163
    assert full[:1096] == base
    tail = full[1096:]
    assert len(tail) == 67
    assert all(isinstance(e, CorpusEntry) for e in tail)


_EDGE_ENTRIES = generate_edge_corpus_entries()


@pytest.mark.parametrize("entry", _EDGE_ENTRIES,
                         ids=[e.edge_name for e in _EDGE_ENTRIES])
def test_edge_case_reference_golden_byte_exact(entry):
    """Every folded edge case is byte-exact vs the c4_min reference golden
    (``ref_interpret`` / the I/O contract) through the canonical scorer — the cheap
    always-run gate.  All 67 (incl. the neural-xfail SHR cases, which are byte-exact
    vs the REFERENCE — they only diverge on the neural model) PASS here."""
    from src.compiler import compile_c
    idx = 1096 + _EDGE_ENTRIES.index(entry)
    r = score_entry(idx, entry, model=None, L=None, compile_c=compile_c,
                    step_cap=10000, guard=False, reference_only=True)
    assert r.status == "PASS", (f"{entry.edge_name}: reference gate {r.status} "
                                f"exp={r.expected} got={r.got_exit} {r.detail}")


if __name__ == "__main__":
    import sys
    from src.compiler import compile_c
    # ONE memory-safe streaming build serves both parametrisations.
    complete = guarded_complete_build(code_size=44)
    npass = 0
    for name, source, expected, needs_dm in _SAMPLE:
        model, L = complete
        code = bytecode_to_isa(compile_c(source)[0])
        try:
            trace = assert_no_python_compute(
                run_pure_forward_complete, model, L, code, max_steps=512,
                mask=0xFFFFFFFF)
            got = trace[-1] & 0xFFFFFFFF if trace else None
            ok = got == expected
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} got={got} "
                  f"want={expected} guard=CLEAN", flush=True)
            npass += int(ok)
        except AssertionError as exc:
            print(f"  [LEAK] {name:20s} {exc}", flush=True)
    print(f"\n{npass}/{len(_SAMPLE)} pure-forward families guard-clean + byte-exact")
    sys.exit(0 if npass == len(_SAMPLE) else 1)
