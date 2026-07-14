"""Recurrent-interpreter tests: ONE baked step-block, applied autoregressively
with per-step integer re-quantisation, runs programs to completion EXACTLY —
including DEEP LOOPS whose real step count far exceeds any bounded unroll.

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_recurrent.py
(or directly: python c4_min/test_recurrent.py).

The load-bearing claim proved here: with per-step re-quantisation the recurrent
form is exact over THOUSANDS of steps (the drift limit of the un-requantised
depth-unroll is gone); WITHOUT re-quantisation the identical loop drifts and the
program eventually fails to terminate.
"""
from __future__ import annotations

from c4_min import isa
from c4_min.recurrent import StepModel, build_step_model, run_recurrent


def _check(prog, max_steps=100000):
    """Compile ONE step-block, run it recurrently, assert == reference interp."""
    code = isa.assemble(prog)
    ref = isa.interpret(code, max_steps=max_steps)
    sm = StepModel(prog)
    assert sm.n_blocks == 4, sm.n_blocks          # exactly one VM step (4 sub-blocks)
    got = sm.run(max_steps=max_steps)
    assert got == ref, f"{prog}: recurrent {got} != ref {ref}"
    return got, ref


# --- the recurrent form reproduces the whole straight-line + control slice ---

def test_straightline_add():
    got, _ = _check([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)])
    assert got[-1] == 8


def test_sub_underflow():
    got, _ = _check([("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0), ("HALT", 0)])
    assert got[-1] == 254


def test_jmp_and_branch():
    _check([("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)])
    _check([("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])
    _check([("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])


def test_computed_if_then_else():
    def ite(a, b, X, Y):
        return [("IMM", a), ("PSH", 0), ("IMM", b), ("SUB", 0),
                ("BZ", 7), ("IMM", Y), ("JMP", 8), ("IMM", X), ("HALT", 0)]
    for a, b, X, Y in [(5, 5, 111, 222), (5, 7, 111, 222)]:
        got, _ = _check(ite(a, b, X, Y))
        assert got[-1] == (X if a == b else Y)


# --- DEEP LOOPS: the bounded unroll cannot do these; the recurrent form can ---

def _countdown(start):
    """AX=start; loop{ AX-=1 } while AX!=0; HALT.  ~4*start+2 VM steps."""
    return [("IMM", start), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("HALT", 0)]


def test_deep_countdown_exact():
    """A loop that runs far past any small bounded unroll, exact end-to-end."""
    got, ref = _check(_countdown(200))          # 802 VM steps
    assert len(ref) == 802 and got[-1] == 0


def test_thousands_of_steps_exact():
    """~5000 steps across chained countdowns — proves effectively-unbounded exact
    iteration with ONE baked step-block (4 physical blocks)."""
    prog = []
    for _ in range(5):
        base = len(prog)
        prog += [("IMM", 255), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                 ("BNZ", base + 1)]
    prog += [("HALT", 0)]
    got, ref = _check(prog, max_steps=1_000_000)
    assert len(ref) > 5000                       # genuinely deep
    assert got == ref


def test_requantization_is_load_bearing():
    """WITHOUT per-step re-quantisation the SAME loop drifts and never halts;
    WITH it, the run is exact. This is the mechanism that unlocks 1096."""
    prog = _countdown(200)
    code = isa.assemble(prog)
    ref = isa.interpret(code, max_steps=100000)
    model, L, _ = build_step_model(code)

    got_q = run_recurrent(model, L, code, max_steps=100000, requantize=True)
    assert got_q == ref                          # exact with requant

    # Without requant the loop drifts and never hits its HALT on time; cap the
    # run just past the true step count so the "failed to terminate" is visible
    # without paying for a runaway to 100k steps.
    cap = len(ref) + 50
    got_nr = run_recurrent(model, L, code, max_steps=cap, requantize=False)
    assert got_nr != ref                         # drifts without it
    assert len(got_nr) == cap                    # never terminated (ran to the cap)


if __name__ == "__main__":
    import traceback
    tests = [
        test_straightline_add, test_sub_underflow, test_jmp_and_branch,
        test_computed_if_then_else, test_deep_countdown_exact,
        test_thousands_of_steps_exact, test_requantization_is_load_bearing,
    ]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} recurrent tests passed")
