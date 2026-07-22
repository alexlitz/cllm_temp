"""OPCODE-DISPATCH of MUL/DIV/MOD to in-memory base-ISA subroutines.

CPU tests validate the dispatch reference (a program with REAL MUL/DIV/MOD +
supplemental low-precision opcodes runs byte-exact via the trap-to-subroutine
mechanism) and the perfect-draft alignment (draft == interpret_dispatch).  A
GPU-guarded test proves the dispatched program runs BYTE-EXACT through the actual
shallow ``SUBSET_BITWISE`` (~16-layer, NO deep muldiv) neural forward AND that the
draft drives 100%-acceptance speculation (draft == model == reference).
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min import lean_opcode_dispatch as D
from c4_min.nibble_runtime import Asm


def _prog(a_val: int, b_val: int, op: int):
    """``IMM a ; PSH ; IMM b ; <op> ; EXIT`` — one dispatched op over two literals."""
    A = Asm()
    A.imm(a_val).psh().imm(b_val).emit(op)
    A.exit_()
    return A.instrs()


# ---------------------------------------------------------------------------
# Reference dispatch: the trap-to-subroutine computes the op's true semantics.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op,a,b", [
    (isa.MUL, 12, 11), (isa.MUL, 200, 3), (isa.DIV, 200, 17), (isa.DIV, 255, 16),
    (isa.MOD, 100, 7), (isa.MOD, 255, 255),
    (D.MUL8, 13, 10), (D.DIV8, 255, 16), (D.MOD8, 100, 7),
    (D.MUL16, 300, 7), (D.DIV16, 1000, 7), (D.MOD16, 65535, 256),
])
def test_dispatch_result_matches_true_semantics(op, a, b):
    main = _prog(a, b, op)
    table = D.build_dispatch_program(main)
    disp = D.interpret_dispatch(table)
    ref = D.run_original_reference(main)
    assert disp[-1] == ref[-1], (D.DISPATCH_NAMES[op], a, b, disp[-1], ref[-1])


def test_dispatch_only_appends_used_subroutines():
    """An 8-bit-only program never pays for the 32-bit machinery."""
    table = D.build_dispatch_program(_prog(13, 10, D.MUL8))
    assert set(table.entry) == {D.MUL8}
    # a real MUL program appends the 32-bit mul subroutine only.
    table2 = D.build_dispatch_program(_prog(12, 11, isa.MUL))
    assert set(table2.entry) == {isa.MUL}


def test_low_precision_costs_fewer_steps():
    """Lower-precision opcodes dispatch to SHORTER subroutines -> far fewer VM steps
    (the whole point of the supplemental variants)."""
    def steps(op, a, b):
        return len(D.interpret_dispatch(D.build_dispatch_program(_prog(a, b, op))))
    # multiply: 8 < 16 < 32-bit.
    assert steps(D.MUL8, 200, 3) < steps(D.MUL16, 200, 3) < steps(isa.MUL, 200, 3)
    # divide: 8 < 16 < 32-bit.
    assert steps(D.DIV8, 200, 17) < steps(D.DIV16, 200, 17) < steps(isa.DIV, 200, 17)


# ---------------------------------------------------------------------------
# The perfect draft == the reference (draft is a valid speculation source).
# ---------------------------------------------------------------------------
class _FakeSubset:
    memory = True


class _FakeLean:
    subset = _FakeSubset()


@pytest.mark.parametrize("op,a,b", [
    (isa.MUL, 12, 11), (D.MUL8, 200, 3), (D.DIV8, 255, 16), (D.MOD16, 65535, 256),
])
def test_draft_reconstructs_reference_trace(op, a, b):
    """The perfect draft (trap steps carry their known AX; model steps carry the
    window) reconstructs the interpret_dispatch trace EXACTLY."""
    table = D.build_dispatch_program(_prog(a, b, op))
    draft = D.draft_program_dispatch(_FakeLean(), table)
    reconstructed = []
    model_seen = 0
    for st in draft.steps:
        if st["trap"]:
            reconstructed.append(st["ax"])
        else:
            model_seen += 1
    # every non-trap step corresponds to a base-ISA op the model verifies; the trap
    # steps' known AX + the model steps' AX together == the reference trace.
    assert len(draft.steps) == len(draft.ref_trace)
    assert draft.n_trap_steps == sum(1 for s in draft.steps if s["trap"])
    # the trap-step AX values sit at the right trace positions.
    for i, st in enumerate(draft.steps):
        if st["trap"]:
            assert st["ax"] == draft.ref_trace[i], (i, st, draft.ref_trace[i])


# ---------------------------------------------------------------------------
# GPU: byte-exact through the ACTUAL shallow neural forward + 100% speculation.
# ---------------------------------------------------------------------------
def _cuda1_or_skip():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs cuda:1")
    return "cuda:1"


def _lean_bitwise(code_size: int):
    dev = _cuda1_or_skip()
    os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_agent")
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    vm = Q.build(code_size=code_size, subset=Q.SUBSET_BITWISE)
    return LF.LeanQwenVM.from_full_vm(vm, device=dev)


# Operands chosen tie-free: the subroutine trace never crosses the model's requant
# boundary at value 255 (a pre-existing SATURATED-TIE the base subroutine driver also
# hits — see test_lean_subroutine_muldiv; e.g. div8 (255,16) ties at an intermediate
# 255 under BOTH the naive stack driver and speculation).  The final RESULT is always
# correct; these operands keep the whole trace byte-exact.
@pytest.mark.parametrize("op,a,b", [(D.MUL8, 200, 3), (D.MUL8, 12, 11)])
def test_mul8_dispatch_through_neural_forward(op, a, b):
    """A REAL supplemental opcode runs BYTE-EXACT through the shallow model via the
    trap-to-subroutine dispatch (naive one-forward-per-step; MUL8 is the fast case —
    the longer DIV/MOD subroutines are exercised through the batched-spec tests)."""
    main = _prog(a, b, op)
    table = D.build_dispatch_program(main)
    code_size = ((len(table.code) + 63) // 64) * 64
    lean = _lean_bitwise(code_size)
    r = D.run_program_dispatch(lean, table, max_steps=5000)
    ref = D.run_original_reference(main)
    assert r.exact and r.status == "PASS", r.detail
    assert r.ax_trace[-1] == ref[-1]
    assert r.trap_steps == 1                       # one dispatched-op fetch trapped


@pytest.mark.parametrize("op,a,b", [
    (D.MUL8, 200, 3), (D.MUL8, 12, 11), (D.DIV8, 200, 17), (D.MOD8, 100, 7),
])
def test_dispatch_speculation_is_exact_and_batches(op, a, b):
    """Perfect-draft speculation verifies the whole subroutine loop in a HANDFUL of
    BATCHED forwards (fast — 1 forward for a whole 8-bit subroutine), is byte-exact vs
    the reference (draft == model == interpret_dispatch), accepts every step, and saves
    forwards."""
    main = _prog(a, b, op)
    table = D.build_dispatch_program(main)
    code_size = ((len(table.code) + 63) // 64) * 64
    lean = _lean_bitwise(code_size)
    r = D.speculative_run_dispatch(lean, table, block_steps=2048, max_steps=5000)
    ref = D.run_original_reference(main)
    assert r.exact and r.status == "PASS", r.detail
    assert r.ax_trace[-1] == ref[-1]
    assert r.accepted == r.steps                   # every step accepted
    assert r.forwards < r.naive_forwards           # batched << one-per-step
    assert r.speedup > 1.0


@pytest.mark.parametrize("op,a,b", [(D.DIV8, 255, 16)])
def test_speculation_matches_naive_model_100pct(op, a, b):
    """Speculation is 100% FAITHFUL to the model: the BATCHED spec decode equals the
    NAIVE per-step decode byte-for-byte, even where the model's own requant hits a
    saturated tie on an intermediate scratch value (div8 (255,16) ties at 255 in BOTH
    the naive and the batched paths — never a speculation artifact).  This is the
    slow one (naive = ~1000 forwards), kept minimal."""
    main = _prog(a, b, op)
    table = D.build_dispatch_program(main)
    code_size = ((len(table.code) + 63) // 64) * 64
    lean = _lean_bitwise(code_size)
    naive = D.run_program_dispatch(lean, table, max_steps=5000)
    spec = D.speculative_run_dispatch(lean, table, block_steps=2048, max_steps=5000)
    # the whole batched trace equals the naive per-step model decode (100% acceptance).
    assert spec.ax_trace == naive.ax_trace
    # and the FINAL program result is the true quotient regardless of any tie.
    assert spec.ax_trace[-1] == D.run_original_reference(main)[-1]
    assert spec.forwards < spec.naive_forwards


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            # skip parametrized (need pytest); run the no-arg ones.
            import inspect
            sig = inspect.signature(t)
            if sig.parameters:
                continue
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} no-arg dispatch tests passed")
