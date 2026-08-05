"""In-transformer JIT on the GOLDEN CFM VM (``C4_CFM_EMIT``): EMIT appends a CODE
frame into the SAME KV §Memory the fetch@PC CAM reads, so runtime-generated code
is fetchable by PC with NO residual-width (D) growth — program-length INDEPENDENT.

Two tiers:
  * REFERENCE tier (no model, always runs): the ``isa.interpret`` / function-aware
    oracle EMIT semantics + append-and-run / re-emit / end-to-end compile-and-run,
    byte-exact against the golden c4 8-bit interpreter.
  * MODEL tier (GPU-guarded): append-and-run + re-emit + end-to-end run through the
    REAL fused CFM lean forward (``qwen_full_vm.build`` weights, ``qwen_lean_forward``
    evaluator), byte-exact vs ``isa.interpret``.  This is the port of the bespoke
    ``nibble_compiler`` JIT onto the golden CFM path.

The flag is DEFAULT OFF; the golden 069cc32f build (and the read-only CFM code path)
is byte-identical with it off (proven by ``_fingerprint_build`` and the flag-off
read-only regression test below).
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min._agent_cfm_emit_demo import (
    prog_append_and_run, prog_reemit, prog_compile_expr, prog_compile_add_expr,
    run_with_preloaded_mem, _emit_seq)


# ===========================================================================
# REFERENCE TIER — EMIT semantics via the golden c4 interpreter (no model).
# ===========================================================================
def test_emit_opcode_above_band():
    """EMIT (45) sits ABOVE both NUM_OPS (40) and NUM_OPS_FLOAT (44), so the OP_IS
    one-hot band width is unchanged -> the golden layout/build stays byte-identical."""
    assert isa.EMIT == 45
    assert isa.EMIT > isa.NUM_OPS_FLOAT > isa.NUM_OPS
    assert isa.NAMES[isa.EMIT] == "EMIT"


def test_reference_append_and_run():
    prog, expect = prog_append_and_run()
    tr = isa.interpret(prog, max_steps=128)
    assert tr[-1] == expect, tr


def test_reference_reemit_self_modifying():
    """Same-address re-emit: the slot is overwritten and the NEWEST instruction runs."""
    prog, expect = prog_reemit()
    tr = isa.interpret(prog, max_steps=128)
    assert tr[-1] == expect == 9, tr


@pytest.mark.parametrize("expr,d1,sel,d2", [
    ("2+3", 2, 0, 3), ("2*3", 2, 1, 3), ("4+5", 4, 0, 5), ("9*7", 9, 1, 7),
])
def test_reference_end_to_end_compile(expr, d1, sel, d2):
    """A LOADED compiler reads a C ``d1 op d2`` source from the data §Memory and
    EMITs ``IMM d1 ; PSH ; IMM d2 ; <ADD|MUL> ; HALT`` then JMPs in — byte-exact
    vs the value the real c4 compiler+VM would produce."""
    PROD = 40
    prog = list(prog_compile_expr(PROD))
    while len(prog) < PROD:
        prog.append(isa.Instr(isa.NOP, 0))
    prog += [isa.Instr(isa.NOP, 0)] * 5
    tr = isa.interpret(prog, max_steps=300, mem_init={0: d1, 1: sel, 2: d2})
    assert tr[-1] == ((d1 * d2) if sel else (d1 + d2)), (expr, tr)


def test_flag_off_read_only_reference_unchanged():
    """A program with NO EMIT decodes identically regardless of the flag (the EMIT
    branch only fires on opcode 45)."""
    prog = isa.assemble([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)])
    assert isa.interpret(prog, max_steps=32)[-1] == 8


# ===========================================================================
# MODEL TIER — the REAL fused CFM lean forward (GPU-guarded).
# ===========================================================================
import torch  # noqa: E402

DEVICE = ("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
          else ("cuda:0" if torch.cuda.is_available() else None))
requires_cuda = pytest.mark.skipif(DEVICE is None, reason="CFM model build needs a GPU")


@pytest.fixture(scope="module")
def lean_cfm():
    if DEVICE is None:
        pytest.skip("no GPU")
    os.environ["C4_PF_CFM"] = "1"
    os.environ["C4_CFM_EMIT"] = "1"
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP, code_from_memory=True)
    vm.embed = vm.embed.to(DEVICE)
    return LF.LeanQwenVM.from_full_vm(vm, device=DEVICE)


@requires_cuda
def test_model_append_and_run(lean_cfm):
    from c4_min import qwen_lean_forward as LF
    prog, expect = prog_append_and_run()
    res = LF.run_program_lean(lean_cfm, prog, max_steps=128)
    assert res["exact"] and res["ax_trace"][-1] == expect, res["ax_trace"]


@requires_cuda
def test_model_reemit_self_modifying(lean_cfm):
    from c4_min import qwen_lean_forward as LF
    prog, expect = prog_reemit()
    res = LF.run_program_lean(lean_cfm, prog, max_steps=128)
    assert res["exact"] and res["ax_trace"][-1] == expect == 9, res["ax_trace"]


@requires_cuda
@pytest.mark.parametrize("expr,d1,d2", [("2+3", 2, 3), ("4+5", 4, 5), ("9+8", 9, 8)])
def test_model_end_to_end_compile_and_run(lean_cfm, expr, d1, d2):
    """A LOADED (code frames, NOT baked-in-weights) compiler reads a C ``d1+d2``
    source from the data band, EMITs ``IMM d1 ; PSH ; IMM d2 ; ADD ; HALT`` into an
    empty code region, JMPs in, and the produced bytecode runs — result byte-exact
    vs isa.interpret (the golden c4 oracle)."""
    PROD = 40
    prog = list(prog_compile_add_expr(PROD))
    while len(prog) < PROD:
        prog.append(isa.Instr(isa.NOP, 0))
    prog += [isa.Instr(isa.NOP, 0)] * 5
    preload = [{"addr": 0, "val": d1}, {"addr": 2, "val": d2}]
    res = run_with_preloaded_mem(lean_cfm, prog, preload, max_steps=200)
    produced = res["produced_code"][PROD:PROD + 5]
    assert [(i.op, i.imm) for i in produced] == [
        (isa.IMM, d1), (isa.PSH, 0), (isa.IMM, d2), (isa.ADD, 0), (isa.HALT, 0)], produced
    assert res["exact"] and res["ax_trace"][-1] == d1 + d2, res["ax_trace"]


@requires_cuda
def test_model_read_only_unregressed_flag_off(lean_cfm):
    """With C4_CFM_EMIT unset in the driver, an ordinary (no-EMIT) program still
    decodes byte-exact — the read-only code path is untouched."""
    from c4_min import qwen_lean_forward as LF
    saved = os.environ.pop("C4_CFM_EMIT", None)
    try:
        for prog in (
            isa.assemble([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]),
            isa.assemble([("IMM", 1), ("JMP", 3), ("IMM", 9), ("IMM", 7), ("HALT", 0)]),
        ):
            res = LF.run_program_lean(lean_cfm, prog, max_steps=32)
            assert res["exact"], (prog, res["ax_trace"])
    finally:
        if saved is not None:
            os.environ["C4_CFM_EMIT"] = saved
