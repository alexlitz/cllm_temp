"""Smoke tests for the program-level DSL spec-conformance gate.

These lock the gate's CLASSIFICATION CONTRACT (PASS / AUTHORING-MISMATCH /
UNINTERPRETABLE) on a handful of programs whose verdicts are structurally
determined:

* ``IMM 42; EXIT`` — pure declarative path, expected nibble reachable ->
  PASS (coverage).
* ``IMM 6; PSH; IMM 7; MUL; EXIT`` — MUL routes through the opaque
  ``layer10_divmod`` install op -> UNINTERPRETABLE.
* ``IMM 84; PSH; IMM 2; DIV; EXIT`` — DIV routes through the opaque
  ``l10_alu_divmod_*`` ops -> UNINTERPRETABLE.

The gate compiles the declarations-only layout once (~8s, CPU-only); these
tests share that build via a module-scoped fixture so the suite stays well
under a minute. No GPU, no disk cache, no model weights.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")

from neural_vm.unified_compiler.symbolic_forward import (  # noqa: E402
    OP_DIV,
    OP_EXIT,
    OP_IMM,
    OP_MUL,
    OP_PSH,
    encode_instr,
)
from tools import dsl_spec_gate as gate  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return gate.build_gate_context(verbose=False)


def _prog(*pairs):
    return [encode_instr(op, imm) for op, imm in pairs]


def test_gate_context_builds(ctx):
    # Layout has the expected shape: most ops carry IR, a minority opaque.
    n = len(ctx.op_infos)
    n_opaque = len(ctx.opaque_ops())
    assert n > 100
    assert 0 < n_opaque < n
    # The DIV/MOD/MUL ALU install ops are opaque (imperative bake).
    opaque_names = {oi.name for oi in ctx.opaque_ops()}
    assert "layer10_divmod" in opaque_names


def test_imm_exit_is_pass(ctx):
    prog = _prog((OP_IMM, 42), (OP_EXIT, 0))
    r = gate.classify_program(ctx, "imm_exit", prog)
    assert r.classification == gate.PASS
    assert (r.oracle_exit or 0) & 0xFF == 42


def test_mul_is_uninterpretable(ctx):
    prog = _prog((OP_IMM, 6), (OP_PSH, 0), (OP_IMM, 7), (OP_MUL, 0),
                 (OP_EXIT, 0))
    r = gate.classify_program(ctx, "mul", prog)
    assert r.classification == gate.UNINTERPRETABLE
    assert any("OP_MUL" in o for o in r.opaque_on_path)


def test_div_is_uninterpretable(ctx):
    prog = _prog((OP_IMM, 84), (OP_PSH, 0), (OP_IMM, 2), (OP_DIV, 0),
                 (OP_EXIT, 0))
    r = gate.classify_program(ctx, "div", prog)
    assert r.classification == gate.UNINTERPRETABLE
    assert any("OP_DIV" in o for o in r.opaque_on_path)


def test_branch_program_classifies(ctx):
    # JMP over a dead IMM, then IMM 42; EXIT. The dim_oracle ReferenceOracle
    # loops on this; the gate uses the real oracle trace, so it must classify
    # cleanly (regression guard for the branch fix).
    from neural_vm.unified_compiler.symbolic_forward import OP_JMP
    prog = _prog((OP_JMP, 2), (OP_IMM, 99), (OP_IMM, 42), (OP_EXIT, 0))
    r = gate.classify_program(ctx, "jmp", prog)
    assert r.classification in (gate.PASS, gate.AUTHORING_MISMATCH)
    assert (r.oracle_exit or 0) & 0xFF == 42
