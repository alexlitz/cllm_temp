"""Smoke tests for the `pure_neural_runner` fixture.

This file verifies the *fixture itself* — not any deep neural-VM behavior.
It exists because many sub-agents have wasted effort on worktrees whose
branch lacked either:
  * the `pure_neural_runner` fixture in `conftest.py`, or
  * the `pure_neural` / `trust_neural_alu` kwargs on `AutoregressiveVMRunner`.

These tests catch both problems at the fastest possible level: instantiation
plus two minimal programs (IMM 5 and ADD 3+4).

Each test calls `pytest.skip(...)` rather than failing if the runner had to
fall back to a non-pure-neural mode (older worktree compatibility). That way
the smoke test reports a *clear, actionable* skip reason instead of a deep
opaque assertion failure.
"""

import pytest

from neural_vm.embedding import Opcode


def _make_bc(prog):
    """Encode a list of (opcode, imm) tuples or bare opcodes into 32-bit words."""
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run(runner, prog, max_steps=30):
    """Run a tiny program and return the EXIT result (or AX equivalent)."""
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, result = runner.run(bc, b"", max_steps=max_steps)
    return result


def _require_pure_neural(runner):
    """Skip the test if the fixture fell back to a non-pure-neural runner."""
    if not getattr(runner, "pure_neural", False):
        pytest.skip(
            "pure_neural_runner fixture fell back to a non-pure-neural runner — "
            "this worktree's AutoregressiveVMRunner does not support pure_neural=True. "
            "See c4_release/docs/AGENT_CONTEXT.md."
        )


def test_fixture_instantiates(pure_neural_runner):
    """The fixture must produce *some* runner with a model attached."""
    assert pure_neural_runner is not None
    assert hasattr(pure_neural_runner, "model")
    assert hasattr(pure_neural_runner, "run")


def test_fixture_resets_memory_state(pure_neural_runner):
    """Per-test state must be empty when the fixture is requested."""
    assert pure_neural_runner._memory == {}
    assert pure_neural_runner._mem_history == {}
    assert pure_neural_runner._mem_access_order == []


def test_fixture_has_no_python_handlers(pure_neural_runner):
    """Session builder strips Python-side handlers for full neural dispatch."""
    assert pure_neural_runner._func_call_handlers == {}
    assert pure_neural_runner._syscall_handlers == {}


def test_smoke_imm_5(pure_neural_runner):
    """IMM 5 → 5 — the canonical smallest pure-neural program."""
    _require_pure_neural(pure_neural_runner)
    result = _run(pure_neural_runner, [(Opcode.IMM, 5), Opcode.EXIT])
    assert result == 5, f"IMM 5 → expected 5, got {result!r}"


def test_smoke_add_3_plus_4(pure_neural_runner):
    """ADD 3+4 → 7 — confirms ALU works in pure-neural + efficient mode."""
    _require_pure_neural(pure_neural_runner)
    # Standard C4 calling convention for binary ALU:
    #   IMM 3; PSH; IMM 4; ADD; EXIT
    # PSH pushes AX onto the stack, then ADD pops it and adds to AX.
    result = _run(pure_neural_runner, [
        (Opcode.IMM, 3),
        Opcode.PSH,
        (Opcode.IMM, 4),
        Opcode.ADD,
        Opcode.EXIT,
    ])
    assert result == 7, f"ADD 3+4 → expected 7, got {result!r}"
