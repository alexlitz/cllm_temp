"""Tests for the ``C4_DISABLE_BATCHED_ALU_RECOVERY`` toggle framework.

The Python ALU recovery overrides at
``neural_vm/batched_pure_neural.py:2149-2256`` rescue broken neural
binary-pop emission by calling ``_serial._compute_alu_legacy``. They mask
the vanilla-transformer thesis: the smoke ~46/51 baseline depends on
Python cheats, not neural compute.

This test suite verifies the incremental-removal framework. It does NOT
remove any cheats — that work is owned by per-op fix agents. It only
verifies that:

1. The env-var parsers correctly turn user input into the in-process
   disable-set.
2. ``_alu_recovery_disabled_for`` honors both the global flag and the
   per-op selector.
3. The collapsed-step recovery path actually consults the toggle: when
   an op is disabled, ``_compute_alu_legacy`` is not invoked and the
   raw (neural) AX is preserved.
4. The non-collapsed recovery path likewise consults the toggle.

These are unit-level: they stub the dispatch dependencies so no GPU /
model build is required.

See ``docs/SERIAL_MODE_DIVERGENCE_ATTRIBUTION_2026_06_09.md`` §7 for the
intended rollout protocol.
"""

from __future__ import annotations

import pytest

from neural_vm.batched_pure_neural import (
    BatchedPureNeuralRunner,
    _ElementState,
    _alu_recovery_disabled_for,
    _parse_disable_alu_recovery_ops,
)
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET


# ---------------------------------------------------------------------------
# 1. Env-var parsing
# ---------------------------------------------------------------------------


def test_parse_disable_ops_empty(monkeypatch):
    """Unset / empty env var yields an empty frozenset."""
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    assert _parse_disable_alu_recovery_ops() == frozenset()
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", "")
    assert _parse_disable_alu_recovery_ops() == frozenset()
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", "   ")
    assert _parse_disable_alu_recovery_ops() == frozenset()


def test_parse_disable_ops_well_formed(monkeypatch):
    """Comma-separated opcode names parse into the corresponding ints."""
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", "ADD,SUB,EQ")
    ops = _parse_disable_alu_recovery_ops()
    assert int(Opcode.ADD) in ops
    assert int(Opcode.SUB) in ops
    assert int(Opcode.EQ) in ops
    assert int(Opcode.MUL) not in ops


def test_parse_disable_ops_whitespace_and_case(monkeypatch):
    """Whitespace and lowercase entries normalize."""
    monkeypatch.setenv(
        "C4_DISABLE_BATCHED_ALU_RECOVERY_OPS",
        "  add , sub ,EQ  ",
    )
    ops = _parse_disable_alu_recovery_ops()
    assert int(Opcode.ADD) in ops
    assert int(Opcode.SUB) in ops
    assert int(Opcode.EQ) in ops


def test_parse_disable_ops_unknown_skipped(monkeypatch):
    """Unknown opcode names are silently skipped (don't crash startup)."""
    monkeypatch.setenv(
        "C4_DISABLE_BATCHED_ALU_RECOVERY_OPS",
        "ADD,NOT_A_REAL_OPCODE,SUB",
    )
    ops = _parse_disable_alu_recovery_ops()
    assert int(Opcode.ADD) in ops
    assert int(Opcode.SUB) in ops
    assert len(ops) == 2


# ---------------------------------------------------------------------------
# 2. _alu_recovery_disabled_for honors both env vars
# ---------------------------------------------------------------------------


def test_default_no_disable(monkeypatch):
    """With both env vars unset, nothing is disabled (cheat stays on)."""
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY", raising=False)
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    for op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.EQ, Opcode.SHL):
        assert not _alu_recovery_disabled_for(int(op))


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_global_disable_flag(monkeypatch, flag):
    """Truthy ``C4_DISABLE_BATCHED_ALU_RECOVERY`` disables every op."""
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY", flag)
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    for op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.EQ, Opcode.SHL):
        assert _alu_recovery_disabled_for(int(op))


def test_per_op_disable_only_listed(monkeypatch):
    """Per-op selector disables only the listed ops; others remain enabled."""
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY", raising=False)
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", "ADD,EQ")
    assert _alu_recovery_disabled_for(int(Opcode.ADD))
    assert _alu_recovery_disabled_for(int(Opcode.EQ))
    assert not _alu_recovery_disabled_for(int(Opcode.SUB))
    assert not _alu_recovery_disabled_for(int(Opcode.MUL))


# ---------------------------------------------------------------------------
# 3. Dispatch paths actually consult the toggle
# ---------------------------------------------------------------------------


class _StubSerial:
    """Minimal stand-in for ``AutoregressiveVMRunner`` used as
    ``runner._serial``. Records every ``_compute_alu_legacy`` call so the
    test can assert recovery was (or wasn't) invoked.
    """

    def __init__(self):
        self.calls = []

    def _compute_alu_legacy(self, op, stack_val, ax_val):
        self.calls.append((int(op), int(stack_val), int(ax_val)))
        # Return a sentinel obviously different from any neural emit so
        # the test can tell which path produced the final AX.
        return 0xDEADBEEF


def _make_runner_with_stub() -> BatchedPureNeuralRunner:
    """Bypass __init__ (which builds a GPU model) and inject a stub serial."""
    runner = BatchedPureNeuralRunner.__new__(BatchedPureNeuralRunner)
    runner._serial = _StubSerial()
    return runner


def _append_marker(ctx: list, marker: int, value: int) -> None:
    """Append ``marker`` followed by 4 little-endian bytes of ``value``."""
    ctx.append(int(marker))
    for j in range(4):
        ctx.append((int(value) >> (j * 8)) & 0xFF)


def _build_step_context(reg_ax: int, reg_pc: int) -> list:
    """Build a minimal context with REG_PC and REG_AX blocks.

    ``_extract_register`` scans backward from the end for each marker, so
    appending PC then AX near the end is sufficient. Empty padding at the
    front matches no marker.
    """
    ctx: list = [0] * 5
    _append_marker(ctx, Token.REG_PC, reg_pc)
    _append_marker(ctx, Token.REG_AX, reg_ax)
    # Trailing padding so the AX read isn't right at the boundary.
    ctx.extend([0, 0, 0, 0])
    return ctx


def _collapsed_state(
    skipped_op: int,
    *,
    neural_ax: int = 0x11111111,
    stack_val: int = 5,
    imm_val: int = 7,
) -> _ElementState:
    """``IMM imm_val; <skipped_op>; EXIT`` shape, ready for the
    collapsed-step recovery probe.

    Bytecode layout: IMM at idx 0, skipped binary-pop at idx 1, EXIT at idx 2.

    Dispatch entry path: ``s.last_pc is None`` so ``exec_pc() = PC_OFFSET``,
    ``exec_idx = 0`` -> ``exec_op = IMM``. The REG_PC token we plant in
    ``context`` carries the post-collapse PC (``idx 2 * INSTR_WIDTH +
    PC_OFFSET``), which the dispatch reads back into ``s.last_pc``. The
    recovery branch then sees ``post_idx = 2`` and ``skipped_idx = 1``
    (== ``exec_idx + 1``), satisfying ``post_idx == skipped_idx + 1``.
    """
    bytecode = [(imm_val << 8) | int(Opcode.IMM), int(skipped_op), int(Opcode.EXIT)]
    post_pc = 2 * INSTR_WIDTH + PC_OFFSET
    s = _ElementState(
        bytecode=bytecode,
        context=_build_step_context(neural_ax, post_pc),
        prefix_len=0,
    )
    # last_pc unset -> exec_pc() returns PC_OFFSET, exec_idx 0 = IMM.
    s.last_pc = None
    s.last_ax = neural_ax
    s.last_pushed_value = stack_val
    s.token_pos = 1
    return s


def _read_ax_from_context(ctx: list) -> int:
    """Re-extract REG_AX value from ``ctx`` (mirrors ``_extract_register``)."""
    scan_back = Token.STEP_TOKENS + 5
    for i in range(len(ctx) - 1, max(0, len(ctx) - scan_back), -1):
        if ctx[i] == Token.REG_AX and i + 4 < len(ctx):
            val = 0
            for j in range(4):
                val |= (ctx[i + 1 + j] & 0xFF) << (j * 8)
            return val
    return -1


def test_collapsed_recovery_runs_by_default(monkeypatch):
    """Default (no env vars): collapsed-step recovery calls
    ``_compute_alu_legacy`` and rewrites AX."""
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY", raising=False)
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    runner = _make_runner_with_stub()
    s = _collapsed_state(int(Opcode.SUB))
    runner._dispatch_pure_neural(s)
    assert any(call[0] == int(Opcode.SUB) for call in runner._serial.calls), (
        "default config should still invoke the legacy ALU cheat for SUB"
    )
    assert _read_ax_from_context(s.context) == 0xDEADBEEF, (
        "default config should override AX with the legacy ALU result"
    )


def test_collapsed_recovery_skipped_when_global_disabled(monkeypatch):
    """Global toggle: collapsed-step recovery NOT invoked.

    Note: exec_op == IMM here, so the unconditional IMM AX rewrite
    (``batched_pure_neural.py:2121-2129``) ALSO runs and replaces REG_AX
    with the bytecode immediate (``imm_val``) — that override is a
    separate cheat outside the ALU-recovery toggle's scope. The thing
    this test guards is the ALU recovery itself: the stub must not see
    a ``_compute_alu_legacy(SUB, ...)`` call, and REG_AX must NOT be
    the sentinel 0xDEADBEEF.
    """
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY", "1")
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    runner = _make_runner_with_stub()
    imm_val = 7
    s = _collapsed_state(int(Opcode.SUB), neural_ax=0x11111111, imm_val=imm_val)
    runner._dispatch_pure_neural(s)
    assert all(call[0] != int(Opcode.SUB) for call in runner._serial.calls), (
        "global disable must skip the SUB ALU cheat"
    )
    ax_after = _read_ax_from_context(s.context)
    assert ax_after != 0xDEADBEEF, (
        f"global disable should not override AX via _compute_alu_legacy, "
        f"got AX=0x{ax_after:x}"
    )


def test_collapsed_recovery_skipped_for_listed_op_only(monkeypatch):
    """Per-op selector: listed op skipped, unlisted op still cheated."""
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY", raising=False)
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", "SUB")
    runner = _make_runner_with_stub()

    # SUB: disabled -> recovery skipped, no SUB call in legacy ALU.
    s_sub = _collapsed_state(int(Opcode.SUB), neural_ax=0xAAAA0000)
    runner._dispatch_pure_neural(s_sub)
    assert all(call[0] != int(Opcode.SUB) for call in runner._serial.calls)
    assert _read_ax_from_context(s_sub.context) != 0xDEADBEEF

    # MUL: not listed -> recovery still runs.
    runner._serial.calls.clear()
    s_mul = _collapsed_state(int(Opcode.MUL), neural_ax=0xBBBB0000)
    runner._dispatch_pure_neural(s_mul)
    assert any(call[0] == int(Opcode.MUL) for call in runner._serial.calls)
    assert _read_ax_from_context(s_mul.context) == 0xDEADBEEF


# ---------------------------------------------------------------------------
# Non-collapsed (binary-op-as-its-own-step) recovery path
# ---------------------------------------------------------------------------


def _non_collapsed_state(
    binop: int,
    *,
    prev_ax: int = 0x33,
    stack_val: int = 0x55,
    neural_ax: int = 0x12345678,
) -> _ElementState:
    """Build a state where the JUST-executed op is the binary-pop ``binop``
    itself (its own step, not collapsed).

    ``s.last_pc is None`` -> ``exec_pc() = PC_OFFSET`` -> ``exec_idx = 0``
    -> ``exec_op = binop`` (placed at bytecode[0]). The non-collapsed
    recovery branch only needs ``exec_op``, ``s.last_pushed_value``, and
    ``prev_ax`` (the pre-overwrite ``s.last_ax``), not ``last_pc``.
    """
    bytecode = [int(binop), int(Opcode.EXIT)]
    # No REG_PC marker -> _extract_register returns None -> s.last_pc stays None.
    s = _ElementState(
        bytecode=bytecode,
        context=_build_step_context(neural_ax, 0),
        prefix_len=0,
    )
    s.last_pc = None
    # Seed s.last_ax with prev_ax; the dispatch snapshots it into the local
    # ``prev_ax`` before the REG_AX extract overwrites s.last_ax.
    s.last_ax = prev_ax
    s.last_pushed_value = stack_val
    s.token_pos = 1
    return s


def test_non_collapsed_recovery_runs_by_default(monkeypatch):
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY", raising=False)
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    runner = _make_runner_with_stub()
    s = _non_collapsed_state(int(Opcode.ADD))
    runner._dispatch_pure_neural(s)
    assert any(call[0] == int(Opcode.ADD) for call in runner._serial.calls), (
        "default config: ADD-as-own-step should hit the recovery"
    )


def test_non_collapsed_recovery_skipped_for_listed_op(monkeypatch):
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY", raising=False)
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", "ADD")
    runner = _make_runner_with_stub()
    s_add = _non_collapsed_state(int(Opcode.ADD))
    runner._dispatch_pure_neural(s_add)
    assert all(call[0] != int(Opcode.ADD) for call in runner._serial.calls), (
        "ADD listed in disable set must skip non-collapsed recovery"
    )

    # XOR not listed: cheat still runs.
    runner._serial.calls.clear()
    s_xor = _non_collapsed_state(int(Opcode.XOR))
    runner._dispatch_pure_neural(s_xor)
    assert any(call[0] == int(Opcode.XOR) for call in runner._serial.calls)


def test_non_collapsed_recovery_skipped_when_global_disabled(monkeypatch):
    monkeypatch.setenv("C4_DISABLE_BATCHED_ALU_RECOVERY", "1")
    monkeypatch.delenv("C4_DISABLE_BATCHED_ALU_RECOVERY_OPS", raising=False)
    runner = _make_runner_with_stub()
    s = _non_collapsed_state(int(Opcode.XOR))
    runner._dispatch_pure_neural(s)
    assert all(call[0] != int(Opcode.XOR) for call in runner._serial.calls), (
        "global disable must skip non-collapsed recovery too"
    )
