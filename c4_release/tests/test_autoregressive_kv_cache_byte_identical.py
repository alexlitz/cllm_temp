"""Byte-identical verification tests for AutoregressiveVMRunner KV cache.

The KV cache reuses K/V tensors across forward calls for the
AutoregressiveVMRunner (pure_neural mode). It MUST produce byte-identical
output (exit code + stdout) to the non-cached path for every test program.

These tests run the same programs twice — once with ``use_kv_cache=True`` and
once with ``use_kv_cache=False`` — and assert that both the exit code and the
captured ``output`` string are exactly equal.

A periodic cache flush (every ``_KV_INCREMENTAL_FLUSH`` incremental calls)
bounds float-non-associativity drift in the L15 memory-lookup attention layer;
this is what makes byte-identity possible despite the cached path using a
chain of single-token matmuls instead of one batch matmul.
"""

import pytest

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run_pair(prog, max_steps=30):
    """Run ``prog`` twice — cache ON and cache OFF — and return both."""
    bc = _make_bc(prog)
    on = AutoregressiveVMRunner(
        pure_neural=True, conversational_io=False, use_kv_cache=True
    )
    off = AutoregressiveVMRunner(
        pure_neural=True, conversational_io=False, use_kv_cache=False
    )
    out_on, exit_on = on.run(bc, b"", max_steps=max_steps)
    out_off, exit_off = off.run(bc, b"", max_steps=max_steps)
    return (out_on, exit_on), (out_off, exit_off)


class TestByteIdenticalCacheOnVsOff:
    """KV cache must be transparent: byte-identical output to non-cached path."""

    @pytest.mark.parametrize(
        "prog,expected",
        [
            ([(Opcode.IMM, 5), Opcode.EXIT], 5),
            ([(Opcode.IMM, 0), Opcode.EXIT], 0),
            ([(Opcode.IMM, 42), Opcode.EXIT], 42),
            ([(Opcode.IMM, 100), Opcode.EXIT], 100),
            ([(Opcode.IMM, 200), Opcode.EXIT], 200),
        ],
    )
    def test_single_imm(self, prog, expected):
        (out_on, exit_on), (out_off, exit_off) = _run_pair(prog)
        assert exit_on == exit_off, (
            f"exit code mismatch: ON={exit_on} OFF={exit_off}"
        )
        assert out_on == out_off, (
            f"output mismatch: ON={out_on!r} OFF={out_off!r}"
        )
        assert exit_off == expected, (
            f"baseline (cache OFF) returned {exit_off}, expected {expected}"
        )

    def test_two_imms(self):
        # IMM 5; IMM 7; EXIT — second IMM should overwrite AX = 7.
        prog = [(Opcode.IMM, 5), (Opcode.IMM, 7), Opcode.EXIT]
        (out_on, exit_on), (out_off, exit_off) = _run_pair(prog)
        assert exit_on == exit_off
        assert out_on == out_off

    def test_three_imms(self):
        prog = [
            (Opcode.IMM, 5),
            (Opcode.IMM, 7),
            (Opcode.IMM, 9),
            Opcode.EXIT,
        ]
        (out_on, exit_on), (out_off, exit_off) = _run_pair(prog)
        assert exit_on == exit_off
        assert out_on == out_off

    def test_imm_then_nop_then_exit(self):
        prog = [(Opcode.IMM, 42), Opcode.NOP, Opcode.EXIT]
        (out_on, exit_on), (out_off, exit_off) = _run_pair(prog)
        assert exit_on == exit_off
        assert out_on == out_off

    def test_nop_then_imm(self):
        prog = [Opcode.NOP, (Opcode.IMM, 42), Opcode.EXIT]
        (out_on, exit_on), (out_off, exit_off) = _run_pair(prog)
        assert exit_on == exit_off
        assert out_on == out_off


class TestCacheStateLifecycle:
    """Cache must be reset cleanly between ``run()`` calls of the same runner."""

    def test_reused_runner_two_programs(self):
        """A single runner with cache enabled must produce correct results
        across multiple ``run()`` calls on different programs."""
        runner = AutoregressiveVMRunner(
            pure_neural=True, conversational_io=False, use_kv_cache=True
        )

        # First program: IMM 5
        _, exit1 = runner.run(_make_bc([(Opcode.IMM, 5), Opcode.EXIT]), b"", max_steps=10)
        # Second program: IMM 9 (different)
        _, exit2 = runner.run(_make_bc([(Opcode.IMM, 9), Opcode.EXIT]), b"", max_steps=10)

        # Compare to fresh-runner baselines
        baseline = AutoregressiveVMRunner(
            pure_neural=True, conversational_io=False, use_kv_cache=False
        )
        _, ref1 = baseline.run(_make_bc([(Opcode.IMM, 5), Opcode.EXIT]), b"", max_steps=10)
        _, ref2 = baseline.run(_make_bc([(Opcode.IMM, 9), Opcode.EXIT]), b"", max_steps=10)

        assert exit1 == ref1, f"first run: cache={exit1}, no-cache={ref1}"
        assert exit2 == ref2, f"second run: cache={exit2}, no-cache={ref2}"

    def test_run_resets_kv_state(self):
        """Each ``run()`` call must reset cached tokens to empty so a new
        program's prefix isn't confused with the previous run's tokens."""
        runner = AutoregressiveVMRunner(
            pure_neural=True, conversational_io=False, use_kv_cache=True
        )

        # First run
        runner.run(_make_bc([(Opcode.IMM, 5), Opcode.EXIT]), b"", max_steps=10)

        # After run, cached_tokens should be empty / reset for next run
        # (the inner loop populates the cache, but the post-run reset must clear).
        # We just check that a subsequent run with a DIFFERENT first opcode does
        # not have stale cached prefix from the previous run.
        _, exit_code = runner.run(
            _make_bc([(Opcode.IMM, 7), Opcode.EXIT]), b"", max_steps=10
        )
        baseline = AutoregressiveVMRunner(
            pure_neural=True, conversational_io=False, use_kv_cache=False
        )
        _, ref = baseline.run(
            _make_bc([(Opcode.IMM, 7), Opcode.EXIT]), b"", max_steps=10
        )
        assert exit_code == ref
