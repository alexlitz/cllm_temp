"""Speculative-decoding correctness tests.

Verifies that the speculative `UltraBatchRunner` produces results that are
byte-identical to the autoregressive runner across multiple values of
`n_speculative_steps`.

Why this matters: multi-step speculation lets DraftVM run N instructions
before the transformer validates, amortizing transformer forward-pass cost
across N steps. Correctness must not depend on N because c4 semantics are
deterministic: rejection on mismatch is well-defined and DraftVM is the
authoritative reference VM.

Tests pin two trivial programs (smoke 2/2 parity) and assert that for every
N in {1, 2, 4, 8, 16} the exit code matches the autoregressive runner.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.embedding import Opcode


def _make_bc(ops):
    """Encode a list of (opcode, imm) tuples or bare opcodes into 32-bit words."""
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


# Two canonical programs from the smoke 2/2 suite.
PROGRAMS = [
    pytest.param(
        _make_bc([(Opcode.IMM, 5), Opcode.EXIT]),
        5,
        id="imm5_exit",
    ),
    pytest.param(
        _make_bc([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 32), Opcode.ADD,
            Opcode.EXIT,
        ]),
        42,
        id="add_42",
    ),
]


N_VALUES = [1, 2, 4, 8, 16]


@pytest.fixture(scope="module")
def autoregressive_results():
    """Pre-compute autoregressive exit codes once per test session.

    Avoids paying the multi-second model-bake cost for every parameterized
    test case.
    """
    from neural_vm.run_vm import AutoregressiveVMRunner

    runner = AutoregressiveVMRunner()
    results = {}
    for bytecode, _expected in [(p.values[0], p.values[1]) for p in PROGRAMS]:
        key = tuple(bytecode)
        _, exit_code = runner.run(bytecode, b"", max_steps=20)
        results[key] = exit_code
    return results


@pytest.fixture(scope="module")
def ultra_runners():
    """Compile one UltraBatchRunner per N value at module scope.

    Sharing module-scope avoids paying the compile cost (~30 s) per test.
    """
    from neural_vm.batch_runner_v2 import UltraBatchRunner

    runners = {}
    for N in N_VALUES:
        runners[N] = UltraBatchRunner(
            batch_size=2, device="cuda", n_speculative_steps=N
        )
    return runners


@pytest.mark.parametrize("bytecode, expected_exit", PROGRAMS)
@pytest.mark.parametrize("N", N_VALUES)
def test_speculative_byte_identity(
    bytecode, expected_exit, N, autoregressive_results, ultra_runners
):
    """For every N in {1,2,4,8,16}, speculative exit code matches autoregressive.

    This is the core byte-identity contract: changing N is a pure performance
    knob and must not alter program semantics. With c4's deterministic
    DraftVM as the reference, rejection at any token only wastes work — the
    final exit code remains DraftVM's deterministic answer, which is also
    the autoregressive answer when the trained transformer is correct.
    """
    runner = ultra_runners[N]
    results = runner.run_batch([bytecode], max_steps=20)

    autoregressive_exit = autoregressive_results[tuple(bytecode)]

    assert results[0] == expected_exit, (
        f"N={N} produced exit={results[0]}, expected {expected_exit}"
    )
    assert results[0] == autoregressive_exit, (
        f"N={N} produced exit={results[0]}, autoregressive produced "
        f"{autoregressive_exit}. Byte-identity broken."
    )


def test_speculation_stats_recorded(ultra_runners):
    """Multi-step runner exposes acceptance statistics via get_speculation_stats()."""
    runner = ultra_runners[4]
    bytecode = _make_bc([(Opcode.IMM, 5), Opcode.EXIT])
    runner.run_batch([bytecode], max_steps=10)

    stats = runner.get_speculation_stats()
    assert "n_speculative_steps" in stats
    assert "total_proposed_steps" in stats
    assert "total_accepted_steps" in stats
    assert "acceptance_rate" in stats
    assert stats["n_speculative_steps"] == 4
    assert stats["total_proposed_steps"] >= 1


def test_default_n_speculative_steps_is_one():
    """Default behavior must be N=1 (legacy parity)."""
    from neural_vm.batch_runner_v2 import UltraBatchRunner

    runner = UltraBatchRunner(batch_size=1, device="cuda")
    assert runner.n_speculative_steps == 1, (
        "UltraBatchRunner default n_speculative_steps must be 1 for "
        "legacy parity. Found: " + str(runner.n_speculative_steps)
    )
