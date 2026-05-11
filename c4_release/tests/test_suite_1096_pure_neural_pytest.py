#!/usr/bin/env python3
"""
1096 Tests - Pure Neural Mode (Parallel Suite)

Pure-neural mirror of tests/test_suite_1096_pytest.py. Runs the same 1096
comprehensive test programs through AutoregressiveVMRunner(pure_neural=True,
trust_neural_alu=True) — the neural network alone drives execution, no
Python overrides.

Per F's Phase 8 scoping doc (docs/PHASE_8_RUNNER_SWITCH_SCOPE.md):
    Option B (Parallel) — keep handler-mode `test_suite_1096_pytest.py`
    unchanged. This parallel suite grows pass-rate as Phases 1-7 land.
    All 1096 tests start marked xfail; as phases complete, the xfail
    decorator gets removed from confirmed-passing subsets.

Realistic target per F's scope (section 3):
    "200-400/1096 passing in pure-neural mode by end of Phase 7. Residual
    failures become Phase 8+ backlog."

Compiled C programs in the 1096 suite typically exercise:
    - PRTF varargs (Phase 6 — blocked)
    - Recursion / deep call stacks (Phase 5 — partially blocked)
    - Malloc/free programs (Phase 7 — blocked)
    - Multi-byte arithmetic (Phase 3 — blocked)
So nearly every test is xfail today; only a small subset of pure
return-constant or pure single-byte arithmetic programs will XPASS.

Usage:
    # Quick check (don't run all 1096 — they're xfail and slow)
    pytest tests/test_suite_1096_pure_neural_pytest.py -v -k "test_program_0" --tb=line

    # Run a single test by ID (to see what XPASSes)
    pytest tests/test_suite_1096_pure_neural_pytest.py -k "add_" --runxfail

    # Verify suite collection (fast, no execution)
    pytest tests/test_suite_1096_pure_neural_pytest.py --collect-only

Date: 2026-05-11
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner


# Generate all test programs once at module load
ALL_TESTS = generate_test_programs()


def make_test_id(test_tuple):
    """Create a readable test ID from test tuple."""
    source, expected, description = test_tuple
    desc = description.replace(" ", "_").replace(":", "")
    return desc[:50]


TEST_IDS = [make_test_id(t) for t in ALL_TESTS]


class TestSuite1096PureNeural:
    """Run all 1096 tests in pure-neural mode (parallel to TestSuite1096).

    Every test starts marked xfail (non-strict) because nearly all compiled
    C programs in the suite exercise opcodes blocked by Phases 2-7. As each
    phase lands, the matching subset of tests gets unmarked. Tests that
    XPASS (i.e. xfail=non-strict and actually pass) are visible in pytest
    output as XPASS — those become candidates for promotion to expected-PASS.

    Class fixture builds a session-scoped pure-neural runner once;
    function-scoped state reset happens inline in test_program.
    """

    @pytest.fixture(scope="class")
    def runner(self):
        """Class-scoped pure-neural runner (no Python overrides).

        Builds AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
        once per class. Strips _func_call_handlers + _syscall_handlers so
        dispatch is fully neural.
        """
        r = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
        r._func_call_handlers = {}
        r._syscall_handlers = {}
        return r

    @pytest.mark.xfail(
        reason="Phase 8 baseline: every 1096 test starts xfail in "
               "pure_neural mode until Phases 1-7 individual subsets are "
               "verified. Per docs/PHASE_8_RUNNER_SWITCH_SCOPE.md the "
               "realistic Phase-7-complete target is 200-400/1096 PASS. "
               "Tests that XPASS are candidates for unmarking.",
        strict=False,
    )
    @pytest.mark.parametrize(
        "source,expected,description",
        ALL_TESTS,
        ids=TEST_IDS
    )
    def test_program(self, runner, source, expected, description):
        """Run a single test program through pure-neural runner."""
        # Per-test state reset (runner is class-scoped, programs are
        # independent).
        runner._memory = {}
        runner._mem_history = {}
        runner._mem_access_order = []

        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=2000)

        assert result == expected, \
            f"{description}: expected {expected}, got {result}"


class TestSuite1096PureNeuralStatistics:
    """Sanity checks on the pure-neural parallel suite.

    These tests do NOT actually run any program — they verify the
    parametrization matches the handler-mode suite exactly so the two
    can be compared run-for-run as phases land.
    """

    def test_suite_has_1096_tests(self):
        """Pure-neural suite mirrors handler suite: 1096 tests."""
        assert len(ALL_TESTS) == 1096, \
            f"Expected 1096 tests, got {len(ALL_TESTS)}"

    def test_test_ids_unique(self):
        """All test IDs are unique (no duplicate parametrizations)."""
        seen = {}
        for i, tid in enumerate(TEST_IDS):
            if tid in seen:
                # Duplicate IDs are tolerable (pytest disambiguates) but
                # they make XPASS reports harder to read.
                seen[tid].append(i)
            else:
                seen[tid] = [i]
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        # Allow small number of dups (the handler suite has some) but flag
        # huge collisions.
        assert len(dups) < 50, \
            f"Too many duplicate test IDs ({len(dups)}); see {list(dups.items())[:5]}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
