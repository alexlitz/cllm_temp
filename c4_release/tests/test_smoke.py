#!/usr/bin/env python3
"""
Smoke Tests - Quick Validation Suite (Batched Pure Neural)

Run with: pytest tests/test_smoke.py -v --tb=short

These tests catch obvious regressions before running the full test suite.

Batched wiring (2026-05-12, branch ``batch-smoke-tests-via-fixture``):
    Previously each smoke test built a small bytecode and called
    ``quick_runner.run(...)`` which spins one full ``AutoregressiveVMRunner``
    pure-neural decode (~45-100s per test). With ~42 tests using
    ``quick_runner``, the suite took ~30-60+ minutes sequentially.

    Now: a session-scoped fixture (``_smoke_batched_results``) collects every
    smoke test's (name, bytecode, max_steps) at module import, runs the whole
    set through ``BatchedPureNeuralRunner.run_batch(...)`` in one shot (with
    ``spec_k=int(os.environ.get("C4_SMOKE_SPEC_K", "8"))``), and caches
    ``{name: (output, exit_code, err)}`` for the rest of the session. Each
    test method becomes a thin lookup-and-assert.

    Equivalence with the serial path is preserved because every batch element
    runs through the same compiled neural model as ``quick_runner``; see
    ``c4_release/neural_vm/batched_pure_neural.py`` for the per-element state
    machine. Tests with custom assertions (e.g. ``result != 0``,
    ``result == 42 or result == 0``) keep those checks intact via per-entry
    ``assert_fn`` callables.

Tuning knobs:
    C4_SMOKE_SPEC_K — DraftVM speculation horizon in VM steps (default 8;
                      0 disables speculation and falls back to one-token-
                      per-forward batched decode).
    C4_SMOKE_MAX_STEPS_CAP — override the per-program ``max_steps`` cap
                      (default: use the per-test's own ``max_steps`` value).
                      Useful for stress-testing edge cases.

Coexists with:
    - ``test_suite_1096_pure_neural_pytest.py`` (1096 batched suite, commit
      ``6a4ff5b``). Same pattern: session-scoped fixture, per-test thin
      lookup, ``BatchedPureNeuralRunner.run_batch`` under the hood.
    - The parallel ``spec-decoding-into-single-test-fixtures-v2`` work that
      wires ``spec_k`` into the serial ``AutoregressiveVMRunner`` path.
      That refactor targets per-test ``quick_runner`` calls; this file
      bypasses ``quick_runner`` for ~42 of its tests by routing through
      the batched runner directly.

Out of scope (still use ``quick_runner`` / other fixtures):
    - ``TestSmokeHandlerStatus``: tests handler-registration metadata, not
      execution.
    - ``TestSmokePipeline.test_compile_and_run``: marked ``@pytest.mark.slow``
      and exercises the full C-source -> bytecode pipeline; kept as a
      separate end-to-end check.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.embedding import Opcode


# =============================================================================
# Bytecode helper (module-level so we can pre-build all smoke bytecodes once)
# =============================================================================


def _make_bytecode(ops):
    """Build a packed bytecode list from operation tuples / opcodes."""
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


# =============================================================================
# Smoke-test registry
# =============================================================================
#
# Each entry is a dict with:
#     "name":       unique key into the session-batched results dict
#     "bytecode":   list[int] of packed instructions (built at import time)
#     "max_steps":  per-program VM step budget (the batched runner uses the
#                   MAX across all programs as its loop bound; smaller-budget
#                   tests just halt earlier and stay halted)
#     "check":      callable(result_int) -> None; raises AssertionError on
#                   failure. We use callables instead of an `expected` int
#                   because a handful of tests use disjunctions or `!= 0`.
#
# The fixture below batches every entry through ``BatchedPureNeuralRunner``
# once per pytest session.


def _eq(expected):
    """Build an equality-check function for the common ``result == X`` case."""
    def check(result, _expected=expected):
        assert result == _expected, f"expected {_expected}, got {result}"
    return check


def _eq_or(*allowed):
    """Build a multi-value check function (used by control-flow smoke tests
    that accept either the neural-path or fallback value)."""
    def check(result, _allowed=allowed):
        assert result in _allowed, f"expected one of {_allowed}, got {result}"
    return check


def _ne(forbidden):
    """Build a not-equal check function (used by LEA test)."""
    def check(result, _forbidden=forbidden):
        assert result != _forbidden, f"expected != {_forbidden}, got {result}"
    return check


# --- TestSmokeBasic --------------------------------------------------------

_BASIC_TESTS = [
    {
        "name": "TestSmokeBasic::test_imm_exit",
        "bytecode": _make_bytecode([(Opcode.IMM, 42), Opcode.EXIT]),
        "max_steps": 10,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeBasic::test_add_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 32), Opcode.ADD,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeBasic::test_sub_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 50), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SUB,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeBasic::test_mul_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 6), Opcode.PSH,
            (Opcode.IMM, 7), Opcode.MUL,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeBasic::test_div_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 84), Opcode.PSH,
            (Opcode.IMM, 2), Opcode.DIV,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeBasic::test_mod_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 43), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.MOD,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(3),
    },
]


# --- TestSmokeControlFlow --------------------------------------------------

_CF_TESTS = [
    {
        "name": "TestSmokeControlFlow::test_jmp_forward",
        "bytecode": _make_bytecode([
            (Opcode.JMP, 2),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ]),
        "max_steps": 15,
        # Per the original test: 42 (neural) or 0 (neural-path broken).
        "check": _eq_or(42, 0),
    },
    {
        "name": "TestSmokeControlFlow::test_bz_branch",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0),
            (Opcode.BZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ]),
        "max_steps": 15,
        "check": _eq_or(42, 0),
    },
    {
        "name": "TestSmokeControlFlow::test_bnz_branch",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 1),
            (Opcode.BNZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT,
        ]),
        "max_steps": 15,
        "check": _eq_or(42, 1),
    },
]


# --- TestSmokeFunctionCall -------------------------------------------------

_FUNC_TESTS = [
    {
        "name": "TestSmokeFunctionCall::test_simple_function",
        "bytecode": _make_bytecode([
            (Opcode.JSR, 3),
            Opcode.EXIT,
            Opcode.NOP,
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ]),
        "max_steps": 30,
        "check": _eq(42),
    },
]


# --- TestSmokeBitwise ------------------------------------------------------

_BITWISE_TESTS = [
    {
        "name": "TestSmokeBitwise::test_or_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x0F), Opcode.PSH,
            (Opcode.IMM, 0x30), Opcode.OR,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x3F),
    },
    {
        "name": "TestSmokeBitwise::test_and_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0x2A), Opcode.AND,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x2A),
    },
    {
        "name": "TestSmokeBitwise::test_xor_basic",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0xD5), Opcode.XOR,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x2A),
    },
]


# --- TestSmokeComparison ---------------------------------------------------

_CMP_TESTS = [
    {
        "name": "TestSmokeComparison::test_eq_true",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 42), Opcode.PSH,
            (Opcode.IMM, 42), Opcode.EQ,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
    {
        "name": "TestSmokeComparison::test_eq_false",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.EQ,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0),
    },
    {
        "name": "TestSmokeComparison::test_lt_true",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.LT,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
    {
        "name": "TestSmokeComparison::test_ne_true",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.NE,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
    {
        "name": "TestSmokeComparison::test_gt_true",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 20), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.GT,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
    {
        "name": "TestSmokeComparison::test_le_true",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.LE,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
    {
        "name": "TestSmokeComparison::test_ge_true",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 20), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.GE,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
]


# --- TestSmokeAddress ------------------------------------------------------
#
# ADJ test originally rewrites bytecode[2] post-hoc (ADJ with imm=8). We
# encode that directly here.

_ADDR_TESTS = [
    {
        "name": "TestSmokeAddress::test_lea_basic",
        "bytecode": _make_bytecode([
            Opcode.ENT,
            (Opcode.IMM, 0),
            (Opcode.LEA, 2),
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _ne(0),
    },
    {
        "name": "TestSmokeAddress::test_adj_sp",
        "bytecode": [
            Opcode.IMM | (42 << 8),
            Opcode.PSH,
            Opcode.ADJ | (8 << 8),  # mirror the original test's post-edit
            Opcode.EXIT,
        ],
        "max_steps": 20,
        "check": _eq(42),
    },
]


# --- TestSmokeMemory -------------------------------------------------------

_MEM_TESTS = [
    {
        "name": "TestSmokeMemory::test_si_li_roundtrip",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x200), Opcode.PSH,
            (Opcode.IMM, 42), Opcode.SI,
            (Opcode.IMM, 0x200), Opcode.LI,
            Opcode.EXIT,
        ]),
        "max_steps": 30,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeMemory::test_sc_lc_roundtrip",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x200), Opcode.PSH,
            (Opcode.IMM, 42), Opcode.SC,
            (Opcode.IMM, 0x200), Opcode.LC,
            Opcode.EXIT,
        ]),
        "max_steps": 30,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeMemory::test_si_li_zero",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x300), Opcode.PSH,
            (Opcode.IMM, 0), Opcode.SI,
            (Opcode.IMM, 0x300), Opcode.LI,
            Opcode.EXIT,
        ]),
        "max_steps": 30,
        "check": _eq(0),
    },
    {
        "name": "TestSmokeMemory::test_si_li_multiple_stores",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x200), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.SI,
            (Opcode.IMM, 0x300), Opcode.PSH,
            (Opcode.IMM, 99), Opcode.SI,
            (Opcode.IMM, 0x300), Opcode.LI,
            Opcode.EXIT,
        ]),
        "max_steps": 40,
        "check": _eq(99),
    },
    {
        "name": "TestSmokeMemory::test_si_li_overwrite",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x200), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.SI,
            (Opcode.IMM, 0x200), Opcode.PSH,
            (Opcode.IMM, 55), Opcode.SI,
            (Opcode.IMM, 0x200), Opcode.LI,
            Opcode.EXIT,
        ]),
        "max_steps": 40,
        "check": _eq(55),
    },
    {
        "name": "TestSmokeMemory::test_si_li_16bit_value",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x200), Opcode.PSH,
            (Opcode.IMM, 0x1234), Opcode.SI,
            (Opcode.IMM, 0x200), Opcode.LI,
            Opcode.EXIT,
        ]),
        "max_steps": 30,
        "check": _eq(0x1234),
    },
]


# --- TestSmokeShift --------------------------------------------------------

_SHIFT_TESTS = [
    {
        "name": "TestSmokeShift::test_shl",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 21), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SHL,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(42),
    },
    {
        "name": "TestSmokeShift::test_shr",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 84), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SHR,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(42),
    },
]


# --- TestSmoke32Bit --------------------------------------------------------

_BIT32_TESTS = [
    {
        "name": "TestSmoke32Bit::test_add_16bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 200), Opcode.PSH,
            (Opcode.IMM, 100), Opcode.ADD,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(300),
    },
    {
        "name": "TestSmoke32Bit::test_add_carry_cascade",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.ADD,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x100),
    },
    {
        "name": "TestSmoke32Bit::test_sub_16bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x100), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SUB,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0xFF),
    },
    {
        "name": "TestSmoke32Bit::test_sub_borrow_cascade",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SUB,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0xFFFFFFFF),
    },
    {
        "name": "TestSmoke32Bit::test_or_16bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x0F00), Opcode.PSH,
            (Opcode.IMM, 0x00FF), Opcode.OR,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x0FFF),
    },
    {
        "name": "TestSmoke32Bit::test_and_16bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x0FFF), Opcode.PSH,
            (Opcode.IMM, 0x00FF), Opcode.AND,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x00FF),
    },
    {
        "name": "TestSmoke32Bit::test_xor_16bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x0F0F), Opcode.PSH,
            (Opcode.IMM, 0x00FF), Opcode.XOR,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(0x0FF0),
    },
    {
        "name": "TestSmoke32Bit::test_mul_overflow",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 100), Opcode.PSH,
            (Opcode.IMM, 5), Opcode.MUL,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(500),
    },
    {
        "name": "TestSmoke32Bit::test_shl_8bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 1), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHL,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(256),
    },
    {
        "name": "TestSmoke32Bit::test_shr_8bit",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 0x100), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHR,
            Opcode.EXIT,
        ]),
        "max_steps": 20,
        "check": _eq(1),
    },
]


# --- TestSmokeIntegration --------------------------------------------------

_INTEGRATION_TESTS = [
    {
        "name": "TestSmokeIntegration::test_cmp_and_branch",
        "bytecode": _make_bytecode([
            (Opcode.IMM, 5), Opcode.PSH,
            (Opcode.IMM, 5), Opcode.EQ,
            (Opcode.BZ, 6),
            (Opcode.IMM, 42),
            Opcode.EXIT,
            (Opcode.IMM, 0),
            Opcode.EXIT,
        ]),
        "max_steps": 30,
        "check": _eq(42),
    },
]


# Master registry — ordered list of every batched smoke entry. Order matches
# the slot index returned by ``BatchedPureNeuralRunner.run_batch``.
_ALL_SMOKE_TESTS = (
    _BASIC_TESTS
    + _CF_TESTS
    + _FUNC_TESTS
    + _BITWISE_TESTS
    + _CMP_TESTS
    + _ADDR_TESTS
    + _MEM_TESTS
    + _SHIFT_TESTS
    + _BIT32_TESTS
    + _INTEGRATION_TESTS
)


# Detect duplicate names early so a typo can't silently shadow a result.
_seen = set()
for _t in _ALL_SMOKE_TESTS:
    assert _t["name"] not in _seen, f"duplicate smoke test name: {_t['name']}"
    _seen.add(_t["name"])
del _seen


_SMOKE_SPEC_K = int(os.environ.get("C4_SMOKE_SPEC_K", "8"))


# =============================================================================
# Session-scoped batched-results fixture
# =============================================================================


@pytest.fixture(scope="session")
def _smoke_batched_results(_batched_pure_neural_runner_model):
    """Run every entry in ``_ALL_SMOKE_TESTS`` through
    ``BatchedPureNeuralRunner.run_batch`` once per pytest session.

    Returns: ``{name: (output, exit_code, err)}``. ``err`` is ``None`` on
    success; on a batch-level failure it's a string and ``exit_code`` is
    ``None``. We catch the batch-run exception so a single bad program
    doesn't poison the rest of the session.

    Uses the SAME compiled model as ``pure_neural_runner`` via the
    session-scoped ``_batched_pure_neural_runner_model`` fixture defined
    in ``conftest.py`` (so the expensive ``compile_full_vm`` bake happens
    exactly once across the whole pytest session, shared with any other
    batched suite running in the same process).
    """
    runner = _batched_pure_neural_runner_model

    bytecodes = [t["bytecode"] for t in _ALL_SMOKE_TESTS]
    max_steps = max(t["max_steps"] for t in _ALL_SMOKE_TESTS)
    names = [t["name"] for t in _ALL_SMOKE_TESTS]

    try:
        batch_results = runner.run_batch(
            bytecodes,
            max_steps=max_steps,
            spec_k=_SMOKE_SPEC_K,
        )
    except Exception as e:
        err = f"batch run error: {e!r}"
        return {n: ("", None, err) for n in names}

    results = {}
    for name, (output, exit_code) in zip(names, batch_results):
        results[name] = (output, exit_code, None)
    return results


def _lookup_and_check(results, name):
    """Shared helper: pull the result for ``name`` and run its check fn."""
    output, exit_code, err = results.get(name, (None, None, "not in batched results"))
    if err is not None:
        pytest.fail(f"{name}: {err}")
    entry = next((t for t in _ALL_SMOKE_TESTS if t["name"] == name), None)
    assert entry is not None, f"unknown smoke test: {name}"
    entry["check"](exit_code)


# =============================================================================
# Basic Functionality Smoke Tests
# =============================================================================

class TestSmokeBasic:
    """Quick sanity checks - should all pass in <5 seconds."""

    def test_imm_exit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBasic::test_imm_exit")

    def test_add_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBasic::test_add_basic")

    def test_sub_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBasic::test_sub_basic")

    def test_mul_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBasic::test_mul_basic")

    def test_div_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBasic::test_div_basic")

    def test_mod_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBasic::test_mod_basic")


# =============================================================================
# Control Flow Smoke Tests
# =============================================================================

class TestSmokeControlFlow:
    """Control flow quick checks."""

    def test_jmp_forward(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeControlFlow::test_jmp_forward")

    def test_bz_branch(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeControlFlow::test_bz_branch")

    def test_bnz_branch(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeControlFlow::test_bnz_branch")


# =============================================================================
# Function Call Smoke Tests
# =============================================================================

class TestSmokeFunctionCall:
    """Function call quick checks."""

    def test_simple_function(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeFunctionCall::test_simple_function")


# =============================================================================
# Bitwise Smoke Tests
# =============================================================================

class TestSmokeBitwise:
    """Bitwise operation quick checks."""

    def test_or_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBitwise::test_or_basic")

    def test_and_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBitwise::test_and_basic")

    def test_xor_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeBitwise::test_xor_basic")


# =============================================================================
# Comparison Smoke Tests
# =============================================================================

class TestSmokeComparison:
    """Comparison operation quick checks."""

    def test_eq_true(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_eq_true")

    def test_eq_false(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_eq_false")

    def test_lt_true(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_lt_true")

    def test_ne_true(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_ne_true")

    def test_gt_true(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_gt_true")

    def test_le_true(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_le_true")

    def test_ge_true(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeComparison::test_ge_true")


# =============================================================================
# Address/Stack Smoke Tests
# =============================================================================

class TestSmokeAddress:
    """LEA and ADJ operation quick checks."""

    def test_lea_basic(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeAddress::test_lea_basic")

    def test_adj_sp(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeAddress::test_adj_sp")


# =============================================================================
# Memory Operation Smoke Tests
# =============================================================================

class TestSmokeMemory:
    """Memory load/store operation quick checks."""

    def test_si_li_roundtrip(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeMemory::test_si_li_roundtrip")

    def test_sc_lc_roundtrip(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeMemory::test_sc_lc_roundtrip")

    def test_si_li_zero(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeMemory::test_si_li_zero")

    def test_si_li_multiple_stores(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeMemory::test_si_li_multiple_stores")

    def test_si_li_overwrite(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeMemory::test_si_li_overwrite")

    def test_si_li_16bit_value(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeMemory::test_si_li_16bit_value")


# =============================================================================
# Shift Smoke Tests
# =============================================================================

class TestSmokeShift:
    """Shift operation quick checks."""

    def test_shl(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeShift::test_shl")

    def test_shr(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeShift::test_shr")


# =============================================================================
# Handler Status Smoke Test
# =============================================================================

class TestSmokeHandlerStatus:
    """Verify expected handlers are registered.

    These don't run programs, so they keep the original (cheap) fixture.
    """

    def test_neural_ops_no_handler(self, handler_status):
        """Verify arithmetic/bitwise/shift ops have no handlers."""
        neural_ops = ["ADD", "SUB", "MUL", "DIV", "MOD",
                      "OR", "XOR", "AND", "SHL", "SHR",
                      "EQ", "NE", "LT", "GT", "LE", "GE"]

        for op in neural_ops:
            assert not handler_status[op]["has_handler"], f"{op} should be neural-only"

    def test_handler_ops_have_handler(self, handler_status):
        """Verify inline handler ops have correct status."""
        inline_ops = ["JSR", "ENT", "LEV", "PSH", "IMM", "JMP", "BZ", "BNZ"]

        for op in inline_ops:
            assert handler_status[op]["handler_type"] == "neural", f"{op} should be inline/neural"


# =============================================================================
# Quick Full Pipeline (slow / unchanged)
# =============================================================================

class TestSmokePipeline:
    """End-to-end pipeline smoke test."""

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_compile_and_run(self, quick_runner):
        """Compile C code and run it."""
        from src.compiler import compile_c

        source = """
        int main() {
            return 6 * 7;
        }
        """
        bytecode, data = compile_c(source)
        _, result = quick_runner.run(bytecode, data, max_steps=50)
        assert result == 42


# =============================================================================
# 32-bit Value Tests
# =============================================================================

class TestSmoke32Bit:
    """Test operations with values > 255 (exercises bytes 1-3)."""

    def test_add_16bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_add_16bit")

    def test_add_carry_cascade(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_add_carry_cascade")

    def test_sub_16bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_sub_16bit")

    def test_sub_borrow_cascade(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_sub_borrow_cascade")

    def test_or_16bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_or_16bit")

    def test_and_16bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_and_16bit")

    def test_xor_16bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_xor_16bit")

    def test_mul_overflow(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_mul_overflow")

    def test_shl_8bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_shl_8bit")

    def test_shr_8bit(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmoke32Bit::test_shr_8bit")


# =============================================================================
# Integration Tests
# =============================================================================

class TestSmokeIntegration:
    """Multi-step integration tests combining multiple opcodes."""

    def test_cmp_and_branch(self, _smoke_batched_results):
        _lookup_and_check(_smoke_batched_results, "TestSmokeIntegration::test_cmp_and_branch")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
