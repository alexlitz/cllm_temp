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

    Initial fix (commit ``944d5df``): a single session-scoped fixture
    batched all ~42 programs through ``BatchedPureNeuralRunner.run_batch``.
    Total wall ~370s. But ``run_batch`` runs until the longest-running
    program halts, so the single longest program (e.g.
    ``TestSmokeMemory::test_si_li_multiple_stores`` at max_steps=40) drove
    the wall time for all 42.

    Refactor (2026-05-12, branch ``smoke-per-class-batching``): one
    session-scoped fixture per test class (``_smoke_basic_results``,
    ``_smoke_controlflow_results``, ``_smoke_memory_results``, etc.). Each
    fixture batches only the programs in its own class through
    ``BatchedPureNeuralRunner.run_batch(...)``. Variance shrinks because
    tests in the same class share similar ``max_steps`` budgets, so each
    per-class batch's longest member is close to the per-class average.
    Total wall = sum of per-class walls, but each per-class wall is much
    smaller than the single ``max(all max_steps)`` we paid before. (See
    the per-class wall-time table in the PR description.)

    Equivalence with the serial path is preserved because every batch element
    runs through the same compiled neural model as ``quick_runner``; see
    ``c4_release/neural_vm/batched_pure_neural.py`` for the per-element state
    machine. Tests with custom assertions (e.g. ``result != 0``) keep those
    checks intact via per-entry ``check`` callables.

Tuning knobs:
    C4_SMOKE_SPEC_K — DraftVM speculation horizon in VM steps (default 0 for
                      neural-authoritative smoke; set >0 only for speculative
                      performance experiments). The default 0 also disables
                      DraftVM length bucketing, so the smoke gate runs the raw
                      one-token-at-a-time neural path. Results are always
                      decoded from model-emitted context, never substituted
                      from DraftVM state.
    C4_SMOKE_MAX_STEPS_CAP — override the per-group ``max_steps`` cap
                      (default: use the max of the group's test budgets).
                      Useful for stress-testing edge cases.
    C4_SMOKE_EXECUTOR — ``batched`` (default) runs class groups through
                      ``BatchedPureNeuralRunner``. ``serial-kv`` runs each
                      entry through the shared serial ``AutoregressiveVMRunner``
                      so speculative verification exercises the KV cache.

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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.embedding import Opcode
from tests.declarative_oracle import declarative_oracle_for_program


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
        "check": _eq(42),
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
        "check": _eq(42),
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
        "check": _eq(42),
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


# Master registry — ordered list of every batched smoke entry, grouped by
# the test class that owns it. Each per-class fixture below batches only its
# own group through ``BatchedPureNeuralRunner.run_batch``. Grouping by class
# keeps tests with similar ``max_steps`` budgets together so each batch's
# longest member is close to the per-class average, which reduces wall-time
# variance vs. one giant 42-program batch (where the single ``max(max_steps)``
# drove the wall for every test).
_SMOKE_GROUPS = {
    "basic":         _BASIC_TESTS,
    "controlflow":   _CF_TESTS,
    "functioncall":  _FUNC_TESTS,
    "bitwise":       _BITWISE_TESTS,
    "comparison":    _CMP_TESTS,
    "address":       _ADDR_TESTS,
    "memory":        _MEM_TESTS,
    "shift":         _SHIFT_TESTS,
    "bit32":         _BIT32_TESTS,
    "integration":   _INTEGRATION_TESTS,
}


# Detect duplicate names early so a typo can't silently shadow a result.
_seen = set()
for _group in _SMOKE_GROUPS.values():
    for _t in _group:
        assert _t["name"] not in _seen, f"duplicate smoke test name: {_t['name']}"
        _seen.add(_t["name"])
del _seen, _group, _t


# Flat list kept for any external introspection (e.g. PR diffs / test
# selection scripts that walk every smoke test by name).
_ALL_SMOKE_TESTS = [t for group in _SMOKE_GROUPS.values() for t in group]


_SMOKE_SPEC_K = int(os.environ.get("C4_SMOKE_SPEC_K", "0"))
_SMOKE_MAX_STEPS_CAP = os.environ.get("C4_SMOKE_MAX_STEPS_CAP")
_SMOKE_MAX_STEPS_CAP = int(_SMOKE_MAX_STEPS_CAP) if _SMOKE_MAX_STEPS_CAP else None
_SMOKE_EXECUTOR = os.environ.get("C4_SMOKE_EXECUTOR", "batched").strip().lower()


# =============================================================================
# Per-class session-scoped batched-results fixtures
# =============================================================================


def _run_group_batch(runner, tests, group_label=""):
    """Run one group of smoke tests through ``run_batch`` and return a
    ``{name: (output, exit_code, err)}`` dict for that group.

    ``err`` is ``None`` on success; on a batch-level failure it's a string
    and ``exit_code`` is ``None``. We catch the batch-run exception so a
    single bad program doesn't poison the rest of the session.

    When ``C4_SMOKE_TIMING=1`` is set, prints a one-line per-group wall-
    time summary to stderr (program count, max_steps, elapsed seconds) so
    benchmarks can compare per-class walls against the prior all-42 wall
    without parsing pytest internals.
    """
    names = [t["name"] for t in tests]
    oracle_results = []
    declarative_errs = {}
    for t in tests:
        oracle = declarative_oracle_for_program(
            t["bytecode"],
            b"",
            suite_check=t["check"],
            label=t["name"],
        )
        oracle_results.append(oracle)
        if oracle.error is not None:
            declarative_errs[t["name"]] = oracle.error

    runnable = [
        (i, t, oracle_results[i])
        for i, t in enumerate(tests)
        if t["name"] not in declarative_errs
    ]
    results = {
        name: ("", None, err)
        for name, err in declarative_errs.items()
    }
    if not runnable:
        return {
            name: results.get(name, ("", None, "not in declarative results"))
            for name in names
        }

    bytecodes = [t["bytecode"] for _i, t, _oracle in runnable]
    runnable_names = [t["name"] for _i, t, _oracle in runnable]
    expected_steps = [oracle.steps for _i, _t, oracle in runnable]
    max_steps = _SMOKE_MAX_STEPS_CAP
    timing_steps = (
        max_steps
        if max_steps is not None
        else max(step or 0 for step in expected_steps)
    )

    t0 = time.perf_counter()
    if _SMOKE_EXECUTOR in ("serial", "serial-kv", "kv") and hasattr(runner, "_serial"):
        serial_runner = getattr(runner, "_serial", runner)
        old_spec_k = getattr(serial_runner, "spec_k", None)
        try:
            if old_spec_k is not None:
                serial_runner.spec_k = _SMOKE_SPEC_K
            batch_results = []
            for bytecode, steps in zip(bytecodes, expected_steps):
                batch_results.append(
                    serial_runner.run(
                        bytecode,
                        b"",
                        max_steps=max_steps if max_steps is not None else steps,
                    )
                )
        except Exception as e:
            err = f"serial run error: {e!r}"
            for n in runnable_names:
                results[n] = ("", None, err)
            return {n: results[n] for n in names}
        finally:
            if old_spec_k is not None:
                serial_runner.spec_k = old_spec_k
        elapsed = time.perf_counter() - t0
        if os.environ.get("C4_SMOKE_TIMING") == "1" and group_label:
            cache_state = "kv" if getattr(serial_runner, "use_kv_cache", False) else "no-kv"
            print(
                f"[smoke-timing] {group_label:>13s}: "
                f"N={len(runnable):2d} decl_steps={timing_steps:3d} "
                f"executor=serial-{cache_state} wall={elapsed:7.2f}s",
                file=sys.stderr,
                flush=True,
            )
        for name, (output, exit_code) in zip(runnable_names, batch_results):
            results[name] = (output, exit_code, None)
        return {n: results[n] for n in names}

    try:
        batch_results = runner.run_batch(
            bytecodes,
            max_steps=max_steps,
            spec_k=_SMOKE_SPEC_K,
            expected_steps_list=expected_steps,
            bucket_by_predicted_length=(_SMOKE_SPEC_K != 0),
        )
    except Exception as e:
        err = f"batch run error: {e!r}"
        for n in runnable_names:
            results[n] = ("", None, err)
        return {n: results[n] for n in names}
    elapsed = time.perf_counter() - t0

    if os.environ.get("C4_SMOKE_TIMING") == "1" and group_label:
        print(
            f"[smoke-timing] {group_label:>13s}: "
            f"N={len(runnable):2d} decl_steps={timing_steps:3d} "
            f"wall={elapsed:7.2f}s",
            file=sys.stderr,
            flush=True,
        )

    for name, (output, exit_code) in zip(runnable_names, batch_results):
        results[name] = (output, exit_code, None)
    return {n: results[n] for n in names}


class TestSmokeHarnessConfig:
    """Cheap harness checks that do not construct the neural model."""

    def test_spec_k_zero_disables_draftvm_bucketing(self, monkeypatch):
        """Strict smoke default must stay raw neural, not DraftVM-assisted."""

        class FakeRunner:
            def __init__(self):
                self.kwargs = None

            def run_batch(self, bytecodes, **kwargs):
                self.kwargs = kwargs
                return [("", 123) for _ in bytecodes]

        fake = FakeRunner()
        monkeypatch.setattr(sys.modules[__name__], "_SMOKE_SPEC_K", 0)

        results = _run_group_batch(
            fake,
            [
                {
                    "name": "harness::raw",
                    "bytecode": _make_bytecode([(Opcode.IMM, 123), Opcode.EXIT]),
                    "max_steps": 2,
                    "check": _eq(123),
                }
            ],
            "harness",
        )

        assert results["harness::raw"] == ("", 123, None)
        assert fake.kwargs["max_steps"] is None
        assert fake.kwargs["expected_steps_list"] == [2]
        assert fake.kwargs["spec_k"] == 0
        assert fake.kwargs["bucket_by_predicted_length"] is False

    def test_positive_spec_k_enables_spec_experiment_bucketing(self, monkeypatch):
        """Speculative smoke experiments keep the length-bucket speed path."""

        class FakeRunner:
            def __init__(self):
                self.kwargs = None

            def run_batch(self, bytecodes, **kwargs):
                self.kwargs = kwargs
                return [("", 7) for _ in bytecodes]

        fake = FakeRunner()
        monkeypatch.setattr(sys.modules[__name__], "_SMOKE_SPEC_K", 8)

        _run_group_batch(
            fake,
            [
                {
                    "name": "harness::spec",
                    "bytecode": _make_bytecode([(Opcode.IMM, 7), Opcode.EXIT]),
                    "max_steps": 2,
                    "check": _eq(7),
                }
            ],
            "harness",
        )

        assert fake.kwargs["spec_k"] == 8
        assert fake.kwargs["expected_steps_list"] == [2]
        assert fake.kwargs["bucket_by_predicted_length"] is True

    def test_max_steps_cap_overrides_group_max(self, monkeypatch):
        """Documented C4_SMOKE_MAX_STEPS_CAP must drive run_batch max_steps."""

        class FakeRunner:
            def __init__(self):
                self.kwargs = None

            def run_batch(self, bytecodes, **kwargs):
                self.kwargs = kwargs
                return [("", 1) for _ in bytecodes]

        fake = FakeRunner()
        monkeypatch.setattr(sys.modules[__name__], "_SMOKE_SPEC_K", 0)
        monkeypatch.setattr(sys.modules[__name__], "_SMOKE_MAX_STEPS_CAP", 77)

        _run_group_batch(
            fake,
            [
                {
                    "name": "harness::cap_a",
                    "bytecode": _make_bytecode([(Opcode.IMM, 1), Opcode.EXIT]),
                    "max_steps": 2,
                    "check": _eq(1),
                },
                {
                    "name": "harness::cap_b",
                    "bytecode": _make_bytecode([(Opcode.IMM, 1), Opcode.EXIT]),
                    "max_steps": 40,
                    "check": _eq(1),
                },
            ],
            "harness",
        )

        assert fake.kwargs["max_steps"] == 77
        assert fake.kwargs["expected_steps_list"] == [2, 2]

    def test_declarative_check_failure_blocks_neural_batch(self, monkeypatch):
        """Smoke declarations are validated before the neural runner is used."""

        class FakeRunner:
            def run_batch(self, bytecodes, **kwargs):
                raise AssertionError("neural runner should not be called")

        monkeypatch.setattr(sys.modules[__name__], "_SMOKE_MAX_STEPS_CAP", None)

        results = _run_group_batch(
            FakeRunner(),
            [
                {
                    "name": "harness::bad_decl_contract",
                    "bytecode": _make_bytecode([(Opcode.IMM, 41), Opcode.EXIT]),
                    "max_steps": 2,
                    "check": _eq(42),
                }
            ],
            "harness",
        )

        output, exit_code, err = results["harness::bad_decl_contract"]
        assert output == ""
        assert exit_code is None
        assert "suite check disagrees with declarative result" in err

    def test_lookup_and_check_exposes_neural_result_mismatch(self, monkeypatch):
        """A wrong raw-neural exit code must fail, not be papered over."""

        entry = {
            "name": "harness::mismatch",
            "bytecode": _make_bytecode([(Opcode.IMM, 42), Opcode.EXIT]),
            "max_steps": 2,
            "check": _eq(42),
        }
        monkeypatch.setattr(sys.modules[__name__], "_ALL_SMOKE_TESTS", [entry])

        with pytest.raises(AssertionError, match="expected 42, got 7"):
            _lookup_and_check({"harness::mismatch": ("", 7, None)}, "harness::mismatch")

    def test_lookup_and_check_reports_batch_errors(self, monkeypatch):
        """Batch-level raw-neural failures must surface as pytest failures."""

        entry = {
            "name": "harness::batch_error",
            "bytecode": _make_bytecode([(Opcode.IMM, 42), Opcode.EXIT]),
            "max_steps": 2,
            "check": _eq(42),
        }
        monkeypatch.setattr(sys.modules[__name__], "_ALL_SMOKE_TESTS", [entry])

        with pytest.raises(pytest.fail.Exception, match="harness::batch_error: batch run error"):
            _lookup_and_check(
                {"harness::batch_error": ("", None, "batch run error: RuntimeError('boom')")},
                "harness::batch_error",
            )

    def test_selected_group_filter_keeps_focused_pytest_runs_narrow(self):
        """Selecting two smoke tests should not execute the whole class batch."""

        class FakeItem:
            def __init__(self, nodeid):
                self.nodeid = nodeid

        class FakeSession:
            items = [
                FakeItem("c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit"),
                FakeItem("c4_release/tests/test_smoke.py::TestSmokeBasic::test_add_basic"),
            ]

        class FakeRequest:
            session = FakeSession()

        selected = _selected_group_tests(FakeRequest(), _BASIC_TESTS)

        assert [t["name"] for t in selected] == [
            "TestSmokeBasic::test_imm_exit",
            "TestSmokeBasic::test_add_basic",
        ]


# All per-class fixtures share the SAME compiled model via the session-
# scoped ``_batched_pure_neural_runner_model`` fixture in ``conftest.py`` —
# so the expensive ``compile_full_vm_dynamic`` bake happens exactly once across the
# whole pytest session, not once per class.


def _selected_group_tests(request, tests):
    """Return only group entries selected by pytest, preserving group order."""
    selected_nodeids = [item.nodeid for item in request.session.items]
    selected = [
        test
        for test in tests
        if any(nodeid.endswith(f"::{test['name']}") for nodeid in selected_nodeids)
    ]
    return selected or tests


@pytest.fixture(scope="session")
def _smoke_basic_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _BASIC_TESTS),
        "basic",
    )


@pytest.fixture(scope="session")
def _smoke_controlflow_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _CF_TESTS),
        "controlflow",
    )


@pytest.fixture(scope="session")
def _smoke_functioncall_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _FUNC_TESTS),
        "functioncall",
    )


@pytest.fixture(scope="session")
def _smoke_bitwise_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _BITWISE_TESTS),
        "bitwise",
    )


@pytest.fixture(scope="session")
def _smoke_comparison_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _CMP_TESTS),
        "comparison",
    )


@pytest.fixture(scope="session")
def _smoke_address_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _ADDR_TESTS),
        "address",
    )


@pytest.fixture(scope="session")
def _smoke_memory_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _MEM_TESTS),
        "memory",
    )


@pytest.fixture(scope="session")
def _smoke_shift_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _SHIFT_TESTS),
        "shift",
    )


@pytest.fixture(scope="session")
def _smoke_bit32_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _BIT32_TESTS),
        "bit32",
    )


@pytest.fixture(scope="session")
def _smoke_integration_results(request, _batched_pure_neural_runner_model):
    return _run_group_batch(
        _batched_pure_neural_runner_model,
        _selected_group_tests(request, _INTEGRATION_TESTS),
        "integration",
    )


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

    def test_imm_exit(self, _smoke_basic_results):
        _lookup_and_check(_smoke_basic_results, "TestSmokeBasic::test_imm_exit")

    def test_add_basic(self, _smoke_basic_results):
        _lookup_and_check(_smoke_basic_results, "TestSmokeBasic::test_add_basic")

    def test_sub_basic(self, _smoke_basic_results):
        _lookup_and_check(_smoke_basic_results, "TestSmokeBasic::test_sub_basic")

    def test_mul_basic(self, _smoke_basic_results):
        _lookup_and_check(_smoke_basic_results, "TestSmokeBasic::test_mul_basic")

    @pytest.mark.xfail(strict=False, reason="MUL/DIV/MOD arch blocked: wide_div_rules per-nibble wrong for cross-nibble dividends, see memory project_mul_div_mod_arch_blocked.md")
    def test_div_basic(self, _smoke_basic_results):
        _lookup_and_check(_smoke_basic_results, "TestSmokeBasic::test_div_basic")

    @pytest.mark.xfail(strict=False, reason="MUL/DIV/MOD arch blocked: wide_div_rules per-nibble wrong for cross-nibble dividends, see memory project_mul_div_mod_arch_blocked.md")
    def test_mod_basic(self, _smoke_basic_results):
        _lookup_and_check(_smoke_basic_results, "TestSmokeBasic::test_mod_basic")


# =============================================================================
# Control Flow Smoke Tests
# =============================================================================

class TestSmokeControlFlow:
    """Control flow quick checks."""

    def test_jmp_forward(self, _smoke_controlflow_results):
        _lookup_and_check(_smoke_controlflow_results, "TestSmokeControlFlow::test_jmp_forward")

    def test_bz_branch(self, _smoke_controlflow_results):
        _lookup_and_check(_smoke_controlflow_results, "TestSmokeControlFlow::test_bz_branch")

    def test_bnz_branch(self, _smoke_controlflow_results):
        _lookup_and_check(_smoke_controlflow_results, "TestSmokeControlFlow::test_bnz_branch")


# =============================================================================
# Function Call Smoke Tests
# =============================================================================

class TestSmokeFunctionCall:
    """Function call quick checks."""

    @pytest.mark.xfail(strict=False, reason="JSR/LEV multi-layer arch blocked, see memory project_jsr_simple_function_arch_blocked.md")
    def test_simple_function(self, _smoke_functioncall_results):
        _lookup_and_check(_smoke_functioncall_results, "TestSmokeFunctionCall::test_simple_function")


# =============================================================================
# Bitwise Smoke Tests
# =============================================================================

class TestSmokeBitwise:
    """Bitwise operation quick checks."""

    def test_or_basic(self, _smoke_bitwise_results):
        _lookup_and_check(_smoke_bitwise_results, "TestSmokeBitwise::test_or_basic")

    def test_and_basic(self, _smoke_bitwise_results):
        _lookup_and_check(_smoke_bitwise_results, "TestSmokeBitwise::test_and_basic")

    def test_xor_basic(self, _smoke_bitwise_results):
        _lookup_and_check(_smoke_bitwise_results, "TestSmokeBitwise::test_xor_basic")


# =============================================================================
# Comparison Smoke Tests
# =============================================================================

class TestSmokeComparison:
    """Comparison operation quick checks."""

    def test_eq_true(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_eq_true")

    def test_eq_false(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_eq_false")

    def test_lt_true(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_lt_true")

    def test_ne_true(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_ne_true")

    def test_gt_true(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_gt_true")

    def test_le_true(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_le_true")

    def test_ge_true(self, _smoke_comparison_results):
        _lookup_and_check(_smoke_comparison_results, "TestSmokeComparison::test_ge_true")


# =============================================================================
# Address/Stack Smoke Tests
# =============================================================================

class TestSmokeAddress:
    """LEA and ADJ operation quick checks."""

    def test_lea_basic(self, _smoke_address_results):
        _lookup_and_check(_smoke_address_results, "TestSmokeAddress::test_lea_basic")

    def test_adj_sp(self, _smoke_address_results):
        _lookup_and_check(_smoke_address_results, "TestSmokeAddress::test_adj_sp")


# =============================================================================
# Memory Operation Smoke Tests
# =============================================================================

class TestSmokeMemory:
    """Memory load/store operation quick checks."""

    def test_si_li_roundtrip(self, _smoke_memory_results):
        _lookup_and_check(_smoke_memory_results, "TestSmokeMemory::test_si_li_roundtrip")

    def test_sc_lc_roundtrip(self, _smoke_memory_results):
        _lookup_and_check(_smoke_memory_results, "TestSmokeMemory::test_sc_lc_roundtrip")

    def test_si_li_zero(self, _smoke_memory_results):
        _lookup_and_check(_smoke_memory_results, "TestSmokeMemory::test_si_li_zero")

    def test_si_li_multiple_stores(self, _smoke_memory_results):
        _lookup_and_check(_smoke_memory_results, "TestSmokeMemory::test_si_li_multiple_stores")

    def test_si_li_overwrite(self, _smoke_memory_results):
        _lookup_and_check(_smoke_memory_results, "TestSmokeMemory::test_si_li_overwrite")

    def test_si_li_16bit_value(self, _smoke_memory_results):
        _lookup_and_check(_smoke_memory_results, "TestSmokeMemory::test_si_li_16bit_value")


# =============================================================================
# Shift Smoke Tests
# =============================================================================

class TestSmokeShift:
    """Shift operation quick checks."""

    def test_shl(self, _smoke_shift_results):
        _lookup_and_check(_smoke_shift_results, "TestSmokeShift::test_shl")

    def test_shr(self, _smoke_shift_results):
        _lookup_and_check(_smoke_shift_results, "TestSmokeShift::test_shr")


# =============================================================================
# Handler Status Smoke Test
# =============================================================================

class TestSmokeHandlerStatus:
    """Verify expected handlers are registered.

    These don't run programs, so they keep the original (cheap) fixture.
    """

    def test_neural_ops_no_handler(self, handler_status):
        """Verify neural-only ops (ALU + control + IMM/EXIT) have no Python handler.

        After V7 REMOVABLE cleanup, only host-boundary syscalls
        (OPEN/READ/CLOS/PRTF) keep Python handlers; everything else is
        neural-only.
        """
        neural_ops = ["ADD", "SUB", "MUL", "DIV", "MOD",
                      "OR", "XOR", "AND", "SHL", "SHR",
                      "EQ", "NE", "LT", "GT", "LE", "GE",
                      "IMM", "EXIT"]

        for op in neural_ops:
            assert not handler_status[op]["has_handler"], f"{op} should be neural-only"

    def test_handler_ops_have_handler(self, handler_status):
        """Verify host-boundary syscall ops have a Python handler.

        The runner's ``_syscall_handlers`` dict currently registers
        OPEN/READ/CLOS/PRTF. GETCHAR/PUTCHAR are reserved boundary ops
        that flow through the conversational-IO path instead and are
        therefore not in the dict today.
        """
        handler_ops = ["PRTF", "OPEN", "CLOS", "READ"]

        for op in handler_ops:
            assert handler_status[op]["has_handler"], f"{op} should have a Python handler"
            assert handler_status[op]["handler_type"] == "syscall", (
                f"{op} should be a syscall handler, got {handler_status[op]['handler_type']}"
            )

    def test_phase6_prtf_read_status_is_explicit(self, handler_status):
        """Document the current PRTF/READ Phase 6 boundary status."""
        for op in ["PRTF", "READ"]:
            assert handler_status[op]["phase6_status"] == "external-shim"
            assert handler_status[op]["neural_complete"] is False
            assert handler_status[op]["diagnostic"]


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

    def test_add_16bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_add_16bit")

    def test_add_carry_cascade(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_add_carry_cascade")

    def test_sub_16bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_sub_16bit")

    def test_sub_borrow_cascade(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_sub_borrow_cascade")

    def test_or_16bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_or_16bit")

    def test_and_16bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_and_16bit")

    def test_xor_16bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_xor_16bit")

    @pytest.mark.xfail(strict=False, reason="MUL/DIV/MOD arch blocked: wide_mul_rules width=2 collides with ADDR_KEY, see memory project_mul_div_mod_arch_blocked.md")
    def test_mul_overflow(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_mul_overflow")

    def test_shl_8bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_shl_8bit")

    def test_shr_8bit(self, _smoke_bit32_results):
        _lookup_and_check(_smoke_bit32_results, "TestSmoke32Bit::test_shr_8bit")


# =============================================================================
# Integration Tests
# =============================================================================

class TestSmokeIntegration:
    """Multi-step integration tests combining multiple opcodes."""

    def test_cmp_and_branch(self, _smoke_integration_results):
        _lookup_and_check(_smoke_integration_results, "TestSmokeIntegration::test_cmp_and_branch")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
