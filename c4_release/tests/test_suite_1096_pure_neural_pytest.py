#!/usr/bin/env python3
"""
1096 Tests - Pure Neural Mode (Parallel Suite, Batched)

Pure-neural mirror of tests/test_suite_1096_pytest.py. Runs the same 1096
comprehensive test programs through `BatchedPureNeuralRunner` so the whole
suite executes in a small number of forward batches rather than ~3 min/test
of serial autoregressive decode.

Per F's Phase 8 scoping doc (docs/PHASE_8_RUNNER_SWITCH_SCOPE.md):
    Option B (Parallel) — keep handler-mode `test_suite_1096_pytest.py`
    unchanged. This parallel suite grows pass-rate as Phases 1-7 land.
    All 1096 tests start marked xfail; as phases complete, the xfail
    decorator gets removed from confirmed-passing subsets.

Realistic target per F's scope (section 3):
    "200-400/1096 passing in pure-neural mode by end of Phase 7. Residual
    failures become Phase 8+ backlog."

Wiring (2026-05-11):
    A session-scoped fixture compiles all 1096 programs and runs them in
    chunks of `C4_BATCH_CHUNK` (default 32) through
    `BatchedPureNeuralRunner.run_batch`. Per-program results are cached;
    each `test_program_<id>` looks up its slot and asserts the expected
    value. This converts the suite from ~55h serial runtime to one set of
    batched forwards (~hours).

    Speculative decoding (2026-05-11 follow-up):
    ``BatchedPureNeuralRunner`` now natively integrates ``DraftVM`` via the
    ``spec_k`` parameter (see ``c4_release/neural_vm/batched_pure_neural.py``).
    Per-element DraftVMs propose K full VM steps (K * 35 tokens) per batched
    forward; the model verifies the SAFE step offsets (registers + MEM
    marker + STEP_END = 27 of every 35 positions) and trusts DraftVM for the
    8 unsafe MEM-section addr/val bytes (where the embedding's MEM_STORE /
    ADDR_KEY injection makes spec-mode and unspec-mode logits diverge). On
    clean Phase-1-style programs (IMM/EXIT etc.) this trades 1 forward-per-
    token for 1 forward-per-K-steps. Byte-identity with ``spec_k=0`` holds
    for any program where DraftVM and the model agree on MEM-section bytes
    (true for IMM/EXIT trivially; true for any store op too because DraftVM's
    `_last_mem_addr/val` are derived from synchronized registers).

Tuning knobs:
    C4_BATCH_CHUNK   — batch chunk size (default 32)
    C4_BATCH_MAX_STEPS — max VM steps per program (default 2000)
    C4_SPEC_K        — DraftVM speculation horizon in VM steps (default 8;
                       0 disables speculation and falls back to one-token-
                       per-forward batched decode)

Usage:
    # Run the whole suite (batched):
    pytest tests/test_suite_1096_pure_neural_pytest.py -v --tb=line

    # Run a tiny subset (still pays one batch-forward cost):
    pytest tests/test_suite_1096_pure_neural_pytest.py -k "test_program_0" \\
        --tb=line

    # Verify collection only:
    pytest tests/test_suite_1096_pure_neural_pytest.py --collect-only

Date: 2026-05-11 (batched wiring)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c


ALL_TESTS = generate_test_programs()


def make_test_id(idx, test_tuple):
    """Create a unique, readable test ID. Index-prefixed so duplicates remain
    selectable via ``-k`` and the result-lookup dict is unambiguous."""
    _src, _expected, description = test_tuple
    desc = description.replace(" ", "_").replace(":", "")
    return f"{idx:04d}_{desc[:48]}"


TEST_IDS = [make_test_id(i, t) for i, t in enumerate(ALL_TESTS)]


_BATCH_CHUNK = int(os.environ.get("C4_BATCH_CHUNK", "32"))
_BATCH_MAX_STEPS = int(os.environ.get("C4_BATCH_MAX_STEPS", "2000"))


# C4_SPEC_K controls the per-element speculative-decoding horizon (VM steps
# proposed by DraftVM per batched forward pass). Accepted values:
#   * Integer ``0``  — disables speculation (one-token-per-forward batched
#     decode).
#   * Positive int   — fixed-K speculation: every element drafts ``K`` full
#     VM steps per forward.
#   * Negative int   — sentinel for adaptive mode (per-element K, starts at
#     ``_ADAPTIVE_START_K`` in batched_pure_neural.py and ramps up/down
#     based on rolling rejection rate).
#   * String ``"adaptive"`` — alias for the negative-int sentinel.
# Default is ``"adaptive"`` (since DraftVM is deterministic — for programs
# the model handles correctly, every proposed token is accepted, giving us
# K-fold speedup per forward and we want to push K as high as the model
# permits per element).
def _parse_spec_k(raw: str) -> int:
    raw = (raw or "").strip().lower()
    if raw == "adaptive":
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


_SPEC_K = _parse_spec_k(os.environ.get("C4_SPEC_K", "adaptive"))


@pytest.fixture(scope="session")
def _pure_neural_1096_results():
    """Compile all 1096 programs and run them in batched chunks once per
    pytest session. Returns a dict ``{test_id: (output, exit_code, err)}``.

    ``err`` is None on success; on per-program failure it's a string. We
    catch per-chunk exceptions so a single bad program doesn't poison the
    entire session.
    """
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    runner = BatchedPureNeuralRunner()

    results = {}
    for chunk_start in range(0, len(ALL_TESTS), _BATCH_CHUNK):
        chunk = ALL_TESTS[chunk_start:chunk_start + _BATCH_CHUNK]
        chunk_ids = TEST_IDS[chunk_start:chunk_start + _BATCH_CHUNK]

        bytecodes = []
        data_list = []
        compile_errs = {}
        for j, (source, _expected, _desc) in enumerate(chunk):
            try:
                bc, data = compile_c(source)
                bytecodes.append(bc)
                data_list.append(data)
            except Exception as e:
                compile_errs[j] = f"compile error: {e!r}"
                bytecodes.append([])
                data_list.append(b"")

        try:
            batch_results = runner.run_batch(
                bytecodes,
                data_list=data_list,
                max_steps=_BATCH_MAX_STEPS,
                spec_k=_SPEC_K,
            )
        except Exception as e:
            err = f"batch run error: {e!r}"
            for j, tid in enumerate(chunk_ids):
                results[tid] = ("", None, compile_errs.get(j, err))
            continue

        for j, tid in enumerate(chunk_ids):
            if j in compile_errs:
                results[tid] = ("", None, compile_errs[j])
            else:
                output, exit_code = batch_results[j]
                results[tid] = (output, exit_code, None)

    return results


class TestSuite1096PureNeural:
    """Run all 1096 tests in pure-neural mode via batched forwards."""

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
    def test_program(self, _pure_neural_1096_results, request, source,
                     expected, description):
        """Assert this program's batched result matches expected."""
        tid = request.node.callspec.id
        output, exit_code, err = _pure_neural_1096_results.get(
            tid, (None, None, "not in batched results")
        )
        if err is not None:
            pytest.fail(f"{description}: {err}")
        assert exit_code == expected, (
            f"{description}: expected {expected}, got {exit_code}"
        )


class TestSuite1096PureNeuralStatistics:
    """Sanity checks on the parallel suite parametrization."""

    def test_suite_has_1096_tests(self):
        assert len(ALL_TESTS) == 1096, (
            f"Expected 1096 tests, got {len(ALL_TESTS)}"
        )

    def test_test_ids_unique(self):
        """Index-prefixed IDs must be globally unique."""
        seen = set()
        for tid in TEST_IDS:
            assert tid not in seen, f"duplicate test ID: {tid}"
            seen.add(tid)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
