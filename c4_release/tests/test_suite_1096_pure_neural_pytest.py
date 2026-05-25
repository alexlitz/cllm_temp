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
    C4_BATCH_MAX_STEPS — max VM steps per program. Default "auto" uses
                       declarative execution to derive each program's exact
                       halt horizon and fails neural overruns as divergence.
    C4_BATCH_CONTEXT_WINDOW — dynamic-token context window retained after
                       the immutable program/data prefix (default 512)
    C4_BATCH_MODEL_MAX_SEQ_LEN — compiled model maximum sequence length
                       (default 4096)
    C4_BATCH_USE_KV_CACHE — enable current-path batched KV reuse when set to 1
    C4_BATCH_KV_VERIFY — KV correctness verifier. Default 0 for throughput.
                       Set 1/full for exhaustive cached-vs-fresh checks,
                       or sample/periodic/N to verify every N KV calls.
    C4_BATCH_KV_MAX_TOKENS — cache window before eviction pressure forces a
                       fresh correctness path (default: model max seq)
    C4_BATCH_ENABLE_MOE_ROUTING / C4_ENABLE_MOE_ROUTING — request a
                       compiler-emitted top-1 MoE model for the batched path
    C4_SPEC_K        — DraftVM speculation horizon in VM steps (default
                       adaptive; 0 disables speculation and falls back to
                       one-token-per-forward batched decode)
    C4_ADAPTIVE_START_K / C4_ADAPTIVE_MAX_K — adaptive speculation start and
                       cap when C4_SPEC_K=adaptive (defaults 32 / 64)
    C4_SPEC_FAIL_FAST — stop an element after persistent first-token DraftVM
                       disagreement (default 1)
    C4_1096_OFFSET / C4_1096_LIMIT — run a contiguous program slice after
                       pytest selection is applied. Useful for quick tuning.
    C4_1096_TIMING   — print per-chunk compile/run timing when set to 1.

Usage:
    # Run the whole suite (batched):
    pytest tests/test_suite_1096_pure_neural_pytest.py -v --tb=line

    # Run a tiny subset (fixture honors pytest selection):
    pytest tests/test_suite_1096_pure_neural_pytest.py -k "test_program_0" \\
        --tb=line

    # Verify collection only:
    pytest tests/test_suite_1096_pure_neural_pytest.py --collect-only

Date: 2026-05-11 (batched wiring)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tests.test_suite_1000 import generate_test_programs
from tests.declarative_oracle import declarative_oracle_for_program
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
_BATCH_MAX_STEPS_RAW = os.environ.get("C4_BATCH_MAX_STEPS", "auto").strip().lower()
_BATCH_MAX_STEPS = (
    None
    if _BATCH_MAX_STEPS_RAW in {"", "0", "auto", "none", "declarative"}
    else int(_BATCH_MAX_STEPS_RAW)
)
_BATCH_CONTEXT_WINDOW = int(os.environ.get("C4_BATCH_CONTEXT_WINDOW", "512"))
_BATCH_MODEL_MAX_SEQ_LEN = int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096"))
_SUBSET_OFFSET = int(os.environ.get("C4_1096_OFFSET", "0"))
_SUBSET_LIMIT_RAW = os.environ.get("C4_1096_LIMIT")
_SUBSET_LIMIT = int(_SUBSET_LIMIT_RAW) if _SUBSET_LIMIT_RAW else None
_TIMING = os.environ.get("C4_1096_TIMING") == "1"
_EMPTY_CUDA_CACHE = os.environ.get(
    "C4_1096_EMPTY_CUDA_CACHE", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def _maybe_empty_cuda_cache():
    if not _EMPTY_CUDA_CACHE:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _env_sliced_programs():
    selected = list(enumerate(zip(TEST_IDS, ALL_TESTS)))
    if _SUBSET_OFFSET:
        selected = selected[_SUBSET_OFFSET:]
    if _SUBSET_LIMIT is not None:
        selected = selected[:_SUBSET_LIMIT]
    return [
        (idx, tid, test_tuple)
        for idx, (tid, test_tuple) in selected
    ]


_PARAM_SELECTED = _env_sliced_programs()
_PARAM_TEST_IDS = [tid for _idx, tid, _test_tuple in _PARAM_SELECTED]
_PARAM_TESTS = [test_tuple for _idx, _tid, test_tuple in _PARAM_SELECTED]


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


def _selected_programs(request):
    """Return ``[(global_idx, test_id, test_tuple), ...]`` for this run.

    Pytest ``-k``/nodeid selection happens before fixtures execute. Honoring
    that selection here lets focused runs pay only focused compile/run cost
    instead of always executing all 1096 programs.
    """
    selected_ids = {
        item.callspec.id
        for item in request.session.items
        if getattr(item, "callspec", None) is not None
        and getattr(item, "originalname", item.name) == "test_program"
    }
    if not selected_ids:
        selected_ids = set(_PARAM_TEST_IDS)

    selected = [
        (idx, tid, test_tuple)
        for idx, tid, test_tuple in _PARAM_SELECTED
        if tid in selected_ids
    ]
    return selected


def _declarative_expected_result(bytecode, data, *, suite_expected=None, label="program"):
    """Return the declarative result for this program.

    The pure-neural suite should not use an arbitrary max-step cap. For
    compiled finite test programs, the declarative program runner gives the
    expected halt horizon and final result. The neural run must match both the
    declaration-derived result and the declaration-derived halt horizon.
    """
    return declarative_oracle_for_program(
        bytecode,
        data,
        suite_expected=suite_expected,
        label=label,
        max_steps=None,
    ).as_dict()


@pytest.fixture(scope="session")
def _pure_neural_1096_results(request):
    """Compile all 1096 programs and run them in batched chunks once per
    pytest session. Returns a dict ``{test_id: (output, exit_code, err)}``.

    ``err`` is None on success; on per-program failure it's a string. We
    catch per-chunk exceptions so a single bad program doesn't poison the
    entire session.
    """
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    runner = BatchedPureNeuralRunner(max_seq_len=_BATCH_MODEL_MAX_SEQ_LEN)

    selected = _selected_programs(request)
    if _TIMING:
        print(
            f"[1096] selected={len(selected)} chunk={_BATCH_CHUNK} "
            f"max_steps={_BATCH_MAX_STEPS if _BATCH_MAX_STEPS is not None else 'declarative'} "
            f"ctx_window={_BATCH_CONTEXT_WINDOW} "
            f"model_max_seq={_BATCH_MODEL_MAX_SEQ_LEN} spec_k={_SPEC_K} "
            f"kv={runner.use_kv_cache} kv_verify={runner.kv_cache_verify} "
            f"kv_verify_interval={runner.kv_cache_verify_interval} "
            f"moe={runner.enable_moe_routing} "
            f"spec_fail_fast={runner.spec_fail_fast} "
            f"empty_cuda_cache={_EMPTY_CUDA_CACHE}",
            file=sys.stderr,
            flush=True,
        )

    results = {}
    for chunk_start in range(0, len(selected), _BATCH_CHUNK):
        chunk_meta = selected[chunk_start:chunk_start + _BATCH_CHUNK]
        chunk = [test_tuple for _idx, _tid, test_tuple in chunk_meta]
        chunk_ids = [tid for _idx, tid, _test_tuple in chunk_meta]
        chunk_global_idxs = [idx for idx, _tid, _test_tuple in chunk_meta]

        bytecodes = []
        data_list = []
        declarative_results = []
        expected_steps = []
        compile_errs = {}
        t_compile0 = time.perf_counter()
        for j, (source, expected, _desc) in enumerate(chunk):
            try:
                bc, data = compile_c(source)
                decl = _declarative_expected_result(
                    bc,
                    data,
                    suite_expected=expected,
                    label=chunk_ids[j],
                )
                declarative_results.append(decl)
                if decl.get("error") is not None:
                    compile_errs[j] = decl["error"]
                    bytecodes.append([38])  # EXIT placeholder; compile_err overrides.
                    data_list.append(b"")
                    expected_steps.append(1 if _BATCH_MAX_STEPS is None else None)
                    continue
                bytecodes.append(bc)
                data_list.append(data)
                expected_steps.append(
                    decl["steps"] if _BATCH_MAX_STEPS is None else None
                )
            except Exception as e:
                compile_errs[j] = f"compile/declarative error: {e!r}"
                bytecodes.append([38])  # EXIT placeholder; compile_err overrides.
                data_list.append(b"")
                declarative_results.append({
                    "output": "",
                    "exit_code": None,
                    "steps": 1,
                    "halted": False,
                    "error": compile_errs[j],
                })
                expected_steps.append(1 if _BATCH_MAX_STEPS is None else None)
        compile_elapsed = time.perf_counter() - t_compile0

        _maybe_empty_cuda_cache()
        t_run0 = time.perf_counter()
        try:
            batch_results = runner.run_batch(
                bytecodes,
                data_list=data_list,
                max_steps=_BATCH_MAX_STEPS,
                max_context_window=_BATCH_CONTEXT_WINDOW,
                spec_k=_SPEC_K,
                expected_steps_list=expected_steps,
            )
        except Exception as e:
            err = f"batch run error: {e!r}"
            for j, tid in enumerate(chunk_ids):
                results[tid] = {
                    "output": "",
                    "exit_code": None,
                    "err": compile_errs.get(j, err),
                    "declarative": declarative_results[j],
                    "suite_expected": chunk[j][1],
                }
            if _TIMING:
                run_elapsed = time.perf_counter() - t_run0
                first_idx = chunk_global_idxs[0] if chunk_global_idxs else -1
                last_idx = chunk_global_idxs[-1] if chunk_global_idxs else -1
                print(
                    f"[1096] chunk {chunk_start // _BATCH_CHUNK:03d} "
                    f"ids={first_idx:04d}-{last_idx:04d} "
                    f"N={len(chunk):3d} compile={compile_elapsed:7.2f}s "
                    f"decl_steps={max((s or 0) for s in expected_steps):4d} "
                    f"run={run_elapsed:7.2f}s ERROR {err}",
                    file=sys.stderr,
                    flush=True,
                )
            _maybe_empty_cuda_cache()
            continue
        run_elapsed = time.perf_counter() - t_run0

        for j, tid in enumerate(chunk_ids):
            if j in compile_errs:
                results[tid] = {
                    "output": "",
                    "exit_code": None,
                    "err": compile_errs[j],
                    "declarative": declarative_results[j],
                    "suite_expected": chunk[j][1],
                }
            else:
                output, exit_code = batch_results[j]
                results[tid] = {
                    "output": output,
                    "exit_code": exit_code,
                    "err": None,
                    "declarative": declarative_results[j],
                    "suite_expected": chunk[j][1],
                }

        if _TIMING:
            ok = 0
            fail = 0
            err_count = 0
            examples = []
            for j, (_source, expected, _desc) in enumerate(chunk):
                result = results[chunk_ids[j]]
                exit_code = result["exit_code"]
                err = result["err"]
                decl = result["declarative"]
                if err is not None:
                    err_count += 1
                    if len(examples) < 5:
                        examples.append(f"{chunk_ids[j]} err={err}")
                elif exit_code == decl["exit_code"]:
                    ok += 1
                else:
                    fail += 1
                    if len(examples) < 5:
                        examples.append(
                            f"{chunk_ids[j]} expected={expected} "
                            f"decl={decl['exit_code']} got={exit_code} "
                            f"decl_steps={decl['steps']}"
                        )
            first_idx = chunk_global_idxs[0] if chunk_global_idxs else -1
            last_idx = chunk_global_idxs[-1] if chunk_global_idxs else -1
            print(
                f"[1096] chunk {chunk_start // _BATCH_CHUNK:03d} "
                f"ids={first_idx:04d}-{last_idx:04d} N={len(chunk):3d} "
                f"compile={compile_elapsed:7.2f}s "
                f"decl_steps={max((s or 0) for s in expected_steps):4d} "
                f"run={run_elapsed:7.2f}s "
                f"ok={ok} fail={fail} err={err_count} "
                f"kv_stats={runner._kv_stats} "
                f"spec_stats={runner._spec_stats}",
                file=sys.stderr,
                flush=True,
            )
            if examples:
                print(
                    "[1096] examples: " + "; ".join(examples),
                    file=sys.stderr,
                    flush=True,
                )
        _maybe_empty_cuda_cache()

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
        _PARAM_TESTS,
        ids=_PARAM_TEST_IDS
    )
    def test_program(self, _pure_neural_1096_results, request, source,
                     expected, description):
        """Assert this program's batched result matches expected."""
        tid = request.node.callspec.id
        result = _pure_neural_1096_results.get(
            tid,
            {
                "output": None,
                "exit_code": None,
                "err": "not in batched results",
                "declarative": {"exit_code": None, "steps": None, "error": None},
                "suite_expected": expected,
            },
        )
        exit_code = result["exit_code"]
        err = result["err"]
        decl = result["declarative"]
        if err is not None:
            pytest.fail(f"{description}: {err}")
        assert decl["exit_code"] == (expected & 0xFFFFFFFF), (
            f"{description}: suite expected {expected}, "
            f"declarative got {decl['exit_code']}"
        )
        assert exit_code == decl["exit_code"], (
            f"{description}: declarative expected {decl['exit_code']} "
            f"after {decl['steps']} steps, neural got {exit_code}"
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
