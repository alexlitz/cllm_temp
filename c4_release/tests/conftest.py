"""
Pytest Configuration for C4 Neural VM Tests

Provides fixtures for fast testing with GPU and speculative decoding.
"""

import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# torch.compile mode helper
# =============================================================================

def _resolve_compile_mode(pure_neural: bool) -> str:
    """Resolve the ``compile_mode`` to pass to ``AutoregressiveVMRunner``.

    Reads ``C4_COMPILE_MODE`` from the environment. When unset, defaults to
    ``"none"`` (opt-in compile only) — first-run torch.compile warmup is
    >8 minutes on the current production model (see
    ``c4_release/docs/TORCH_COMPILE_BENCHMARK.md``). The legacy KV-cache
    byte-identity tests also depend on the un-compiled incremental forward
    path, which ``compile_mode != "none"`` disables.

    Accepted values match ``torch.compile(mode=...)`` plus ``"none"`` /
    ``"None"`` to opt out explicitly. Tests/CI can opt into compile by
    setting ``C4_COMPILE_MODE=reduce-overhead`` (or ``max-autotune``).
    """
    # Default stays "none" pending compile-warmup improvements (see
    # TORCH_COMPILE_BENCHMARK.md). ``pure_neural`` is accepted for forward
    # compatibility so callers don't need to change once the default flips.
    del pure_neural  # currently unused; kept for forward compat
    mode = os.environ.get("C4_COMPILE_MODE", "none")
    if mode in ("None", "none", ""):
        return "none"
    return mode


# =============================================================================
# GPU Detection
# =============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
HAS_GPU = DEVICE.type in ("cuda", "mps")


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "neural: marks tests that test pure neural execution")
    config.addinivalue_line("markers", "handler: marks tests that use handlers")
    config.addinivalue_line("markers", "quine: marks quine-specific tests")
    config.addinivalue_line("markers", "bundler: marks bundler tests")
    config.addinivalue_line("markers", "dual: marks tests that run with both weight modes")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if no GPU available."""
    if not HAS_GPU:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# =============================================================================
# Fixtures - Fast Runners
# =============================================================================

@pytest.fixture(scope="session")
def device():
    """Get the compute device."""
    return DEVICE


@pytest.fixture(scope="session")
def fast_runner():
    """Fast runner using speculative decoding + KV cache.

    ~35x faster than pure autoregressive.
    Session-scoped to reuse model across tests.
    """
    from neural_vm.batch_runner import BatchedSpeculativeRunner

    runner = BatchedSpeculativeRunner(
        use_kv_cache=True,
        kv_cache_max_tokens=64,
        device=DEVICE,
    )
    return runner


@pytest.fixture(scope="session")
def neural_runner():
    """Pure neural runner (no speculative decoding).

    Use for testing neural correctness.

    PHASE 8 (2026-05-11): Default flipped to pure_neural=True,
    trust_neural_alu=True. Many tests will fail until Phases 1-7 complete;
    that is the intended signal for what the neural path still cannot do.

    torch.compile is OFF by default (first-run compile is >8 min on this
    model). Opt in via ``C4_COMPILE_MODE=reduce-overhead`` (or another
    ``torch.compile(mode=...)`` value). See
    ``c4_release/docs/TORCH_COMPILE_BENCHMARK.md`` for measurements.
    """
    from neural_vm.run_vm import AutoregressiveVMRunner
    compile_mode = _resolve_compile_mode(pure_neural=True)
    return AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        compile_mode=compile_mode,
    )


@pytest.fixture(scope="session")
def batch_runner():
    """Batch runner for parallel test execution.

    Can run 256 programs in parallel.
    """
    from neural_vm.ultra_batch import UltraBatchRunner

    batch_size = 64 if HAS_GPU else 8
    return UltraBatchRunner(batch_size=batch_size, device=DEVICE)


# =============================================================================
# Fixtures - Quick Runners (Limited Steps)
# =============================================================================

@pytest.fixture
def quick_runner():
    """Quick runner with limited steps for unit tests.

    PHASE 8 (2026-05-11): Default flipped to pure_neural=True,
    trust_neural_alu=True. The headline smoke suite now exercises the neural
    path with Python overrides disabled. Tests that the neural network cannot
    yet handle (most of Phases 2-7) will fail honestly; that is the new CI
    signal driving Phases 1-7 to completion.

    torch.compile is OFF by default (first-run compile is >8 min on this
    model). Opt in via ``C4_COMPILE_MODE=reduce-overhead`` (or another
    ``torch.compile(mode=...)`` value). See
    ``c4_release/docs/TORCH_COMPILE_BENCHMARK.md`` for measurements.
    """
    from neural_vm.run_vm import AutoregressiveVMRunner

    compile_mode = _resolve_compile_mode(pure_neural=True)
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        compile_mode=compile_mode,
    )
    # Limit max_steps for quick tests
    return runner


@pytest.fixture
def smoke_runner():
    """Minimal runner for smoke tests (max 20 steps)."""
    from neural_vm.batch_runner import BatchedSpeculativeRunner

    return BatchedSpeculativeRunner(
        use_kv_cache=True,
        kv_cache_max_tokens=32,
        device=DEVICE,
    )


# =============================================================================
# Fixtures - Test Helpers
# =============================================================================

@pytest.fixture
def make_bytecode():
    """Helper to create bytecode from operation list."""
    def _make_bytecode(ops):
        bytecode = []
        for op in ops:
            if isinstance(op, tuple):
                opcode, imm = op
                bytecode.append(opcode | (imm << 8))
            else:
                bytecode.append(op)
        return bytecode
    return _make_bytecode


@pytest.fixture
def opcodes():
    """Provide opcode constants."""
    from neural_vm.embedding import Opcode
    return Opcode


# =============================================================================
# Fixtures - Neural vs Handler Testing
# =============================================================================

@pytest.fixture(scope="session")
def _neural_only_runner_model():
    """Session-scoped model construction for neural-only tests."""
    from neural_vm.run_vm import AutoregressiveVMRunner

    runner = AutoregressiveVMRunner(trust_neural_alu=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    return runner


@pytest.fixture(scope="session")
def _pure_neural_runner_model():
    """Session-scoped model construction for pure neural tests (no Python overrides)."""
    from neural_vm.run_vm import AutoregressiveVMRunner

    runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    return runner


@pytest.fixture
def pure_neural_runner(_pure_neural_runner_model):
    """Runner with NO Python overrides at all (100% neural).

    Use for testing that neural network alone can execute programs.
    This skips ALL dispatch logic - only neural network outputs matter.
    Resets per-call state on the session-scoped runner.
    """
    runner = _pure_neural_runner_model
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    return runner


@pytest.fixture
def neural_only_runner(_neural_only_runner_model):
    """Runner with ALL handlers removed (pure neural).

    Use for testing neural path completeness.
    Warning: Many opcodes will fail without handlers.
    Resets per-call state on the session-scoped runner.
    """
    runner = _neural_only_runner_model
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    return runner


@pytest.fixture
def neural_only_runner(_neural_only_runner_model):
    """Runner with ALL handlers removed (pure neural).

    Use for testing neural path completeness.
    Warning: Many opcodes will fail without handlers.
    Resets per-call state on the session-scoped runner.
    """
    runner = _neural_only_runner_model
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    return runner


# =============================================================================
# Pure-neural fixture (stable across worktrees)
# =============================================================================
#
# The `pure_neural_runner` fixture below is the canonical entry point for any
# test that needs to verify behavior with NO Python overrides — the neural
# network alone drives execution.
#
# Design notes:
#   * Session-scoped model build (`_pure_neural_runner_model`):
#       The model bake is expensive (~6.8s cold). Building it once per pytest
#       session and reusing the same `AutoregressiveVMRunner` keeps test time
#       reasonable across long suites. Per-test isolation is provided by the
#       function-scoped wrapper below, which clears mutable runner state.
#   * Per-test state reset (function-scoped `pure_neural_runner`):
#       Before yielding the runner to a test, the wrapper resets:
#           - `_memory`            (sparse byte-addressable shadow memory)
#           - `_mem_history`       (token sequences for memory addresses)
#           - `_mem_access_order`  (LRU eviction tracking)
#       This prevents one test's stack/heap writes from leaking into the next.
#       The runner's `_func_call_handlers` and `_syscall_handlers` are also
#       cleared once at session-build time so dispatch is 100% neural.
#   * Kwargs used:
#       `AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)`.
#       Older worktrees may not yet have one or both kwargs. The session
#       builder catches the resulting `TypeError`, emits a `RuntimeWarning`,
#       and falls back to whatever combination the runner accepts so the
#       fixture can still construct *some* runner. Tests that strictly need
#       pure_neural behavior should branch on `runner.pure_neural` and call
#       `pytest.skip(...)` when it is False (see `test_pure_neural_fixture.py`
#       for the canonical pattern).

@pytest.fixture(scope="session")
def _pure_neural_runner_model():
    """Session-scoped model construction for pure neural tests.

    Tries `AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)`
    first. If either kwarg is unsupported on the current branch (older
    worktrees), falls back to whatever combination the runner accepts and
    emits a warning so the test author can decide whether to skip.
    """
    import warnings
    from neural_vm.run_vm import AutoregressiveVMRunner

    kwargs_attempts = [
        {"pure_neural": True, "trust_neural_alu": True},
        {"trust_neural_alu": True},   # branch missing pure_neural kwarg
        {"pure_neural": True},         # branch missing trust_neural_alu kwarg
        {},                             # branch missing both kwargs
    ]
    runner = None
    last_err = None
    for kwargs in kwargs_attempts:
        try:
            runner = AutoregressiveVMRunner(**kwargs)
            if kwargs != kwargs_attempts[0]:
                warnings.warn(
                    f"pure_neural_runner: AutoregressiveVMRunner does not accept "
                    f"the full kwargs {kwargs_attempts[0]!r}. "
                    f"Fell back to {kwargs!r}. Tests that strictly require "
                    f"pure_neural=True should check runner.pure_neural and "
                    f"pytest.skip if False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            break
        except TypeError as e:
            last_err = e
            continue
    if runner is None:
        raise RuntimeError(
            f"pure_neural_runner: AutoregressiveVMRunner could not be "
            f"instantiated with any tried kwargs. Last error: {last_err!r}"
        )

    # Strip Python-side handlers so dispatch is fully neural (or as close as
    # the current branch supports). Done once at session build time.
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    return runner


@pytest.fixture
def pure_neural_runner(_pure_neural_runner_model):
    """Runner with NO Python overrides at all (100% neural when supported).

    Session-scoped model is shared via `_pure_neural_runner_model`; this
    function-scoped wrapper resets per-test state (`_memory`, `_mem_history`,
    `_mem_access_order`) so individual tests do not leak state into each other.

    Use for testing that the neural network alone can execute programs.
    See the design-notes block above the session fixture for details on
    kwargs fallback behavior on older worktrees.
    """
    runner = _pure_neural_runner_model
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    return runner


# =============================================================================
# Batched pure-neural runner (optional, additive fixture)
# =============================================================================
#
# `BatchedPureNeuralRunner` runs N programs through one batched forward pass
# per token instead of N serial passes. Tests that want to opt in can take the
# `batched_pure_neural_runner` fixture below; it shares the SAME compiled model
# as `_pure_neural_runner_model` (constructed once at session build time), so
# fixture cost is amortized identically.
#
# Equivalence: every batch element must produce the same `(output, exit_code)`
# as the serial runner running that program alone. See `test_batched_pure_neural`
# for element-by-element verification on a small batch.
#
# Existing tests are unchanged — this is purely additive. Future refactors can
# group N tests into a single batched call to cut wall-clock for phases with
# many uniformly-sized tests (e.g. Phase 1 PC arithmetic, Phase 7 heap+DIV).

@pytest.fixture(scope="session")
def _batched_pure_neural_runner_model(_pure_neural_runner_model):
    """Session-scoped BatchedPureNeuralRunner sharing the pure_neural model.

    Wraps the same compiled AutoregressiveVM as `pure_neural_runner` so the
    expensive bake (`compile_full_vm` + `set_vm_weights`) happens once.
    """
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    return BatchedPureNeuralRunner(model_runner=_pure_neural_runner_model)


@pytest.fixture
def batched_pure_neural_runner(_batched_pure_neural_runner_model):
    """Batched pure-neural runner; opt-in alternative to `pure_neural_runner`.

    Use when a test (or test group) has multiple bytecodes that can run in one
    batched forward pass — e.g. parameterized over immediate values, or a
    "phase" gate that runs ~10-25 similar programs. Call:

        results = batched_pure_neural_runner.run_batch(bytecodes, max_steps=N)

    Returns a list of `(output_string, exit_code)` tuples in input order.

    For single-program tests, `pure_neural_runner` is still simpler and just
    as fast since batch=1 collapses to one forward per token anyway.
    """
    return _batched_pure_neural_runner_model


@pytest.fixture
def handler_status():
    """Get current handler registration status."""
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.embedding import Opcode

    runner = AutoregressiveVMRunner()

    all_ops = [
        ("LEA", Opcode.LEA), ("IMM", Opcode.IMM), ("JMP", Opcode.JMP),
        ("JSR", Opcode.JSR), ("BZ", Opcode.BZ), ("BNZ", Opcode.BNZ),
        ("ENT", Opcode.ENT), ("ADJ", Opcode.ADJ), ("LEV", Opcode.LEV),
        ("LI", Opcode.LI), ("LC", Opcode.LC), ("SI", Opcode.SI),
        ("SC", Opcode.SC), ("PSH", Opcode.PSH),
        ("OR", Opcode.OR), ("XOR", Opcode.XOR), ("AND", Opcode.AND),
        ("EQ", Opcode.EQ), ("NE", Opcode.NE), ("LT", Opcode.LT),
        ("GT", Opcode.GT), ("LE", Opcode.LE), ("GE", Opcode.GE),
        ("SHL", Opcode.SHL), ("SHR", Opcode.SHR),
        ("ADD", Opcode.ADD), ("SUB", Opcode.SUB), ("MUL", Opcode.MUL),
        ("DIV", Opcode.DIV), ("MOD", Opcode.MOD),
        ("OPEN", Opcode.OPEN), ("READ", Opcode.READ), ("CLOS", Opcode.CLOS),
        ("PRTF", Opcode.PRTF), ("MALC", Opcode.MALC), ("FREE", Opcode.FREE),
        ("MSET", Opcode.MSET), ("MCMP", Opcode.MCMP), ("EXIT", Opcode.EXIT),
        ("NOP", Opcode.NOP),
    ]

    status = {}
    for name, op in all_ops:
        has_func = op in runner._func_call_handlers
        has_sys = op in runner._syscall_handlers
        status[name] = {
            "opcode": op,
            "has_handler": has_func or has_sys,
            "handler_type": "func" if has_func else ("syscall" if has_sys else "neural"),
        }

    return status


# =============================================================================
# Fixtures - Weight Mode Testing
# =============================================================================

def get_available_weight_modes():
    """Get weight modes that are currently working."""
    from neural_vm.weight_setter import WeightMode, set_weights
    from neural_vm.vm_step import AutoregressiveVM

    modes = [WeightMode.HAND_SET]

    # Test if compiled weights work
    try:
        test_model = AutoregressiveVM(
            d_model=512, n_layers=16, n_heads=8, ffn_hidden=4096
        )
        set_weights(test_model, mode=WeightMode.COMPILED, verify_purity=False)
        modes.append(WeightMode.COMPILED)
    except Exception:
        pass  # COMPILED not yet working

    return modes


AVAILABLE_WEIGHT_MODES = get_available_weight_modes()


@pytest.fixture(params=AVAILABLE_WEIGHT_MODES, ids=[m.value for m in AVAILABLE_WEIGHT_MODES])
def weight_mode(request):
    """Parametrized fixture for available weight modes.

    Use this fixture to run tests with all working weight modes.
    Currently only HAND_SET works; COMPILED will be added when ready.
    """
    return request.param


@pytest.fixture
def compile_program():
    """Fixture to compile C programs."""
    from src.compiler import compile_c
    return compile_c


# =============================================================================
# Session Info
# =============================================================================

def pytest_report_header(config):
    """Add device info to pytest header."""
    from neural_vm.weight_setter import WeightMode

    compiled_status = "Available" if WeightMode.COMPILED in AVAILABLE_WEIGHT_MODES else "Not implemented"

    return [
        f"Device: {DEVICE}",
        f"GPU Available: {HAS_GPU}",
        f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
        f"Compiled Weights: {compiled_status}",
    ]
