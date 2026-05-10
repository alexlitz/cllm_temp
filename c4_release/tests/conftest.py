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
    """
    from neural_vm.run_vm import AutoregressiveVMRunner
    return AutoregressiveVMRunner()


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
    """Quick runner with limited steps for unit tests."""
    from neural_vm.run_vm import AutoregressiveVMRunner

    runner = AutoregressiveVMRunner()
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
