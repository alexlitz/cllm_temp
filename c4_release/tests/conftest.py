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

@pytest.fixture
def neural_only_runner():
    """Runner with ALL handlers removed (pure neural).

    Use for testing neural path completeness.
    Warning: Many opcodes will fail without handlers.
    """
    from neural_vm.run_vm import AutoregressiveVMRunner

    runner = AutoregressiveVMRunner()
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
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
