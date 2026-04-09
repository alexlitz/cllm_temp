#!/usr/bin/env python3
"""
VM Runner Abstraction Layer

Provides a unified interface for different C4 VM execution backends:
- FastVMRunner: Pure Python VM (fast, reference implementation)
- TransformerVMRunner: Neural transformer VM (full C4 model)
- ONNXVMRunner: ONNX exported model
- CRuntimeRunner: Compiled C runtime
- BundlerRunner: Bundled standalone executable

This abstraction enables testing the same programs across all backends
to ensure consistency and correctness.
"""

import sys
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from src.speculator import FastLogicalVM
from src.baked_c4 import BakedC4Transformer


class VMRunner(ABC):
    """Abstract base class for VM execution backends."""

    def __init__(self):
        """Initialize runner."""
        self._initialized = False

    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the runner.

        Returns:
            True if setup successful, False if dependencies unavailable
        """
        pass

    @abstractmethod
    def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
        """
        Compile and execute a C program.

        Args:
            source: C source code
            max_steps: Maximum VM steps before timeout

        Returns:
            Tuple of (exit_code, execution_time_seconds)

        Raises:
            RuntimeError: If execution fails
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources (optional, override if needed)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend display name."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if backend is available and initialized."""
        return self._initialized

    def __enter__(self):
        """Context manager entry."""
        if not self.setup():
            raise RuntimeError(f"{self.name} setup failed - dependencies unavailable")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class FastVMRunner(VMRunner):
    """Fast pure-Python VM runner (reference implementation)."""

    def __init__(self):
        super().__init__()
        self.vm = None

    def setup(self) -> bool:
        """Initialize FastLogicalVM."""
        try:
            self.vm = None  # Create fresh VM for each test
            self._initialized = True
            return True
        except Exception as e:
            print(f"FastVMRunner setup failed: {e}")
            return False

    def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
        """Execute program using FastLogicalVM."""
        if not self._initialized:
            raise RuntimeError("FastVMRunner not initialized")

        start_time = time.time()

        try:
            # Compile source to bytecode
            bytecode, data = compile_c(source)

            # Create fresh VM and execute
            vm = FastLogicalVM()
            vm.load(bytecode, data)
            result = vm.run(max_steps=max_steps)

            elapsed = time.time() - start_time
            return result, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            raise RuntimeError(f"FastVMRunner execution failed: {e}") from e

    @property
    def name(self) -> str:
        return "FastLogicalVM"


class TransformerVMRunner(VMRunner):
    """Neural transformer VM runner (full C4 model)."""

    def __init__(self):
        super().__init__()
        self.c4 = None

    def setup(self) -> bool:
        """Initialize BakedC4Transformer."""
        try:
            print("Loading BakedC4Transformer (this may take 1-2 minutes)...")
            self.c4 = BakedC4Transformer(use_speculator=True)
            self._initialized = True
            print("BakedC4Transformer loaded successfully")
            return True
        except Exception as e:
            print(f"TransformerVMRunner setup failed: {e}")
            return False

    def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
        """Execute program using BakedC4Transformer."""
        if not self._initialized:
            raise RuntimeError("TransformerVMRunner not initialized")

        start_time = time.time()

        try:
            result = self.c4.run_c(source, max_steps=max_steps)
            elapsed = time.time() - start_time
            return result, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            raise RuntimeError(f"TransformerVMRunner execution failed: {e}") from e

    def cleanup(self) -> None:
        """Cleanup transformer resources."""
        if self.c4 is not None:
            # Free GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def name(self) -> str:
        return "BakedC4Transformer"


# Import concrete implementations from tests/runners/
try:
    from tests.runners.onnx_runner import ONNXVMRunner
except ImportError:
    # Fallback if runners module not available
    class ONNXVMRunner(VMRunner):
        def setup(self) -> bool:
            print("ONNXVMRunner not available (import failed)")
            return False
        def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
            raise NotImplementedError("ONNXVMRunner not available")
        @property
        def name(self) -> str:
            return "ONNX Runtime"

try:
    from tests.runners.c_runtime_runner import CRuntimeRunner
except ImportError:
    class CRuntimeRunner(VMRunner):
        def setup(self) -> bool:
            print("CRuntimeRunner not available (import failed)")
            return False
        def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
            raise NotImplementedError("CRuntimeRunner not available")
        @property
        def name(self) -> str:
            return "C Runtime"

try:
    from tests.runners.bundler_runner import BundlerRunner
except ImportError:
    class BundlerRunner(VMRunner):
        def setup(self) -> bool:
            print("BundlerRunner not available (import failed)")
            return False
        def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
            raise NotImplementedError("BundlerRunner not available")
        @property
        def name(self) -> str:
            return "Bundler"


# Factory function for creating runners by mode name
def create_runner(mode: str) -> VMRunner:
    """
    Create a VMRunner instance by mode name.

    Args:
        mode: One of 'fast', 'transformer', 'onnx', 'c-runtime', 'bundler'

    Returns:
        VMRunner instance

    Raises:
        ValueError: If mode is unknown
    """
    runners = {
        'fast': FastVMRunner,
        'transformer': TransformerVMRunner,
        'onnx': ONNXVMRunner,
        'c-runtime': CRuntimeRunner,
        'bundler': BundlerRunner,
    }

    if mode not in runners:
        available = ', '.join(runners.keys())
        raise ValueError(f"Unknown mode '{mode}'. Available modes: {available}")

    return runners[mode]()
