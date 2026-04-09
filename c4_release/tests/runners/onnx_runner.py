#!/usr/bin/env python3
"""
ONNX Runtime VM Runner

Executes C4 VM programs using ONNX exported model.
Requires: onnxruntime package
"""

import sys
import os
from typing import Tuple
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.vm_runners import VMRunner
from src.compiler import compile_c


class ONNXVMRunner(VMRunner):
    """ONNX runtime VM runner."""

    def __init__(self):
        super().__init__()
        self.session = None
        self.runner = None

    def setup(self) -> bool:
        """Initialize ONNX runtime and load model."""
        try:
            # Check for onnxruntime
            try:
                import onnxruntime as ort
            except ImportError:
                print("ERROR: ONNX mode requires onnxruntime")
                print("Install: pip install onnxruntime")
                return False

            # ONNX runtime execution for C4 VM is not fully implemented
            # The codebase uses .arvm binary format, not ONNX for runtime execution
            # ONNX export exists but is used for model inspection, not execution
            print("NOTE: ONNX runtime execution not implemented")
            print("The C4 VM uses .arvm binary format for weights, not ONNX")
            print("ONNX export is available for model inspection but not execution")
            print("Use --mode fast or --mode transformer for testing")
            return False

        except Exception as e:
            print(f"ONNXVMRunner setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
        """Execute program using ONNX runtime."""
        if not self._initialized:
            raise RuntimeError("ONNXVMRunner not initialized")

        start_time = time.time()

        try:
            # Compile source to bytecode
            bytecode, data = compile_c(source)

            # Execute using ONNX runner
            result = self.runner.run(bytecode, data, max_steps=max_steps)

            elapsed = time.time() - start_time
            return result, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            raise RuntimeError(f"ONNXVMRunner execution failed: {e}") from e

    @property
    def name(self) -> str:
        return "ONNX Runtime"
