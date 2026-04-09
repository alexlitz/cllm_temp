"""
VM Runner implementations for different execution backends.

Available runners:
- ONNXVMRunner: ONNX runtime execution
- CRuntimeRunner: Compiled C runtime
- BundlerRunner: Bundled standalone executable
"""

from .onnx_runner import ONNXVMRunner
from .c_runtime_runner import CRuntimeRunner
from .bundler_runner import BundlerRunner

__all__ = ['ONNXVMRunner', 'CRuntimeRunner', 'BundlerRunner']
