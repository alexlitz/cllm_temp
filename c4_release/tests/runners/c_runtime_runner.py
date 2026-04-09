#!/usr/bin/env python3
"""
C Runtime VM Runner

Compiles C runtime and executes C4 VM programs.
Requires: gcc
"""

import sys
import os
from typing import Tuple
import time
import subprocess
import tempfile
import hashlib
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.vm_runners import VMRunner
from src.compiler import compile_c


class CRuntimeRunner(VMRunner):
    """Compiled C runtime VM runner."""

    def __init__(self, runtime_name='neural_runtime'):
        super().__init__()
        self.runtime_name = runtime_name
        self.runtime_binary = None
        self.cache_dir = os.path.expanduser('~/.cache/c4_test_runner/c_runtimes')
        self.temp_dir = None

    def _check_gcc(self) -> bool:
        """Check if gcc is available."""
        try:
            result = subprocess.run(['gcc', '--version'],
                                    capture_output=True,
                                    timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_runtime_path(self) -> str:
        """Get path to C runtime source file."""
        candidates = [
            f'vm/{self.runtime_name}.c',
            f'bundler/{self.runtime_name}.c',
        ]

        base_dir = os.path.join(os.path.dirname(__file__), '../..')

        for candidate in candidates:
            path = os.path.join(base_dir, candidate)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"C runtime not found: {self.runtime_name}.c")

    def _hash_runtime(self, runtime_path: str) -> str:
        """Compute hash of runtime source for caching."""
        with open(runtime_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]

    def _compile_runtime(self, runtime_path: str) -> str:
        """Compile C runtime and return path to binary."""
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check cache
        runtime_hash = self._hash_runtime(runtime_path)
        cached_binary = os.path.join(self.cache_dir, f'{self.runtime_name}_{runtime_hash}')

        if os.path.exists(cached_binary):
            print(f"Using cached runtime: {cached_binary}")
            return cached_binary

        # Compile runtime
        print(f"Compiling C runtime: {runtime_path}...")
        start_time = time.time()

        try:
            result = subprocess.run([
                'gcc',
                '-O2',
                '-o', cached_binary,
                runtime_path,
                '-lm'
            ], capture_output=True, timeout=60, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed:\n{result.stderr}")

            elapsed = time.time() - start_time
            print(f"Runtime compiled successfully in {elapsed:.1f}s")
            return cached_binary

        except subprocess.TimeoutExpired:
            raise RuntimeError("Compilation timeout (60s)")

    def setup(self) -> bool:
        """Initialize C runtime."""
        try:
            # Check for gcc
            if not self._check_gcc():
                print("ERROR: C runtime mode requires gcc")
                print("Install: apt-get install gcc  OR  brew install gcc")
                return False

            # Find runtime source
            try:
                runtime_path = self._get_runtime_path()
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                return False

            # Compile runtime
            try:
                self.runtime_binary = self._compile_runtime(runtime_path)
            except RuntimeError as e:
                print(f"ERROR: {e}")
                return False

            # Create temp directory for test execution
            self.temp_dir = tempfile.mkdtemp(prefix='c4_test_')

            self._initialized = True
            return True

        except Exception as e:
            print(f"CRuntimeRunner setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
        """Execute program using C runtime."""
        if not self._initialized:
            raise RuntimeError("CRuntimeRunner not initialized")

        start_time = time.time()

        try:
            # Compile source to bytecode
            bytecode, data = compile_c(source)

            # Write bytecode to temp file
            bytecode_file = os.path.join(self.temp_dir, 'bytecode.dat')
            with open(bytecode_file, 'wb') as f:
                # Write bytecode as binary
                for instr in bytecode:
                    f.write(instr.to_bytes(8, 'little', signed=False))

            # Write data to temp file
            data_file = os.path.join(self.temp_dir, 'data.dat')
            with open(data_file, 'wb') as f:
                f.write(data)

            # Execute runtime
            result = subprocess.run([
                self.runtime_binary,
                bytecode_file,
                data_file
            ], capture_output=True, timeout=30, text=True)

            # Parse result from stdout
            try:
                # Expected format: "Exit code: <n>"
                output = result.stdout.strip()
                if 'Exit code:' in output:
                    exit_code = int(output.split('Exit code:')[1].strip())
                else:
                    # Try to parse last line as integer
                    exit_code = int(output.split('\n')[-1])
            except (ValueError, IndexError):
                raise RuntimeError(f"Failed to parse output: {result.stdout}")

            elapsed = time.time() - start_time
            return exit_code, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            raise RuntimeError("Execution timeout (30s)")
        except Exception as e:
            elapsed = time.time() - start_time
            raise RuntimeError(f"CRuntimeRunner execution failed: {e}") from e

    def cleanup(self) -> None:
        """Cleanup temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @property
    def name(self) -> str:
        return f"C Runtime ({self.runtime_name})"
