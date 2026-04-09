#!/usr/bin/env python3
"""
Bundler VM Runner

Bundles model + bytecode into standalone executable and runs it.
Requires: gcc, model weights
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


class BundlerRunner(VMRunner):
    """Bundler VM runner - creates standalone executables."""

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path
        self.cache_dir = os.path.expanduser('~/.cache/c4_test_runner/bundled_programs')
        self.temp_dir = None
        self.bundler = None

    def _check_gcc(self) -> bool:
        """Check if gcc is available."""
        try:
            result = subprocess.run(['gcc', '--version'],
                                    capture_output=True,
                                    timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _find_model(self) -> str:
        """Find model file (.c4onnx format)."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path

        # Search for .c4onnx model
        candidates = [
            'models/transformer_vm.c4onnx',
            'models/from_onnx.c4onnx',
            'models/baked_quine.c4onnx',
            'model.c4onnx',
        ]

        base_dir = os.path.join(os.path.dirname(__file__), '../..')

        for candidate in candidates:
            path = os.path.join(base_dir, candidate)
            if os.path.exists(path):
                return path

        raise FileNotFoundError("No .c4onnx model file found")

    def setup(self) -> bool:
        """Initialize bundler."""
        try:
            # Check for gcc
            if not self._check_gcc():
                print("ERROR: Bundler mode requires gcc")
                print("Install: apt-get install gcc  OR  brew install gcc")
                return False

            # Find model
            try:
                self.model_path = self._find_model()
                print(f"Using model: {self.model_path}")
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                return False

            # Import bundler
            try:
                from bundler import neural_bundler
                self.bundler = neural_bundler
            except ImportError as e:
                print(f"ERROR: Failed to import bundler: {e}")
                return False

            # Create cache and temp directories
            os.makedirs(self.cache_dir, exist_ok=True)
            self.temp_dir = tempfile.mkdtemp(prefix='c4_bundler_')

            self._initialized = True
            return True

        except Exception as e:
            print(f"BundlerRunner setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_program(self, source: str, max_steps: int = 10000) -> Tuple[int, float]:
        """Execute program using bundler."""
        if not self._initialized:
            raise RuntimeError("BundlerRunner not initialized")

        start_time = time.time()

        try:
            # Hash source for caching
            source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
            cached_binary = os.path.join(self.cache_dir, f'bundled_{source_hash}')

            if os.path.exists(cached_binary):
                # Use cached binary
                result = subprocess.run([cached_binary],
                                        capture_output=True,
                                        timeout=30,
                                        text=True)
            else:
                # Write source to temp file (bundle() expects file path)
                source_file = os.path.join(self.temp_dir, f'source_{source_hash}.c')
                with open(source_file, 'w') as f:
                    f.write(source)

                # Create bundled C file using bundle() function
                bundled_c = os.path.join(self.temp_dir, f'bundled_{source_hash}.c')

                with open(bundled_c, 'w') as out_file:
                    # Use bundler.bundle() to generate the bundled C code
                    self.bundler.bundle(
                        model_path=self.model_path,
                        source_path=source_file,
                        output_file=out_file,
                        runtime_name='neural',  # Use neural runtime (now with 32-bit support)
                        use_argv=False
                    )

                # Compile bundled C file
                compile_result = subprocess.run([
                    'gcc',
                    '-O2',
                    '-o', cached_binary,
                    bundled_c,
                    '-lm'
                ], capture_output=True, timeout=120, text=True)

                if compile_result.returncode != 0:
                    raise RuntimeError(f"Compilation failed:\n{compile_result.stderr}")

                # Execute
                result = subprocess.run([cached_binary],
                                        capture_output=True,
                                        timeout=30,
                                        text=True)

            # Parse result from stdout (neural runtime prints exit code)
            # Format: single integer on stdout
            try:
                stdout = result.stdout.strip()
                if stdout:
                    exit_code = int(stdout.split('\n')[-1])
                else:
                    exit_code = result.returncode
            except (ValueError, IndexError):
                # Fallback to returncode if parsing fails
                exit_code = result.returncode

            elapsed = time.time() - start_time
            return exit_code, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            raise RuntimeError("Execution timeout")
        except Exception as e:
            elapsed = time.time() - start_time
            raise RuntimeError(f"BundlerRunner execution failed: {e}") from e

    def cleanup(self) -> None:
        """Cleanup temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @property
    def name(self) -> str:
        return "Bundler"
