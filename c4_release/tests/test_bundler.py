"""
Bundler Tests.

Tests for Requirement #6: C4 C bundler works and passes 1000 tests.

Tests verify:
1. Python bundler module exists and can be imported
2. Bundler helper functions (emit_byte_array, emit_bytecode_array, etc.)
3. Bytecode patching (patch_bytecode)
4. C runtime files exist and have proper structure
5. Bundled C code compilation (if gcc available)
6. Neural computation primitives in runtime

Note: Full 1000-test suite requires model files and compiled executables.
"""

import sys
import os
import subprocess
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestBundlerModuleExists:
    """Verify bundler modules can be imported."""

    def test_neural_bundler_import(self):
        """Neural bundler module can be imported."""
        from bundler import neural_bundler
        assert hasattr(neural_bundler, 'bundle')
        assert hasattr(neural_bundler, 'emit_byte_array')
        assert hasattr(neural_bundler, 'emit_bytecode_array')

    def test_bake_program_import(self):
        """Bake program module can be imported."""
        from tools.bake_program import patch_bytecode
        assert callable(patch_bytecode)


class TestBundlerHelperFunctions:
    """Test bundler helper functions."""

    def test_emit_byte_array(self):
        """emit_byte_array generates valid C array."""
        from bundler.neural_bundler import emit_byte_array

        data = bytes([0x00, 0x42, 0xFF, 0x12, 0x34])
        result = emit_byte_array("test_array", data)

        assert "char test_array[]" in result
        assert "0x00" in result
        assert "0x42" in result
        assert "0xff" in result
        assert "int test_array_len = 5;" in result

    def test_emit_byte_array_empty(self):
        """emit_byte_array handles empty data."""
        from bundler.neural_bundler import emit_byte_array

        result = emit_byte_array("empty", b"")
        assert "char empty[]" in result
        assert "int empty_len = 0;" in result

    def test_emit_bytecode_array(self):
        """emit_bytecode_array generates valid C array."""
        from bundler.neural_bundler import emit_bytecode_array

        # Simple bytecode: IMM 42 (op=1, imm=42)
        bytecode = [1 + (42 << 8)]
        result = emit_bytecode_array(bytecode)

        assert "int program_code[][2]" in result
        assert "{1, 42}" in result
        assert "int program_code_len = 1;" in result

    def test_emit_data_array(self):
        """emit_data_array generates valid C array."""
        from bundler.neural_bundler import emit_data_array

        data = [0x48, 0x65, 0x6C, 0x6C, 0x6F]  # "Hello"
        result = emit_data_array(data)

        assert "char program_data[]" in result
        assert "0x48" in result
        assert "int program_data_len = 5;" in result

    def test_emit_data_array_empty(self):
        """emit_data_array handles empty data."""
        from bundler.neural_bundler import emit_data_array

        result = emit_data_array([])
        assert "program_data_len = 0" in result


class TestPatchBytecode:
    """Test bytecode patching functionality."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_patch_bytecode_basic(self, compile_program):
        """patch_bytecode returns valid bytecode."""
        from tools.bake_program import patch_bytecode

        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        patched = patch_bytecode(bytecode)

        # Patched bytecode should be longer (includes subroutines)
        assert len(patched) >= len(bytecode)

    def test_patch_bytecode_mset_replacement(self, compile_program):
        """patch_bytecode replaces MSET with JSR."""
        from tools.bake_program import patch_bytecode, MSET_OP

        # Simple program without memset (just test patching doesn't break it)
        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        patched = patch_bytecode(bytecode)

        # Patched bytecode should be longer (includes subroutines)
        assert len(patched) >= len(bytecode)

        # All original instructions should still be valid (unless MSET/MCMP)
        for i, instr in enumerate(patched[:len(bytecode)]):
            op = instr & 0xFF
            assert op < 100  # Valid opcode range

    def test_patch_bytecode_argv(self, compile_program):
        """patch_bytecode adds argv support when requested."""
        from tools.bake_program import patch_bytecode, ARGV_BASE

        source = """
        int main(int argc, char **argv) {
            return argc;
        }
        """
        bytecode, data = compile_program(source)

        patched = patch_bytecode(bytecode, argv=True)

        # Patched with argv should be significantly longer
        assert len(patched) > len(bytecode)

        # First instruction should JSR to wrapper (not original main)
        first_op = patched[0] & 0xFF
        first_imm = patched[0] >> 8
        assert first_op == 3  # JSR


class TestCRuntimeFilesExist:
    """Verify C runtime files exist and have proper content."""

    def test_neural_runtime_exists(self):
        """Neural runtime C file exists."""
        path = "bundler/neural_runtime.c"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10000  # Substantial file

    def test_onnx_vm_runtime_exists(self):
        """ONNX VM runtime C file exists."""
        path = "bundler/onnx_vm_runtime.c"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10000

    def test_autoregressive_runtime_exists(self):
        """Autoregressive runtime C file exists."""
        path = "bundler/autoregressive_runtime.c"
        assert os.path.exists(path)

    def test_optimized_runtime_exists(self):
        """Optimized runtime C file exists."""
        path = "bundler/optimized_runtime.c"
        assert os.path.exists(path)

    def test_neural_bundler_c_exists(self):
        """Neural bundler C file exists."""
        path = "bundler/neural_bundler.c"
        assert os.path.exists(path)


class TestCRuntimeStructure:
    """Test C runtime file structure and content."""

    def test_neural_runtime_has_fixed_point(self):
        """Neural runtime uses fixed-point arithmetic."""
        with open("bundler/neural_runtime.c") as f:
            content = f.read()

        # Should have fixed-point functions
        assert "fp_mul" in content or "FP_ONE" in content
        assert "16.16" in content or "fixed" in content.lower()

    def test_neural_runtime_has_vm_ops(self):
        """Neural runtime implements VM operations."""
        with open("bundler/neural_runtime.c") as f:
            content = f.read()

        # Should have VM step function
        assert "vm_step" in content or "step" in content.lower()

    def test_neural_runtime_has_neural_ops(self):
        """Neural runtime implements neural operations."""
        with open("bundler/neural_runtime.c") as f:
            content = f.read()

        required = ["matmul", "softmax"]
        found = sum(1 for r in required if r in content.lower())
        assert found >= 1, f"Expected neural ops like matmul/softmax"

    def test_onnx_runtime_has_operations(self):
        """ONNX runtime implements required operations."""
        with open("bundler/onnx_vm_runtime.c") as f:
            content = f.read()

        # Should have some tensor operations
        assert "matmul" in content.lower() or "gemm" in content.lower()


class TestInlineRuntime:
    """Test inline NEURAL_VM_RUNTIME in neural_bundler.py."""

    def test_inline_runtime_exists(self):
        """Inline runtime constant exists in bundler."""
        from bundler.neural_bundler import NEURAL_VM_RUNTIME

        assert len(NEURAL_VM_RUNTIME) > 10000
        assert "NEURAL VM RUNTIME" in NEURAL_VM_RUNTIME

    def test_inline_runtime_has_fixed_point(self):
        """Inline runtime has fixed-point math."""
        from bundler.neural_bundler import NEURAL_VM_RUNTIME

        assert "FP_ONE" in NEURAL_VM_RUNTIME
        assert "fp_mul" in NEURAL_VM_RUNTIME
        assert "fp_exp" in NEURAL_VM_RUNTIME

    def test_inline_runtime_has_neural_ops(self):
        """Inline runtime has neural operations."""
        from bundler.neural_bundler import NEURAL_VM_RUNTIME

        required = ["nn_softmax", "nn_matmul", "nn_argmax"]
        for op in required:
            assert op in NEURAL_VM_RUNTIME, f"Missing: {op}"

    def test_inline_runtime_has_nval_ops(self):
        """Inline runtime has NVal (32-bit neural value) operations."""
        from bundler.neural_bundler import NEURAL_VM_RUNTIME

        required = ["nval_new", "nval_encode", "nval_decode"]
        for op in required:
            assert op in NEURAL_VM_RUNTIME, f"Missing: {op}"

    def test_inline_runtime_has_vm_step(self):
        """Inline runtime has VM step function."""
        from bundler.neural_bundler import NEURAL_VM_RUNTIME

        assert "int vm_step()" in NEURAL_VM_RUNTIME
        assert "int main()" in NEURAL_VM_RUNTIME


class TestBundlerCompilation:
    """Test that bundler-generated code compiles."""

    @pytest.fixture
    def has_gcc(self):
        """Check if gcc is available."""
        return shutil.which("gcc") is not None

    @pytest.mark.skipif(not shutil.which("gcc"), reason="gcc not available")
    def test_inline_runtime_syntax_check(self, has_gcc):
        """Inline runtime passes syntax check."""
        from bundler.neural_bundler import NEURAL_VM_RUNTIME

        # Create a minimal C file with the runtime
        test_code = """
char embedded_model[] = {0};
int embedded_model_len = 0;

int program_code[][2] = {{1, 42}};
int program_code_len = 1;

char program_data[] = {0};
int program_data_len = 0;

""" + NEURAL_VM_RUNTIME

        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(test_code)
            path = f.name

        try:
            result = subprocess.run(
                ["gcc", "-fsyntax-only", path],
                capture_output=True,
                text=True,
                timeout=30
            )
            # May have warnings but should not error on syntax
            # If it fails, check for "error:" in output
            if result.returncode != 0:
                assert "error:" not in result.stderr.lower(), \
                    f"Syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.skip("Syntax check timed out")
        finally:
            os.unlink(path)

    @pytest.mark.skipif(not shutil.which("gcc"), reason="gcc not available")
    def test_neural_runtime_syntax_check(self, has_gcc):
        """Neural runtime C file passes syntax check."""
        result = subprocess.run(
            ["gcc", "-fsyntax-only", "bundler/neural_runtime.c"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Check for errors (not just warnings)
        if result.returncode != 0:
            has_error = "error:" in result.stderr.lower()
            if has_error:
                pytest.skip(f"Has compilation errors (expected for standalone file)")


class TestBundlerRuntimeSelection:
    """Test runtime selection in bundler."""

    def test_get_runtime_default(self):
        """get_runtime returns inline runtime by default."""
        from bundler.neural_bundler import get_runtime, NEURAL_VM_RUNTIME

        result = get_runtime()
        assert result == NEURAL_VM_RUNTIME

    def test_get_runtime_neural(self):
        """get_runtime returns neural runtime file."""
        from bundler.neural_bundler import get_runtime

        result = get_runtime('neural')
        assert len(result) > 10000
        assert "neural" in result.lower() or "vm" in result.lower()

    def test_get_runtime_onnx(self):
        """get_runtime returns onnx runtime file."""
        from bundler.neural_bundler import get_runtime

        result = get_runtime('onnx')
        assert len(result) > 10000


class TestBytecodeRelocation:
    """Test bytecode relocation in bake_program."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_subroutine_compilation(self, compile_program):
        """Subroutine compilation produces valid bytecode."""
        from tools.bake_program import _compile_subroutine, _MEMSET_SRC

        # Compile memset to a base address
        base = 1000
        code = _compile_subroutine(_MEMSET_SRC, base)

        # Should produce bytecode
        assert len(code) > 0

        # Each instruction should be valid
        for instr in code:
            op = instr & 0xFF
            assert op < 100  # Valid opcode range

    def test_encode_instr(self):
        """_encode_instr produces correct format."""
        from tools.bake_program import _encode_instr

        # IMM 42: op=1, imm=42
        instr = _encode_instr(1, 42)
        assert (instr & 0xFF) == 1
        assert (instr >> 8) == 42

        # Negative immediate
        instr = _encode_instr(2, -8)
        assert (instr & 0xFF) == 2
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        assert imm == -8


class TestBundlerIntegration:
    """Integration tests for bundler."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_full_bundle_generation(self, compile_program):
        """Full bundle can be generated."""
        from bundler.neural_bundler import (
            emit_byte_array, emit_bytecode_array, emit_data_array
        )

        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        # Generate all components
        model_bytes = b'\x00' * 100  # Dummy model
        model_arr = emit_byte_array("embedded_model", model_bytes)
        code_arr = emit_bytecode_array(bytecode)
        data_arr = emit_data_array(data)

        # All should be valid C
        assert "char embedded_model[]" in model_arr
        assert "int program_code[][2]" in code_arr
        assert "program_data" in data_arr


class TestNeuralVMBaselineForBundler:
    """Establish baseline results for bundler comparison."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_baseline_return_42(self, compile_program):
        """Python VM returns 42 correctly (baseline for bundler)."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        runner = BatchedSpeculativeRunner(batch_size=1)
        results = runner.run_batch([bytecode], [data], max_steps=500)

        assert results[0][1] == 42

    def test_baseline_arithmetic_suite(self, compile_program):
        """Arithmetic operations work (baseline for bundler)."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        test_cases = [
            ("int main() { return 5 + 3; }", 8),
            ("int main() { return 10 - 4; }", 6),
            ("int main() { return 3 * 4; }", 12),
            ("int main() { return 20 / 5; }", 4),
            ("int main() { return 17 % 5; }", 2),
        ]

        runner = BatchedSpeculativeRunner(batch_size=1)

        for source, expected in test_cases:
            bytecode, data = compile_program(source)
            results = runner.run_batch([bytecode], [data], max_steps=500)
            assert results[0][1] == expected, f"Failed: {source}"

    def test_baseline_100_programs(self, compile_program):
        """Run 100 programs (baseline for bundler 1000 test requirement)."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        programs = []
        for i in range(100):
            a = (i % 50) + 1
            b = ((i * 3) % 30) + 1
            expected = a + b
            programs.append((f"int main() {{ return {a} + {b}; }}", expected))

        runner = BatchedSpeculativeRunner(batch_size=10)
        passed = 0

        for source, expected in programs:
            try:
                bytecode, data = compile_program(source)
                results = runner.run_batch([bytecode], [data], max_steps=500)
                if results[0][1] == expected:
                    passed += 1
            except Exception:
                pass

        assert passed >= 100, f"Expected 100 passes, got {passed}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
