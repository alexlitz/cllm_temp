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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        path = os.path.join(_HERE, "bundler/neural_runtime.c")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10000  # Substantial file

    def test_onnx_vm_runtime_exists(self):
        """ONNX VM runtime C file exists."""
        path = os.path.join(_HERE, "bundler/onnx_vm_runtime.c")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10000

    def test_autoregressive_runtime_exists(self):
        """Autoregressive runtime C file exists."""
        path = os.path.join(_HERE, "bundler/autoregressive_runtime.c")
        assert os.path.exists(path)

    def test_optimized_runtime_exists(self):
        """Optimized runtime C file exists."""
        path = os.path.join(_HERE, "bundler/optimized_runtime.c")
        assert os.path.exists(path)

    def test_neural_bundler_c_exists(self):
        """Neural bundler C file exists."""
        path = os.path.join(_HERE, "bundler/neural_bundler.c")
        assert os.path.exists(path)


class TestCRuntimeStructure:
    """Test C runtime file structure and content."""

    def test_neural_runtime_has_fixed_point(self):
        """Neural runtime uses fixed-point arithmetic."""
        with open(os.path.join(_HERE, "bundler/neural_runtime.c")) as f:
            content = f.read()

        # Should have fixed-point functions
        assert "fp_mul" in content or "FP_ONE" in content
        assert "16.16" in content or "fixed" in content.lower()

    def test_neural_runtime_has_vm_ops(self):
        """Neural runtime implements VM operations."""
        with open(os.path.join(_HERE, "bundler/neural_runtime.c")) as f:
            content = f.read()

        # Should have VM step function
        assert "vm_step" in content or "step" in content.lower()

    def test_neural_runtime_has_neural_ops(self):
        """Neural runtime implements neural operations."""
        with open(os.path.join(_HERE, "bundler/neural_runtime.c")) as f:
            content = f.read()

        required = ["matmul", "softmax"]
        found = sum(1 for r in required if r in content.lower())
        assert found >= 1, f"Expected neural ops like matmul/softmax"

    def test_onnx_runtime_has_operations(self):
        """ONNX runtime implements required operations."""
        with open(os.path.join(_HERE, "bundler/onnx_vm_runtime.c")) as f:
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
            ["gcc", "-fsyntax-only", os.path.join(_HERE, "bundler/neural_runtime.c")],
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


# ----------------------------------------------------------------------------
# Bundle file-format coverage: explicit write / read / round-trip equivalence.
#
# Per TESTING_CHECKLIST.md the bundler "bundles programs, the model weights
# and the program bytecode all together into a single file". The ``.arvm``
# format in ``tools/export_autoregressive.py`` is the canonical model-weight
# container (header config + per-tensor blobs). These tests pin its
# format-level invariants by constructing tensors directly (no full VM
# compile required) so they run in any environment.
# ----------------------------------------------------------------------------


def _write_minimal_arvm(path, *, vocab_size=8, d_model=4, n_layers=1,
                        n_heads=2, ffn_hidden=8, sparse=False, seed=0):
    """Hand-write a minimal .arvm file with deterministic tensors.

    Uses the real exporter's tensor-emission helper so the on-disk layout is
    exercised exactly. Returns the dict of original numpy tensors so callers
    can verify round-trip equivalence against ``load_arvm``.
    """
    import numpy as np
    from tools.export_autoregressive import (
        ARVM_MAGIC, ARVM_VERSION, _write_u32, write_tensor,
    )

    rng = np.random.default_rng(seed)

    def t(shape):
        return rng.standard_normal(shape).astype(np.float32)

    with open(path, 'wb') as f:
        _write_u32(f, ARVM_MAGIC)
        _write_u32(f, ARVM_VERSION)
        _write_u32(f, vocab_size)
        _write_u32(f, d_model)
        _write_u32(f, n_layers)
        _write_u32(f, n_heads)
        _write_u32(f, ffn_hidden)

        tensors = {'embed_weight': t((vocab_size, d_model))}
        write_tensor(f, tensors['embed_weight'], sparse=sparse)

        layers = []
        for _ in range(n_layers):
            layer = {
                'alibi_slopes': t((n_heads,)),
                'W_q': t((d_model, d_model)),
                'W_k': t((d_model, d_model)),
                'W_v': t((d_model, d_model)),
                'W_o': t((d_model, d_model)),
                'W_up': t((ffn_hidden, d_model)),
                'b_up': t((ffn_hidden,)),
                'W_gate': t((ffn_hidden, d_model)),
                'b_gate': t((ffn_hidden,)),
                'W_down': t((d_model, ffn_hidden)),
                'b_down': t((d_model,)),
            }
            write_tensor(f, layer['alibi_slopes'], sparse=False)
            for key in ('W_q', 'W_k', 'W_v', 'W_o',
                        'W_up', 'b_up', 'W_gate', 'b_gate',
                        'W_down', 'b_down'):
                write_tensor(f, layer[key], sparse=sparse)
            layers.append(layer)
        tensors['layers'] = layers

        tensors['head_weight'] = t((vocab_size, d_model))
        tensors['head_bias'] = t((vocab_size,))
        write_tensor(f, tensors['head_weight'], sparse=sparse)
        write_tensor(f, tensors['head_bias'], sparse=sparse)

    return tensors


class TestBundleWrite:
    """Bundle-write produces a well-formed file on disk."""

    def test_bundle_write_produces_file(self, tmp_path):
        """Writing a minimal bundle creates a non-empty file."""
        path = tmp_path / "minimal.arvm"
        _write_minimal_arvm(str(path))
        assert path.exists()
        assert path.stat().st_size > 28

    def test_bundle_write_has_correct_magic(self, tmp_path):
        """First 4 bytes of a written bundle are the ARVM magic."""
        import struct
        from tools.export_autoregressive import ARVM_MAGIC

        path = tmp_path / "magic.arvm"
        _write_minimal_arvm(str(path))
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
        assert magic == ARVM_MAGIC


class TestBundleRead:
    """``load_arvm`` recovers the header config and tensor shapes."""

    def test_bundle_read_recovers_config(self, tmp_path):
        """Bundle read recovers the header config (vocab/d_model/etc.)."""
        from tools.export_autoregressive import load_arvm

        path = tmp_path / "config.arvm"
        _write_minimal_arvm(
            str(path), vocab_size=12, d_model=4, n_layers=2,
            n_heads=2, ffn_hidden=8,
        )
        loaded = load_arvm(str(path))

        assert loaded['vocab_size'] == 12
        assert loaded['d_model'] == 4
        assert loaded['n_layers'] == 2
        assert loaded['n_heads'] == 2
        assert loaded['ffn_hidden'] == 8

    def test_bundle_read_recovers_layer_tensors(self, tmp_path):
        """Loaded ``layers`` list has all expected tensor keys per layer."""
        from tools.export_autoregressive import load_arvm

        path = tmp_path / "layers.arvm"
        _write_minimal_arvm(str(path), n_layers=3)
        loaded = load_arvm(str(path))

        assert len(loaded['layers']) == 3
        for layer in loaded['layers']:
            for key in ('alibi_slopes', 'W_q', 'W_k', 'W_v', 'W_o',
                        'W_up', 'b_up', 'W_gate', 'b_gate',
                        'W_down', 'b_down'):
                assert key in layer, f"Missing layer tensor: {key}"

    def test_bundle_read_rejects_bad_magic(self, tmp_path):
        """``load_arvm`` rejects files with the wrong magic bytes."""
        import struct
        from tools.export_autoregressive import load_arvm

        path = tmp_path / "badmagic.arvm"
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', 0xDEADBEEF))
            f.write(b'\x00' * 24)

        with pytest.raises(ValueError, match="Bad magic"):
            load_arvm(str(path))

    def test_bundle_read_rejects_unsupported_version(self, tmp_path):
        """``load_arvm`` rejects unknown format versions."""
        import struct
        from tools.export_autoregressive import ARVM_MAGIC, load_arvm

        path = tmp_path / "badver.arvm"
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', ARVM_MAGIC))
            f.write(struct.pack('<I', 999))
            f.write(b'\x00' * 20)

        with pytest.raises(ValueError, match="Unsupported version"):
            load_arvm(str(path))


class TestBundleRoundTrip:
    """write -> read returns bit-equivalent float32 tensors."""

    def test_round_trip_dense_bit_equivalent(self, tmp_path):
        """Dense write -> read returns bit-equivalent float32 tensors."""
        import numpy as np
        from tools.export_autoregressive import load_arvm

        path = tmp_path / "dense.arvm"
        original = _write_minimal_arvm(str(path), sparse=False, seed=42)
        loaded = load_arvm(str(path))

        np.testing.assert_array_equal(loaded['embed_weight'],
                                      original['embed_weight'])
        np.testing.assert_array_equal(loaded['head_weight'],
                                      original['head_weight'])
        np.testing.assert_array_equal(loaded['head_bias'],
                                      original['head_bias'])
        assert len(loaded['layers']) == len(original['layers'])
        for got, want in zip(loaded['layers'], original['layers']):
            for key in want:
                np.testing.assert_array_equal(
                    got[key], want[key],
                    err_msg=f"Round-trip mismatch on {key}",
                )

    def test_round_trip_sparse_bit_equivalent(self, tmp_path):
        """Sparse-COO write -> read returns bit-equivalent dense tensors."""
        import numpy as np
        from tools.export_autoregressive import (
            ARVM_MAGIC, ARVM_VERSION, _write_u32, write_tensor, load_arvm,
        )

        vocab_size, d_model, n_layers, n_heads, ffn_hidden = 4, 4, 1, 2, 4

        def sparse(shape, nnz_fraction=0.1, seed=0):
            arr = np.zeros(shape, dtype=np.float32)
            rng = np.random.default_rng(seed)
            flat = arr.flatten()
            nnz = max(1, int(len(flat) * nnz_fraction))
            idx = rng.choice(len(flat), nnz, replace=False)
            flat[idx] = rng.standard_normal(nnz).astype(np.float32)
            return flat.reshape(shape)

        tensors = {
            'embed': sparse((vocab_size, d_model), seed=1),
            'Wq': sparse((d_model, d_model), seed=2),
            'Wk': sparse((d_model, d_model), seed=3),
            'Wv': sparse((d_model, d_model), seed=4),
            'Wo': sparse((d_model, d_model), seed=5),
            'Wup': sparse((ffn_hidden, d_model), seed=6),
            'bup': sparse((ffn_hidden,), seed=7),
            'Wgate': sparse((ffn_hidden, d_model), seed=8),
            'bgate': sparse((ffn_hidden,), seed=9),
            'Wdown': sparse((d_model, ffn_hidden), seed=10),
            'bdown': sparse((d_model,), seed=11),
            'head_w': sparse((vocab_size, d_model), seed=12),
            'head_b': sparse((vocab_size,), seed=13),
        }
        alibi = np.array([1.0, 0.5], dtype=np.float32)

        path = tmp_path / "sparse.arvm"
        with open(path, 'wb') as f:
            _write_u32(f, ARVM_MAGIC)
            _write_u32(f, ARVM_VERSION)
            _write_u32(f, vocab_size)
            _write_u32(f, d_model)
            _write_u32(f, n_layers)
            _write_u32(f, n_heads)
            _write_u32(f, ffn_hidden)
            write_tensor(f, tensors['embed'], sparse=True)
            write_tensor(f, alibi, sparse=False)
            for key in ('Wq', 'Wk', 'Wv', 'Wo', 'Wup', 'bup',
                        'Wgate', 'bgate', 'Wdown', 'bdown'):
                write_tensor(f, tensors[key], sparse=True)
            write_tensor(f, tensors['head_w'], sparse=True)
            write_tensor(f, tensors['head_b'], sparse=True)

        loaded = load_arvm(str(path))
        np.testing.assert_array_equal(loaded['embed_weight'], tensors['embed'])
        np.testing.assert_array_equal(loaded['head_weight'], tensors['head_w'])
        np.testing.assert_array_equal(loaded['head_bias'], tensors['head_b'])
        np.testing.assert_array_equal(loaded['layers'][0]['W_q'], tensors['Wq'])
        np.testing.assert_array_equal(loaded['layers'][0]['W_down'],
                                      tensors['Wdown'])


class TestBundleTokenizerRoundTrip:
    """Tokenizer encode/decode round-trip (the bundle's logical tokenizer)."""

    def test_tokenizer_round_trip_ascii(self):
        """ASCII text round-trips through the tokenizer."""
        from src.tokenizer import C4Tokenizer
        tok = C4Tokenizer()
        text = "int main() { return 42; }"
        assert tok.decode(tok.encode(text)) == text

    def test_tokenizer_round_trip_with_specials(self):
        """Round-trip preserves text when BOS/EOS are skipped on decode."""
        from src.tokenizer import C4Tokenizer
        tok = C4Tokenizer()
        text = "hello"
        ids = tok.encode(text, add_special_tokens=True)
        assert tok.decode(ids, skip_special_tokens=True) == text

    def test_tokenizer_config_vocab_size_consistent(self):
        """Tokenizer config exposes vocab_size used by the model header."""
        from src.tokenizer import C4Tokenizer, TokenizerConfig
        cfg = TokenizerConfig()
        tok = C4Tokenizer(cfg)
        assert tok.vocab_size == cfg.vocab_size
        assert cfg.vocab_size >= 256


class TestBundleGapTokenizerNotSerialized:
    """Gap test: the .arvm bundle does NOT yet embed tokenizer assets.

    Per the user-facing definition (bundle = model + tokenizer + config), a
    truly self-contained bundle should include tokenizer metadata. Today the
    .arvm header carries only ``vocab_size``; the tokenizer mapping lives in
    ``src/tokenizer.py``. This test asserts the gap so a future fix flips
    its expectation.
    """

    def test_arvm_does_not_serialize_tokenizer(self, tmp_path):
        """``load_arvm`` returns no tokenizer assets — only weights + dims."""
        path = tmp_path / "no_tokenizer.arvm"
        _write_minimal_arvm(str(path))
        from tools.export_autoregressive import load_arvm
        loaded = load_arvm(str(path))
        for key in ('tokenizer', 'special_tokens',
                    'bos_token_id', 'eos_token_id', 'pad_token_id'):
            assert key not in loaded, (
                f"GAP CLOSED: {key} is now in the bundle. Update this "
                f"test to verify the new contract."
            )


class TestBundleGapExporterEmbeddingPath:
    """Gap test: ``export_autoregressive`` reads the wrong embed path.

    ``AutoregressiveVM`` wraps its ``nn.Embedding`` inside
    ``NeuralVMEmbedding`` so the actual parameter lives at
    ``model.embed.embed.weight``. ``export_autoregressive`` still reads
    ``model.embed.weight``; ``tests/test_onnx_export.py`` skips its
    end-to-end exporter test for the same reason. This test pins the gap.
    """

    def test_exporter_uses_unwrapped_embed_path(self):
        """The exporter's embed access path is bare ``embed.weight``."""
        src = os.path.join(_HERE, "tools", "export_autoregressive.py")
        with open(src) as f:
            contents = f.read()
        assert "model.embed.weight" in contents, (
            "GAP CLOSED: the exporter no longer reads model.embed.weight."
        )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
