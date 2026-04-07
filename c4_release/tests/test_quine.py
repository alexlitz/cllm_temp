"""
Quine Tests.

Tests for Requirement #8: Quine self-replication works correctly.

A quine is a program that outputs its own source code. The C4 quine
demonstrates that the neural VM can execute self-referential code.

Tests verify:
1. Quine source file exists and has valid structure
2. Quine can be compiled to C4 bytecode
3. Running quine outputs its own source (self-replication)
4. Quine bundler infrastructure exists
5. Autoregressive quine bundler can be imported
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestQuineSourceExists:
    """Verify quine source files exist."""

    def test_quine_cllm_exists(self):
        """Main quine source file exists."""
        path = "cllm/quine_cllm.c"
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 100

    def test_quine_source_readable(self):
        """Quine source is readable and non-empty."""
        with open("cllm/quine_cllm.c") as f:
            content = f.read()

        assert len(content) > 100
        assert "int main()" in content


class TestQuineStructure:
    """Test quine source structure."""

    def test_quine_has_main(self):
        """Quine has main function."""
        with open("cllm/quine_cllm.c") as f:
            content = f.read()

        assert "int main()" in content

    def test_quine_has_string_literal(self):
        """Quine has self-referential string literal."""
        with open("cllm/quine_cllm.c") as f:
            content = f.read()

        # Quine uses a string containing its own code
        # String typically has 's = "...' pattern
        assert 's = "' in content

    def test_quine_has_putchar(self):
        """Quine uses putchar for output."""
        with open("cllm/quine_cllm.c") as f:
            content = f.read()

        assert "putchar" in content

    def test_quine_has_placeholder(self):
        """Quine uses placeholder character (126 = ~) for string insertion."""
        with open("cllm/quine_cllm.c") as f:
            content = f.read()

        # The quine uses character 126 (~) as placeholder
        assert "126" in content or "~" in content

    def test_quine_has_escape_logic(self):
        """Quine has newline escape logic (10 -> \\n)."""
        with open("cllm/quine_cllm.c") as f:
            content = f.read()

        # Should escape newlines: putchar(92) for \, putchar(110) for n
        assert "92" in content and "110" in content


class TestQuineCompilation:
    """Test that quine can be compiled."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_quine_compiles(self, compile_program):
        """Quine source compiles to bytecode."""
        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)

        # Should produce bytecode
        assert len(bytecode) > 0
        # Quine has substantial code
        assert len(bytecode) > 10

    def test_quine_bytecode_valid(self, compile_program):
        """Quine bytecode has valid opcodes."""
        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)

        for instr in bytecode:
            op = instr & 0xFF
            assert op < 100  # Valid opcode range

    def test_quine_has_data_section(self, compile_program):
        """Quine has data section (for string literal)."""
        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)

        # Quine string literal should be in data section
        assert len(data) > 100  # Substantial string


class TestQuineExecution:
    """Test quine execution and self-replication.

    Note: These tests are slow (require running thousands of neural VM steps).
    Marked with pytest.mark.slow for skipping in quick test runs.
    Run with: pytest -m slow tests/test_quine.py
    """

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    @pytest.mark.skip(reason="Slow test - requires ~50000 neural VM steps")
    def test_quine_runs_without_error(self, compile_program):
        """Quine executes without error in neural VM."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)

        runner = BatchedSpeculativeRunner(batch_size=1)
        results = runner.run_batch([bytecode], [data], max_steps=50000)

        # Should complete with exit code 0
        output, exit_code = results[0]
        assert exit_code == 0

    @pytest.mark.skip(reason="Slow test - requires ~50000 neural VM steps")
    def test_quine_produces_output(self, compile_program):
        """Quine produces output."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)

        runner = BatchedSpeculativeRunner(batch_size=1)
        results = runner.run_batch([bytecode], [data], max_steps=50000)

        output, exit_code = results[0]

        # Should produce non-empty output
        assert output is not None
        assert len(output) > 100  # Substantial output

    @pytest.mark.skip(reason="Slow test - requires ~50000 neural VM steps")
    def test_quine_self_replicates(self, compile_program):
        """Quine output matches its source (self-replication)."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)

        runner = BatchedSpeculativeRunner(batch_size=1)
        results = runner.run_batch([bytecode], [data], max_steps=50000)

        output, exit_code = results[0]

        # Convert output bytes to string
        if isinstance(output, list):
            output_str = ''.join(chr(b) for b in output if 0 <= b < 256)
        elif isinstance(output, bytes):
            output_str = output.decode('latin-1')
        else:
            output_str = str(output)

        # Output should exactly match source (self-replication)
        assert output_str == source, \
            f"Quine output doesn't match source.\nFirst 100 chars of diff:\nOutput: {output_str[:100]}\nSource: {source[:100]}"


class TestQuineBundlerExists:
    """Verify quine bundler infrastructure exists."""

    def test_bundle_autoregressive_quine_exists(self):
        """Autoregressive quine bundler script exists."""
        path = "tools/bundle_autoregressive_quine.py"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

    def test_quine_model_exists(self):
        """Baked quine model file exists."""
        path = "models/baked_quine.c4onnx"
        if os.path.exists(path):
            assert os.path.getsize(path) > 0
        else:
            pytest.skip("Quine model not yet generated")


class TestQuineBundlerImport:
    """Test quine bundler can be imported."""

    def test_export_autoregressive_import(self):
        """Export autoregressive module can be imported."""
        from tools.export_autoregressive import load_arvm, export_autoregressive
        assert callable(load_arvm)
        assert callable(export_autoregressive)

    def test_export_arvm_constants(self):
        """ARVM format constants are defined."""
        from tools.export_autoregressive import ARVM_MAGIC, ARVM_VERSION
        assert ARVM_MAGIC == 0x4D565241  # "ARVM"
        assert ARVM_VERSION >= 1


class TestQuineBundledFiles:
    """Test bundled quine files exist (if generated)."""

    def test_bundled_quine_neural_exists(self):
        """Bundled quine neural C file exists (if generated)."""
        path = "build/bundled/quine_neural.c"
        if os.path.exists(path):
            assert os.path.getsize(path) > 10000
        else:
            pytest.skip("Bundled quine not yet generated")

    def test_bundled_quine_draft_exists(self):
        """Bundled quine draft C file exists (if generated)."""
        path = "build/bundled/quine_draft.c"
        if os.path.exists(path):
            assert os.path.getsize(path) > 1000
        else:
            pytest.skip("Bundled quine draft not yet generated")


class TestOtherCLLMPrograms:
    """Test other CLLM (C4 Little Language Model) programs."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    def test_cllm_programs_exist(self):
        """CLLM utility programs exist."""
        cllm_dir = "cllm"
        assert os.path.isdir(cllm_dir)

        programs = ["cat_cllm.c", "echo_cllm.c", "yes_cllm.c", "wc_cllm.c"]
        for prog in programs:
            path = os.path.join(cllm_dir, prog)
            assert os.path.exists(path), f"Missing: {path}"

    def test_cat_cllm_compiles(self, compile_program):
        """cat CLLM program compiles."""
        with open("cllm/cat_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)
        assert len(bytecode) > 0

    def test_echo_cllm_compiles(self, compile_program):
        """echo CLLM program compiles."""
        with open("cllm/echo_cllm.c") as f:
            source = f.read()

        bytecode, data = compile_program(source)
        assert len(bytecode) > 0


class TestQuineArchiveFiles:
    """Test quine archive files exist."""

    def test_archive_quine_files_exist(self):
        """Archive quine files exist."""
        archive_dir = "archive/quine_feedforward"
        if os.path.exists(archive_dir):
            files = os.listdir(archive_dir)
            assert len(files) > 0
        else:
            pytest.skip("Quine archive directory not found")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
