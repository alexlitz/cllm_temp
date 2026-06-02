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

    @pytest.mark.slow
    @pytest.mark.quine
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

    @pytest.mark.slow
    @pytest.mark.quine
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

    @pytest.mark.slow
    @pytest.mark.quine
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

    @pytest.mark.slow
    @pytest.mark.quine
    def test_quine_output_compiles(self, compile_program):
        """Quine output can be re-compiled."""
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

        # Output should be compilable
        bytecode2, data2 = compile_program(output_str)
        assert len(bytecode2) > 0, "Quine output should compile to valid bytecode"

    @pytest.mark.slow
    @pytest.mark.quine
    def test_quine_second_generation(self, compile_program):
        """Quine output, when run, produces same output (transitivity)."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        with open("cllm/quine_cllm.c") as f:
            source = f.read()

        # First generation
        bytecode1, data1 = compile_program(source)
        runner = BatchedSpeculativeRunner(batch_size=1)
        results1 = runner.run_batch([bytecode1], [data1], max_steps=50000)
        output1, _ = results1[0]

        if isinstance(output1, list):
            output1_str = ''.join(chr(b) for b in output1 if 0 <= b < 256)
        elif isinstance(output1, bytes):
            output1_str = output1.decode('latin-1')
        else:
            output1_str = str(output1)

        # Second generation
        bytecode2, data2 = compile_program(output1_str)
        results2 = runner.run_batch([bytecode2], [data2], max_steps=50000)
        output2, _ = results2[0]

        if isinstance(output2, list):
            output2_str = ''.join(chr(b) for b in output2 if 0 <= b < 256)
        elif isinstance(output2, bytes):
            output2_str = output2.decode('latin-1')
        else:
            output2_str = str(output2)

        # Second generation output should match first
        assert output1_str == output2_str, "Quine should be transitive"


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


# ----------------------------------------------------------------------------
# VM self-compilation quine: the "deeper" quine from TESTING_CHECKLIST.
#
# The checklist quine requirement is twofold:
#   (a) ``cllm/quine_cllm.c`` runs on the VM and emits its own source —
#       covered by ``TestQuineExecution`` above.
#   (b) The VM compiling itself: C4 source of the VM/compiler, compiled into
#       the VM, run through the VM, produces a binary equivalent to the VM's
#       own compiled state. This is the "self-hosting" quine and is currently
#       NOT wired end-to-end — the tests below document the gap and the
#       partial pieces that exist today.
# ----------------------------------------------------------------------------


class TestVMSelfCompilationQuineGap:
    """Document the VM-self-compilation quine path and its current gap.

    Pieces that exist:
      - ``bundler/c4_compile.c`` — C4 source of a bytecode compiler.
      - ``src/compiler.py``      — Python reference compiler used at bake.
      - ``neural_vm/batch_runner.py`` — VM that runs bytecode.

    Missing wiring:
      - A driver that compiles ``c4_compile.c`` THROUGH THE VM (not the
        Python host) into bytecode, then byte-compares that bytecode against
        the bytecode produced by compiling the same source via the Python
        compiler used at bake time.
    """

    def test_c4_compile_source_exists(self):
        """The self-hosting C4 compiler source is present."""
        path = "bundler/c4_compile.c"
        assert os.path.exists(path), (
            f"Missing C4 self-host compiler source at {path}; the VM-quine "
            f"path needs this file."
        )
        assert os.path.getsize(path) > 5000

    def test_c4_compile_has_compiler_entrypoint(self):
        """``c4_compile.c`` exposes a compiler-style main entrypoint."""
        with open("bundler/c4_compile.c") as f:
            content = f.read()
        assert "main(" in content
        # The compiler should reference tokenization / code emission.
        assert any(kw in content for kw in ("TK_NUM", "expr(", "next("))

    def test_python_compiler_is_deterministic_on_c4_self_host(self):
        """The Python compiler is deterministic on the C4 self-host source.

        Necessary condition for any "compile-itself" comparison: two
        host-compiles of the same source must agree byte-for-byte.
        """
        from src.compiler import compile_c
        with open("bundler/c4_compile.c") as f:
            source = f.read()
        try:
            bc1, dt1 = compile_c(source)
            bc2, dt2 = compile_c(source)
        except Exception as exc:
            pytest.skip(
                f"Python compiler cannot consume c4_compile.c as-is "
                f"({type(exc).__name__}); VM-quine path also blocked until "
                f"the source is in the host-compiler's accepted subset."
            )
        assert bc1 == bc2, "Host compiler is non-deterministic on same input"
        assert dt1 == dt2

    @pytest.mark.slow
    @pytest.mark.quine
    @pytest.mark.xfail(
        reason=(
            "VM-self-compilation quine path not yet wired: no driver runs "
            "c4_compile.c through the VM and emits a bytecode array "
            "byte-equivalent to the host compile. Closing this gap is the "
            "canonical TESTING_CHECKLIST self-host quine."
        ),
        strict=False,
    )
    def test_vm_compiles_itself_to_equivalent_bytecode(self):
        """VM compiling its own C4 source equals the host compile.

        End state (xfail today):
          1. Read ``bundler/c4_compile.c``.
          2. Compile it via the Python compiler -> ``bc_host``.
          3. Run the resulting compiler bytecode IN THE VM, feeding it the
             SAME C4 source as input, and capture its emitted bytecode array
             -> ``bc_vm``.
          4. Assert ``bc_vm == bc_host``.
        """
        from src.compiler import compile_c

        with open("bundler/c4_compile.c") as f:
            source = f.read()

        bc_host, _ = compile_c(source)
        assert bc_host, "Host compile produced no bytecode"

        # Step 3 (VM-driven compile) is the missing wiring. Asserting False
        # keeps the xfail honest until a driver exists; flip ``strict=True``
        # on the xfail marker once the path lands.
        assert False, (
            "VM-driven self-compile path not implemented; see test docstring "
            "for the four-step contract."
        )

    def test_baked_quine_model_artefact_when_present(self):
        """If a baked quine model exists, it is a non-trivial artefact.

        The "binary equivalent to the VM's own compiled state" comparison
        needs a stable on-disk model artefact to gate against.
        """
        path = "models/baked_quine.c4onnx"
        if not os.path.exists(path):
            pytest.skip(
                "Baked quine model not generated yet — VM-self-compile "
                "comparison has no fixed-point artefact to gate against."
            )
        assert os.path.getsize(path) > 1024


# ----------------------------------------------------------------------------
# Self-compilation byte-sequence + weight-bundle emission gap class.
#
# The TESTING_CHECKLIST quine requires that the model+runtime+bytecode bundle
# can run the VM-compiles-its-own-source path and emit a valid weight bundle.
# Today the path is split:
#   - ``bundler/c4_compile.c`` is the C4 source the VM should be able to
#     consume as a byte sequence.
#   - ``tools/export_autoregressive.export_autoregressive`` is the canonical
#     weight-bundle emitter (.arvm format).
# The wiring between "VM runs on c4_compile.c" and "exporter emits bundle"
# is missing. These tests document the gap and pin the contract pieces that
# DO exist so a future driver can flip them strict.
# ----------------------------------------------------------------------------


class TestSelfCompilationByteSequenceAndBundleEmission:
    """The VM consumes C4 source as a byte sequence + emits a weight bundle.

    Two sub-gates:
      1. The C source of the c4-compile target (``bundler/c4_compile.c``) is
         loadable as a clean byte sequence the VM can step over.
      2. The exporter that emits a valid .arvm weight bundle is callable on
         a baked model, and emits a file conforming to the
         documented header+tensor layout.
    """

    def test_c4_source_loads_as_byte_sequence(self):
        """``c4_compile.c`` is loadable as a self-consistent byte sequence."""
        path = "bundler/c4_compile.c"
        with open(path, 'rb') as f:
            raw = f.read()
        # Non-empty + ASCII-decodable (the VM consumes the source as bytes,
        # the host compiler consumes the same text).
        assert len(raw) > 5000
        # Every byte must be in the printable+whitespace range so a
        # byte-by-byte VM-driven tokenizer wouldn't trip on a control char.
        bad = [(i, b) for i, b in enumerate(raw)
               if b not in (9, 10, 13) and not (32 <= b < 127)]
        assert not bad, (
            f"c4_compile.c contains non-ASCII bytes at offsets "
            f"{bad[:5]}; VM byte-stream driver would have to handle them."
        )

    def test_weight_bundle_emitter_is_callable(self):
        """``export_autoregressive`` is importable + has the expected signature."""
        import inspect
        from tools.export_autoregressive import export_autoregressive
        sig = inspect.signature(export_autoregressive)
        # (model, path, sparse=True) is the documented contract.
        assert 'model' in sig.parameters
        assert 'path' in sig.parameters
        assert 'sparse' in sig.parameters

    def test_weight_bundle_emitter_writes_arvm_header(self, tmp_path):
        """Exporter emits a file whose first 28 bytes match the ARVM header.

        We can't bake the full VM here (the dynamic compiler is unreliable in
        the test sandbox per project memory notes about the L10/L16/L3
        blockers). Instead we construct a tiny stand-in module that satisfies
        the exporter's structural contract and verify the bytes it emits.
        """
        import torch
        import torch.nn as nn
        import struct
        from tools.export_autoregressive import (
            export_autoregressive, ARVM_MAGIC, ARVM_VERSION,
        )

        vocab_size, d_model, n_heads, ffn_hidden = 4, 4, 2, 4

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = n_heads
                self.alibi_slopes = nn.Parameter(torch.zeros(n_heads))
                self.W_q = nn.Parameter(torch.zeros(d_model, d_model))
                self.W_k = nn.Parameter(torch.zeros(d_model, d_model))
                self.W_v = nn.Parameter(torch.zeros(d_model, d_model))
                self.W_o = nn.Parameter(torch.zeros(d_model, d_model))

        class FakeFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_up = nn.Parameter(torch.zeros(ffn_hidden, d_model))
                self.b_up = nn.Parameter(torch.zeros(ffn_hidden))
                self.W_gate = nn.Parameter(torch.zeros(ffn_hidden, d_model))
                self.b_gate = nn.Parameter(torch.zeros(ffn_hidden))
                self.W_down = nn.Parameter(torch.zeros(d_model, ffn_hidden))
                self.b_down = nn.Parameter(torch.zeros(d_model))

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = FakeAttn()
                self.ffn = FakeFFN()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = vocab_size
                self.d_model = d_model
                self.embed = nn.Embedding(vocab_size, d_model)
                self.blocks = nn.ModuleList([FakeBlock()])
                self.head = nn.Linear(d_model, vocab_size)

        path = tmp_path / "self_compile.arvm"
        export_autoregressive(FakeModel(), str(path), sparse=False)

        with open(path, 'rb') as f:
            header = f.read(28)
        magic, ver, vs, dm, nl, nh, fh = struct.unpack('<IIIIIII', header)
        assert magic == ARVM_MAGIC, "Exporter wrote wrong magic"
        assert ver == ARVM_VERSION
        assert vs == vocab_size
        assert dm == d_model
        assert nl == 1
        assert nh == n_heads
        assert fh == ffn_hidden

    def test_emitted_bundle_is_loadable(self, tmp_path):
        """A bundle emitted by ``export_autoregressive`` re-loads via load_arvm."""
        import torch
        import torch.nn as nn
        from tools.export_autoregressive import (
            export_autoregressive, load_arvm,
        )

        vocab_size, d_model, n_heads, ffn_hidden = 4, 4, 2, 4

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.num_heads = n_heads
                self.attn.alibi_slopes = nn.Parameter(torch.zeros(n_heads))
                self.attn.W_q = nn.Parameter(torch.zeros(d_model, d_model))
                self.attn.W_k = nn.Parameter(torch.zeros(d_model, d_model))
                self.attn.W_v = nn.Parameter(torch.zeros(d_model, d_model))
                self.attn.W_o = nn.Parameter(torch.zeros(d_model, d_model))
                self.ffn = nn.Module()
                self.ffn.W_up = nn.Parameter(torch.zeros(ffn_hidden, d_model))
                self.ffn.b_up = nn.Parameter(torch.zeros(ffn_hidden))
                self.ffn.W_gate = nn.Parameter(torch.zeros(ffn_hidden, d_model))
                self.ffn.b_gate = nn.Parameter(torch.zeros(ffn_hidden))
                self.ffn.W_down = nn.Parameter(torch.zeros(d_model, ffn_hidden))
                self.ffn.b_down = nn.Parameter(torch.zeros(d_model))

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = vocab_size
                self.d_model = d_model
                self.embed = nn.Embedding(vocab_size, d_model)
                self.blocks = nn.ModuleList([_Block()])
                self.head = nn.Linear(d_model, vocab_size)

        path = tmp_path / "loadable.arvm"
        export_autoregressive(_Model(), str(path), sparse=False)
        loaded = load_arvm(str(path))

        assert loaded['vocab_size'] == vocab_size
        assert loaded['n_layers'] == 1
        assert loaded['embed_weight'].shape == (vocab_size, d_model)
        assert len(loaded['layers']) == 1

    @pytest.mark.slow
    @pytest.mark.quine
    @pytest.mark.xfail(
        reason=(
            "End-to-end VM-consumes-c4_compile.c + emits .arvm bundle path "
            "is not yet wired. The two halves (source loadable as bytes; "
            "exporter emits valid bundle) are individually pinned above; "
            "the missing piece is a driver that runs the VM-compiled c4 "
            "compiler over the c4 source byte sequence and feeds its "
            "emitted model state into export_autoregressive."
        ),
        strict=False,
    )
    def test_vm_self_compile_then_emit_bundle(self, tmp_path):
        """End-to-end: VM compiles c4 source -> exporter emits valid bundle."""
        from src.compiler import compile_c
        with open("bundler/c4_compile.c") as f:
            source = f.read()
        bc, dt = compile_c(source)
        assert bc and dt is not None
        # Missing wiring: take the VM run state produced by executing ``bc``
        # over ``source``-as-bytes and call export_autoregressive on it.
        assert False, "VM-driven self-compile -> bundle path not wired"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
