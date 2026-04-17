#!/usr/bin/env python3
"""
End-to-end tests for autoregressive VM ARVM export + C runtime.

Tests:
  1. Export round-trip: bake -> export -> reload -> verify forward pass matches
  2. C runtime: bake -> export -> bundle -> gcc -> run -> verify exit code
  3. Python generation sanity check

Usage:
    python test_onnx_autoregressive.py
"""

import os
import sys
import re
import subprocess
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner
from tools.export_autoregressive import export_autoregressive, load_arvm
from test_bake_v2 import bake_v2


def make_bytecode(*instructions):
    """Build bytecode list from (op, imm) pairs."""
    return [op | (imm << 8) for op, imm in instructions]


def _make_baked_model(n_layers=8):
    """Create and bake an AutoregressiveVM using bake_v2 (tested, working)."""
    model = AutoregressiveVM(d_model=256, n_layers=n_layers, n_heads=4,
                             ffn_hidden=1024)
    bake_v2(model)
    model.eval()
    return model


class TestExportRoundTrip(unittest.TestCase):
    """Test that export -> load round-trips weights correctly."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_baked_model()
        cls.tmpdir = tempfile.mkdtemp()
        cls.arvm_path = os.path.join(cls.tmpdir, 'test.arvm')
        export_autoregressive(cls.model, cls.arvm_path, sparse=True)
        cls.loaded = load_arvm(cls.arvm_path)

    def test_header(self):
        """Verify header fields match model config."""
        self.assertEqual(self.loaded['vocab_size'], Token.VOCAB_SIZE)
        self.assertEqual(self.loaded['d_model'], 256)
        self.assertEqual(self.loaded['n_layers'], 8)
        self.assertEqual(self.loaded['n_heads'], 4)
        self.assertEqual(self.loaded['ffn_hidden'], 1024)

    def test_embed_weight(self):
        """Verify embedding weights round-trip."""
        orig = self.model.embed.weight.detach().cpu().numpy()
        loaded = self.loaded['embed_weight']
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_head_weight(self):
        """Verify head weights round-trip."""
        orig = self.model.head.weight.detach().cpu().numpy()
        loaded = self.loaded['head_weight']
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_head_bias(self):
        """Verify head bias round-trips."""
        orig = self.model.head.bias.detach().cpu().numpy()
        loaded = self.loaded['head_bias']
        np.testing.assert_allclose(loaded, orig, atol=1e-6)

    def test_layer_weights(self):
        """Verify all layer weights round-trip."""
        n_layers = len(self.model.blocks)
        for i in range(n_layers):
            block = self.model.blocks[i]
            layer = self.loaded['layers'][i]

            for name in ['W_q', 'W_k', 'W_v', 'W_o']:
                orig = getattr(block.attn, name).detach().cpu().numpy()
                np.testing.assert_allclose(
                    layer[name], orig, atol=1e-6,
                    err_msg=f"Layer {i} {name} mismatch"
                )

            orig_slopes = block.attn.alibi_slopes.detach().cpu().numpy()
            np.testing.assert_allclose(
                layer['alibi_slopes'], orig_slopes, atol=1e-6,
                err_msg=f"Layer {i} alibi_slopes mismatch"
            )

            for name in ['W_up', 'b_up', 'W_gate', 'b_gate', 'W_down', 'b_down']:
                orig = getattr(block.ffn, name).detach().cpu().numpy()
                np.testing.assert_allclose(
                    layer[name], orig, atol=1e-6,
                    err_msg=f"Layer {i} {name} mismatch"
                )

    def test_forward_pass_matches(self):
        """Verify forward pass with loaded weights matches original model."""
        bytecode = make_bytecode((Opcode.IMM, 42), (Opcode.EXIT, 0))
        runner = AutoregressiveVMRunner()
        ctx = runner._build_context(bytecode, b'', [])

        token_ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            orig_logits = self.model(token_ids)[0, -1, :].numpy()

        # Rebuild model from loaded weights
        n_layers = self.loaded['n_layers']
        ffn_hidden = self.loaded['ffn_hidden']
        model2 = AutoregressiveVM(d_model=256, n_layers=n_layers, n_heads=4,
                                  ffn_hidden=ffn_hidden)
        w = self.loaded
        with torch.no_grad():
            model2.embed.weight.copy_(torch.from_numpy(w['embed_weight']))
            model2.head.weight.copy_(torch.from_numpy(w['head_weight']))
            model2.head.bias.copy_(torch.from_numpy(w['head_bias']))
            for i in range(n_layers):
                layer = w['layers'][i]
                block = model2.blocks[i]
                block.attn.alibi_slopes.copy_(torch.from_numpy(layer['alibi_slopes']))
                for name in ['W_q', 'W_k', 'W_v', 'W_o']:
                    getattr(block.attn, name).copy_(torch.from_numpy(layer[name]))
                for name in ['W_up', 'b_up', 'W_gate', 'b_gate', 'W_down', 'b_down']:
                    getattr(block.ffn, name).copy_(torch.from_numpy(layer[name]))

        model2.eval()
        with torch.no_grad():
            loaded_logits = model2(token_ids)[0, -1, :].numpy()

        np.testing.assert_allclose(loaded_logits, orig_logits, atol=1e-5)

    def test_sparse_file_smaller(self):
        """Verify sparse export is smaller than dense."""
        dense_path = os.path.join(self.tmpdir, 'test_dense.arvm')
        export_autoregressive(self.model, dense_path, sparse=False)

        sparse_size = os.path.getsize(self.arvm_path)
        dense_size = os.path.getsize(dense_path)
        print(f"  Sparse: {sparse_size} bytes, Dense: {dense_size} bytes, "
              f"Ratio: {sparse_size/dense_size:.2%}")
        self.assertLess(sparse_size, dense_size)


class TestCRuntime(unittest.TestCase):
    """Test C runtime compilation and execution."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_baked_model()
        cls.tmpdir = tempfile.mkdtemp()
        cls.arvm_path = os.path.join(cls.tmpdir, 'test.arvm')
        export_autoregressive(cls.model, cls.arvm_path, sparse=True)

        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.bundler_py = os.path.join(cls.base_dir, 'bundler', 'neural_bundler.py')

    def _run_program(self, bytecode, label="test"):
        """Bundle + compile + run a program, return exit code."""
        # Write minimal C source (bundler compiles it to bytecode)
        c_source = os.path.join(self.tmpdir, f'{label}.c')
        with open(c_source, 'w') as f:
            f.write('int main() { return 0; }\n')

        bundled_c = os.path.join(self.tmpdir, f'{label}_bundled.c')
        bundled_bin = os.path.join(self.tmpdir, f'{label}_bundled')

        # Bundle
        result = subprocess.run(
            [sys.executable, self.bundler_py, self.arvm_path, c_source,
             '--runtime', 'autoregressive', '-o', bundled_c],
            capture_output=True, text=True, cwd=self.base_dir,
        )
        if result.returncode != 0:
            self.fail(f"Bundle failed: {result.stderr}")

        # Patch bytecode to our specific program
        self._patch_bytecode(bundled_c, bytecode)

        # Compile
        result = subprocess.run(
            ['gcc', '-O2', '-o', bundled_bin, bundled_c, '-lm'],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.fail(f"Compile failed: {result.stderr}")

        # Run (generous timeout for full recomputation)
        result = subprocess.run(
            [bundled_bin], capture_output=True, timeout=300,
        )
        return result.returncode

    def _patch_bytecode(self, bundled_c_path, bytecode):
        """Replace the program_code array in the bundled C file."""
        with open(bundled_c_path, 'r') as f:
            content = f.read()

        entries = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= 0x80000000:
                imm -= 0x100000000
            entries.append(f"  {{{op}, {imm}}}")

        new_array = "int program_code[][2] = {\n" + ",\n".join(entries) + "\n};\n"
        new_array += f"int program_code_len = {len(bytecode)};"

        # [^;]* matches across inner braces (entries like {3,16} have no semicolons)
        content = re.sub(
            r'int program_code\[\]\[2\] = \{[^;]*\};\s*int program_code_len = \d+;',
            new_array,
            content,
        )

        with open(bundled_c_path, 'w') as f:
            f.write(content)

    def test_imm42_exit(self):
        """IMM 42; EXIT -> exit code 42."""
        bytecode = make_bytecode((Opcode.IMM, 42), (Opcode.EXIT, 0))
        exit_code = self._run_program(bytecode, "imm42")
        self.assertEqual(exit_code, 42)

    def test_imm5_imm3_exit(self):
        """IMM 5; IMM 3; EXIT -> exit code 3 (last IMM wins)."""
        bytecode = make_bytecode(
            (Opcode.IMM, 5), (Opcode.IMM, 3), (Opcode.EXIT, 0)
        )
        exit_code = self._run_program(bytecode, "imm53")
        self.assertEqual(exit_code, 3)

    def test_imm0_exit(self):
        """IMM 0; EXIT -> exit code 0."""
        bytecode = make_bytecode((Opcode.IMM, 0), (Opcode.EXIT, 0))
        exit_code = self._run_program(bytecode, "imm0")
        self.assertEqual(exit_code, 0)


class TestPythonGeneration(unittest.TestCase):
    """Verify Python-side generation produces correct exit codes."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_baked_model()

    def _generate_exit_code(self, bytecode):
        """Build context, generate tokens, extract exit code."""
        runner = AutoregressiveVMRunner()
        runner.model = self.model
        ctx = runner._build_context(bytecode, b'', [])

        for _ in range(200):
            next_tok = self.model.generate_next(ctx)
            ctx.append(next_tok)
            if next_tok == Token.HALT:
                break

        return runner._decode_exit_code(ctx)

    def test_imm42(self):
        bytecode = make_bytecode((Opcode.IMM, 42), (Opcode.EXIT, 0))
        self.assertEqual(self._generate_exit_code(bytecode), 42)

    def test_imm0(self):
        bytecode = make_bytecode((Opcode.IMM, 0), (Opcode.EXIT, 0))
        self.assertEqual(self._generate_exit_code(bytecode), 0)

    def test_imm5_imm3(self):
        bytecode = make_bytecode(
            (Opcode.IMM, 5), (Opcode.IMM, 3), (Opcode.EXIT, 0)
        )
        self.assertEqual(self._generate_exit_code(bytecode), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
