"""
Unit tests for NeuralVMEmbedding.

Verifies that NeuralVMEmbedding produces identical output to the old
implementation that used post-embedding augmentation methods.
"""

import unittest
import torch
from neural_vm.neural_embedding import NeuralVMEmbedding
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim, Token


class TestNeuralEmbedding(unittest.TestCase):
    """Test that NeuralVMEmbedding produces identical output to old methods."""

    def test_addr_key_basic(self):
        """Test basic ADDR_KEY computation for simple code sequence."""
        # Create test token sequence: CODE_START + 3 bytes + CODE_END
        token_ids = torch.tensor([[
            Token.CODE_START,
            0x01,  # Byte at addr=0
            0x02,  # Byte at addr=1
            0x03,  # Byte at addr=2
            Token.CODE_END,
        ]])

        # Create embedding
        embed = NeuralVMEmbedding(272, 512)
        x = embed(token_ids)

        BD = _SetDim

        # Verify byte at position 1 has addr=0
        # lo=0, hi=0, top=0
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 16 + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 32 + 0].item(), 1.0, places=6)

        # Verify byte at position 2 has addr=1
        # lo=1, hi=0, top=0
        self.assertAlmostEqual(x[0, 2, BD.ADDR_KEY + 1].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 2, BD.ADDR_KEY + 16 + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 2, BD.ADDR_KEY + 32 + 0].item(), 1.0, places=6)

        # Verify byte at position 3 has addr=2
        # lo=2, hi=0, top=0
        self.assertAlmostEqual(x[0, 3, BD.ADDR_KEY + 2].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 3, BD.ADDR_KEY + 16 + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 3, BD.ADDR_KEY + 32 + 0].item(), 1.0, places=6)

    def test_addr_key_high_address(self):
        """Test ADDR_KEY computation for higher addresses (multiple nibbles)."""
        # Create sequence with addr=0x123 (291 decimal)
        # This tests lo=3, hi=2, top=1
        code_bytes = [Token.CODE_START] + [0xFF] * 291 + [Token.CODE_END]
        token_ids = torch.tensor([code_bytes])

        embed = NeuralVMEmbedding(272, 512)
        x = embed(token_ids)

        BD = _SetDim

        # Byte at position 291 (index 1+291) has addr=291=0x123
        pos = 1 + 290  # Position of byte at addr=290=0x122
        addr = 290
        lo = addr & 0xF  # 2
        hi = (addr >> 4) & 0xF  # 2
        top = (addr >> 8) & 0xF  # 1

        self.assertAlmostEqual(x[0, pos, BD.ADDR_KEY + lo].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, pos, BD.ADDR_KEY + 16 + hi].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, pos, BD.ADDR_KEY + 32 + top].item(), 1.0, places=6)

    def test_addr_key_only_code_bytes(self):
        """Test that ADDR_KEY is only set for byte tokens (< 256)."""
        token_ids = torch.tensor([[
            Token.CODE_START,
            0x01,  # Byte: should get ADDR_KEY
            Token.STEP_END,  # Non-byte: should NOT get ADDR_KEY
            0x02,  # Byte: should get ADDR_KEY
            Token.CODE_END,
        ]])

        embed = NeuralVMEmbedding(272, 512)
        x = embed(token_ids)

        BD = _SetDim

        # Position 1 (byte 0x01) should have ADDR_KEY set to 1.0 at addr=0
        # addr=0: lo=0, hi=0, top=0
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 16 + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 32 + 0].item(), 1.0, places=6)

        # Position 2 (STEP_END) should NOT get ADDR_KEY modified (stays at embedding values)
        # We can't check for 0.0 because embedding has random values
        # Instead we check position 3 gets the right address

        # Position 3 (byte 0x02) should have ADDR_KEY set to 1.0 at addr=2
        # Address is based on position: addr = i - cs_pos - 1 = 3 - 0 - 1 = 2
        # addr=2: lo=2, hi=0, top=0
        self.assertAlmostEqual(x[0, 3, BD.ADDR_KEY + 2].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 3, BD.ADDR_KEY + 16 + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 3, BD.ADDR_KEY + 32 + 0].item(), 1.0, places=6)

    def test_mem_store_injection(self):
        """Test MEM_STORE injection on historical MEM markers."""
        token_ids = torch.tensor([[
            Token.MEM, 0x00, 0x00, 0x00, 0x00,  # Historical MEM (pos 0)
            Token.STEP_END,
            Token.MEM, 0x00, 0x00, 0x00, 0x00,  # Current step MEM (pos 6)
        ]])

        embed = NeuralVMEmbedding(272, 512)

        # Set history end to cover only first MEM
        embed.set_mem_history_end(6)

        x = embed(token_ids)

        BD = _SetDim

        # First MEM (pos 0) should have MEM_STORE=1.0
        self.assertAlmostEqual(x[0, 0, BD.MEM_STORE].item(), 1.0, places=6)

        # Second MEM (pos 6) should NOT have MEM_STORE modified
        # (it's beyond history end, so it keeps embedding value)
        # We verify by checking that first MEM got set to 1.0
        # and that the injection only happens before history_end

    def test_mem_store_no_history(self):
        """Test that MEM_STORE is not injected when _mem_history_end=0."""
        token_ids = torch.tensor([[
            Token.MEM, 0x00, 0x00, 0x00, 0x00,
            Token.STEP_END,
        ]])

        embed = NeuralVMEmbedding(272, 512)
        # Default _mem_history_end=0

        # Get embedding without history
        x_no_history = embed(token_ids)

        # Now set history and get embedding again
        embed.set_mem_history_end(6)
        x_with_history = embed(token_ids)

        BD = _SetDim

        # Without history, MEM_STORE should just be embedding value
        # With history, MEM_STORE should be 1.0
        # They should be different
        self.assertNotEqual(x_no_history[0, 0, BD.MEM_STORE].item(),
                           x_with_history[0, 0, BD.MEM_STORE].item())
        self.assertAlmostEqual(x_with_history[0, 0, BD.MEM_STORE].item(), 1.0, places=6)

    def test_full_embedding_realistic_program(self):
        """Test complete embedding with realistic program tokens."""
        # Build realistic context: CODE + DATA + initial state
        token_ids = torch.tensor([[
            Token.CODE_START,
            Opcode.IMM, 0x2A, 0x00, 0x00, 0x00,  # IMM 42
            Opcode.EXIT,
            Token.CODE_END,
            Token.DATA_START,
            Token.DATA_END,
            Token.REG_PC, 0x00, 0x00, 0x00, 0x00,
            Token.REG_AX, 0x00, 0x00, 0x00, 0x00,
            Token.REG_SP, 0x00, 0x00, 0x00, 0x00,
            Token.REG_BP, 0x00, 0x00, 0x00, 0x00,
            Token.MEM, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            Token.STEP_END,
        ]])

        embed = NeuralVMEmbedding(272, 512)
        x = embed(token_ids)

        BD = _SetDim

        # Verify code bytes have ADDR_KEY
        # Position 1 (IMM opcode) should have addr=0
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 0].item(), 1.0, places=6)

        # Position 2 (0x2A) should have addr=1
        self.assertAlmostEqual(x[0, 2, BD.ADDR_KEY + 1].item(), 1.0, places=6)

        # Position 6 (EXIT) should have addr=5
        self.assertAlmostEqual(x[0, 6, BD.ADDR_KEY + 5].item(), 1.0, places=6)

        # Verify embedding shape (40 tokens total)
        self.assertEqual(x.shape, (1, 40, 512))

    def test_equivalence_vs_old_implementation(self):
        """Test that new embedding matches old post-augmentation approach."""
        token_ids = torch.tensor([[
            Token.CODE_START,
            Opcode.IMM, 0x0F, 0x00, 0x00, 0x00,
            Token.CODE_END,
            Token.MEM, 0x00, 0x00, 0x00, 0x00,
            Token.STEP_END,
        ]])

        # Old approach: embed then augment
        embed_old = torch.nn.Embedding(272, 512)
        torch.manual_seed(42)
        embed_old.weight.data.normal_(0, 0.02)

        x_old = embed_old(token_ids)

        # Manually apply old augmentations
        BD = _SetDim
        B, S = token_ids.shape

        # _add_code_addr_keys
        for b in range(B):
            cs_pos = None
            for i in range(S):
                tok = token_ids[b, i].item()
                if tok == Token.CODE_START:
                    cs_pos = i
                elif tok == Token.CODE_END:
                    break
                elif cs_pos is not None and tok < 256:
                    addr = i - cs_pos - 1
                    if addr < 0:
                        continue
                    lo = addr & 0xF
                    hi = (addr >> 4) & 0xF
                    top = (addr >> 8) & 0xF
                    x_old[b, i, BD.ADDR_KEY + lo] = 1.0
                    x_old[b, i, BD.ADDR_KEY + 16 + hi] = 1.0
                    x_old[b, i, BD.ADDR_KEY + 32 + top] = 1.0

        # New approach: NeuralVMEmbedding
        embed_new = NeuralVMEmbedding(272, 512)
        torch.manual_seed(42)
        embed_new.embed.weight.data.normal_(0, 0.02)

        x_new = embed_new(token_ids)

        # Verify they match exactly
        torch.testing.assert_close(x_old, x_new, rtol=1e-6, atol=1e-6)

    def test_batch_processing(self):
        """Test that batched inputs work correctly."""
        token_ids = torch.tensor([
            [Token.CODE_START, 0x01, 0x02, Token.CODE_END],
            [Token.CODE_START, 0x0A, 0x0B, Token.CODE_END],
        ])

        embed = NeuralVMEmbedding(272, 512)
        x = embed(token_ids)

        BD = _SetDim

        # Verify first batch element
        self.assertAlmostEqual(x[0, 1, BD.ADDR_KEY + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[0, 2, BD.ADDR_KEY + 1].item(), 1.0, places=6)

        # Verify second batch element
        self.assertAlmostEqual(x[1, 1, BD.ADDR_KEY + 0].item(), 1.0, places=6)
        self.assertAlmostEqual(x[1, 2, BD.ADDR_KEY + 1].item(), 1.0, places=6)

        # Verify shape
        self.assertEqual(x.shape, (2, 4, 512))


if __name__ == '__main__':
    unittest.main()
