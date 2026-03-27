"""
Strict Neural Prediction Tests

These tests verify that the transformer actually predicts the correct tokens,
not just that DraftVM produces correct results.

CRITICAL: These tests should FAIL if the neural VM weights are incorrect.
This is different from the other tests which use DraftVM execution and pass
even when transformer predictions are wrong.
"""

import unittest
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM


def build_context(bytecode):
    """Build context tokens for bytecode."""
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens


class StrictNeuralPredictionTest(unittest.TestCase):
    """Base class for strict neural prediction tests."""

    @classmethod
    def setUpClass(cls):
        cls.model = AutoregressiveVM()
        set_vm_weights(cls.model)
        cls.model.compact(block_size=32)
        cls.model.compact_moe()
        cls.model.eval()

    def assert_all_tokens_match(self, bytecode, opcode_name):
        """
        Assert that transformer predictions match DraftVM tokens.

        This is the strict validation - if transformer predictions don't match,
        the test MUST fail.

        Uses SPECULATIVE DECODING semantics:
        - DraftVM generates draft tokens (speculation)
        - Transformer verifies each draft token
        - Transformer predictions are AUTHORITATIVE
        - If transformer disagrees, that's a weight bug
        """
        # Generate draft tokens with DraftVM (speculation)
        draft_vm = DraftVM(bytecode)
        context = build_context(bytecode)
        draft_vm.step()
        draft_tokens = draft_vm.draft_tokens()

        # Verify each draft token with the transformer (autoritative verification)
        current_context = context[:]
        mismatches = []

        with torch.no_grad():
            for i in range(35):  # Verify 35 draft tokens (one VM step)
                # Forward pass with current context
                ctx_tensor = torch.tensor([current_context], dtype=torch.long)
                logits = self.model.forward(ctx_tensor)

                # Get transformer's prediction (authoritative)
                predicted = logits[0, -1, :].argmax().item()
                expected = draft_tokens[i]

                # Check if transformer accepts the draft
                if predicted != expected:
                    token_names = (
                        ['REG_PC'] + [f'PC_b{j}' for j in range(4)] +
                        ['REG_AX'] + [f'AX_b{j}' for j in range(4)] +
                        ['REG_SP'] + [f'SP_b{j}' for j in range(4)] +
                        ['REG_BP'] + [f'BP_b{j}' for j in range(4)] +
                        ['STACK0'] + [f'ST_b{j}' for j in range(4)] +
                        ['MEM'] + [f'MEM_a{j}' for j in range(4)] + [f'MEM_v{j}' for j in range(4)] +
                        ['END']
                    )
                    token_name = token_names[i] if i < len(token_names) else f'Token{i}'
                    mismatches.append((i, token_name, expected, predicted))

                # CRITICAL: Use draft token for speculation (not transformer's prediction)
                # This matches speculative decoding: we speculate with draft, verify with transformer
                current_context.append(expected)

        # Report mismatches
        if mismatches:
            error_msg = f"\nTransformer REJECTED draft tokens for {opcode_name}:\n"
            error_msg += f"  Bytecode: {[hex(x) for x in bytecode]}\n"
            error_msg += f"  Mismatches: {len(mismatches)}/35 tokens\n\n"
            for i, name, exp, pred in mismatches[:5]:  # Show first 5
                error_msg += f"  Token {i:2d} ({name:8s}): draft={exp:3d}, transformer={pred:3d}\n"
            if len(mismatches) > 5:
                error_msg += f"  ... and {len(mismatches) - 5} more mismatches\n"
            error_msg += "\nThis means the neural VM weights are INCORRECT."
            error_msg += f"\nTransformer disagreed on {len(mismatches)} tokens (should accept all drafts)."
            self.fail(error_msg)


class TestNOP(StrictNeuralPredictionTest):
    """Test NOP instruction neural predictions."""

    def test_nop(self):
        """NOP should increment PC from 0 to 8."""
        self.assert_all_tokens_match([Opcode.NOP], "NOP")


class TestIMM(StrictNeuralPredictionTest):
    """Test IMM instruction neural predictions."""

    def test_imm_0(self):
        """IMM 0 should set AX=0."""
        self.assert_all_tokens_match([Opcode.IMM | (0 << 8)], "IMM 0")

    def test_imm_42(self):
        """IMM 42 should set AX=42."""
        self.assert_all_tokens_match([Opcode.IMM | (42 << 8)], "IMM 42")

    def test_imm_255(self):
        """IMM 255 should set AX=255."""
        self.assert_all_tokens_match([Opcode.IMM | (255 << 8)], "IMM 255")


class TestJMP(StrictNeuralPredictionTest):
    """Test JMP instruction neural predictions."""

    def test_jmp_16(self):
        """JMP 16 should set PC=16."""
        self.assert_all_tokens_match([Opcode.JMP | (16 << 8)], "JMP 16")

    def test_jmp_8(self):
        """JMP 8 should set PC=8."""
        self.assert_all_tokens_match([Opcode.JMP | (8 << 8)], "JMP 8")


class TestEXIT(StrictNeuralPredictionTest):
    """Test EXIT instruction neural predictions."""

    def test_exit(self):
        """EXIT should emit HALT token (263)."""
        self.assert_all_tokens_match([Opcode.EXIT], "EXIT")


class TestLEA(StrictNeuralPredictionTest):
    """Test LEA instruction neural predictions."""

    def test_lea_8(self):
        """LEA 8 should set AX=BP+8."""
        self.assert_all_tokens_match([Opcode.LEA | (8 << 8)], "LEA 8")


class TestADD(StrictNeuralPredictionTest):
    """Test ADD instruction neural predictions."""

    def test_add_simple(self):
        """Test ADD with simple values."""
        bytecode = [
            Opcode.IMM | (5 << 8),
            Opcode.PSH,
            Opcode.IMM | (3 << 8),
            Opcode.ADD
        ]
        self.assert_all_tokens_match(bytecode, "ADD 5+3")


if __name__ == '__main__':
    unittest.main()
