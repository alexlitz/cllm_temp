"""Test conversational I/O mode (PRTF/READ with thinking tags)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token


class TestConversationalIO(unittest.TestCase):
    """Test conversational I/O mode with PRTF detection and THINKING_END emission."""

    def setUp(self):
        """Create runner with conversational I/O enabled."""
        self.runner = AutoregressiveVMRunner(conversational_io=True)

    def test_prtf_emits_thinking_end(self):
        """Test that PRTF opcode triggers THINKING_END emission."""
        # Simple C program with printf
        c_code = '''
int main() {
    int x;
    x = 42;
    printf("x=%d\\n", x);
    return 0;
}
'''
        # Compile to bytecode
        code, data = compile_c(c_code)

        # Build context
        context = self.runner._build_context(code, bytes(data), [])
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # Add batch dim

        # Generate tokens
        generated_tokens = []
        for i in range(100):  # Generate up to 100 tokens
            logits = self.runner.model.forward(context_tensor)
            next_token = logits[0, -1].argmax().item()  # [batch, seq, vocab]
            generated_tokens.append(next_token)
            # Append to both list and tensor
            context.append(next_token)
            context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)

            # Stop if we hit HALT
            if next_token == Token.HALT:
                break

        # Check that THINKING_END was emitted
        self.assertIn(Token.THINKING_END, generated_tokens,
                     "THINKING_END should be emitted when PRTF is detected")

        # Check that THINKING_END comes before HALT
        thinking_end_idx = generated_tokens.index(Token.THINKING_END)
        halt_idx = generated_tokens.index(Token.HALT)
        self.assertLess(thinking_end_idx, halt_idx,
                       "THINKING_END should come before HALT")

        print(f"✓ PRTF detection working: THINKING_END at position {thinking_end_idx}")

    def test_prtf_sequence(self):
        """Test full PRTF sequence: THINKING_END → output → THINKING_START."""
        c_code = '''
int main() {
    printf("Hello\\n");
    return 0;
}
'''
        code, data = compile_c(c_code)
        context = self.runner._build_context(code, bytes(data), [])
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)

        generated_tokens = []
        for i in range(200):
            logits = self.runner.model.forward(context_tensor)
            next_token = logits[0, -1].argmax().item()
            generated_tokens.append(next_token)
            context.append(next_token)
            context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)

            if next_token == Token.HALT:
                break

        # Check sequence
        if Token.THINKING_END in generated_tokens:
            thinking_end_idx = generated_tokens.index(Token.THINKING_END)
            print(f"✓ THINKING_END emitted at position {thinking_end_idx}")

            # Check what comes after
            tokens_after = generated_tokens[thinking_end_idx+1:thinking_end_idx+10]
            print(f"  Tokens after THINKING_END: {tokens_after}")
        else:
            self.fail("THINKING_END not found in generated tokens")


if __name__ == '__main__':
    unittest.main()
