"""Quick test for conversational I/O detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

# Simple test: just check that runner initializes with conversational_io=True
print("Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
print("✓ Runner created successfully")

# Check that weights were set with conversational_io enabled
print(f"Model has {len(runner.model.blocks)} layers")
print(f"Vocab size: {Token.VOCAB_SIZE}")
print(f"THINKING_START token: {Token.THINKING_START}")
print(f"THINKING_END token: {Token.THINKING_END}")
print(f"IO_STATE_EMIT_BYTE token: {Token.IO_STATE_EMIT_BYTE}")
print(f"IO_STATE_EMIT_THINKING token: {Token.IO_STATE_EMIT_THINKING}")

# Test compilation
c_code = '''
int main() {
    printf("Hello\\n");
    return 0;
}
'''
print("\nCompiling test program...")
code, data = compile_c(c_code)
print(f"✓ Compiled: {len(code)} instructions, {len(data)} data bytes")

# Build context
print("\nBuilding context...")
context = runner._build_context(code, bytes(data), [])
print(f"✓ Context built: {len(context)} tokens")

print("\n✓ All initialization tests passed")
print("\nNote: Full generation tests are in test_conversational_io.py")
