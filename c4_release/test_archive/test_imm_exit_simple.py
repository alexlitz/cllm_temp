"""
Simple test for IMM+EXIT.
"""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token

print("Building runner...")
runner = AutoregressiveVMRunner()

print("Compiling bytecode...")
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("Generating step 0...")
for i in range(100):
    tok = runner.model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"  Step 0 ended at token {i}")
        break

# Check step 0 output
ax_idx = None
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.REG_AX:
        ax_idx = i
        break

if ax_idx:
    ax_bytes = context[ax_idx+1:ax_idx+5]
    ax_value = (ax_bytes[3] << 24) | (ax_bytes[2] << 16) | (ax_bytes[1] << 8) | ax_bytes[0]
    print(f"  Step 0 AX = 0x{ax_value:08x} (expected 0x2a)")

print("\nGenerating step 1...")
for i in range(100):
    tok = runner.model.generate_next(context)
    context.append(tok)
    print(f"  Token {i}: 0x{tok:02x}")
    if tok == Token.HALT:
        print(f"  HALT at token {i}")
        break
    if i >= 20:  # Safety limit
        print(f"  Stopped at 20 tokens")
        break

print(f"\nFinal context length: {len(context)}")
