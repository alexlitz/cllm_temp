"""Test if ACTIVE_OPCODE_PRTF is being set correctly."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

# Find PRTF instruction
for i, instr in enumerate(code):
    op = instr & 0xFF
    if op == 33:  # PRTF
        print(f"PRTF at instruction {i}, PC=0x{i*6:04x}")

runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Check embeddings after setting active_opcode
print("\nTesting active_opcode injection:")
runner.model.set_active_opcode(33)  # PRTF

# Create a dummy context
test_tokens = torch.tensor([[Token.REG_PC, 0, 0, 0, 0]], device=runner.model.embed.embed.weight.device)

# Get embeddings
with torch.no_grad():
    x = runner.model.embed(test_tokens, active_opcode=33)

    print(f"Context shape: {x.shape}")
    print(f"ACTIVE_OPCODE_PRTF at position 0: {x[0, 0, BD.ACTIVE_OPCODE_PRTF].item():.2f}")
    print(f"ACTIVE_OPCODE_PRTF at position 1: {x[0, 1, BD.ACTIVE_OPCODE_PRTF].item():.2f}")
    print(f"ACTIVE_OPCODE_PRTF at position 2: {x[0, 2, BD.ACTIVE_OPCODE_PRTF].item():.2f}")
    print(f"ACTIVE_OPCODE_PRTF at position 3: {x[0, 3, BD.ACTIVE_OPCODE_PRTF].item():.2f}")
    print(f"ACTIVE_OPCODE_PRTF at position 4: {x[0, 4, BD.ACTIVE_OPCODE_PRTF].item():.2f}")

    if x[0, 0, BD.ACTIVE_OPCODE_PRTF].item() > 0.5:
        print("\n✓ ACTIVE_OPCODE_PRTF is being set correctly!")
    else:
        print("\n✗ ACTIVE_OPCODE_PRTF is NOT set")

# Now test through L5 FFN
print("\nTesting through L5 FFN:")
with torch.no_grad():
    x = runner.model.embed(test_tokens, active_opcode=33)

    # Run through first 6 layers
    for i in range(6):
        x = runner.model.blocks[i](x)

    print(f"After L5:")
    for pos in range(5):
        print(f"  IO_IS_PRTF at position {pos}: {x[0, pos, BD.IO_IS_PRTF].item():.2f}")

    # Check if ANY position has IO_IS_PRTF > 0.5
    max_io_is_prtf = x[0, :, BD.IO_IS_PRTF].max().item()
    if max_io_is_prtf > 0.5:
        print(f"\n✓ IO_IS_PRTF is being set by L5 FFN! (max={max_io_is_prtf:.2f})")
    else:
        print(f"\n✗ IO_IS_PRTF is NOT set by L5 FFN")
        print(f"   (expected ~5.0, got max {max_io_is_prtf:.2f})")
