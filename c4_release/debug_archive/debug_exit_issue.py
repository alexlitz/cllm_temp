"""
Debug EXIT instruction: why PC=0x32 and AX=0x00 instead of PC=0x0a, AX=0x2a?
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model
print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model = model.cuda()
model.eval()

# Build bytecode
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate step 0
print("\nGenerating step 0...")
for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Extract step 0 state
step0_end = len(context) - 1
print(f"Step 0 ends at position {step0_end}")

# Find PC and AX from step 0
reg_pc_idx = None
reg_ax_idx = None
for i in range(step0_end, -1, -1):
    if context[i] == Token.REG_PC and reg_pc_idx is None:
        reg_pc_idx = i
    if context[i] == Token.REG_AX and reg_ax_idx is None:
        reg_ax_idx = i
    if reg_pc_idx and reg_ax_idx:
        break

pc0 = context[reg_pc_idx+1] | (context[reg_pc_idx+2] << 8) | (context[reg_pc_idx+3] << 16) | (context[reg_pc_idx+4] << 24)
ax0 = context[reg_ax_idx+1] | (context[reg_ax_idx+2] << 8) | (context[reg_ax_idx+3] << 16) | (context[reg_ax_idx+4] << 24)

print(f"\nStep 0 state:")
print(f"  PC: 0x{pc0:08x}")
print(f"  AX: 0x{ax0:08x}")

# Generate step 1 REG_PC marker
print(f"\nGenerating step 1...")
tok = model.generate_next(context)
context.append(tok)
print(f"  Token: {tok} (REG_PC={Token.REG_PC})")

# Generate PC bytes
pc_bytes = []
for i in range(4):
    tok = model.generate_next(context)
    context.append(tok)
    pc_bytes.append(tok)

pc1 = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
print(f"\nStep 1 PC: 0x{pc1:08x}")
print(f"  Expected: 0x{pc0:08x} (EXIT doesn't increment PC)")
print(f"  Difference: {pc1 - pc0} bytes ({(pc1 - pc0) // 8} instruction slots)")

# Check what the PC increment logic should do
print(f"\nPC increment analysis:")
print(f"  INSTR_WIDTH = 8")
print(f"  Normal increment: PC + 8 = 0x{pc0 + 8:08x}")
print(f"  Actual: 0x{pc1:08x}")
print(f"  Extra increment: {pc1 - pc0 - 8} bytes")

# Generate REG_AX marker and bytes
for _ in range(1):  # REG_AX marker
    tok = model.generate_next(context)
    context.append(tok)

ax_bytes = []
for i in range(4):
    tok = model.generate_next(context)
    context.append(tok)
    ax_bytes.append(tok)

ax1 = ax_bytes[0] | (ax_bytes[1] << 8) | (ax_bytes[2] << 16) | (ax_bytes[3] << 24)
print(f"\nStep 1 AX: 0x{ax1:08x}")
print(f"  Expected: 0x{ax0:08x} (EXIT preserves AX)")
if ax1 == 0:
    print(f"  ✗ AX was cleared to 0!")
elif ax1 == ax0:
    print(f"  ✓ AX preserved correctly")
else:
    print(f"  ✗ AX changed unexpectedly")
