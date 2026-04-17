"""
Debug what PC value is generated at step 1.
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

print(f"\nBytecode (Python):")
print(f"  [0] IMM | (0x{value:02x} << 8) = 0x{bytecode[0]:08x}")
print(f"  [1] EXIT = 0x{bytecode[1]:08x}")

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate step 0
print("\nGenerating step 0...")
while True:
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

print(f"Step 0 complete, context length: {len(context)}")

# Find step 0 PC value
print("\nStep 0 state:")
step_end_idx = context.index(Token.STEP_END)
reg_pc_idx = context.index(Token.REG_PC)
pc_bytes = context[reg_pc_idx+1:reg_pc_idx+5]
print(f"  PC bytes: {[f'{b:02x}' for b in pc_bytes]}")
pc_val = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
print(f"  PC value: 0x{pc_val:08x}")

ax_idx = context.index(Token.REG_AX)
ax_bytes = context[ax_idx+1:ax_idx+5]
print(f"  AX bytes: {[f'{b:02x}' for b in ax_bytes]}")
ax_val = ax_bytes[0] | (ax_bytes[1] << 8) | (ax_bytes[2] << 16) | (ax_bytes[3] << 24)
print(f"  AX value: 0x{ax_val:08x}")

# Generate step 1 up to AX bytes
print("\nGenerating step 1...")
step1_start = len(context)
for _ in range(10):  # REG_PC + 4 PC bytes + REG_AX + first few AX bytes
    tok = model.generate_next(context)
    context.append(tok)

# Find step 1 PC
step1_pc_idx = step1_start  # Should be REG_PC
if context[step1_pc_idx] == Token.REG_PC:
    pc_bytes = context[step1_pc_idx+1:step1_pc_idx+5]
    print(f"  PC bytes: {[f'{b:02x}' for b in pc_bytes]}")
    pc_val = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
    print(f"  PC value: 0x{pc_val:08x}")
    
    # Check what instruction is at this PC
    print(f"\n  Instruction at PC=0x{pc_val:08x}:")
    if pc_val == 0:
        print(f"    IMM 0x2a")
    elif pc_val == 4:
        print(f"    EXIT")
    else:
        print(f"    Unknown (PC out of expected range)")
        
    # What byte should be fetched from bytecode?
    print(f"\n  Memory layout (assuming 4-byte words):")
    print(f"    [0x00-0x03]: IMM | (0x2a << 8) = 0x{bytecode[0]:08x}")
    print(f"    [0x04-0x07]: EXIT = 0x{bytecode[1]:08x}")
    
    # Decode as little-endian bytes
    word0 = bytecode[0]
    word1 = bytecode[1]
    print(f"\n  Byte-by-byte (little-endian):")
    print(f"    [0x00]: 0x{(word0 >> 0) & 0xFF:02x}")
    print(f"    [0x01]: 0x{(word0 >> 8) & 0xFF:02x}")
    print(f"    [0x02]: 0x{(word0 >> 16) & 0xFF:02x}")
    print(f"    [0x03]: 0x{(word0 >> 24) & 0xFF:02x}")
    print(f"    [0x04]: 0x{(word1 >> 0) & 0xFF:02x}")
    print(f"    [0x05]: 0x{(word1 >> 8) & 0xFF:02x}")
    
    byte_at_pc = 0
    if pc_val < 4:
        byte_at_pc = (word0 >> (pc_val * 8)) & 0xFF
    elif pc_val < 8:
        byte_at_pc = (word1 >> ((pc_val - 4) * 8)) & 0xFF
    print(f"\n  Byte at PC=0x{pc_val:08x}: 0x{byte_at_pc:02x}")
