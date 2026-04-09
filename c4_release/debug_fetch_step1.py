"""
Debug Layer 5 fetch at step 1 - why is it fetching 0x00 instead of 0x26?
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.constants import opcode_address

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

print(f"\nBytecode:")
print(f"  [0] 0x{bytecode[0]:08x} (IMM)")
print(f"  [1] 0x{bytecode[1]:08x} (EXIT)")

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Show context bytecode section
code_start = context.index(Token.CODE_START)
code_end = context.index(Token.CODE_END)
code_bytes = context[code_start+1:code_end]

print(f"\nContext bytecode section (bytes {code_start+1} to {code_end-1}):")
for i, b in enumerate(code_bytes):
    print(f"  [{i}]: 0x{b:02x}", end="")
    if i == 0:
        print(" ← IMM opcode", end="")
    elif i == 8:
        print(" ← EXIT opcode", end="")
    print()

# Generate step 0
print("\nGenerating step 0...")
while True:
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Extract PC from step 0
reg_pc_idx = None
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.REG_PC:
        reg_pc_idx = i
        break

pc_bytes = context[reg_pc_idx+1:reg_pc_idx+5]
pc_val = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
print(f"Step 0 PC: 0x{pc_val:08x}")

opcode_addr = opcode_address(pc_val)
print(f"Expected opcode fetch address: {opcode_addr}")
print(f"Expected opcode byte: 0x{code_bytes[opcode_addr]:02x}")

# Generate step 1 up to after L5
print("\nGenerating step 1 REG_PC + 4 bytes + REG_AX...")
for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)

# Forward pass
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    
    # Forward to Layer 5
    for i in range(5):
        x = model.blocks[i](x, kv_cache=None)
    
    reg_pc_pos = len(context) - 1  # REG_AX position
    
    print(f"\nBefore Layer 5 (position {reg_pc_pos}, REG_AX marker):")
    
    # Check EMBED values (should contain PC from L4 relay)
    print(f"  EMBED_LO/HI (PC value relayed from L4):")
    for k in range(16):
        lo = x[0, reg_pc_pos, BD.EMBED_LO + k].item()
        hi = x[0, reg_pc_pos, BD.EMBED_HI + k].item()
        if abs(lo) > 0.1 or abs(hi) > 0.1:
            print(f"    LO[{k}]: {lo:.2f}, HI[{k}]: {hi:.2f}")
    
    # Decode PC value from EMBED
    pc_lo_idx = None
    pc_hi_idx = None
    for k in range(16):
        if x[0, reg_pc_pos, BD.EMBED_LO + k].item() > 0.5:
            pc_lo_idx = k
        if x[0, reg_pc_pos, BD.EMBED_HI + k].item() > 0.5:
            pc_hi_idx = k
    
    if pc_lo_idx is not None and pc_hi_idx is not None:
        pc_from_embed = pc_lo_idx | (pc_hi_idx << 4)
        print(f"\n  Decoded PC from EMBED: 0x{pc_from_embed:02x}")
        fetch_addr = opcode_address(pc_from_embed)
        print(f"  Opcode fetch address: {fetch_addr}")
        print(f"  Expected opcode: 0x{code_bytes[fetch_addr]:02x}")
    
    # Layer 5 attention - this should fetch the opcode
    x_l5_attn = x + model.blocks[5].attn(x, kv_cache=None)
    
    print(f"\n  After Layer 5 attention:")
    print(f"  OPCODE_BYTE values:")
    for k in range(16):
        lo = x_l5_attn[0, reg_pc_pos, BD.OPCODE_BYTE_LO + k].item()
        hi = x_l5_attn[0, reg_pc_pos, BD.OPCODE_BYTE_HI + k].item()
        if abs(lo) > 0.1 or abs(hi) > 0.1:
            print(f"    LO[{k}]: {lo:.2f}, HI[{k}]: {hi:.2f}")
    
    # Decode opcode byte
    op_lo = None
    op_hi = None
    for k in range(16):
        if x_l5_attn[0, reg_pc_pos, BD.OPCODE_BYTE_LO + k].item() > 0.5:
            op_lo = k
        if x_l5_attn[0, reg_pc_pos, BD.OPCODE_BYTE_HI + k].item() > 0.5:
            op_hi = k
    
    if op_lo is not None and op_hi is not None:
        fetched_opcode = op_lo | (op_hi << 4)
        print(f"\n  Decoded opcode: 0x{fetched_opcode:02x}")
        if fetched_opcode == 0x26:
            print(f"    ✓ CORRECT! (EXIT)")
        else:
            print(f"    ✗ WRONG! Expected 0x26 (EXIT)")
