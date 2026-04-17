"""
Check if Layer 3 AX carry-forward is working.
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
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate step 0
for _ in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find step 0 AX byte 0
ax_idx = None
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.REG_AX:
        ax_idx = i
        break

ax_byte0_pos = ax_idx + 1
print(f"Step 0 AX byte 0 at position {ax_byte0_pos}, value: 0x{context[ax_byte0_pos]:02x}")

# Generate step 1 up to REG_AX marker
print(f"\nGenerating step 1 up to REG_AX...")
for _ in range(6):  # REG_PC + 4 PC bytes + REG_AX
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        break

reg_ax_pos = len(context) - 1
print(f"Step 1 REG_AX marker at position {reg_ax_pos}")

# Forward pass to check Layer 3 carry-forward
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    
    # Check L1H1/L1H0 at step 0 AX byte 0
    for i in range(2):
        x = model.blocks[i](x, kv_cache=None)
    
    print(f"\nAfter Layer 1, at step 0 AX byte 0 (pos {ax_byte0_pos}):")
    print(f"  L1H1[AX] (marker_idx=1): {x[0, ax_byte0_pos, BD.L1H1 + 1].item():.3f}")
    print(f"  L1H0[AX] (marker_idx=1): {x[0, ax_byte0_pos, BD.L1H0 + 1].item():.3f}")
    print(f"  EMBED_LO[10]: {x[0, ax_byte0_pos, BD.EMBED_LO + 10].item():.3f}")
    print(f"  EMBED_HI[2]: {x[0, ax_byte0_pos, BD.EMBED_HI + 2].item():.3f}")
    
    # Forward through Layer 2
    x = model.blocks[2](x, kv_cache=None)
    
    # Layer 3 carry-forward
    x_before_l3 = x.clone()
    x = model.blocks[3](x, kv_cache=None)
    
    print(f"\nAfter Layer 3, at step 1 REG_AX marker (pos {reg_ax_pos}):")
    print(f"  AX_CARRY_LO[10]: {x[0, reg_ax_pos, BD.AX_CARRY_LO + 10].item():.3f}")
    print(f"  AX_CARRY_HI[2]: {x[0, reg_ax_pos, BD.AX_CARRY_HI + 2].item():.3f}")
    
    if abs(x[0, reg_ax_pos, BD.AX_CARRY_LO + 10].item() - 1.0) < 0.1 and \
       abs(x[0, reg_ax_pos, BD.AX_CARRY_HI + 2].item() - 1.0) < 0.1:
        print(f"  ✓ AX_CARRY set correctly!")
    else:
        print(f"  ✗ AX_CARRY not set!")
        
        # Debug: Check all AX_CARRY values
        print(f"\n  All AX_CARRY values:")
        for k in range(16):
            lo = x[0, reg_ax_pos, BD.AX_CARRY_LO + k].item()
            hi = x[0, reg_ax_pos, BD.AX_CARRY_HI + k].item()
            if abs(lo) > 0.1 or abs(hi) > 0.1:
                print(f"    LO[{k}]: {lo:.2f}, HI[{k}]: {hi:.2f}")
