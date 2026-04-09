"""
Check if byte tokens have EMBED_LO/HI set.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

# Build model
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Test byte token 0x2a
byte_val = 0x2a
context = [byte_val]

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    
    print(f"Byte token 0x{byte_val:02x} embedding:")
    print(f"  EMBED_LO values:")
    for k in range(16):
        val = x[0, 0, BD.EMBED_LO + k].item()
        if abs(val) > 0.1:
            print(f"    [{k}]: {val:.2f}")
    
    print(f"  EMBED_HI values:")
    for k in range(16):
        val = x[0, 0, BD.EMBED_HI + k].item()
        if abs(val) > 0.1:
            print(f"    [{k}]: {val:.2f}")
    
    # Expected: LO[10]=1 (0xA), HI[2]=1 (0x2)
    lo_nibble = byte_val & 0xF
    hi_nibble = (byte_val >> 4) & 0xF
    print(f"\n  Expected: EMBED_LO[{lo_nibble}]=1, EMBED_HI[{hi_nibble}]=1")
    
    lo_val = x[0, 0, BD.EMBED_LO + lo_nibble].item()
    hi_val = x[0, 0, BD.EMBED_HI + hi_nibble].item()
    
    if abs(lo_val - 1.0) < 0.1 and abs(hi_val - 1.0) < 0.1:
        print(f"  ✓ Byte embedding has correct EMBED_LO/HI!")
    else:
        print(f"  ✗ Byte embedding missing EMBED_LO/HI!")
        print(f"    Actual: EMBED_LO[{lo_nibble}]={lo_val:.2f}, EMBED_HI[{hi_nibble}]={hi_val:.2f}")
