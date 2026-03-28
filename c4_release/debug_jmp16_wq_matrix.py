"""Debug JMP 16 - Check W_q matrix for L5 head 3."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode

model = AutoregressiveVM()
set_vm_weights(model)

print("JMP 16 - L5 Head 3 W_q Matrix Check")
print("=" * 70)
print()

attn = model.blocks[5].attn
HD = attn.head_dim
head_idx = 3
base = head_idx * HD

# Check W_q for head 3
W_q = attn.W_q[base:base+HD, :]

print(f"Checking W_q for head 3 (rows {base} to {base+HD-1}):")
print()

# Check specific dimensions that should be set
print("Expected non-zero rows:")
print(f"  Row {base+1} (Q[1], for ADDR_KEY_LO[1]): should read CONST")
print(f"  Row {base+16} (Q[16], for ADDR_KEY_HI[0]): should read CONST")
print(f"  Row {base+32} (Q[32], third nibble): should read MARK_PC")
print(f"  Row {base+33} (Q[33], anti-leakage): should read MARK_PC and CONST")
print()

# Check row base+0 (Q[0]) - this should be ZERO but we're seeing Q[0]=20
print(f"Row {base+0} (Q[0] - SHOULD BE ZERO):")
w_q_0 = W_q[0, :]
non_zero = torch.nonzero(torch.abs(w_q_0) > 0.01)
if len(non_zero) > 0:
    print(f"  ⚠️  NON-ZERO! Has {len(non_zero)} non-zero elements:")
    for idx in non_zero[:10]:  # Show first 10
        dim = idx.item()
        val = w_q_0[dim].item()
        # Identify the dimension
        if dim == BD.CONST:
            name = "CONST"
        elif dim == BD.MARK_PC:
            name = "MARK_PC"
        elif dim == BD.MARK_AX:
            name = "MARK_AX"
        elif dim == BD.HAS_SE:
            name = "HAS_SE"
        else:
            name = f"dim_{dim}"
        print(f"    {name}: {val:.2f}")
else:
    print("  ✓ Zero (as expected)")
print()

# Check row base+1 (Q[1])
print(f"Row {base+1} (Q[1] - should read CONST):")
w_q_1 = W_q[1, :]
if abs(w_q_1[BD.CONST].item() - 20.0) < 0.1:
    print(f"  ✓ CONST weight = {w_q_1[BD.CONST].item():.2f}")
else:
    print(f"  ✗ CONST weight = {w_q_1[BD.CONST].item():.2f} (expected 20.0)")
print()

print("Analysis:")
print("  If W_q[base+0] has non-zero weights, that's the bug!")
print("  This causes Q[0] to be set when it shouldn't be.")
print("  Position 1 (address 0) then matches via K[0] instead of K[1].")
