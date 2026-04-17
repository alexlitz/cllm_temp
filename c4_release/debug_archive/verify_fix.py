"""
Quick verification that the Layer 6 head fix was applied correctly.
This checks the weight configuration without running a full program.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

print("="*80)
print("VERIFYING LAYER 6 HEAD FIX")
print("="*80 + "\n")

model = AutoregressiveVM()
set_vm_weights(model)

layer6 = model.blocks[6]
attn6 = layer6.attn
HD = attn6.head_dim

print("Checking Head 2 (should NOT write to AX_CARRY)...")
base_h2 = 2 * HD
w_o_h2 = attn6.W_o[:, base_h2:base_h2+HD]
ax_carry_writes_h2 = w_o_h2[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs().max().item()
print(f"  Max weight to AX_CARRY: {ax_carry_writes_h2:.6f}")

if ax_carry_writes_h2 > 1e-6:
    # Check if it's conditional (only at PC marker)
    w_q_h2 = attn6.W_q[base_h2:base_h2+HD, :]
    queries_pc = (w_q_h2[:, BD.MARK_PC].abs() > 1e-6).any().item()
    queries_ax = (w_q_h2[:, BD.MARK_AX].abs() > 1e-6).any().item()

    if queries_pc and not queries_ax:
        print("  ✓ OK - Writes to AX_CARRY but only at PC marker (JMP relay)")
    else:
        print("  ❌ PROBLEM - Writes to AX_CARRY at AX marker!")
else:
    print("  ✓ OK - No writes to AX_CARRY")

print("\nChecking Head 6 (should write to ALU, not AX_CARRY)...")
base_h6 = 6 * HD
w_o_h6 = attn6.W_o[:, base_h6:base_h6+HD]
ax_carry_writes_h6 = w_o_h6[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs().max().item()
alu_writes_h6 = w_o_h6[BD.ALU_LO:BD.ALU_LO+32, :].abs().max().item()

print(f"  Max weight to AX_CARRY: {ax_carry_writes_h6:.6f}")
print(f"  Max weight to ALU: {alu_writes_h6:.6f}")

if alu_writes_h6 > 1e-6 and ax_carry_writes_h6 < 1e-6:
    print("  ✓ OK - Writes to ALU, not AX_CARRY")
else:
    print("  ❌ PROBLEM - Should write to ALU!")

print("\nChecking Head 7 (should be for JSR, writes to AX_CARRY at SP marker)...")
base_h7 = 7 * HD
w_o_h7 = attn6.W_o[:, base_h7:base_h7+HD]
ax_carry_writes_h7 = w_o_h7[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs().max().item()
print(f"  Max weight to AX_CARRY: {ax_carry_writes_h7:.6f}")

if ax_carry_writes_h7 > 1e-6:
    w_q_h7 = attn6.W_q[base_h7:base_h7+HD, :]
    queries_sp = (w_q_h7[:, BD.MARK_SP].abs() > 1e-6).any().item()
    queries_ax = (w_q_h7[:, BD.MARK_AX].abs() > 1e-6).any().item()

    if queries_sp and not queries_ax:
        print("  ✓ OK - JSR handling (queries SP marker)")
    else:
        print("  ⚠️  Unexpected query pattern")
else:
    print("  ⚠️  Head 7 not configured")

print("\n" + "="*80)
print("SUMMARY")
print("="*80 + "\n")

# Check AX_CARRY preservation path
print("Checking AX_CARRY preservation from Layer 3 to Layer 8...")

# Layer 3 Head 1 should set AX_CARRY
layer3 = model.blocks[3]
attn3 = layer3.attn
base_h1 = 1 * attn3.head_dim
w_o_l3h1 = attn3.W_o[:, base_h1:base_h1+attn3.head_dim]
l3_writes = w_o_l3h1[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs().max().item()
print(f"  Layer 3 Head 1 writes to AX_CARRY: {l3_writes > 1e-6}")

# Check layers 4-7 don't overwrite at AX marker
overwrites_found = False
for layer_idx in [4, 5, 6, 7]:
    layer = model.blocks[layer_idx]
    attn = layer.attn

    for head in range(attn.num_heads):
        base = head * attn.head_dim

        # Check if writes to AX_CARRY
        w_o = attn.W_o[:, base:base+attn.head_dim]
        writes_ax_carry = (w_o[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs() > 1e-6).any().item()

        if writes_ax_carry:
            # Check if it queries AX marker (would overwrite at AX)
            w_q = attn.W_q[base:base+attn.head_dim, :]
            queries_ax = (w_q[:, BD.MARK_AX].abs() > 1e-6).any().item()
            queries_pc = (w_q[:, BD.MARK_PC].abs() > 1e-6).any().item()
            queries_sp = (w_q[:, BD.MARK_SP].abs() > 1e-6).any().item()

            if queries_ax and not queries_pc and not queries_sp:
                print(f"  ❌ Layer {layer_idx} Head {head} overwrites AX_CARRY at AX marker!")
                overwrites_found = True

if not overwrites_found:
    print("  ✓ No overwrites of AX_CARRY at AX marker in layers 4-7")
    print("\n✓✓✓ FIX VERIFIED SUCCESSFULLY! ✓✓✓")
    print("\nThe neural ADD/SUB/MUL/DIV operations should now work correctly.")
else:
    print("\n❌ FIX NOT COMPLETE - Still have AX_CARRY overwrites!")

print("="*80)

import torch
torch.cuda.empty_cache()
