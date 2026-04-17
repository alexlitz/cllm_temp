"""
Verify that no Layer 6 head writes to AX_CARRY **at the AX marker**.

Heads can write to AX_CARRY at PC, SP, or STACK0 markers (for JMP, JSR, etc.),
but they must NOT write to AX_CARRY at the AX marker because that would overwrite
the carry-forward value set by Layer 3 Head 1.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

print("="*80)
print("VERIFYING AX_CARRY PRESERVATION AT AX MARKER")
print("="*80 + "\n")

model = AutoregressiveVM()
set_vm_weights(model)

layer6 = model.blocks[6]
attn6 = layer6.attn
HD = attn6.head_dim

print("Checking each Layer 6 head:\n")

problems_found = False

for head in range(attn6.num_heads):
    base = head * HD

    # Check if this head writes to AX_CARRY
    w_o = attn6.W_o[:, base:base+HD]
    writes_ax_carry = (w_o[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs() > 1e-6).any().item()

    if not writes_ax_carry:
        print(f"Head {head}: ✓ Does not write to AX_CARRY")
        continue

    # This head writes to AX_CARRY - check WHERE it fires
    w_q = attn6.W_q[base:base+HD, :]

    # Check query weights for markers
    q_ax = w_q[:, BD.MARK_AX].abs().max().item()
    q_pc = w_q[:, BD.MARK_PC].abs().max().item()
    q_sp = w_q[:, BD.MARK_SP].abs().max().item()
    q_stack0 = w_q[:, BD.MARK_STACK0].abs().max().item()

    # Check sign of MARK_AX query
    ax_sign = w_q[:, BD.MARK_AX].max().item()

    fires_at = []
    if q_pc > 1e-6:
        fires_at.append("PC")
    if q_sp > 1e-6:
        fires_at.append("SP")
    if q_stack0 > 1e-6:
        fires_at.append("STACK0")
    if q_ax > 1e-6 and ax_sign > 0:
        fires_at.append("AX")

    blocks_ax = (q_ax > 1e-6 and ax_sign < 0)

    print(f"Head {head}: Writes AX_CARRY", end="")

    if blocks_ax:
        print(f" - ✓ BLOCKS at AX marker (queries {', '.join(fires_at)})")
    elif "AX" in fires_at:
        print(f" - ❌ FIRES at AX marker! This will corrupt AX_CARRY!")
        problems_found = True
    else:
        # Doesn't query AX at all
        print(f" - ✓ Fires at {', '.join(fires_at)} (not AX)")

print("\n" + "="*80)
print("AX_CARRY CARRY-FORWARD PATH CHECK")
print("="*80 + "\n")

# Check Layer 3 Head 1 sets AX_CARRY
layer3 = model.blocks[3]
attn3 = layer3.attn
base_h1 = 1 * attn3.head_dim
w_o_l3h1 = attn3.W_o[:, base_h1:base_h1+attn3.head_dim]
l3_writes = (w_o_l3h1[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs() > 1e-6).any().item()
print(f"Layer 3 Head 1 sets AX_CARRY: {l3_writes}")

# Check it queries the AX marker
w_q_l3h1 = attn3.W_q[base_h1:base_h1+attn3.head_dim, :]
l3_queries_ax = (w_q_l3h1[:, BD.MARK_AX].abs() > 1e-6).any().item()
print(f"Layer 3 Head 1 fires at AX marker: {l3_queries_ax}")

print("\n" + "="*80)
if not problems_found:
    print("✓✓✓ SUCCESS! No heads corrupt AX_CARRY at AX marker! ✓✓✓")
    print("\nThe neural ADD/SUB/MUL/DIV operations should work correctly.")
    print("AX_CARRY values set by L3 H1 will reach L8 FFN intact.")
else:
    print("❌ FAILURE! Some heads still corrupt AX_CARRY at AX marker!")
    print("\nArithmetic operations will fail until this is fixed.")

print("="*80)

import torch
torch.cuda.empty_cache()
