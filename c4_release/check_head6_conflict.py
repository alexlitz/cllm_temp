"""
Check if Head 6 has a configuration conflict between:
1. _set_layer6_relay_heads (STACK0←AX relay, writes to ALU)
2. _set_opcode_relay_head (opcode relay AX→SP/STACK0, writes to CMP)
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

print("="*80)
print("CHECKING HEAD 6 CONFIGURATION")
print("="*80 + "\n")

model = AutoregressiveVM()
set_vm_weights(model)

layer6 = model.blocks[6]
attn6 = layer6.attn
HD = attn6.head_dim

base = 6 * HD

# Check Query weights
w_q = attn6.W_q[base:base+HD, :]
print("Head 6 Query weights:")
q_dims = (w_q.abs() > 1e-6).any(dim=0).nonzero(as_tuple=True)[0]
for d in q_dims[:10]:
    d_val = d.item()
    weight = w_q[:, d_val].abs().max().item()
    sign = "+" if w_q[:, d_val].max().item() > 0 else "-"

    name = "?"
    if d_val == BD.MARK_AX: name = "MARK_AX"
    elif d_val == BD.MARK_PC: name = "MARK_PC"
    elif d_val == BD.MARK_SP: name = "MARK_SP"
    elif d_val == BD.MARK_STACK0: name = "MARK_STACK0"
    elif d_val == BD.MARK_BP: name = "MARK_BP"
    elif d_val == BD.MARK_MEM: name = "MARK_MEM"

    print(f"  {sign}{weight:.1f} on dim {d_val} ({name})")

print("\nHead 6 Key weights:")
w_k = attn6.W_k[base:base+HD, :]
k_dims = (w_k.abs() > 1e-6).any(dim=0).nonzero(as_tuple=True)[0]
for d in k_dims[:5]:
    d_val = d.item()
    weight = w_k[:, d_val].abs().max().item()

    name = "?"
    if d_val == BD.MARK_AX: name = "MARK_AX"
    elif d_val == BD.MARK_PC: name = "MARK_PC"
    elif d_val == BD.MARK_SP: name = "MARK_SP"
    elif d_val == BD.MARK_STACK0: name = "MARK_STACK0"

    print(f"  +{weight:.1f} on dim {d_val} ({name})")

print("\nHead 6 Output weights:")
w_o = attn6.W_o[:, base:base+HD]
o_dims = (w_o.abs() > 1e-6).any(dim=1).nonzero(as_tuple=True)[0]
for d in o_dims[:20]:
    d_val = d.item()
    weight = w_o[d_val, :].abs().max().item()

    name = "?"
    if d_val >= BD.ALU_LO and d_val < BD.ALU_LO + 32:
        name = f"ALU (byte {(d_val - BD.ALU_LO) // 16})"
    elif d_val >= BD.CMP and d_val < BD.CMP + 16:
        name = f"CMP[{d_val - BD.CMP}]"
    elif d_val >= BD.AX_CARRY_LO and d_val < BD.AX_CARRY_LO + 32:
        name = "AX_CARRY"

    print(f"  +{weight:.2f} on dim {d_val} ({name})")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80 + "\n")

# Check if both functions' weights are present
writes_alu = (w_o[BD.ALU_LO:BD.ALU_LO+32, :].abs() > 1e-6).any().item()
writes_cmp = (w_o[BD.CMP:BD.CMP+16, :].abs() > 1e-6).any().item()

queries_stack0 = (w_q[:, BD.MARK_STACK0].abs() > 1e-6).any().item()
queries_sp = (w_q[:, BD.MARK_SP].abs() > 1e-6).any().item()
queries_bp = (w_q[:, BD.MARK_BP].abs() > 1e-6).any().item()
queries_mem = (w_q[:, BD.MARK_MEM].abs() > 1e-6).any().item()

print(f"Writes to ALU: {writes_alu} (from _set_layer6_relay_heads)")
print(f"Writes to CMP: {writes_cmp} (from _set_opcode_relay_head)")
print()
print(f"Queries STACK0: {queries_stack0}")
print(f"Queries SP: {queries_sp}")
print(f"Queries BP: {queries_bp}")
print(f"Queries MEM: {queries_mem}")

if writes_alu and writes_cmp:
    print("\n⚠️  HEAD 6 IS CONFIGURED BY BOTH FUNCTIONS!")
    print("This is a weight collision - both functions write to the same head.")
    print("The second function call (_set_opcode_relay_head) overwrites the first.")
    print("\nFIX: Need to use different heads for these two functions.")
else:
    print("\n✓ No conflict detected")

print("="*80)

import torch
torch.cuda.empty_cache()
