"""
Analyze which Layer 6 heads are actually configured and find available heads.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

print("="*80)
print("LAYER 6 HEAD USAGE ANALYSIS")
print("="*80 + "\n")

model = AutoregressiveVM()
set_vm_weights(model)

layer6 = model.blocks[6]
attn6 = layer6.attn

print(f"Layer 6 has {attn6.num_heads} heads\n")
print(f"Head dimension: {attn6.head_dim}\n")

# Check each head
for head in range(attn6.num_heads):
    head_dim = attn6.head_dim
    base = head * head_dim

    # Check if head has any configuration
    w_q = attn6.W_q[base:base+head_dim, :]
    w_k = attn6.W_k[base:base+head_dim, :]
    w_v = attn6.W_v[base:base+head_dim, :]
    w_o = attn6.W_o[:, base:base+head_dim]

    has_q = (w_q.abs() > 1e-6).any().item()
    has_k = (w_k.abs() > 1e-6).any().item()
    has_v = (w_v.abs() > 1e-6).any().item()
    has_o = (w_o.abs() > 1e-6).any().item()

    is_configured = has_q or has_k or has_v or has_o

    print(f"Head {head}: {'CONFIGURED' if is_configured else 'UNUSED'}")

    if is_configured:
        # Find what it queries
        q_dims = (w_q.abs() > 1e-6).any(dim=0).nonzero(as_tuple=True)[0]
        q_names = []
        for d in q_dims[:5]:
            d = d.item()
            if d == BD.MARK_PC: q_names.append("MARK_PC")
            elif d == BD.MARK_AX: q_names.append("MARK_AX")
            elif d == BD.MARK_SP: q_names.append("MARK_SP")
            elif d == BD.MARK_STACK0: q_names.append("MARK_STACK0")
            elif d == BD.NEXT_SE: q_names.append("NEXT_SE")
            elif d == BD.HAS_SE: q_names.append("HAS_SE")

        if q_names:
            print(f"  Queries: {', '.join(q_names)}")

        # Find what it writes to
        o_dims = (w_o.abs() > 1e-6).any(dim=1).nonzero(as_tuple=True)[0]
        o_names = []
        for d in o_dims[:5]:
            d = d.item()
            if d >= BD.AX_CARRY_LO and d < BD.AX_CARRY_LO + 32:
                o_names.append("AX_CARRY")
            elif d >= BD.ALU_LO and d < BD.ALU_LO + 32:
                o_names.append("ALU")
            elif d >= BD.CMP and d < BD.CMP + 16:
                o_names.append(f"CMP[{d-BD.CMP}]")
            elif d >= BD.TEMP and d < BD.TEMP + 32:
                o_names.append("TEMP")

        if o_names:
            print(f"  Writes to: {', '.join(set(o_names))}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80 + "\n")

print("Based on the analysis:")
print("  - Heads 0-5 are configured by _set_layer6_attn")
print("  - Head 6 might be unused")
print("  - Head 7 is configured for JSR handling")
print()
print("Options:")
print("  1. Use Head 6 alone for both STACK0 and SP relays (if possible)")
print("  2. Find a different layer to do these relays")
print("  3. Change JSR handling to use a different dimension instead of AX_CARRY")
print("  4. Reorganize Layer 6 to consolidate functionality")

import torch
torch.cuda.empty_cache()
