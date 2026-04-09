"""
Identify exactly which layers/heads are overwriting AX_CARRY and where in the code.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

print("="*80)
print("IDENTIFYING AX_CARRY OVERWRITES")
print("="*80 + "\n")

model = AutoregressiveVM()
set_vm_weights(model)

# Check each problematic layer in detail
for layer_idx in [5, 6]:
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx}")
    print(f"{'='*80}\n")

    layer = model.blocks[layer_idx]
    attn = layer.attn

    for head in range(attn.num_heads):
        head_dim = attn.head_dim
        base = head * head_dim

        # Check W_o for writes to AX_CARRY
        w_o_ax_carry_lo = attn.W_o[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16, base:base+head_dim]
        w_o_ax_carry_hi = attn.W_o[BD.AX_CARRY_HI:BD.AX_CARRY_HI+16, base:base+head_dim]

        has_lo = (w_o_ax_carry_lo.abs() > 1e-6).any().item()
        has_hi = (w_o_ax_carry_hi.abs() > 1e-6).any().item()

        if has_lo or has_hi:
            print(f"Head {head}:")
            print(f"  Writes to AX_CARRY_LO: {has_lo}")
            print(f"  Writes to AX_CARRY_HI: {has_hi}")

            # Check what it queries (W_q)
            w_q = attn.W_q[base:base+head_dim, :]
            nonzero_q = (w_q.abs() > 1e-6).any(dim=0).nonzero(as_tuple=True)[0]
            print(f"  Query dims (W_q): {', '.join([str(d.item()) for d in nonzero_q[:10]])}")

            # Map to dimension names
            q_markers = []
            for d in nonzero_q[:20]:
                d = d.item()
                if d == BD.MARK_PC:
                    q_markers.append("MARK_PC")
                elif d == BD.MARK_AX:
                    q_markers.append("MARK_AX")
                elif d == BD.MARK_SP:
                    q_markers.append("MARK_SP")
                elif d == BD.MARK_BP:
                    q_markers.append("MARK_BP")
                elif d == BD.MARK_STACK0:
                    q_markers.append("MARK_STACK0")
                elif d == BD.HAS_SE:
                    q_markers.append("HAS_SE")
                elif d == 8:  # CONST
                    q_markers.append("CONST")
            if q_markers:
                print(f"  Query markers: {', '.join(q_markers)}")

            # Check what it retrieves (W_v)
            w_v = attn.W_v[base:base+head_dim, :]
            nonzero_v = (w_v.abs() > 1e-6).any(dim=0).nonzero(as_tuple=True)[0]
            print(f"  Value dims (W_v): {', '.join([str(d.item()) for d in nonzero_v[:10]])}")

            # Map to dimension names
            v_sources = []
            for d in nonzero_v[:20]:
                d = d.item()
                if BD.EMBED_LO <= d < BD.EMBED_LO + 16:
                    v_sources.append(f"EMBED_LO[{d-BD.EMBED_LO}]")
                elif BD.EMBED_HI <= d < BD.EMBED_HI + 16:
                    v_sources.append(f"EMBED_HI[{d-BD.EMBED_HI}]")
                elif BD.OUTPUT_LO <= d < BD.OUTPUT_LO + 16:
                    v_sources.append(f"OUTPUT_LO[{d-BD.OUTPUT_LO}]")
                elif BD.OUTPUT_HI <= d < BD.OUTPUT_HI + 16:
                    v_sources.append(f"OUTPUT_HI[{d-BD.OUTPUT_HI}]")
                elif BD.AX_CARRY_LO <= d < BD.AX_CARRY_LO + 16:
                    v_sources.append(f"AX_CARRY_LO[{d-BD.AX_CARRY_LO}]")
                elif BD.AX_CARRY_HI <= d < BD.AX_CARRY_HI + 16:
                    v_sources.append(f"AX_CARRY_HI[{d-BD.AX_CARRY_HI}]")
            if v_sources:
                print(f"  Value sources: {', '.join(v_sources[:5])}")

            # Check where it writes (W_o)
            w_o = attn.W_o[:, base:base+head_dim]
            nonzero_o_rows = (w_o.abs() > 1e-6).any(dim=1).nonzero(as_tuple=True)[0]
            print(f"  Output dims (W_o): {', '.join([str(d.item()) for d in nonzero_o_rows[:10]])}")

            # Map output destinations
            o_dests = []
            for d in nonzero_o_rows:
                d = d.item()
                if BD.AX_CARRY_LO <= d < BD.AX_CARRY_LO + 16:
                    o_dests.append(f"AX_CARRY_LO[{d-BD.AX_CARRY_LO}]")
                elif BD.AX_CARRY_HI <= d < BD.AX_CARRY_HI + 16:
                    o_dests.append(f"AX_CARRY_HI[{d-BD.AX_CARRY_HI}]")
                elif BD.ALU_LO <= d < BD.ALU_LO + 16:
                    o_dests.append(f"ALU_LO[{d-BD.ALU_LO}]")
                elif BD.ALU_HI <= d < BD.ALU_HI + 16:
                    o_dests.append(f"ALU_HI[{d-BD.ALU_HI}]")
            if o_dests:
                print(f"  Output destinations: {', '.join(o_dests[:5])}")

            print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80 + "\n")

print("The issue is that these layers write to AX_CARRY_LO/HI dimensions:")
print("  - Layer 5 Head 3")
print("  - Layer 6 Head 0")
print("  - Layer 6 Head 2")
print("  - Layer 6 Head 7")
print()
print("This overwrites the value set by Layer 3 Head 1, so it doesn't reach Layer 8.")
print()
print("Solution: These heads should write to different dimensions (e.g., ALU or TEMP)")
print("  or should be conditioned to not fire at the AX marker position.")

import torch
torch.cuda.empty_cache()
