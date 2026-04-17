"""
Analyze the AX_CARRY dataflow path through layers 3-8.
Check weight configurations to understand why AX_CARRY doesn't reach Layer 8.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

print("="*80)
print("ANALYZING AX_CARRY DATAFLOW PATH")
print("="*80 + "\n")

# Load model
model = AutoregressiveVM()
set_vm_weights(model)

print("Key Dimensions:")
print(f"  AX_CARRY_LO: {BD.AX_CARRY_LO}-{BD.AX_CARRY_LO+15} (dims {BD.AX_CARRY_LO}:{BD.AX_CARRY_LO+16})")
print(f"  AX_CARRY_HI: {BD.AX_CARRY_HI}-{BD.AX_CARRY_HI+15} (dims {BD.AX_CARRY_HI}:{BD.AX_CARRY_HI+16})")
print(f"  ALU_LO: {BD.ALU_LO}-{BD.ALU_LO+15} (dims {BD.ALU_LO}:{BD.ALU_LO+16})")
print(f"  ALU_HI: {BD.ALU_HI}-{BD.ALU_HI+15} (dims {BD.ALU_HI}:{BD.ALU_HI+16})")
print()

# Check Layer 3: Does it write to AX_CARRY?
print("="*80)
print("LAYER 3: CARRY-FORWARD (should SET AX_CARRY)")
print("="*80 + "\n")

layer3 = model.blocks[3]
attn3 = layer3.attn

# Check all heads for writes to AX_CARRY_LO
for head in range(attn3.num_heads):
    head_dim = attn3.head_dim
    base = head * head_dim

    # Check W_o for this head writing to AX_CARRY_LO
    w_o_ax_carry_lo = attn3.W_o[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16, base:base+head_dim]
    w_o_ax_carry_hi = attn3.W_o[BD.AX_CARRY_HI:BD.AX_CARRY_HI+16, base:base+head_dim]

    has_lo = (w_o_ax_carry_lo.abs() > 1e-6).any().item()
    has_hi = (w_o_ax_carry_hi.abs() > 1e-6).any().item()

    if has_lo or has_hi:
        print(f"Head {head}: Writes to AX_CARRY")
        if has_lo:
            print(f"  AX_CARRY_LO: max weight = {w_o_ax_carry_lo.abs().max().item():.3f}")
        if has_hi:
            print(f"  AX_CARRY_HI: max weight = {w_o_ax_carry_hi.abs().max().item():.3f}")

        # Check what this head reads from (W_v)
        v_weights = attn3.W_v[base:base+head_dim, :]
        nonzero_cols = (v_weights.abs() > 1e-6).any(dim=0).nonzero(as_tuple=True)[0]
        print(f"  Reads from dimensions: {nonzero_cols[:10].tolist()}...")

print()

# Check Layer 4-6: Do they preserve or overwrite AX_CARRY?
print("="*80)
print("LAYERS 4-6: INTERMEDIATE (should PRESERVE AX_CARRY)")
print("="*80 + "\n")

for layer_idx in [4, 5, 6]:
    layer = model.blocks[layer_idx]
    attn = layer.attn
    ffn = layer.ffn

    # Check attention heads
    attn_writes_ax_carry = False
    for head in range(attn.num_heads):
        head_dim = attn.head_dim
        base = head * head_dim
        w_o = attn.W_o[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, base:base+head_dim]
        if (w_o.abs() > 1e-6).any().item():
            attn_writes_ax_carry = True
            print(f"Layer {layer_idx} Attention Head {head}: Writes to AX_CARRY (may interfere!)")

    # Check FFN
    ffn_writes_ax_carry = False
    w_down = ffn.W_down[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, :]
    if (w_down.abs() > 1e-6).any().item():
        ffn_writes_ax_carry = True
        print(f"Layer {layer_idx} FFN: Writes to AX_CARRY (may interfere!)")

    if not attn_writes_ax_carry and not ffn_writes_ax_carry:
        print(f"Layer {layer_idx}: ✓ Does NOT write to AX_CARRY (good - preserves via residual)")

print()

# Check Layer 7: Does it overwrite AX_CARRY?
print("="*80)
print("LAYER 7: OPERAND GATHER (should PRESERVE AX_CARRY)")
print("="*80 + "\n")

layer7 = model.blocks[7]
attn7 = layer7.attn

for head in range(attn7.num_heads):
    head_dim = attn7.head_dim
    base = head * head_dim

    # Check if writes to AX_CARRY
    w_o_ax_carry = attn7.W_o[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, base:base+head_dim]
    has_writes = (w_o_ax_carry.abs() > 1e-6).any().item()

    # Check if writes to ALU
    w_o_alu = attn7.W_o[BD.ALU_LO:BD.ALU_HI+16, base:base+head_dim]
    has_alu_writes = (w_o_alu.abs() > 1e-6).any().item()

    if has_writes:
        print(f"Head {head}: ❌ WRITES to AX_CARRY (THIS IS BAD - overwrites!)")
    elif has_alu_writes:
        print(f"Head {head}: Writes to ALU_LO/HI (OK - different dimensions)")
    else:
        print(f"Head {head}: No relevant writes")

print()

# Check Layer 8 FFN: Does it read AX_CARRY?
print("="*80)
print("LAYER 8 FFN: ALU OPERATIONS (needs AX_CARRY)")
print("="*80 + "\n")

layer8 = model.blocks[8]
ffn8 = layer8.ffn

# Check W_up for reads from AX_CARRY_LO
w_up_ax_carry_lo = ffn8.W_up[:, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
w_up_alu_lo = ffn8.W_up[:, BD.ALU_LO:BD.ALU_LO+16]

units_reading_ax_carry = (w_up_ax_carry_lo.abs() > 1e-6).any(dim=1).sum().item()
units_reading_alu = (w_up_alu_lo.abs() > 1e-6).any(dim=1).sum().item()

print(f"FFN units reading AX_CARRY_LO: {units_reading_ax_carry}")
print(f"FFN units reading ALU_LO: {units_reading_alu}")

# Check if ADD units are configured
add_units = 0
for unit in range(256):  # First 256 units should be ADD lo nibble
    # Check if this unit has the 3-way AND pattern
    has_mark_ax = ffn8.W_up[unit, BD.MARK_AX].abs().item() > 1e-6
    has_alu_lo = (ffn8.W_up[unit, BD.ALU_LO:BD.ALU_LO+16].abs() > 1e-6).any().item()
    has_ax_carry_lo = (ffn8.W_up[unit, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].abs() > 1e-6).any().item()
    has_op_add_gate = ffn8.W_gate[unit, BD.OP_ADD].abs().item() > 1e-6

    if has_mark_ax and has_alu_lo and has_ax_carry_lo and has_op_add_gate:
        add_units += 1

print(f"ADD units with 3-way AND (MARK_AX + ALU_LO + AX_CARRY_LO): {add_units}")
print(f"Expected: 256 (16x16 for all nibble combinations)")

if add_units == 256:
    print("✓ Layer 8 FFN is correctly configured for ADD")
else:
    print("❌ Layer 8 FFN configuration issue!")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80 + "\n")

print("Expected dataflow:")
print("  1. Layer 3 Head 1: Previous step's AX → AX_CARRY_LO/HI at AX marker")
print("  2. Layers 4-6: Preserve AX_CARRY via residual stream")
print("  3. Layer 7: Set ALU_LO/HI (different dims), preserve AX_CARRY")
print("  4. Layer 8 FFN: Read both ALU_LO and AX_CARRY_LO at AX marker")
print()

print("Key question:")
print("  Does Layer 3 set AX_CARRY at the RIGHT POSITION?")
print("  → It should set it at the AX marker of the CURRENT step")
print("  → But it should contain the value from PREVIOUS step's AX")
print()

print("Next steps:")
print("  1. Run execution trace to see actual values at each layer")
print("  2. Check if Layer 3 is querying the right position (previous AX)")
print("  3. Check if values persist through residual connections")

import torch
torch.cuda.empty_cache()
