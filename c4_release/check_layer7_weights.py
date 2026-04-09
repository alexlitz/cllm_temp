"""Check Layer 7 attention weight configuration for reading AX_CARRY."""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim
import torch
import sys

BD = _SetDim

print("=" * 70, file=sys.stderr)
print("LAYER 7 ATTENTION WEIGHT CONFIGURATION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

model = AutoregressiveVM()
set_vm_weights(model)

attn7 = model.blocks[7].attn

print(f"Layer 7 attention:", file=sys.stderr)
print(f"  Number of heads: {attn7.num_heads}", file=sys.stderr)
print(f"  Head dim: {attn7.head_dim}", file=sys.stderr)
print("", file=sys.stderr)

HD = attn7.head_dim

# Check each head for AX_CARRY usage
print("Checking which heads read from AX_CARRY dimensions:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

for head in range(attn7.num_heads):
    base = head * HD

    # Check if W_q or W_k reads from AX_CARRY
    wq_ax_carry = attn7.W_q[base:base+HD, BD.AX_CARRY_LO:BD.AX_CARRY_HI+16]
    wk_ax_carry = attn7.W_k[base:base+HD, BD.AX_CARRY_LO:BD.AX_CARRY_HI+16]

    wq_nonzero = (wq_ax_carry.abs() > 1e-6).sum().item()
    wk_nonzero = (wk_ax_carry.abs() > 1e-6).sum().item()

    if wq_nonzero > 0 or wk_nonzero > 0:
        print(f"  Head {head}:", file=sys.stderr)
        print(f"    W_q[{base}:{base+HD}, AX_CARRY] non-zero: {wq_nonzero}", file=sys.stderr)
        print(f"    W_k[{base}:{base+HD}, AX_CARRY] non-zero: {wk_nonzero}", file=sys.stderr)

print("", file=sys.stderr)

# Check which heads write to ALU dimensions (where AX_CARRY should be copied)
print("Checking which heads write to ALU dimensions:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

for head in range(attn7.num_heads):
    base = head * HD

    # Check if this head writes to ALU_LO/HI via W_o
    wo_alu_lo = attn7.W_o[BD.ALU_LO:BD.ALU_LO+16, base:base+HD]
    wo_alu_hi = attn7.W_o[BD.ALU_HI:BD.ALU_HI+16, base:base+HD]

    alu_lo_nonzero = (wo_alu_lo.abs() > 1e-6).sum().item()
    alu_hi_nonzero = (wo_alu_hi.abs() > 1e-6).sum().item()

    if alu_lo_nonzero > 0 or alu_hi_nonzero > 0:
        print(f"  Head {head}:", file=sys.stderr)
        print(f"    W_o[ALU_LO, {base}:{base+HD}] non-zero: {alu_lo_nonzero}", file=sys.stderr)
        print(f"    W_o[ALU_HI, {base}:{base+HD}] non-zero: {alu_hi_nonzero}", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Expected: Layer 7 should copy AX_CARRY → ALU dimensions for binary ops
# Check if there's a head configured for this

print("\nExpected behavior:", file=sys.stderr)
print("  Layer 7 should copy AX_CARRY_LO/HI → ALU_LO/HI", file=sys.stderr)
print("  This provides the second operand for binary operations", file=sys.stderr)
print("", file=sys.stderr)

# Check if any head does: query from MARK_AX/IS_BYTE, key from something, value from AX_CARRY
found_copy_head = False
for head in range(attn7.num_heads):
    base = head * HD

    # Check V matrix: does it read from AX_CARRY?
    wv_ax_carry = attn7.W_v[base:base+HD, BD.AX_CARRY_LO:BD.AX_CARRY_HI+16]
    wv_nonzero = (wv_ax_carry.abs() > 1e-6).sum().item()

    # Check O matrix: does it write to ALU?
    wo_alu = attn7.W_o[BD.ALU_LO:BD.ALU_HI+16, base:base+HD]
    wo_alu_nonzero = (wo_alu.abs() > 1e-6).sum().item()

    if wv_nonzero > 0 and wo_alu_nonzero > 0:
        print(f"✓ Head {head} copies AX_CARRY → ALU:", file=sys.stderr)
        print(f"    W_v reads from AX_CARRY: {wv_nonzero} non-zero weights", file=sys.stderr)
        print(f"    W_o writes to ALU: {wo_alu_nonzero} non-zero weights", file=sys.stderr)
        found_copy_head = True

        # Check Q and K configuration
        wq = attn7.W_q[base:base+HD, :]
        wk = attn7.W_k[base:base+HD, :]

        # Find which dimensions Q and K read from
        wq_max_dims = []
        for dim in range(wq.shape[1]):
            if wq[:, dim].abs().max() > 1.0:
                wq_max_dims.append(dim)

        wk_max_dims = []
        for dim in range(wk.shape[1]):
            if wk[:, dim].abs().max() > 1.0:
                wk_max_dims.append(dim)

        print(f"    W_q reads from dims: {wq_max_dims[:10]}", file=sys.stderr)
        print(f"    W_k reads from dims: {wk_max_dims[:10]}", file=sys.stderr)

if not found_copy_head:
    print("❌ NO head copies AX_CARRY → ALU!", file=sys.stderr)
    print("   This is the root cause - Layer 7 needs configuration to:", file=sys.stderr)
    print("   1. Read AX_CARRY_LO/HI via W_v", file=sys.stderr)
    print("   2. Write to ALU_LO/HI via W_o", file=sys.stderr)
    print("   3. Attend to the right positions (where AX_CARRY is set)", file=sys.stderr)

print("=" * 70, file=sys.stderr)
