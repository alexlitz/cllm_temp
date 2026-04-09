"""Check if Layer 7 attention writes to AX_CARRY dimensions."""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim
import sys

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)

attn7 = model.blocks[7].attn

print("=" * 70, file=sys.stderr)
print("CHECKING IF LAYER 7 WRITES TO AX_CARRY DIMENSIONS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Check W_o matrix - does any head write to AX_CARRY dims?
wo_ax_carry = attn7.W_o[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, :]
nonzero = (wo_ax_carry.abs() > 1e-6).sum().item()
total = wo_ax_carry.numel()

print(f"\nLayer 7 W_o[AX_CARRY_LO:AX_CARRY_HI, :] non-zero: {nonzero} / {total}", file=sys.stderr)

if nonzero > 0:
    print("\n❌ Layer 7 DOES write to AX_CARRY dimensions!", file=sys.stderr)
    print("   This will overwrite the values from Layer 3.", file=sys.stderr)

    # Find which heads
    print("\n  Heads writing to AX_CARRY:", file=sys.stderr)
    HD = attn7.head_dim
    for head in range(attn7.num_heads):
        base = head * HD
        head_writes = attn7.W_o[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, base:base+HD]
        head_nonzero = (head_writes.abs() > 1e-6).sum().item()
        if head_nonzero > 0:
            print(f"    Head {head}: {head_nonzero} non-zero weights", file=sys.stderr)
else:
    print("\n✓ Layer 7 does NOT write to AX_CARRY dimensions", file=sys.stderr)
    print("  AX_CARRY should propagate through residual connection", file=sys.stderr)

print("=" * 70, file=sys.stderr)
