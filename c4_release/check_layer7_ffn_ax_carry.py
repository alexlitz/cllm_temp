"""Check if Layer 7 FFN writes to AX_CARRY dimensions."""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim
import sys

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)

ffn7 = model.blocks[7].ffn

print("=" * 70, file=sys.stderr)
print("CHECKING IF LAYER 7 FFN WRITES TO AX_CARRY DIMENSIONS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Check W_down matrix - does FFN write to AX_CARRY dims?
w_down_ax_carry = ffn7.W_down[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, :]
nonzero = (w_down_ax_carry.abs() > 1e-6).sum().item()
total = w_down_ax_carry.numel()

print(f"\nLayer 7 FFN W_down[AX_CARRY_LO:AX_CARRY_HI, :] non-zero: {nonzero} / {total}", file=sys.stderr)

if nonzero > 0:
    print("\n❌ Layer 7 FFN DOES write to AX_CARRY dimensions!", file=sys.stderr)
    print("   This will overwrite the values from Layer 3.", file=sys.stderr)
    print(f"   {nonzero} non-zero weights found", file=sys.stderr)

    # Show max magnitude
    max_val = w_down_ax_carry.abs().max().item()
    print(f"   Max abs weight: {max_val:.3f}", file=sys.stderr)
else:
    print("\n✓ Layer 7 FFN does NOT write to AX_CARRY dimensions", file=sys.stderr)
    print("  AX_CARRY should propagate through residual connection", file=sys.stderr)

print("=" * 70, file=sys.stderr)
