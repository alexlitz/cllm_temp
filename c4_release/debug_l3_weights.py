"""Check L3 FFN weights for first-step default."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

# Load model
model = AutoregressiveVM()
set_vm_weights(model)

ffn = model.blocks[3].ffn

print("L3 FFN First-Step Default Weights")
print("="*70)
print(f"OUTPUT_LO base dim: {BD.OUTPUT_LO}")
print(f"Expected: W_down[{BD.OUTPUT_LO + 0}, unit] should be set")
print()

# Find the first unit that writes to OUTPUT_LO range
# The first-step default should be in the first few units (unit 0 or 1)
print("Checking first few units:")
for unit in range(10):
    # Check if this unit writes to any OUTPUT_LO dimension
    w_down_slice = ffn.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, unit]
    nonzero = (w_down_slice.abs() > 0.1).nonzero(as_tuple=True)[0]

    if len(nonzero) > 0:
        # Check the activation conditions
        w_up_mark_pc = ffn.W_up[unit, BD.MARK_PC].item()
        w_up_has_se = ffn.W_up[unit, BD.HAS_SE].item()
        b_up = ffn.b_up[unit].item()
        b_gate = ffn.b_gate[unit].item()

        print(f"\nUnit {unit}:")
        print(f"  W_up[MARK_PC]: {w_up_mark_pc:.2f}")
        print(f"  W_up[HAS_SE]: {w_up_has_se:.2f}")
        print(f"  b_up: {b_up:.2f}")
        print(f"  b_gate: {b_gate:.2f}")
        print(f"  Writes to OUTPUT_LO indices: {nonzero.tolist()}")
        for idx in nonzero:
            val = w_down_slice[idx].item()
            print(f"    OUTPUT_LO[{idx.item()}]: {val:.4f}")

        # Determine what this unit does
        if abs(w_up_mark_pc - 50.0) < 0.1 and abs(w_up_has_se) < 0.1 and abs(b_gate - 1.0) < 0.1:
            print(f"  → This looks like FIRST-STEP DEFAULT (MARK_PC=50, HAS_SE=0, b_gate=1)")
            if nonzero[0].item() == 0:
                print(f"     ✓ Correctly writes to OUTPUT_LO[0]")
            else:
                print(f"     ✗ BUG: Writes to OUTPUT_LO[{nonzero[0].item()}] instead of OUTPUT_LO[0]!")
        elif abs(w_up_mark_pc - 50.0) < 0.1 and abs(w_up_has_se - 50.0) < 0.1:
            print(f"  → This looks like UNDO unit (MARK_PC AND HAS_SE)")

print("\n" + "="*70)
print("Checking W_down matrix dimensions:")
print(f"  W_down shape: {ffn.W_down.shape}")
print(f"  Expected: [d_model={model.embed.embedding_dim}, d_ff={ffn.W_down.shape[1]}]")
print()
print("Checking if indices are transposed:")
# The code does ffn.W_down[BD.OUTPUT_LO + 0, unit] = value
# This assumes W_down has shape [d_model, d_ff]
# Let's verify this
print(f"  Accessing W_down[{BD.OUTPUT_LO}, 0]:")
print(f"    Value: {ffn.W_down[BD.OUTPUT_LO, 0].item():.4f}")
print(f"  Accessing W_down[{BD.OUTPUT_LO + 2}, 0]:")
print(f"    Value: {ffn.W_down[BD.OUTPUT_LO + 2, 0].item():.4f}")

# Check if maybe the weight is set at a different dimension due to transpose
print("\nChecking transpose possibility:")
print(f"  W_down[0, {BD.OUTPUT_LO}]: {ffn.W_down[0, BD.OUTPUT_LO].item():.4f}")
print(f"  W_down[0, {BD.OUTPUT_LO + 2}]: {ffn.W_down[0, BD.OUTPUT_LO + 2].item():.4f}")
