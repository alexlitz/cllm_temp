"""Check all L3 FFN units that affect OUTPUT_LO[0]."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
set_vm_weights(model)

l3_ffn = model.blocks[3].ffn

print("=== L3 FFN units writing to OUTPUT_LO[0] ===")
output_lo_0_col = l3_ffn.W_down[BD.OUTPUT_LO, :]
nonzero_units = (output_lo_0_col.abs() > 0.001).nonzero(as_tuple=True)[0]

print(f"Found {len(nonzero_units)} units writing to OUTPUT_LO[0]:")
for unit_idx in nonzero_units:
    unit = unit_idx.item()
    weight = output_lo_0_col[unit].item()
    print(f"\n  Unit {unit}: W_down[OUTPUT_LO[0], unit] = {weight:.6f}")

    # Check what activates this unit
    w_up_nonzero = (l3_ffn.W_up[unit, :].abs() > 0.001).nonzero(as_tuple=True)[0]
    w_gate_nonzero = (l3_ffn.W_gate[unit, :].abs() > 0.001).nonzero(as_tuple=True)[0]

    if len(w_up_nonzero) > 0:
        print(f"    W_up non-zero at: {w_up_nonzero.tolist()}")
        for dim in w_up_nonzero[:3]:  # Show first 3
            print(f"      dim {dim.item()}: {l3_ffn.W_up[unit, dim].item():.1f}")
    print(f"    b_up: {l3_ffn.b_up[unit].item():.1f}")

    if len(w_gate_nonzero) > 0:
        print(f"    W_gate non-zero at: {w_gate_nonzero.tolist()}")
        for dim in w_gate_nonzero[:3]:  # Show first 3
            print(f"      dim {dim.item()}: {l3_ffn.W_gate[unit, dim].item():.3f}")
    print(f"    b_gate: {l3_ffn.b_gate[unit].item():.3f}")
