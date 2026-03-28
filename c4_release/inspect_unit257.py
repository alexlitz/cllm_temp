#!/usr/bin/env python3
"""Inspect unit 257 in Layer 6 FFN."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

ffn6 = model.blocks[6].ffn
unit = 257

print(f"Unit 257 in Layer 6 FFN:")
print(f"\n  W_up[257, :] non-zero dims:")
W_up = ffn6.W_up.data[unit, :]
for dim in W_up.nonzero(as_tuple=True)[0][:20]:
    print(f"    dim {dim}: {W_up[dim].item():.4f}")

print(f"\n  b_up[257] = {ffn6.b_up.data[unit].item():.4f}")

print(f"\n  W_gate[257, :] non-zero dims:")
W_gate = ffn6.W_gate.data[unit, :]
for dim in W_gate.nonzero(as_tuple=True)[0][:20]:
    print(f"    dim {dim}: {W_gate[dim].item():.4f}")

print(f"\n  b_gate[257] = {ffn6.b_gate.data[unit].item():.4f}")

print(f"\n  W_down[:, 257] non-zero dims:")
W_down = ffn6.W_down.data[:, unit]
for dim in W_down.nonzero(as_tuple=True)[0]:
    dim_name = f"TEMP+0" if dim == BD.TEMP else str(dim)
    print(f"    dim {dim} ({dim_name}): {W_down[dim].item():.6f}")

# Test what this unit outputs at PC marker with TEMP[0] = 5.525
print("\n  Simulating at PC marker with TEMP[0]=5.525:")
x_test = torch.zeros(512)
x_test[BD.MARK_PC] = 1.0
x_test[BD.TEMP + 0] = 5.525

up_val = (W_up * x_test).sum().item() + ffn6.b_up[unit].item()
gate_val = (W_gate * x_test).sum().item() + ffn6.b_gate[unit].item()
hidden_val = torch.nn.functional.silu(torch.tensor(up_val)).item() * gate_val

print(f"    up = {up_val:.4f}")
print(f"    gate = {gate_val:.4f}")
print(f"    hidden = silu(up) * gate = {hidden_val:.4f}")
print(f"    contribution to TEMP[0] = {hidden_val * W_down[BD.TEMP].item():.6f}")
