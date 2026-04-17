#!/usr/bin/env python3
"""Find which layer has the OP flag relays."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner()
model = runner.model

print(f"Model has {len(model.blocks)} blocks")
print("\nSearching for OP_IMM in V matrices...")

for layer_idx, block in enumerate(model.blocks):
    attn = block.attn
    # Check all heads
    for head in range(8):
        base = head * 64
        weights = attn.W_v[base:base+64, BD.OP_IMM].abs()
        if weights.max().item() > 0.1:
            print(f"  Layer {layer_idx}, Head {head}: Found OP_IMM")

print("\nThe layer with OP flag relays is where we need to add OP_JSR!")
