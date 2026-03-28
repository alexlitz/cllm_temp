#!/usr/bin/env python3
"""Test OR operation in complete isolation."""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

# Create VM with just 1 layer
vm = AutoregressiveVM(d_model=1352, n_layers=1, ffn_hidden=6000)
vm.eval()

# Compile and load ONLY OR
compiler = OpcodeNibbleCompiler(num_positions=8, ffn_hidden=6000)
or_weights = compiler.compile_opcode(Opcode.OR, unit_offset=0)

# Load OR weights into layer 0
with torch.no_grad():
    vm.blocks[0].ffn.W_up.data[:] = or_weights['W_up']
    vm.blocks[0].ffn.b_up.data[:] = or_weights['b_up']
    vm.blocks[0].ffn.W_gate.data[:] = or_weights['W_gate']
    vm.blocks[0].ffn.b_gate.data[:] = or_weights['b_gate']
    vm.blocks[0].ffn.W_down.data[:] = or_weights['W_down']
    vm.blocks[0].ffn.b_down.data[:] = or_weights['b_down']

embed = NibbleVMEmbedding(d_model=1352)

print("Testing OR(5, 3) = 7 with ONLY OR loaded")
print()

# Encode input
input_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

# Run through layer 0
with torch.no_grad():
    x = input_emb.unsqueeze(1)
    x = vm.blocks[0](x)
    result = embed.decode_result_nibbles(x.squeeze(1))

print(f"Result: {result}")
if result == 7:
    print("✓ PASS")
else:
    print(f"✗ FAIL (expected 7)")

# Check result at position 0
output = x.squeeze()
print(f"\nResult nibbles:")
for pos in range(8):
    base_idx = pos * 169
    result_val = output[base_idx + 13].item()  # E.RESULT = 13
    print(f"  Position {pos}: {result_val:.2f}")
