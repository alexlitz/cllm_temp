#!/usr/bin/env python3
"""Debug MoE VM weight loading."""

import torch
from neural_vm.moe_vm import MoEAutoregressiveVM
from neural_vm.moe_weight_loader import MoEWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

print("Creating weight loader...")
loader = MoEWeightLoader(verbose=False)
experts_per_layer = loader.create_expert_configs()

print(f"\nChecking expert weights for layer 0...")
for expert in experts_per_layer[0]:
    print(f"\n  Expert: {expert.name} (opcode={expert.opcode}, hidden={expert.ffn_hidden})")
    if expert.weights is not None:
        for key, tensor in expert.weights.items():
            nonzero = (tensor.abs() > 1e-9).sum().item()
            print(f"    {key:10s}: shape={str(tensor.shape):20s} nonzero={nonzero}")
    else:
        print("    WARNING: No weights!")

print("\nCreating MoE VM...")
vm = MoEAutoregressiveVM(
    d_model=1352,
    n_layers=loader.n_layers,
    n_heads=8,
    experts_per_layer=experts_per_layer
)

print("\nChecking VM expert FFNs for layer 0...")
moe_layer = vm.blocks[0].moe
for expert_idx, expert_ffn in enumerate(moe_layer.expert_ffns):
    print(f"\n  Expert {expert_idx}:")
    print(f"    W_up:   nonzero={(expert_ffn.W_up.abs() > 1e-9).sum().item()}, shape={expert_ffn.W_up.shape}")
    print(f"    W_gate: nonzero={(expert_ffn.W_gate.abs() > 1e-9).sum().item()}, shape={expert_ffn.W_gate.shape}")
    print(f"    W_down: nonzero={(expert_ffn.W_down.abs() > 1e-9).sum().item()}, shape={expert_ffn.W_down.shape}")

# Test OR operation
embed = NibbleVMEmbedding(d_model=1352)
input_emb = embed.encode_vm_state(
    pc=0, ax=3, sp=0, bp=0,
    opcode=Opcode.OR, stack_top=5, batch_size=1
)

print("\n\nTesting OR(5, 3):")
print(f"Input embedding shape: {input_emb.shape}")
print(f"Input embedding nonzero: {(input_emb.abs() > 1e-9).sum().item()}")

with torch.no_grad():
    x = input_emb.unsqueeze(1)  # [1, 1, 1352]
    print(f"Input x shape: {x.shape}")

    # Extract opcode onehot
    from neural_vm.moe_layer import extract_opcode_onehot
    from neural_vm.embedding import E
    opcode_onehot = extract_opcode_onehot(x, E.OP_START, E.NUM_OPS)
    print(f"Opcode onehot shape: {opcode_onehot.shape}")
    print(f"Opcode value: {opcode_onehot.argmax().item()}")

    # Run just the first block
    block = vm.blocks[0]
    x_out = block(x, opcode_onehot)
    print(f"After block 0 shape: {x_out.shape}")
    print(f"After block 0 nonzero: {(x_out.abs() > 1e-9).sum().item()}")
    print(f"After block 0 mean: {x_out.mean().item():.6f}")
    print(f"After block 0 std: {x_out.std().item():.6f}")

    # Decode
    result = embed.decode_result_nibbles(x_out.squeeze(1))
    print(f"Decoded result: {result}")
