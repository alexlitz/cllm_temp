"""
Debug single operation execution.
"""

import torch
from neural_vm.single_op_executor import SingleOperationExecutor
from neural_vm.embedding import Opcode, E

def debug_add():
    """Debug ADD operation in detail."""
    
    print("="*70)
    print("DEBUG: ADD OPERATION")
    print("="*70)
    print()
    
    executor = SingleOperationExecutor()
    
    # Test case: 2 + 3 = 5
    a, b = 2, 3
    expected = 5
    
    print(f"Test: {a} + {b} = {expected}")
    print()
    
    # Step 1: Check input encoding
    print("Step 1: Input Encoding")
    print("-" * 70)
    
    input_embedding = executor.embed.encode_vm_state(
        pc=0, ax=a, sp=4096, bp=4096,
        opcode=Opcode.ADD, stack_top=b, batch_size=1,
    )
    
    print(f"Input shape: {input_embedding.shape}")
    
    # Check nibbles
    print("\nInput nibbles:")
    for pos in range(3):  # Just first 3 positions
        base_idx = pos * 160
        nib_a = input_embedding[0, base_idx + E.NIB_A].item()
        nib_b = input_embedding[0, base_idx + E.NIB_B].item()
        print(f"  Pos {pos}: NIB_A={nib_a:.1f}, NIB_B={nib_b:.1f}")
    
    # Check opcode
    print(f"\nOpcode encoding (ADD={Opcode.ADD}):")
    for pos in range(2):
        base_idx = pos * 160
        opcode_idx = base_idx + E.OP_START + Opcode.ADD
        if opcode_idx < base_idx + 160:
            val = input_embedding[0, opcode_idx].item()
            print(f"  Pos {pos}: index={opcode_idx}, value={val:.1f}")
    
    # Check non-zero count
    nonzero = (input_embedding.abs() > 1e-6).sum().item()
    print(f"\nNon-zero inputs: {nonzero} / 1280")
    
    print()
    
    # Step 2: Check layer weights
    print("Step 2: Layer Weights")
    print("-" * 70)
    
    layer_idx = executor.layer_map['PRIMARY_ALU']
    layer = executor.vm.blocks[layer_idx]
    
    w_up_nonzero = (layer.ffn.W_up.abs() > 1e-9).sum().item()
    w_gate_nonzero = (layer.ffn.W_gate.abs() > 1e-9).sum().item()
    w_down_nonzero = (layer.ffn.W_down.abs() > 1e-9).sum().item()
    
    print(f"W_up non-zero: {w_up_nonzero:,} / {layer.ffn.W_up.numel():,}")
    print(f"W_gate non-zero: {w_gate_nonzero:,} / {layer.ffn.W_gate.numel():,}")
    print(f"W_down non-zero: {w_down_nonzero:,} / {layer.ffn.W_down.numel():,}")
    
    print()
    
    # Step 3: Check forward pass
    print("Step 3: Forward Pass")
    print("-" * 70)
    
    with torch.no_grad():
        # Up projection
        up = layer.ffn.W_up @ input_embedding.T + layer.ffn.b_up.unsqueeze(1)
        print(f"Up output shape: {up.shape}")
        print(f"Up non-zero: {(up.abs() > 1e-6).sum().item()} / {up.numel()}")
        print(f"Up range: [{up.min().item():.2f}, {up.max().item():.2f}]")
        
        # Gate projection
        gate = layer.ffn.W_gate @ input_embedding.T + layer.ffn.b_gate.unsqueeze(1)
        print(f"\nGate output shape: {gate.shape}")
        print(f"Gate non-zero: {(gate.abs() > 1e-6).sum().item()} / {gate.numel()}")
        print(f"Gate range: [{gate.min().item():.2f}, {gate.max().item():.2f}]")
        
        # Check which units are active (gated)
        active_units = (gate.abs() > 0.5).sum().item()
        print(f"Active units (gate > 0.5): {active_units}")
        
        # SwiGLU
        hidden = torch.nn.functional.silu(up) * gate
        print(f"\nHidden shape: {hidden.shape}")
        print(f"Hidden non-zero: {(hidden.abs() > 1e-6).sum().item()} / {hidden.numel()}")
        print(f"Hidden range: [{hidden.min().item():.2f}, {hidden.max().item():.2f}]")
        
        # Down projection
        output = layer.ffn.W_down @ hidden + layer.ffn.b_down.unsqueeze(1)
        print(f"\nOutput shape: {output.shape}")
        print(f"Output non-zero: {(output.abs() > 1e-6).sum().item()} / {output.numel()}")
        print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
        
        output_embedding = output.T
    
    print()
    
    # Step 4: Check output nibbles
    print("Step 4: Output Nibbles")
    print("-" * 70)
    
    print("\nRESULT slot values:")
    for pos in range(8):
        base_idx = pos * 160
        result_val = output_embedding[0, base_idx + E.RESULT].item()
        expected_nibble = (expected >> (pos * 4)) & 0xF
        print(f"  Pos {pos}: {result_val:.2f} (expected {expected_nibble})")
    
    # Decode
    decoded = executor.embed.decode_result_nibbles(output_embedding)
    print(f"\nDecoded result: {decoded}")
    print(f"Expected: {expected}")
    print(f"Match: {'✅' if decoded == expected else '❌'}")
    
    print()
    print("="*70)

if __name__ == "__main__":
    debug_add()
