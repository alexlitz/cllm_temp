"""
Test ADD in complete isolation - only ADD weights loaded.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E

def test_isolated_add():
    """Test ADD with ONLY ADD weights loaded (no other opcodes)."""
    
    print("="*70)
    print("ISOLATED ADD TEST")
    print("="*70)
    print()
    
    # Create VM
    print("Creating VM...")
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()
    
    # Load ONLY ADD weights into layer 9
    print("Loading ONLY ADD weights...")
    compiler = OpcodeNibbleCompiler()
    add_weights = compiler.compile_opcode(Opcode.ADD)
    
    layer = vm.blocks[9].ffn
    layer.W_up.data = add_weights['W_up']
    layer.b_up.data = add_weights['b_up']
    layer.W_gate.data = add_weights['W_gate']
    layer.b_gate.data = add_weights['b_gate']
    layer.W_down.data = add_weights['W_down']
    layer.b_down.data = add_weights['b_down']
    
    nonzero = sum((w.abs() > 1e-9).sum().item() for w in add_weights.values())
    print(f"  Non-zero params: {nonzero}")
    print()
    
    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)
    
    # Test 1: 2 + 3 = 5
    print("Test 1: 2 + 3 = 5")
    print("-" * 70)
    
    input_embedding = embed.encode_vm_state(
        pc=0, ax=2, sp=4096, bp=4096,
        opcode=Opcode.ADD, stack_top=3, batch_size=1,
    )
    
    with torch.no_grad():
        # FFN forward with residual
        up = layer.W_up @ input_embedding.T + layer.b_up.unsqueeze(1)
        gate = layer.W_gate @ input_embedding.T + layer.b_gate.unsqueeze(1)
        hidden = torch.nn.functional.silu(up) * gate
        delta = layer.W_down @ hidden + layer.b_down.unsqueeze(1)
        output = input_embedding.T + delta
        output_embedding = output.T
    
    # Decode
    result = embed.decode_result_nibbles(output_embedding)
    
    print(f"  Input: NIB_A=2, NIB_B=3")
    print(f"  Result: {result}")
    print(f"  Expected: 5")
    print(f"  {'✅ PASS' if result == 5 else '❌ FAIL'}")
    
    # Show RESULT slot values
    print()
    print("RESULT slot values by position:")
    for pos in range(8):
        base_idx = pos * 160
        val = output_embedding[0, base_idx + E.RESULT].item()
        exp_nibble = (5 >> (pos * 4)) & 0xF
        print(f"  Pos {pos}: {val:.2f} (expected {exp_nibble})")
    
    print()

    # Test 2: 100 + 200 = 300
    print("Test 2: 100 + 200 = 300")
    print("-" * 70)

    input_embedding2 = embed.encode_vm_state(
        pc=0, ax=100, sp=4096, bp=4096,
        opcode=Opcode.ADD, stack_top=200, batch_size=1,
    )

    with torch.no_grad():
        # FFN forward with residual
        up2 = layer.W_up @ input_embedding2.T + layer.b_up.unsqueeze(1)
        gate2 = layer.W_gate @ input_embedding2.T + layer.b_gate.unsqueeze(1)
        hidden2 = torch.nn.functional.silu(up2) * gate2
        delta2 = layer.W_down @ hidden2 + layer.b_down.unsqueeze(1)
        output2 = input_embedding2.T + delta2
        output_embedding2 = output2.T

    # Decode
    result2 = embed.decode_result_nibbles(output_embedding2)

    print(f"  Input: NIB_A=100, NIB_B=200")
    print(f"  Result: {result2}")
    print(f"  Expected: 300")
    print(f"  {'✅ PASS' if result2 == 300 else '❌ FAIL'}")

    # Show RESULT slot values
    print()
    print("RESULT slot values by position:")
    for pos in range(8):
        base_idx = pos * 160
        val = output_embedding2[0, base_idx + E.RESULT].item()
        exp_nibble = (300 >> (pos * 4)) & 0xF
        print(f"  Pos {pos}: {val:.2f} (expected {exp_nibble})")

    print()
    print("="*70)

    return result == 5 and result2 == 300

if __name__ == "__main__":
    success = test_isolated_add()
    exit(0 if success else 1)
