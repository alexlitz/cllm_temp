"""
Debug ADD operation in detail.

Shows activations at each step to understand why large values fail.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode, E

def test_add_detailed(a, b, expected):
    """Test ADD with detailed activation output."""

    print("="*70)
    print(f"DETAILED ADD TEST: {a} + {b} (expected {expected})")
    print("="*70)
    print()

    # Create VM
    vm = AutoregressiveVM(d_model=1280, n_layers=16, ffn_hidden=4096)
    vm.eval()

    # Load all weights
    loader = CompiledWeightLoader()
    loader.load_all_weights(vm, verbose=False)

    # Create embedding
    embed = NibbleVMEmbedding(d_model=1280)

    # Encode input
    print(f"Input: a={a}, b={b}")
    print()

    # Show input nibbles
    print("Input nibbles:")
    for pos in range(8):
        a_nibble = (a >> (pos * 4)) & 0xF
        b_nibble = (b >> (pos * 4)) & 0xF
        print(f"  Pos {pos}: a={a_nibble:2d} (0x{a_nibble:X}), b={b_nibble:2d} (0x{b_nibble:X})")
    print()

    input_embedding = embed.encode_vm_state(
        pc=0, ax=a, sp=4096, bp=4096,
        opcode=Opcode.ADD, stack_top=b, batch_size=1,
    )

    # Get layer 9 (PRIMARY_ALU)
    layer_idx = loader.layer_map.PRIMARY_ALU
    layer = vm.blocks[layer_idx].ffn

    with torch.no_grad():
        # FFN forward with residual
        up = layer.W_up @ input_embedding.T + layer.b_up.unsqueeze(1)     # [4096, 1]
        gate = layer.W_gate @ input_embedding.T + layer.b_gate.unsqueeze(1)  # [4096, 1]
        hidden = torch.nn.functional.silu(up) * gate                      # [4096, 1]
        delta = layer.W_down @ hidden + layer.b_down.unsqueeze(1)         # [1280, 1]
        output = input_embedding.T + delta                                # [1280, 1]
        output_embedding = output.T                                       # [1, 1280]

    # Decode result
    result = embed.decode_result_nibbles(output_embedding)

    print(f"Result: {result} (expected {expected})")
    print()

    # Show output nibbles
    print("Output nibbles (RESULT slot):")
    for pos in range(8):
        base_idx = pos * 160
        val = output_embedding[0, base_idx + E.RESULT].item()
        exp_nibble = (expected >> (pos * 4)) & 0xF
        res_nibble = (result >> (pos * 4)) & 0xF
        print(f"  Pos {pos}: val={val:7.2f}, decoded={res_nibble:2d} (0x{res_nibble:X}), expected={exp_nibble:2d} (0x{exp_nibble:X}) {'✅' if res_nibble == exp_nibble else '❌'}")
    print()

    # Show carry propagation
    print("Carry values (CARRY_OUT slot):")
    for pos in range(8):
        base_idx = pos * 160
        carry = output_embedding[0, base_idx + E.CARRY_OUT].item()
        print(f"  Pos {pos}: carry_out={carry:7.2f}")
    print()

    # Show hidden unit activations for ADD units (0-63)
    print("Hidden unit activations (ADD units 0-63):")
    nonzero_units = []
    for unit in range(64):
        if hidden[unit, 0].item() > 0.1:  # Significant activation
            nonzero_units.append((unit, hidden[unit, 0].item()))

    if nonzero_units:
        print(f"  {len(nonzero_units)} units active:")
        for unit, val in nonzero_units[:20]:  # Show first 20
            print(f"    Unit {unit:3d}: {val:.3f}")
        if len(nonzero_units) > 20:
            print(f"    ... and {len(nonzero_units) - 20} more")
    else:
        print("  No significant activations!")
    print()

    # Status
    if result == expected:
        print("✅ PASS")
    else:
        print(f"❌ FAIL - got {result}, expected {expected}")

    print("="*70)
    print()

    return result == expected

if __name__ == "__main__":
    # Test small value (known to work)
    test_add_detailed(2, 3, 5)

    # Test larger value (known to fail)
    test_add_detailed(100, 200, 300)

    # Test another value
    test_add_detailed(10, 20, 30)
