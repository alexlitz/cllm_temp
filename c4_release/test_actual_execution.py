#!/usr/bin/env python3
"""
Test Actual Execution with Compiled Weights

Verify that compiled weights actually execute correctly in the VM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.embedding import Opcode


def test_weight_dimensions():
    """Test: Verify compiled weight dimensions."""
    print("="*70)
    print("TEST: Weight Dimensions")
    print("="*70)

    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_opcode(Opcode.ADD)

    print(f"\nCompiled ADD weights:")
    for k, v in weights.items():
        print(f"  {k}: {list(v.shape)}")

    # Check if they're the right shape for d_model=1280
    expected_d_model = 1280
    expected_hidden = 4096

    correct = (
        weights['W_up'].shape == (expected_hidden, expected_d_model) and
        weights['W_down'].shape == (expected_d_model, expected_hidden)
    )

    if correct:
        print(f"\n✅ Weights are correctly shaped for d_model={expected_d_model}")
        return True
    else:
        print(f"\n❌ Weight shapes don't match expected dimensions")
        return False


def test_nibble_simulation():
    """Test: Simulate nibble ADD operation manually."""
    print("\n" + "="*70)
    print("TEST: Manual Nibble Simulation")
    print("="*70)

    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_opcode(Opcode.ADD)

    # Simulate: 0x0F + 0x01 = 0x10 (15 + 1 = 16)
    # In nibbles: [F,0,0,0] + [1,0,0,0] = [0,1,0,0]

    print("\nSimulating: 0x0F + 0x01 = 0x10")
    print("  Nibble 0: F + 1 = 0 (carry 1)")
    print("  Nibble 1: 0 + 0 + carry(1) = 1")
    print("  Nibble 2-7: 0")

    # Create input embedding (simplified - just the relevant nibble dims)
    from neural_vm.nibble_weight_compiler import NibbleRegisterMap
    reg_map = NibbleRegisterMap()

    d_model = 1280
    x = torch.zeros(1, d_model)  # [1, 1280]

    # Set nibble 0: A=15, B=1
    x[0, reg_map._flatten_index(0, reg_map.NIB_A)] = 15.0
    x[0, reg_map._flatten_index(0, reg_map.NIB_B)] = 1.0

    # Set opcode gate for ADD (Opcode.ADD = 25)
    from neural_vm.embedding import E
    x[0, reg_map._flatten_index(0, E.OP_START + Opcode.ADD)] = 1.0

    print("\n  Input prepared:")
    print(f"    NIB_A[0] = 15")
    print(f"    NIB_B[0] = 1")
    print(f"    Opcode gate[{Opcode.ADD}] = 1")

    # Apply FFN layers (simplified - just the core computation)
    W_up = weights['W_up']      # [4096, 1280]
    b_up = weights['b_up']      # [4096]
    W_gate = weights['W_gate']  # [4096, 1280]
    b_gate = weights['b_gate']  # [4096]
    W_down = weights['W_down']  # [1280, 4096]
    b_down = weights['b_down']  # [1280]

    # Up projection
    up = torch.matmul(x, W_up.T) + b_up  # [1, 4096]
    gate = torch.matmul(x, W_gate.T) + b_gate  # [1, 4096]

    # SwiGLU activation
    hidden = torch.nn.functional.silu(gate) * up

    # Down projection
    output = torch.matmul(hidden, W_down.T) + b_down  # [1, 1280]

    # Add residual
    output = output + x

    # Read result nibbles
    result_nibbles = []
    for pos in range(8):
        result_idx = reg_map._flatten_index(pos, reg_map.RESULT)
        result_val = output[0, result_idx].item()
        result_nibbles.append(result_val)

    print(f"\n  Output nibbles:")
    for i, val in enumerate(result_nibbles):
        print(f"    RESULT[{i}] = {val:.2f}")

    # Check nibble 0 (should be 0) and nibble 1 (should be 1)
    correct = (
        abs(result_nibbles[0] - 0.0) < 1.0 and  # F+1 mod 16 = 0
        abs(result_nibbles[1] - 1.0) < 1.0      # Carry propagated
    )

    if correct:
        print("\n✅ Manual simulation matches expected result!")
        return True
    else:
        print(f"\n⚠️  Result doesn't match expected (got {result_nibbles[0]:.2f}, {result_nibbles[1]:.2f})")
        print("     (This might be OK - full carry propagation needs multi-layer)")
        return False


def test_opcode_coverage():
    """Test: Check which opcodes are actually implemented."""
    print("\n" + "="*70)
    print("TEST: Opcode Coverage")
    print("="*70)

    from neural_vm.embedding import Opcode
    compiler = OpcodeNibbleCompiler()

    # Core C4 opcodes 0-38
    core_opcodes = [
        (0, "LEA"), (1, "IMM"), (2, "JMP"), (3, "JSR"), (4, "BZ"),
        (5, "BNZ"), (6, "ENT"), (7, "ADJ"), (8, "LEV"), (9, "LI"),
        (10, "LC"), (11, "SI"), (12, "SC"), (13, "PSH"), (14, "OR"),
        (15, "XOR"), (16, "AND"), (17, "EQ"), (18, "NE"), (19, "LT"),
        (20, "GT"), (21, "LE"), (22, "GE"), (23, "SHL"), (24, "SHR"),
        (25, "ADD"), (26, "SUB"), (27, "MUL"), (28, "DIV"), (29, "MOD"),
        (30, "OPEN"), (31, "READ"), (32, "CLOS"), (33, "PRTF"),
        (34, "MALC"), (35, "FREE"), (36, "MSET"), (37, "MCMP"),
        (38, "EXIT"),
    ]

    compiled = []
    graphed = []
    missing = []

    from neural_vm.full_opcode_mapper import FullOpcodeMapper
    mapper = FullOpcodeMapper()

    for opcode, name in core_opcodes:
        # Check if compilable
        is_compilable = compiler.is_compilable(opcode)

        # Check if graphed
        try:
            if opcode in [0, 1, 2, 3, 4, 5, 6, 7, 34, 38]:
                graph = mapper.map_opcode(opcode, imm=0)
                is_graphed = True
            elif opcode in [9, 10, 11, 12, 13, 8]:
                multi = mapper.map_opcode_multilayer(opcode)
                is_graphed = True
            else:
                graph = mapper.map_opcode(opcode)
                is_graphed = True
        except:
            is_graphed = False

        if is_compilable:
            compiled.append(name)
        if is_graphed:
            graphed.append(name)
        if not is_graphed:
            missing.append(f"{opcode}:{name}")

    print(f"\nCore C4 opcodes (0-38): 39 total")
    print(f"  Graphed: {len(graphed)}/39")
    print(f"  Compilable to FFN: {len(compiled)}/39")
    print(f"  Missing: {len(missing)}/39")

    if missing:
        print(f"\n  Missing opcodes: {', '.join(missing)}")

    coverage = len(graphed) / 39 * 100
    print(f"\n  Overall coverage: {coverage:.1f}%")

    return len(missing) == 0


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "Actual Execution Tests" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝\n")

    results = []

    # Test 1: Dimensions
    try:
        passed = test_weight_dimensions()
        results.append(("Weight Dimensions", passed))
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Weight Dimensions", False))

    # Test 2: Manual simulation
    try:
        passed = test_nibble_simulation()
        results.append(("Nibble Simulation", passed))
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Nibble Simulation", False))

    # Test 3: Coverage
    try:
        passed = test_opcode_coverage()
        results.append(("Opcode Coverage", passed))
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Opcode Coverage", False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed_count}/{len(results)} tests passing")
