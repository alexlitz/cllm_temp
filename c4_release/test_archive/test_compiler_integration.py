#!/usr/bin/env python3
"""
Test Graph Weight Compiler Integration

Compares compiler-generated weights against reference implementations for
the 22 operations that have been implemented.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import OpType, IRNode, ComputationGraph, WeightEmitter
from neural_vm.embedding import E


def test_operation(op_type, inputs, expected, description, base=None):
    """Test a single operation against expected output.

    Args:
        op_type: OpType enum value
        inputs: List of input values (scalars or one-hot indices)
        expected: Expected output value (scalar or one-hot index)
        description: Test description
        base: Base for one-hot operations (16 for nibbles)

    Returns:
        (success, actual_value, error_message)
    """
    try:
        # Determine if this is a one-hot operation
        is_onehot = op_type in [OpType.MUL, OpType.DIV, OpType.MOD, OpType.BIT_AND, OpType.BIT_OR, OpType.BIT_XOR, OpType.SHL, OpType.SHR]

        # Set dimensions
        if is_onehot:
            base = base or 16
            dim = 3 * base  # a, b, result
            # MUL and bitwise ops use all base*base combinations
            # DIV and MOD exclude division/mod by zero
            if op_type in [OpType.MUL, OpType.BIT_AND, OpType.BIT_OR, OpType.BIT_XOR, OpType.SHL, OpType.SHR]:
                hidden_dim = 2 * base * base
            else:  # DIV, MOD
                hidden_dim = 2 * base * (base - 1)
        else:
            dim = len(inputs) + 1  # inputs + output
            hidden_dim = 512  # Plenty for any scalar operation

        scale = E.SCALE

        # Create graph
        graph = ComputationGraph()

        # For one-hot operations, don't create CONST nodes - just reference the inputs
        # For scalar operations, we still need to handle them
        if is_onehot:
            # Create placeholder input nodes without emitting weights
            input_node_ids = []
            for idx in range(len(inputs)):
                node_id = len(graph.nodes)
                graph.nodes[node_id] = IRNode(
                    id=node_id,
                    op=OpType.CONST,
                    inputs=[],
                    output_reg=f"input{idx}",
                    params={},
                    physical_reg=idx * base
                )
                input_node_ids.append(node_id)

            # Create operation node
            op_node_id = len(graph.nodes)
            out_reg = len(inputs) * base

            op_node = IRNode(
                id=op_node_id,
                op=op_type,
                inputs=input_node_ids,
                output_reg="result",
                params={'base': base},
                physical_reg=out_reg,
                gate=None
            )
            graph.nodes[op_node_id] = op_node

            # Compile only the operation node (skip CONST nodes)
            emitter = WeightEmitter(dim, hidden_dim, scale)
            if op_type == OpType.MUL:
                emitter.emit_mul(op_node, graph)
            elif op_type == OpType.DIV:
                emitter.emit_div(op_node, graph)
            elif op_type == OpType.MOD:
                emitter.emit_mod(op_node, graph)
            elif op_type == OpType.BIT_AND:
                emitter.emit_bit_and(op_node, graph)
            elif op_type == OpType.BIT_OR:
                emitter.emit_bit_or(op_node, graph)
            elif op_type == OpType.BIT_XOR:
                emitter.emit_bit_xor(op_node, graph)
            elif op_type == OpType.SHL:
                emitter.emit_shl(op_node, graph)
            elif op_type == OpType.SHR:
                emitter.emit_shr(op_node, graph)
        else:
            # Scalar operations - emit full graph
            input_node_ids = []
            for idx, val in enumerate(inputs):
                node_id = len(graph.nodes)
                graph.nodes[node_id] = IRNode(
                    id=node_id,
                    op=OpType.CONST,
                    inputs=[],
                    output_reg=f"input{idx}",
                    params={'value': val},
                    physical_reg=idx
                )
                input_node_ids.append(node_id)

            # Create operation node
            op_node_id = len(graph.nodes)
            out_reg = len(inputs)

            params = {}
            if base is not None:
                params['base'] = base

            op_node = IRNode(
                id=op_node_id,
                op=op_type,
                inputs=input_node_ids,
                output_reg="result",
                params=params,
                physical_reg=out_reg,
                gate=None
            )
            graph.nodes[op_node_id] = op_node

            # Compile
            emitter = WeightEmitter(dim, hidden_dim, scale)
            emitter.emit_graph(graph)

        weights = {
            'W_up': emitter.W_up,
            'b_up': emitter.b_up,
            'W_gate': emitter.W_gate,
            'b_gate': emitter.b_gate,
            'W_down': emitter.W_down,
            'b_down': emitter.b_down,
        }

        # Create input tensor
        x = torch.zeros(1, 1, dim)

        if is_onehot:
            # One-hot encoding for each input
            for idx, val in enumerate(inputs):
                if 0 <= val < base:
                    x[0, 0, idx * base + val] = 1.0
        else:
            # Scalar values
            for idx, val in enumerate(inputs):
                x[0, 0, idx] = val

        # Forward pass
        up = F.linear(x, weights['W_up'], weights['b_up'])
        gate = F.linear(x, weights['W_gate'], weights['b_gate'])
        hidden = F.silu(up) * gate
        output_delta = F.linear(hidden, weights['W_down'], weights['b_down'])
        output = x + output_delta

        # Extract result
        if is_onehot:
            result_vec = output[0, 0, out_reg:out_reg + base]
            result = torch.argmax(result_vec).item()
        else:
            result = output[0, 0, out_reg].item()

        # Check match
        if is_onehot:
            match = (result == expected)
            error = 0 if match else 1
        else:
            match = abs(result - expected) < 0.1
            error = abs(result - expected)

        if match:
            return (True, result, None)
        else:
            return (False, result, f"Expected {expected}, got {result:.4f}, error {error:.4f}")

    except Exception as e:
        return (False, None, f"Exception: {str(e)}")


def run_compiler_tests():
    """Run comprehensive tests for all 22 implemented operations."""

    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "Compiler Integration Tests" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = {}

    # ==========================================================================
    # Arithmetic Operations (Scalar)
    # ==========================================================================

    print("=" * 70)
    print("ARITHMETIC OPERATIONS (Scalar)")
    print("=" * 70)

    tests = [
        # ADD
        (OpType.ADD, [42, 58], 100, "ADD(42, 58) = 100"),
        (OpType.ADD, [0, 0], 0, "ADD(0, 0) = 0"),
        (OpType.ADD, [-10, 20], 10, "ADD(-10, 20) = 10"),

        # SUB
        (OpType.SUB, [100, 42], 58, "SUB(100, 42) = 58"),
        (OpType.SUB, [0, 0], 0, "SUB(0, 0) = 0"),
        (OpType.SUB, [20, 30], -10, "SUB(20, 30) = -10"),
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["Arithmetic (Scalar)"] = (passing, len(tests))

    # ==========================================================================
    # Comparison Operations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("COMPARISON OPERATIONS")
    print("=" * 70)

    tests = [
        # CMP_EQ
        (OpType.CMP_EQ, [42, 42], 1, "CMP_EQ(42, 42) = 1"),
        (OpType.CMP_EQ, [42, 43], 0, "CMP_EQ(42, 43) = 0"),

        # CMP_NE
        (OpType.CMP_NE, [42, 43], 1, "CMP_NE(42, 43) = 1"),
        (OpType.CMP_NE, [42, 42], 0, "CMP_NE(42, 42) = 0"),

        # CMP_LT
        (OpType.CMP_LT, [10, 20], 1, "CMP_LT(10, 20) = 1"),
        (OpType.CMP_LT, [20, 10], 0, "CMP_LT(20, 10) = 0"),

        # CMP_LE
        (OpType.CMP_LE, [10, 20], 1, "CMP_LE(10, 20) = 1"),
        (OpType.CMP_LE, [20, 20], 1, "CMP_LE(20, 20) = 1"),
        (OpType.CMP_LE, [20, 10], 0, "CMP_LE(20, 10) = 0"),

        # CMP_GT
        (OpType.CMP_GT, [20, 10], 1, "CMP_GT(20, 10) = 1"),
        (OpType.CMP_GT, [10, 20], 0, "CMP_GT(10, 20) = 0"),

        # CMP_GE
        (OpType.CMP_GE, [20, 10], 1, "CMP_GE(20, 10) = 1"),
        (OpType.CMP_GE, [20, 20], 1, "CMP_GE(20, 20) = 1"),
        (OpType.CMP_GE, [10, 20], 0, "CMP_GE(10, 20) = 0"),
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["Comparisons"] = (passing, len(tests))

    # ==========================================================================
    # Logical Operations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("LOGICAL OPERATIONS")
    print("=" * 70)

    tests = [
        # AND
        (OpType.AND, [1, 1], 1, "AND(1, 1) = 1"),
        (OpType.AND, [1, 0], 0, "AND(1, 0) = 0"),
        (OpType.AND, [0, 0], 0, "AND(0, 0) = 0"),

        # OR
        (OpType.OR, [1, 1], 1, "OR(1, 1) = 1"),
        (OpType.OR, [1, 0], 1, "OR(1, 0) = 1"),
        (OpType.OR, [0, 0], 0, "OR(0, 0) = 0"),

        # NOT
        (OpType.NOT, [0], 1, "NOT(0) = 1"),
        (OpType.NOT, [1], 0, "NOT(1) = 0"),

        # XOR
        (OpType.XOR, [1, 0], 1, "XOR(1, 0) = 1"),
        (OpType.XOR, [0, 1], 1, "XOR(0, 1) = 1"),
        (OpType.XOR, [1, 1], 0, "XOR(1, 1) = 0"),
        (OpType.XOR, [0, 0], 0, "XOR(0, 0) = 0"),
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["Logical"] = (passing, len(tests))

    # ==========================================================================
    # Register Operations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("REGISTER OPERATIONS")
    print("=" * 70)

    tests = [
        # MOVE
        (OpType.MOVE, [42], 42, "MOVE(42) = 42"),
        (OpType.MOVE, [-10], -10, "MOVE(-10) = -10"),
        (OpType.MOVE, [0], 0, "MOVE(0) = 0"),

        # CLEAR
        (OpType.CLEAR, [42], 0, "CLEAR(42) = 0"),
        (OpType.CLEAR, [-10], 0, "CLEAR(-10) = 0"),
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["Register"] = (passing, len(tests))

    # ==========================================================================
    # Conditional Operations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("CONDITIONAL OPERATIONS")
    print("=" * 70)

    tests = [
        # SELECT
        (OpType.SELECT, [1, 42, 99], 42, "SELECT(1, 42, 99) = 42"),
        (OpType.SELECT, [0, 42, 99], 99, "SELECT(0, 42, 99) = 99"),

        # IF_THEN (needs initial value in output register - test framework limitation)
        # Skip IF_THEN for now as it requires special handling
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["Conditional"] = (passing, len(tests))

    # ==========================================================================
    # One-Hot Operations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("ONE-HOT OPERATIONS (Nibbles)")
    print("=" * 70)

    tests = [
        # MUL
        (OpType.MUL, [3, 4], 12, "MUL(3, 4) = 12"),
        (OpType.MUL, [7, 7], 1, "MUL(7, 7) = 1 (49 % 16)"),
        (OpType.MUL, [0, 10], 0, "MUL(0, 10) = 0"),

        # DIV
        (OpType.DIV, [12, 3], 4, "DIV(12, 3) = 4"),
        (OpType.DIV, [15, 4], 3, "DIV(15, 4) = 3"),
        (OpType.DIV, [0, 5], 0, "DIV(0, 5) = 0"),

        # MOD
        (OpType.MOD, [12, 5], 2, "MOD(12, 5) = 2"),
        (OpType.MOD, [15, 4], 3, "MOD(15, 4) = 3"),
        (OpType.MOD, [10, 10], 0, "MOD(10, 10) = 0"),
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc, base=16)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["One-Hot (Nibbles)"] = (passing, len(tests))

    # ==========================================================================
    # Bitwise Operations
    # ==========================================================================

    print("\n" + "=" * 70)
    print("BITWISE OPERATIONS (Nibbles)")
    print("=" * 70)

    tests = [
        # BIT_AND
        (OpType.BIT_AND, [0b1100, 0b1010], 0b1000, "BIT_AND(0b1100, 0b1010) = 0b1000"),
        (OpType.BIT_AND, [0b1111, 0b0000], 0b0000, "BIT_AND(0b1111, 0b0000) = 0b0000"),

        # BIT_OR
        (OpType.BIT_OR, [0b1100, 0b1010], 0b1110, "BIT_OR(0b1100, 0b1010) = 0b1110"),
        (OpType.BIT_OR, [0b0101, 0b0110], 0b0111, "BIT_OR(0b0101, 0b0110) = 0b0111"),

        # BIT_XOR
        (OpType.BIT_XOR, [0b1100, 0b1010], 0b0110, "BIT_XOR(0b1100, 0b1010) = 0b0110"),
        (OpType.BIT_XOR, [0b1111, 0b1111], 0b0000, "BIT_XOR(0b1111, 0b1111) = 0b0000"),

        # SHL
        (OpType.SHL, [0b0001, 1], 0b0010, "SHL(0b0001, 1) = 0b0010"),
        (OpType.SHL, [0b0011, 2], 0b1100, "SHL(0b0011, 2) = 0b1100"),
        (OpType.SHL, [0b1111, 1], 0b1110, "SHL(0b1111, 1) = 0b1110 (overflow)"),

        # SHR
        (OpType.SHR, [0b1000, 1], 0b0100, "SHR(0b1000, 1) = 0b0100"),
        (OpType.SHR, [0b1111, 2], 0b0011, "SHR(0b1111, 2) = 0b0011"),
        (OpType.SHR, [0b1100, 2], 0b0011, "SHR(0b1100, 2) = 0b0011"),
    ]

    passing = 0
    for op_type, inputs, expected, desc in tests:
        success, result, error = test_operation(op_type, inputs, expected, desc, base=16)
        status = "✓" if success else "✗"
        print(f"  {status} {desc}: {result if result is not None else 'ERROR'}")
        if not success and error:
            print(f"      {error}")
        if success:
            passing += 1

    results["Bitwise (Nibbles)"] = (passing, len(tests))

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_passing = 0
    total_tests = 0

    for category, (passing, total) in results.items():
        pct = 100.0 * passing / total if total > 0 else 0
        status = "✓" if passing == total else "✗"
        print(f"  {status} {category}: {passing}/{total} ({pct:.1f}%)")
        total_passing += passing
        total_tests += total

    print(f"\n  Overall: {total_passing}/{total_tests} tests passing ({100.0 * total_passing / total_tests:.1f}%)")

    if total_passing == total_tests:
        print("\n✅ All compiler-generated operations match expected behavior!")
        print("   The graph weight compiler is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - total_passing} test(s) failed")
        print("   Some operations need debugging.")

    print("=" * 70)

    return total_passing == total_tests


if __name__ == "__main__":
    success = run_compiler_tests()
    sys.exit(0 if success else 1)
