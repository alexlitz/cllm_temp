#!/usr/bin/env python3
"""
Test Program Execution with Weight Compilation

Compiles complex computation graphs to weights and executes them to verify
numerical correctness. Tests multi-operation sequences end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.graph_weight_compiler import (
    OpType, ComputationGraph, WeightEmitter, GraphWeightCompiler
)
from neural_vm.embedding import E


def forward_pass_swiglu(W_up, b_up, W_gate, b_gate, W_down, b_down, x):
    """Execute SwiGLU forward pass."""
    up = W_up @ x + b_up
    gate = W_gate @ x + b_gate
    hidden = F.silu(up) * gate
    delta = W_down @ hidden + b_down
    return x + delta


def execute_graph(graph: ComputationGraph, inputs: dict, dim: int = None,
                 hidden_dim: int = 512) -> torch.Tensor:
    """
    Compile graph to weights and execute.

    Args:
        graph: Computation graph
        inputs: Dict mapping input names to values
        dim: Input dimension (auto-calculated if None)
        hidden_dim: Hidden layer dimension

    Returns:
        Output tensor with results
    """
    # Calculate dimension if not provided
    if dim is None:
        # Count unique registers (inputs + outputs)
        registers = set()
        for node in graph.nodes.values():
            if node.output_reg:
                registers.add(node.output_reg)
            for inp_id in node.inputs:
                inp_node = graph.nodes[inp_id]
                if inp_node.output_reg:
                    registers.add(inp_node.output_reg)
        dim = max(len(registers), len(inputs) + 2)

    # Create input tensor
    x = torch.zeros(dim, dtype=torch.float32)

    # Set input values (map by name to register positions)
    reg_map = {}
    for i, name in enumerate(sorted(inputs.keys())):
        reg_map[name] = i
        x[i] = inputs[name]

    # Allocate physical registers
    next_reg = len(inputs)
    for node in graph.nodes.values():
        if node.output_reg and node.output_reg not in reg_map:
            if node.params.get('input'):  # Input nodes already allocated
                continue
            reg_map[node.output_reg] = next_reg
            node.physical_reg = next_reg
            next_reg += 1

    # Assign physical registers to input nodes
    for node in graph.nodes.values():
        if node.output_reg in inputs:
            node.physical_reg = reg_map[node.output_reg]

    # Compile graph to weights
    emitter = WeightEmitter(dim, hidden_dim, E.SCALE)

    # Process nodes in order
    for node_id in graph.topological_sort():
        node = graph.nodes[node_id]

        # Skip input/const nodes that don't emit weights
        if node.op == OpType.CONST and node.params.get('input'):
            continue

        # Emit weights for this node
        if node.op == OpType.CONST:
            # Constant nodes: emit MOVE-like operation with constant value
            value = node.params.get('value', 0)
            x[node.physical_reg] = value
            continue

        # Map inputs to physical registers
        input_nodes = [graph.nodes[i] for i in node.inputs]
        if not all(hasattr(n, 'physical_reg') and n.physical_reg is not None
                   for n in input_nodes):
            continue  # Skip if inputs not ready

        # Emit operation
        try:
            if node.op == OpType.ADD:
                emitter.emit_add(node, graph)
            elif node.op == OpType.SUB:
                emitter.emit_sub(node, graph)
            elif node.op == OpType.MUL:
                emitter.emit_mul(node, graph)
            elif node.op == OpType.DIV:
                emitter.emit_div(node, graph)
            elif node.op == OpType.MOD:
                emitter.emit_mod(node, graph)
            elif node.op == OpType.CMP_EQ:
                emitter.emit_cmp_eq(node, graph)
            elif node.op == OpType.CMP_NE:
                emitter.emit_cmp_ne(node, graph)
            elif node.op == OpType.CMP_LT:
                emitter.emit_cmp_lt(node, graph)
            elif node.op == OpType.CMP_GT:
                emitter.emit_cmp_gt(node, graph)
            elif node.op == OpType.CMP_LE:
                emitter.emit_cmp_le(node, graph)
            elif node.op == OpType.CMP_GE:
                emitter.emit_cmp_ge(node, graph)
            elif node.op == OpType.AND:
                emitter.emit_logical_and(node, graph)
            elif node.op == OpType.OR:
                emitter.emit_logical_or(node, graph)
            elif node.op == OpType.NOT:
                emitter.emit_logical_not(node, graph)
            elif node.op == OpType.BIT_AND:
                # Bitwise ops need one-hot encoding
                continue  # Skip for now (needs proper setup)
            elif node.op == OpType.SELECT:
                emitter.emit_select(node, graph)
            elif node.op == OpType.MOVE:
                emitter.emit_move(node, graph)
        except Exception as e:
            print(f"  ⚠️  Failed to emit {node.op.name}: {e}")
            continue

    # Execute forward pass
    weights = emitter.get_weights()
    result = forward_pass_swiglu(
        weights['W_up'], weights['b_up'],
        weights['W_gate'], weights['b_gate'],
        weights['W_down'], weights['b_down'],
        x
    )

    return result, reg_map


def test_max_function():
    """Test: max(a, b) = (a > b) ? a : b"""
    print("\n" + "="*70)
    print("TEST: Max Function Execution")
    print("="*70)

    graph = ComputationGraph()
    a = graph.add_input("a")
    b = graph.add_input("b")

    # a > b
    cond = graph.add_op(OpType.CMP_GT, [a, b], "cond")

    # Select: cond ? a : b
    result = graph.add_op(OpType.SELECT, [cond, a, b], "result")

    test_cases = [
        ({"a": 10, "b": 5}, 10, "max(10, 5) = 10"),
        ({"a": 5, "b": 10}, 10, "max(5, 10) = 10"),
        ({"a": 7, "b": 7}, 7, "max(7, 7) = 7"),
        ({"a": -5, "b": 3}, 3, "max(-5, 3) = 3"),
    ]

    passed = 0
    for inputs, expected, desc in test_cases:
        output, reg_map = execute_graph(graph, inputs, dim=10)
        result_reg = reg_map.get("result")
        if result_reg is not None:
            actual = output[result_reg].item()
            # Check if close (floating point comparison)
            if abs(actual - expected) < 0.1:
                print(f"  ✅ {desc}: {actual:.1f}")
                passed += 1
            else:
                print(f"  ❌ {desc}: got {actual:.1f}, expected {expected}")
        else:
            print(f"  ❌ {desc}: result register not found")

    return passed == len(test_cases)


def test_abs_function():
    """Test: abs(x) = (x < 0) ? -x : x"""
    print("\n" + "="*70)
    print("TEST: Absolute Value Execution")
    print("="*70)

    graph = ComputationGraph()
    x = graph.add_input("x")
    zero = graph.add_const(0)

    # x < 0
    is_neg = graph.add_op(OpType.CMP_LT, [x, zero], "is_neg")

    # -x = 0 - x
    neg_x = graph.add_op(OpType.SUB, [zero, x], "neg_x")

    # Select: is_neg ? -x : x
    result = graph.add_op(OpType.SELECT, [is_neg, neg_x, x], "abs")

    test_cases = [
        ({"x": 10}, 10, "abs(10) = 10"),
        ({"x": -10}, 10, "abs(-10) = 10"),
        ({"x": 0}, 0, "abs(0) = 0"),
        ({"x": -1}, 1, "abs(-1) = 1"),
    ]

    passed = 0
    for inputs, expected, desc in test_cases:
        output, reg_map = execute_graph(graph, inputs, dim=10)
        result_reg = reg_map.get("abs")
        if result_reg is not None:
            actual = output[result_reg].item()
            if abs(actual - expected) < 0.1:
                print(f"  ✅ {desc}: {actual:.1f}")
                passed += 1
            else:
                print(f"  ❌ {desc}: got {actual:.1f}, expected {expected}")
        else:
            print(f"  ❌ {desc}: result register not found")

    return passed == len(test_cases)


def test_arithmetic_chain():
    """Test: (a + b) - (c + d)"""
    print("\n" + "="*70)
    print("TEST: Arithmetic Chain Execution")
    print("="*70)

    graph = ComputationGraph()
    a = graph.add_input("a")
    b = graph.add_input("b")
    c = graph.add_input("c")
    d = graph.add_input("d")

    # a + b
    sum1 = graph.add_op(OpType.ADD, [a, b], "sum1")

    # c + d
    sum2 = graph.add_op(OpType.ADD, [c, d], "sum2")

    # (a + b) - (c + d)
    result = graph.add_op(OpType.SUB, [sum1, sum2], "result")

    test_cases = [
        ({"a": 10, "b": 5, "c": 3, "d": 2}, 10, "(10+5)-(3+2) = 10"),
        ({"a": 20, "b": 30, "c": 15, "d": 25}, 10, "(20+30)-(15+25) = 10"),
        ({"a": 5, "b": 5, "c": 5, "d": 5}, 0, "(5+5)-(5+5) = 0"),
    ]

    passed = 0
    for inputs, expected, desc in test_cases:
        output, reg_map = execute_graph(graph, inputs, dim=10)
        result_reg = reg_map.get("result")
        if result_reg is not None:
            actual = output[result_reg].item()
            if abs(actual - expected) < 0.1:
                print(f"  ✅ {desc}: {actual:.1f}")
                passed += 1
            else:
                print(f"  ❌ {desc}: got {actual:.1f}, expected {expected}")
        else:
            print(f"  ❌ {desc}: result register not found")

    return passed == len(test_cases)


def test_comparison_and_logic():
    """Test: (a > b) && (c <= d)"""
    print("\n" + "="*70)
    print("TEST: Comparison and Logic Execution")
    print("="*70)

    graph = ComputationGraph()
    a = graph.add_input("a")
    b = graph.add_input("b")
    c = graph.add_input("c")
    d = graph.add_input("d")

    # a > b
    cond1 = graph.add_op(OpType.CMP_GT, [a, b], "a_gt_b")

    # c <= d
    cond2 = graph.add_op(OpType.CMP_LE, [c, d], "c_le_d")

    # (a > b) && (c <= d)
    result = graph.add_op(OpType.AND, [cond1, cond2], "result")

    test_cases = [
        ({"a": 10, "b": 5, "c": 3, "d": 7}, 1, "(10>5) && (3<=7) = true"),
        ({"a": 10, "b": 5, "c": 10, "d": 7}, 0, "(10>5) && (10<=7) = false"),
        ({"a": 3, "b": 5, "c": 3, "d": 7}, 0, "(3>5) && (3<=7) = false"),
        ({"a": 3, "b": 5, "c": 10, "d": 7}, 0, "(3>5) && (10<=7) = false"),
    ]

    passed = 0
    for inputs, expected, desc in test_cases:
        output, reg_map = execute_graph(graph, inputs, dim=10)
        result_reg = reg_map.get("result")
        if result_reg is not None:
            actual = output[result_reg].item()
            # Boolean result: 0 or 1
            actual_bool = 1 if actual > 0.5 else 0
            if actual_bool == expected:
                print(f"  ✅ {desc}: {actual_bool}")
                passed += 1
            else:
                print(f"  ❌ {desc}: got {actual_bool}, expected {expected}")
        else:
            print(f"  ❌ {desc}: result register not found")

    return passed == len(test_cases)


def test_nested_selects():
    """Test: Nested conditional (3-way select)"""
    print("\n" + "="*70)
    print("TEST: Nested Selects (3-way)")
    print("="*70)

    # result = (a < 0) ? -1 : ((a > 0) ? 1 : 0)
    # This is sign(a)

    graph = ComputationGraph()
    a = graph.add_input("a")
    zero = graph.add_const(0)
    neg_one = graph.add_const(-1)
    pos_one = graph.add_const(1)

    # a < 0
    is_neg = graph.add_op(OpType.CMP_LT, [a, zero], "is_neg")

    # a > 0
    is_pos = graph.add_op(OpType.CMP_GT, [a, zero], "is_pos")

    # Inner select: (a > 0) ? 1 : 0
    inner = graph.add_op(OpType.SELECT, [is_pos, pos_one, zero], "inner")

    # Outer select: (a < 0) ? -1 : inner
    result = graph.add_op(OpType.SELECT, [is_neg, neg_one, inner], "sign")

    test_cases = [
        ({"a": 10}, 1, "sign(10) = 1"),
        ({"a": -10}, -1, "sign(-10) = -1"),
        ({"a": 0}, 0, "sign(0) = 0"),
    ]

    passed = 0
    for inputs, expected, desc in test_cases:
        output, reg_map = execute_graph(graph, inputs, dim=10)
        result_reg = reg_map.get("sign")
        if result_reg is not None:
            actual = output[result_reg].item()
            if abs(actual - expected) < 0.1:
                print(f"  ✅ {desc}: {actual:.1f}")
                passed += 1
            else:
                print(f"  ❌ {desc}: got {actual:.1f}, expected {expected}")
        else:
            print(f"  ❌ {desc}: result register not found")

    return passed == len(test_cases)


def run_all_tests():
    """Run all execution tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "Program Execution Tests" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        ("Max Function", test_max_function),
        ("Absolute Value", test_abs_function),
        ("Arithmetic Chain", test_arithmetic_chain),
        ("Comparison and Logic", test_comparison_and_logic),
        ("Nested Selects", test_nested_selects),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("EXECUTION TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\n  Total: {passed}/{total} tests passing ({100*passed/total:.1f}%)")

    if passed == total:
        print("\n✅ All execution tests passing!")
        print("   Complex programs compile to weights and execute correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
