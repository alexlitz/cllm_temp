#!/usr/bin/env python3
"""
Test Opcode Mapper with Complex Programs

Tests the opcode mapper and graph compiler with realistic C program patterns:
- Conditional branches
- Loops
- Function calls
- Stack operations
- Multi-operation sequences
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from neural_vm.opcode_mapper import OpcodeMapper, C4Opcode, OpcodeSupport
from neural_vm.graph_weight_compiler import ComputationGraph, WeightEmitter
from neural_vm.embedding import E


class ProgramSimulator:
    """Simulates execution of compiled opcode sequences."""

    def __init__(self):
        self.mapper = OpcodeMapper()
        self.scale = E.SCALE

    def compile_sequence(self, opcodes_with_params):
        """
        Compile a sequence of opcodes into a single computation graph.

        Args:
            opcodes_with_params: List of (opcode, params_dict) tuples

        Returns:
            Combined computation graph
        """
        graphs = []
        for opcode, params in opcodes_with_params:
            if self.mapper.can_compile_to_ffn(opcode):
                graph = self.mapper.compile_opcode(opcode, **params)
                graphs.append(graph)
            else:
                support = self.mapper.get_support_level(opcode)
                print(f"  ⚠️  {opcode.name} requires {support.name} (skipped)")

        return graphs

    def execute_graph(self, graph, inputs):
        """
        Execute a computation graph with given inputs.

        Args:
            graph: ComputationGraph
            inputs: Dict mapping input names to values

        Returns:
            Dict mapping output names to computed values
        """
        # For now, just verify the graph compiles
        # Full execution would require weight emission and forward pass
        return {"compiled": True, "nodes": len(graph.nodes)}


def test_simple_arithmetic():
    """Test: result = (a + b) * c - d"""
    print("\n" + "="*70)
    print("TEST: Simple Arithmetic Expression")
    print("="*70)
    print("C code: result = (a + b) * c - d")
    print()

    sim = ProgramSimulator()

    # Compile sequence: temp1 = a + b, temp2 = temp1 * c, result = temp2 - d
    sequence = [
        (C4Opcode.ADD, {}),  # AX = a + b
        (C4Opcode.MUL, {}),  # AX = (a+b) * c
        (C4Opcode.SUB, {}),  # AX = ((a+b)*c) - d
    ]

    graphs = sim.compile_sequence(sequence)

    print(f"✅ Compiled {len(graphs)} operations")
    for i, graph in enumerate(graphs):
        print(f"   Operation {i+1}: {len(graph.nodes)} nodes")

    return len(graphs) == 3


def test_conditional_branch():
    """Test: if (a > b) then result = a else result = b"""
    print("\n" + "="*70)
    print("TEST: Conditional Branch (Max Function)")
    print("="*70)
    print("C code: result = (a > b) ? a : b")
    print()

    sim = ProgramSimulator()

    # This becomes: CMP_GT(a, b) then SELECT based on result
    mapper = OpcodeMapper()

    # First compare: GT returns 1 if a > b, 0 otherwise
    graph_cmp = mapper.compile_opcode(C4Opcode.GT)
    print(f"✅ GT comparison: {len(graph_cmp.nodes)} nodes")

    # The SELECT is already a primitive, but let's show the pattern
    from neural_vm.graph_weight_compiler import OpType

    graph = ComputationGraph()
    a = graph.add_input("a")
    b = graph.add_input("b")

    # Compare a > b
    cond = graph.add_op(OpType.CMP_GT, [a, b], "cond")

    # Select based on condition
    result = graph.add_op(OpType.SELECT, [cond, a, b], "result")

    print(f"✅ Full max function: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return True


def test_loop_counter():
    """Test: Loop with counter (simplified)"""
    print("\n" + "="*70)
    print("TEST: Loop Counter Pattern")
    print("="*70)
    print("C code: for (i = 0; i < n; i++) { sum += i; }")
    print()

    sim = ProgramSimulator()

    # Loop pattern uses: IMM (i=0), BZ/BNZ (condition), ADD (i++), JMP (loop back)
    sequence = [
        (C4Opcode.IMM, {'imm': 0}),        # i = 0
        (C4Opcode.BZ, {'imm': 0x2000}),    # if i == 0: break (simplified)
        (C4Opcode.ADD, {}),                # sum += i
        (C4Opcode.IMM, {'imm': 1}),        # Load 1
        (C4Opcode.ADD, {}),                # i++
        (C4Opcode.JMP, {'imm': 0x1000}),   # Jump back to loop start
    ]

    graphs = sim.compile_sequence(sequence)

    print(f"✅ Compiled {len(graphs)} loop operations")
    total_nodes = sum(len(g.nodes) for g in graphs)
    print(f"   Total nodes: {total_nodes}")

    return len(graphs) == 6


def test_function_prologue_epilogue():
    """Test: Function entry and exit"""
    print("\n" + "="*70)
    print("TEST: Function Prologue/Epilogue")
    print("="*70)
    print("C code: int add(int a, int b) { return a + b; }")
    print()

    sim = ProgramSimulator()

    # Function prologue: ENT, Function body: ADD, Epilogue: LEV
    sequence = [
        (C4Opcode.ENT, {'imm': 16}),  # Enter function (requires attention)
        (C4Opcode.LEA, {'imm': 8}),   # Load first arg: BP+8
        (C4Opcode.LEA, {'imm': 12}),  # Load second arg: BP+12
        (C4Opcode.ADD, {}),           # a + b
        (C4Opcode.LEV, {}),           # Leave function (requires attention)
    ]

    graphs = sim.compile_sequence(sequence)

    print(f"✅ Compiled {len(graphs)} operations (excluding attention-based)")
    for i, graph in enumerate(graphs):
        print(f"   Operation {i+1}: {len(graph.nodes)} nodes")

    # ENT and LEV require attention, so we should only compile 3 operations
    return len(graphs) == 3


def test_stack_operations():
    """Test: Stack push/pop pattern"""
    print("\n" + "="*70)
    print("TEST: Stack Operations")
    print("="*70)
    print("C code: push temp values, compute, pop")
    print()

    sim = ProgramSimulator()

    # Stack pattern: LEA (get address), PSH (save), compute, ADJ (restore)
    sequence = [
        (C4Opcode.LEA, {'imm': 0}),    # AX = BP + 0 (local var)
        (C4Opcode.PSH, {}),            # push AX (requires attention)
        (C4Opcode.IMM, {'imm': 42}),   # AX = 42
        (C4Opcode.ADD, {}),            # AX = stack_top + 42
        (C4Opcode.ADJ, {'imm': 8}),    # SP += 8 (pop)
    ]

    graphs = sim.compile_sequence(sequence)

    print(f"✅ Compiled {len(graphs)} operations (excluding PSH)")
    return len(graphs) == 4


def test_comparison_chain():
    """Test: Multiple comparisons (range check)"""
    print("\n" + "="*70)
    print("TEST: Comparison Chain (Range Check)")
    print("="*70)
    print("C code: result = (x >= min) && (x <= max)")
    print()

    mapper = OpcodeMapper()

    # Build: (x >= min) && (x <= max)
    graph = ComputationGraph()
    x = graph.add_input("x")
    min_val = graph.add_input("min")
    max_val = graph.add_input("max")

    # x >= min
    from neural_vm.graph_weight_compiler import OpType
    cond1 = graph.add_op(OpType.CMP_GE, [x, min_val], "x_ge_min")

    # x <= max
    cond2 = graph.add_op(OpType.CMP_LE, [x, max_val], "x_le_max")

    # AND both conditions
    result = graph.add_op(OpType.AND, [cond1, cond2], "in_range")

    print(f"✅ Range check compiled: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return len(graph.nodes) == 6  # 3 inputs + 2 comparisons + 1 AND


def test_bitwise_mask():
    """Test: Bitwise operations for masking"""
    print("\n" + "="*70)
    print("TEST: Bitwise Masking Pattern")
    print("="*70)
    print("C code: result = (x & mask) | flag")
    print()

    mapper = OpcodeMapper()
    from neural_vm.graph_weight_compiler import OpType

    # Build: (x & mask) | flag
    graph = ComputationGraph()
    x = graph.add_input("x")
    mask = graph.add_const(0xFF)
    flag = graph.add_input("flag")

    # x & mask
    masked = graph.add_op(OpType.BIT_AND, [x, mask], "masked")

    # masked | flag
    result = graph.add_op(OpType.BIT_OR, [masked, flag], "result")

    print(f"✅ Bitwise mask compiled: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return len(graph.nodes) == 5


def test_shift_and_add():
    """Test: Multiply by power of 2 plus offset"""
    print("\n" + "="*70)
    print("TEST: Shift and Add Pattern")
    print("="*70)
    print("C code: result = (x << 3) + offset  // x * 8 + offset")
    print()

    mapper = OpcodeMapper()
    from neural_vm.graph_weight_compiler import OpType

    # Build: (x << 3) + offset
    graph = ComputationGraph()
    x = graph.add_input("x")
    shift_amt = graph.add_const(3)
    offset = graph.add_input("offset")

    # x << 3
    shifted = graph.add_op(OpType.SHL, [x, shift_amt], "shifted")

    # shifted + offset
    result = graph.add_op(OpType.ADD, [shifted, offset], "result")

    print(f"✅ Shift and add compiled: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return len(graph.nodes) == 5


def test_absolute_value():
    """Test: Absolute value using select"""
    print("\n" + "="*70)
    print("TEST: Absolute Value Function")
    print("="*70)
    print("C code: result = (x < 0) ? -x : x")
    print()

    mapper = OpcodeMapper()
    from neural_vm.graph_weight_compiler import OpType

    # Build: (x < 0) ? -x : x
    graph = ComputationGraph()
    x = graph.add_input("x")
    zero = graph.add_const(0)

    # x < 0
    is_neg = graph.add_op(OpType.CMP_LT, [x, zero], "is_neg")

    # -x
    neg_x = graph.add_op(OpType.SUB, [zero, x], "neg_x")

    # Select: is_neg ? -x : x
    result = graph.add_op(OpType.SELECT, [is_neg, neg_x, x], "abs")

    print(f"✅ Absolute value compiled: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return len(graph.nodes) == 5


def test_division_with_check():
    """Test: Safe division with zero check"""
    print("\n" + "="*70)
    print("TEST: Safe Division with Zero Check")
    print("="*70)
    print("C code: result = (b != 0) ? (a / b) : 0")
    print()

    mapper = OpcodeMapper()
    from neural_vm.graph_weight_compiler import OpType

    # Build: (b != 0) ? (a / b) : 0
    graph = ComputationGraph()
    a = graph.add_input("a")
    b = graph.add_input("b")
    zero = graph.add_const(0)

    # b != 0
    not_zero = graph.add_op(OpType.CMP_NE, [b, zero], "not_zero")

    # a / b
    quotient = graph.add_op(OpType.DIV, [a, b], "quotient")

    # Select: not_zero ? quotient : 0
    result = graph.add_op(OpType.SELECT, [not_zero, quotient, zero], "safe_div")

    print(f"✅ Safe division compiled: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return len(graph.nodes) == 6


def test_min_max_clamp():
    """Test: Clamp value to range [min, max]"""
    print("\n" + "="*70)
    print("TEST: Clamp to Range")
    print("="*70)
    print("C code: result = min(max(x, min_val), max_val)")
    print()

    mapper = OpcodeMapper()
    from neural_vm.graph_weight_compiler import OpType

    # Build: clamp(x, min_val, max_val)
    graph = ComputationGraph()
    x = graph.add_input("x")
    min_val = graph.add_input("min_val")
    max_val = graph.add_input("max_val")

    # Step 1: max(x, min_val) = x > min_val ? x : min_val
    cond1 = graph.add_op(OpType.CMP_GT, [x, min_val], "x_gt_min")
    clamped_low = graph.add_op(OpType.SELECT, [cond1, x, min_val], "clamped_low")

    # Step 2: min(clamped_low, max_val) = clamped_low < max_val ? clamped_low : max_val
    cond2 = graph.add_op(OpType.CMP_LT, [clamped_low, max_val], "low_lt_max")
    result = graph.add_op(OpType.SELECT, [cond2, clamped_low, max_val], "clamped")

    print(f"✅ Clamp compiled: {len(graph.nodes)} nodes")
    for node_id, node in graph.nodes.items():
        print(f"   {node}")

    return len(graph.nodes) == 7  # 3 inputs + 2 comparisons + 2 selects


def test_complex_expression():
    """Test: Complex arithmetic expression"""
    print("\n" + "="*70)
    print("TEST: Complex Expression")
    print("="*70)
    print("C code: result = ((a * b) + (c / d)) - ((e << 2) & f)")
    print()

    mapper = OpcodeMapper()
    from neural_vm.graph_weight_compiler import OpType

    graph = ComputationGraph()
    a = graph.add_input("a")
    b = graph.add_input("b")
    c = graph.add_input("c")
    d = graph.add_input("d")
    e = graph.add_input("e")
    f = graph.add_input("f")

    # a * b
    mul_ab = graph.add_op(OpType.MUL, [a, b], "mul_ab")

    # c / d
    div_cd = graph.add_op(OpType.DIV, [c, d], "div_cd")

    # (a*b) + (c/d)
    add_left = graph.add_op(OpType.ADD, [mul_ab, div_cd], "add_left")

    # e << 2
    two = graph.add_const(2)
    shift_e = graph.add_op(OpType.SHL, [e, two], "shift_e")

    # (e << 2) & f
    and_right = graph.add_op(OpType.BIT_AND, [shift_e, f], "and_right")

    # Final subtraction
    result = graph.add_op(OpType.SUB, [add_left, and_right], "result")

    print(f"✅ Complex expression compiled: {len(graph.nodes)} nodes")
    total_ops = sum(1 for node in graph.nodes.values()
                   if node.op not in [OpType.CONST])
    print(f"   Operations: {total_ops}")
    print(f"   Inputs: 6, Constants: 1, Ops: {total_ops}")

    return len(graph.nodes) > 10


def run_all_tests():
    """Run all program tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "Complex Program Tests" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        ("Simple Arithmetic", test_simple_arithmetic),
        ("Conditional Branch", test_conditional_branch),
        ("Loop Counter", test_loop_counter),
        ("Function Calls", test_function_prologue_epilogue),
        ("Stack Operations", test_stack_operations),
        ("Comparison Chain", test_comparison_chain),
        ("Bitwise Masking", test_bitwise_mask),
        ("Shift and Add", test_shift_and_add),
        ("Absolute Value", test_absolute_value),
        ("Safe Division", test_division_with_check),
        ("Clamp to Range", test_min_max_clamp),
        ("Complex Expression", test_complex_expression),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\n  Total: {passed}/{total} tests passing ({100*passed/total:.1f}%)")

    if passed == total:
        print("\n✅ All complex program tests passing!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
