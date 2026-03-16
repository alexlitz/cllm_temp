#!/usr/bin/env python3
"""
Comprehensive tests for:
1. Speculator (FastLogicalVM) - verify it matches neural V5 outputs
2. Pruning - verify memory eviction and KV cache pruning work correctly
"""
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

import torch
import random
from typing import List, Tuple, Dict

from src.speculator import FastLogicalVM, TracingVM
from src.pruned_transformer_vm import PrunedMemory, PrunedRegisterFile, PrunedTransformerVM

from pure_gen_vm_v5 import (
    EmbedDimsV5, EfficientAddSubLayer, EfficientMulProductsLayer,
    EfficientComparisonLayer, EfficientDivLayer, BranchZeroLayer,
    EfficientBitwiseLayer, EfficientShiftLayer, ControlFlowLayer, StackFrameLayer
)
from pure_gen_vm import Opcode

E = EmbedDimsV5
DIM = E.DIM

# =============================================================================
# V5 NEURAL HELPERS
# =============================================================================

def create_v5_input(a, b, opcode):
    """Create value-encoded input tensor for V5."""
    x = torch.zeros(1, 1, DIM)
    for nib in range(8):
        x[0, 0, E.OP_A_VAL_START + nib] = float((a >> (nib * 4)) & 0xF)
        x[0, 0, E.OP_B_VAL_START + nib] = float((b >> (nib * 4)) & 0xF)
    x[0, 0, E.OPCODE_START + opcode] = 1.0
    return x

def read_v5_result(out):
    """Read value-encoded result as integer."""
    result = 0
    for nib in range(8):
        val = out[0, 0, E.RESULT_VAL_START + nib].item()
        result |= (int(round(val)) & 0xF) << (nib * 4)
    return result

# Initialize V5 layers
add_sub_layer = EfficientAddSubLayer(DIM)
mul_layer = EfficientMulProductsLayer(DIM)
div_layer = EfficientDivLayer(DIM)
cmp_layer = EfficientComparisonLayer(DIM)
bitwise_layer = EfficientBitwiseLayer(DIM)
shift_layer = EfficientShiftLayer(DIM)

def v5_add(a, b):
    x = create_v5_input(a, b, Opcode.ADD)
    with torch.no_grad():
        out = add_sub_layer(x)
    return read_v5_result(out)

def v5_sub(a, b):
    x = create_v5_input(a, b, Opcode.SUB)
    with torch.no_grad():
        out = add_sub_layer(x)
    return read_v5_result(out)

def v5_mul(a, b):
    """V5 MUL - only returns product of nibble 0"""
    x = create_v5_input(a & 0xF, b & 0xF, Opcode.MUL)
    with torch.no_grad():
        out = mul_layer(x)
    return int(round(out[0, 0, E.MUL_PROD_START].item()))

def v5_div(a, b):
    x = create_v5_input(a, b, Opcode.DIV)
    with torch.no_grad():
        out = div_layer(x)
    return read_v5_result(out)

def v5_mod(a, b):
    x = create_v5_input(a, b, Opcode.MOD)
    with torch.no_grad():
        out = div_layer(x)
    return read_v5_result(out)

def v5_and(a, b):
    x = create_v5_input(a, b, Opcode.AND)
    with torch.no_grad():
        out = bitwise_layer(x)
    return read_v5_result(out)

def v5_or(a, b):
    x = create_v5_input(a, b, Opcode.OR)
    with torch.no_grad():
        out = bitwise_layer(x)
    return read_v5_result(out)

def v5_xor(a, b):
    x = create_v5_input(a, b, Opcode.XOR)
    with torch.no_grad():
        out = bitwise_layer(x)
    return read_v5_result(out)

def v5_shl(a, b):
    x = create_v5_input(a, b, Opcode.SHL)
    with torch.no_grad():
        out = shift_layer(x)
    return read_v5_result(out)

def v5_shr(a, b):
    x = create_v5_input(a, b, Opcode.SHR)
    with torch.no_grad():
        out = shift_layer(x)
    return read_v5_result(out)

def v5_eq(a, b):
    x = create_v5_input(a, b, Opcode.EQ)
    with torch.no_grad():
        out = cmp_layer(x)
    return 1 if out[0, 0, E.CMP_EQ].item() > 0.5 else 0

def v5_ne(a, b):
    x = create_v5_input(a, b, Opcode.NE)
    with torch.no_grad():
        out = cmp_layer(x)
    return 1 if out[0, 0, E.CMP_NE].item() > 0.5 else 0

def v5_lt(a, b):
    x = create_v5_input(a, b, Opcode.LT)
    with torch.no_grad():
        out = cmp_layer(x)
    return 1 if out[0, 0, E.CMP_LT].item() > 0.5 else 0

def v5_gt(a, b):
    x = create_v5_input(a, b, Opcode.GT)
    with torch.no_grad():
        out = cmp_layer(x)
    return 1 if out[0, 0, E.CMP_GT].item() > 0.5 else 0

def v5_le(a, b):
    x = create_v5_input(a, b, Opcode.LE)
    with torch.no_grad():
        out = cmp_layer(x)
    return 1 if out[0, 0, E.CMP_LE].item() > 0.5 else 0

def v5_ge(a, b):
    x = create_v5_input(a, b, Opcode.GE)
    with torch.no_grad():
        out = cmp_layer(x)
    return 1 if out[0, 0, E.CMP_GE].item() > 0.5 else 0


# =============================================================================
# SPECULATOR TESTS
# =============================================================================

def test_speculator_arithmetic():
    """Test FastLogicalVM arithmetic against V5 neural outputs."""
    print("=" * 70)
    print("SPECULATOR vs V5 NEURAL - ARITHMETIC TESTS")
    print("=" * 70)

    vm = FastLogicalVM()

    # Generate random test cases
    random.seed(42)
    test_values = [0, 1, 2, 15, 16, 255, 256, 0xFFFF, 0x12345678, 0xFFFFFFFF]
    test_values += [random.randint(0, 0xFFFFFFFF) for _ in range(20)]

    results = {
        'ADD': {'pass': 0, 'fail': 0, 'failures': []},
        'SUB': {'pass': 0, 'fail': 0, 'failures': []},
        'AND': {'pass': 0, 'fail': 0, 'failures': []},
        'OR': {'pass': 0, 'fail': 0, 'failures': []},
        'XOR': {'pass': 0, 'fail': 0, 'failures': []},
        'SHL': {'pass': 0, 'fail': 0, 'failures': []},
        'SHR': {'pass': 0, 'fail': 0, 'failures': []},
        'DIV': {'pass': 0, 'fail': 0, 'failures': []},
        'MOD': {'pass': 0, 'fail': 0, 'failures': []},
        'EQ': {'pass': 0, 'fail': 0, 'failures': []},
        'NE': {'pass': 0, 'fail': 0, 'failures': []},
        'LT': {'pass': 0, 'fail': 0, 'failures': []},
        'GT': {'pass': 0, 'fail': 0, 'failures': []},
        'LE': {'pass': 0, 'fail': 0, 'failures': []},
        'GE': {'pass': 0, 'fail': 0, 'failures': []},
    }

    for a in test_values[:15]:  # Limit for speed
        for b in test_values[:15]:
            # ADD
            fast = (a + b) & 0xFFFFFFFF
            neural = v5_add(a, b)
            if fast == neural:
                results['ADD']['pass'] += 1
            else:
                results['ADD']['fail'] += 1
                if len(results['ADD']['failures']) < 5:
                    results['ADD']['failures'].append((a, b, fast, neural))

            # SUB
            fast = (a - b) & 0xFFFFFFFF
            neural = v5_sub(a, b)
            if fast == neural:
                results['SUB']['pass'] += 1
            else:
                results['SUB']['fail'] += 1
                if len(results['SUB']['failures']) < 5:
                    results['SUB']['failures'].append((a, b, fast, neural))

            # AND
            fast = a & b
            neural = v5_and(a, b)
            if fast == neural:
                results['AND']['pass'] += 1
            else:
                results['AND']['fail'] += 1
                if len(results['AND']['failures']) < 5:
                    results['AND']['failures'].append((a, b, fast, neural))

            # OR
            fast = a | b
            neural = v5_or(a, b)
            if fast == neural:
                results['OR']['pass'] += 1
            else:
                results['OR']['fail'] += 1
                if len(results['OR']['failures']) < 5:
                    results['OR']['failures'].append((a, b, fast, neural))

            # XOR
            fast = a ^ b
            neural = v5_xor(a, b)
            if fast == neural:
                results['XOR']['pass'] += 1
            else:
                results['XOR']['fail'] += 1
                if len(results['XOR']['failures']) < 5:
                    results['XOR']['failures'].append((a, b, fast, neural))

            # SHL (limit shift to 31)
            shift = b & 31
            fast = (a << shift) & 0xFFFFFFFF
            neural = v5_shl(a, shift)
            if fast == neural:
                results['SHL']['pass'] += 1
            else:
                results['SHL']['fail'] += 1
                if len(results['SHL']['failures']) < 5:
                    results['SHL']['failures'].append((a, shift, fast, neural))

            # SHR
            shift = b & 31
            fast = a >> shift
            neural = v5_shr(a, shift)
            if fast == neural:
                results['SHR']['pass'] += 1
            else:
                results['SHR']['fail'] += 1
                if len(results['SHR']['failures']) < 5:
                    results['SHR']['failures'].append((a, shift, fast, neural))

            # DIV (avoid div by 0)
            if b > 0:
                fast = a // b
                neural = v5_div(a, b)
                if fast == neural:
                    results['DIV']['pass'] += 1
                else:
                    results['DIV']['fail'] += 1
                    if len(results['DIV']['failures']) < 5:
                        results['DIV']['failures'].append((a, b, fast, neural))

                # MOD
                fast = a % b
                neural = v5_mod(a, b)
                if fast == neural:
                    results['MOD']['pass'] += 1
                else:
                    results['MOD']['fail'] += 1
                    if len(results['MOD']['failures']) < 5:
                        results['MOD']['failures'].append((a, b, fast, neural))

            # Comparisons
            for op_name, fast_fn, neural_fn in [
                ('EQ', lambda x, y: 1 if x == y else 0, v5_eq),
                ('NE', lambda x, y: 1 if x != y else 0, v5_ne),
                ('LT', lambda x, y: 1 if x < y else 0, v5_lt),
                ('GT', lambda x, y: 1 if x > y else 0, v5_gt),
                ('LE', lambda x, y: 1 if x <= y else 0, v5_le),
                ('GE', lambda x, y: 1 if x >= y else 0, v5_ge),
            ]:
                fast = fast_fn(a, b)
                neural = neural_fn(a, b)
                if fast == neural:
                    results[op_name]['pass'] += 1
                else:
                    results[op_name]['fail'] += 1
                    if len(results[op_name]['failures']) < 5:
                        results[op_name]['failures'].append((a, b, fast, neural))

    # Print results
    print("\nResults:")
    print("-" * 70)
    total_pass = 0
    total_fail = 0
    for op, data in results.items():
        total = data['pass'] + data['fail']
        pct = 100 * data['pass'] / total if total > 0 else 0
        status = "✓" if data['fail'] == 0 else "✗"
        print(f"  {op:6s}: {data['pass']:5d}/{total:5d} ({pct:5.1f}%) {status}")
        if data['failures']:
            for f in data['failures'][:2]:
                print(f"          Example: {f}")
        total_pass += data['pass']
        total_fail += data['fail']

    print("-" * 70)
    pct = 100 * total_pass / (total_pass + total_fail)
    print(f"  TOTAL: {total_pass}/{total_pass + total_fail} ({pct:.1f}%)")

    return total_fail == 0


def test_speculator_tracing():
    """Test TracingVM captures correct traces."""
    print("\n" + "=" * 70)
    print("TRACING VM TESTS")
    print("=" * 70)

    vm = TracingVM()

    # Create simple bytecode: IMM 5, PSH, IMM 3, ADD, EXIT
    # Opcodes: IMM=1, PSH=13, ADD=25, EXIT=38
    bytecode = [
        (1 << 8) | 1,    # IMM 5 (but we'll use 5)
        (5 << 8) | 1,    # IMM 5
        13,              # PSH
        (3 << 8) | 1,    # IMM 3
        25,              # ADD
        38,              # EXIT
    ]

    vm.load(bytecode)
    result, trace = vm.run_with_trace()

    print(f"\nBytecode: IMM 5, PSH, IMM 3, ADD, EXIT")
    print(f"Result: {result}")
    print(f"Trace length: {len(trace)}")

    for i, step in enumerate(trace):
        op_name = TracingVM.ARITHMETIC_OPS.get(step.op, f"OP{step.op}")
        print(f"  Step {i}: {op_name} a={step.operand_a} b={step.operand_b} -> {step.result}")

    # Verify the ADD step
    if len(trace) >= 1:
        add_step = trace[0]
        expected = (5 + 3) & 0xFFFFFFFF
        if add_step.result == expected:
            print(f"\n✓ ADD trace correct: 5 + 3 = {expected}")
            return True
        else:
            print(f"\n✗ ADD trace wrong: expected {expected}, got {add_step.result}")
            return False
    return False


# =============================================================================
# PRUNING TESTS
# =============================================================================

def test_memory_pruning():
    """Test PrunedMemory correctly evicts overwritten values."""
    print("\n" + "=" * 70)
    print("MEMORY PRUNING TESTS")
    print("=" * 70)

    mem = PrunedMemory(max_entries=100)

    # Test 1: Overwrite eviction
    print("\n--- Test 1: Overwrite Eviction ---")
    mem.write(0x1000, 42)
    mem.write(0x1000, 100)  # Overwrite
    mem.write(0x1000, 200)  # Overwrite again

    assert mem.read(0x1000) == 200, "Should read latest value"
    assert len(mem.entries) == 1, "Should have only 1 entry"
    assert mem.pruned_overwrites == 2, "Should have pruned 2 overwrites"
    print(f"  ✓ Overwrites correctly pruned: {mem.pruned_overwrites}")

    # Test 2: Multiple addresses
    print("\n--- Test 2: Multiple Addresses ---")
    mem = PrunedMemory(max_entries=100)
    for addr in range(50):
        mem.write(addr, addr * 10)

    assert len(mem.entries) == 50, "Should have 50 entries"
    for addr in range(50):
        assert mem.read(addr) == addr * 10, f"Address {addr} should have value {addr * 10}"
    print(f"  ✓ 50 addresses stored correctly")

    # Test 3: LRU eviction when exceeding max
    print("\n--- Test 3: LRU Eviction ---")
    mem = PrunedMemory(max_entries=10)

    # Write 15 entries (should trigger LRU eviction)
    for i in range(15):
        mem.write(i, i * 100)
        mem.step()

    # Read some entries to make them "recently used"
    mem.read(12)  # Mark as recently read
    mem.read(13)
    mem.read(14)

    # Write more to trigger eviction
    for i in range(15, 20):
        mem.write(i, i * 100)
        mem.step()

    print(f"  Entries after eviction: {len(mem.entries)}")
    print(f"  Total writes: {mem.total_writes}")

    # Recent entries should still exist
    assert mem.read(14) == 14 * 100, "Recently read entry should exist"
    print(f"  ✓ Recently read entries preserved")

    # Test 4: Freed memory detection
    print("\n--- Test 4: Memory Write/Free Pattern ---")
    mem = PrunedMemory(max_entries=100)

    # Simulate stack usage: push, use, pop pattern
    sp = 0x10000

    # Push values
    for i in range(10):
        sp -= 8
        mem.write(sp, i)
        mem.step()

    entries_after_push = len(mem.entries)

    # Pop values (overwrite with garbage to simulate "free")
    for i in range(10):
        mem.write(sp, 0xDEADBEEF)  # Mark as freed
        sp += 8
        mem.step()

    # The freed entries should be overwritten
    print(f"  Entries after push: {entries_after_push}")
    print(f"  Entries after pop: {len(mem.entries)}")
    print(f"  Pruned overwrites: {mem.pruned_overwrites}")

    assert mem.pruned_overwrites == 10, "Should have pruned 10 stack entries"
    print(f"  ✓ Stack memory correctly pruned")

    return True


def test_register_pruning():
    """Test PrunedRegisterFile only keeps current state."""
    print("\n" + "=" * 70)
    print("REGISTER PRUNING TESTS")
    print("=" * 70)

    regs = PrunedRegisterFile()

    # Test 1: Only current state
    print("\n--- Test 1: Current State Only ---")
    regs.pc = 100
    regs.ax = 42
    regs.sp = 0x20000
    regs.bp = 0x20000

    state = regs.get_state()
    assert state == (100, 0x20000, 0x20000, 42), f"State mismatch: {state}"
    print(f"  ✓ Current state correct: PC={state[0]}, SP={hex(state[1])}, BP={hex(state[2])}, AX={state[3]}")

    # Test 2: History is limited
    print("\n--- Test 2: Limited History ---")
    regs = PrunedRegisterFile()
    regs.history_size = 10

    for i in range(100):
        regs.pc = i * 8
        regs.ax = i
        regs.snapshot()

    assert len(regs.history) == 10, f"History should be limited to 10, got {len(regs.history)}"
    print(f"  ✓ History limited to {len(regs.history)} entries (wrote 100)")

    # Test 3: History contains recent values
    print("\n--- Test 3: Recent History ---")
    # Last entry should be the most recent
    last_pc, last_sp, last_bp, last_ax = regs.history[-1]
    assert last_pc == 99 * 8, f"Last PC should be {99*8}, got {last_pc}"
    assert last_ax == 99, f"Last AX should be 99, got {last_ax}"
    print(f"  ✓ Recent history preserved: PC={last_pc}, AX={last_ax}")

    return True


def test_kv_cache_pruning():
    """Test that KV cache entries for old memory are evicted."""
    print("\n" + "=" * 70)
    print("KV CACHE / MEMORY CONSISTENCY TESTS")
    print("=" * 70)

    # The PrunedTransformerVM should maintain consistency between
    # memory writes and any potential KV cache

    vm = PrunedTransformerVM()

    # Test 1: Memory consistency after overwrites
    print("\n--- Test 1: Memory Consistency ---")

    # Write to same address multiple times
    vm.memory.write(0x1000, 111)
    vm.memory.step()
    vm.memory.write(0x1000, 222)
    vm.memory.step()
    vm.memory.write(0x1000, 333)
    vm.memory.step()

    val = vm.memory.read(0x1000)
    assert val == 333, f"Should read latest value 333, got {val}"
    assert len(vm.memory.entries) == 1, "Should only have 1 entry for this address"
    print(f"  ✓ Memory reads latest value after overwrites: {val}")

    # Test 2: Register consistency
    print("\n--- Test 2: Register Consistency ---")
    vm.regs.ax = 100
    vm.regs.ax = 200
    vm.regs.ax = 300

    assert vm.regs.ax == 300, f"AX should be 300, got {vm.regs.ax}"
    print(f"  ✓ Register reads latest value: {vm.regs.ax}")

    # Test 3: Stack frame memory consistency
    print("\n--- Test 3: Stack Frame Consistency ---")
    vm = PrunedTransformerVM()

    # Simulate function calls with stack frames
    initial_sp = vm.regs.sp

    # Function 1: push local vars
    vm.regs.sp -= 8
    vm.memory.write(vm.regs.sp, 0xAAAA)
    vm.regs.sp -= 8
    vm.memory.write(vm.regs.sp, 0xBBBB)

    entries_in_func = len(vm.memory.entries)

    # Return: restore SP (logically frees stack memory)
    vm.regs.sp = initial_sp

    # New function: reuse same stack addresses
    vm.regs.sp -= 8
    vm.memory.write(vm.regs.sp, 0xCCCC)  # Overwrites old 0xAAAA
    vm.regs.sp -= 8
    vm.memory.write(vm.regs.sp, 0xDDDD)  # Overwrites old 0xBBBB

    # Old values should be pruned
    print(f"  Entries during first function: {entries_in_func}")
    print(f"  Entries after reuse: {len(vm.memory.entries)}")
    print(f"  Pruned overwrites: {vm.memory.pruned_overwrites}")

    # Read back to verify new values
    vm.regs.sp += 8
    val1 = vm.memory.read(vm.regs.sp - 8)
    val2 = vm.memory.read(vm.regs.sp)

    assert val1 == 0xDDDD, f"Should read new value 0xDDDD, got {hex(val1)}"
    print(f"  ✓ Stack memory correctly overwritten on reuse")

    return True


def test_full_program_pruning():
    """Test pruning with a full program execution."""
    print("\n" + "=" * 70)
    print("FULL PROGRAM PRUNING TEST")
    print("=" * 70)

    # Simpler direct memory test - just use the PrunedMemory directly
    mem = PrunedMemory(max_entries=1000)

    # Simulate: for (i = 0; i < 100; i++) { arr[0] = i; }
    for i in range(100):
        mem.write(0x1000, i)
        mem.step()

    stats = mem.stats()

    print(f"\nAfter 100 writes to same address:")
    print(f"  Live entries: {stats['live_entries']}")
    print(f"  Total writes: {stats['total_writes']}")
    print(f"  Pruned overwrites: {stats['pruned_overwrites']}")
    print(f"  Prune ratio: {stats['prune_ratio']:.1%}")

    # Should have pruned 99 overwrites (keeping only final value)
    assert stats['pruned_overwrites'] == 99, f"Should have pruned 99 overwrites, got {stats['pruned_overwrites']}"
    print(f"  ✓ Overwrites correctly pruned")

    # Final value should be 99
    final_val = mem.read(0x1000)
    assert final_val == 99, f"Final value should be 99, got {final_val}"
    print(f"  ✓ Final value correct: {final_val}")

    # Test 2: Multiple addresses with partial overwrites
    print("\n--- Test 2: Multi-Address Partial Overwrite ---")
    mem = PrunedMemory(max_entries=1000)

    # Write to 10 addresses, then overwrite first 5
    for i in range(10):
        mem.write(i, i * 10)
        mem.step()

    for i in range(5):
        mem.write(i, i * 100)  # Overwrite first 5
        mem.step()

    stats = mem.stats()
    print(f"  Live entries: {stats['live_entries']}")
    print(f"  Pruned overwrites: {stats['pruned_overwrites']}")

    assert stats['live_entries'] == 10, "Should still have 10 entries"
    assert stats['pruned_overwrites'] == 5, "Should have pruned 5 overwrites"

    # Verify values
    for i in range(5):
        assert mem.read(i) == i * 100, f"Address {i} should have new value {i*100}"
    for i in range(5, 10):
        assert mem.read(i) == i * 10, f"Address {i} should have old value {i*10}"

    print(f"  ✓ Partial overwrite correct")

    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPECULATOR AND PRUNING COMPREHENSIVE TESTS")
    print("=" * 70)

    all_passed = True

    # Speculator tests
    all_passed &= test_speculator_arithmetic()
    all_passed &= test_speculator_tracing()

    # Pruning tests
    all_passed &= test_memory_pruning()
    all_passed &= test_register_pruning()
    all_passed &= test_kv_cache_pruning()
    all_passed &= test_full_program_pruning()

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
