#!/usr/bin/env python3
"""
Analyze Autoregressive Status and Weight Generation Potential

Categorizes all 39 opcodes by:
1. Autoregressive purity
2. Weight generation feasibility
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.embedding import Opcode


def analyze_opcodes():
    """Analyze all 39 opcodes for autoregressive status and weight generation."""

    # Pure FFN - 100% autoregressive, weight generation complete
    pure_ffn = {
        "Arithmetic": [
            (Opcode.ADD, "ADD", "✅ Done"),
            (Opcode.SUB, "SUB", "✅ Done"),
            (Opcode.MUL, "MUL", "✅ Done"),
            (Opcode.DIV, "DIV", "✅ Done"),
            (Opcode.MOD, "MOD", "✅ Done"),
        ],
        "Comparison": [
            (Opcode.EQ, "EQ", "✅ Done"),
            (Opcode.NE, "NE", "✅ Done"),
            (Opcode.LT, "LT", "✅ Done"),
            (Opcode.GT, "GT", "✅ Done"),
            (Opcode.LE, "LE", "✅ Done"),
            (Opcode.GE, "GE", "✅ Done"),
        ],
        "Bitwise": [
            (Opcode.OR, "OR", "✅ Done"),
            (Opcode.XOR, "XOR", "✅ Done"),
            (Opcode.AND, "AND", "✅ Done"),
        ],
        "Shift": [
            (Opcode.SHL, "SHL", "✅ Done"),
            (Opcode.SHR, "SHR", "✅ Done"),
        ],
        "Register": [
            (Opcode.LEA, "LEA", "✅ Done"),
            (Opcode.IMM, "IMM", "✅ Done"),
        ]
    }

    # Multi-layer (FFN + Attention) - 100% autoregressive, needs implementation
    multi_layer = {
        "Memory Ops": [
            (Opcode.LI, "LI", "🔄 Autoregressive attention"),
            (Opcode.LC, "LC", "🔄 Autoregressive attention"),
            (Opcode.SI, "SI", "🔄 Autoregressive attention"),
            (Opcode.SC, "SC", "🔄 Autoregressive attention"),
        ],
        "Stack Ops": [
            (Opcode.PSH, "PSH", "🔄 Autoregressive attention"),
        ],
        "Function Calls": [
            (Opcode.JSR, "JSR", "🔄 Autoregressive attention"),
            (Opcode.ENT, "ENT", "🔄 Autoregressive attention"),
            (Opcode.LEV, "LEV", "🔄 Autoregressive attention"),
        ]
    }

    # Control flow - Can be autoregressive with compiled weights
    control_flow = {
        "Control": [
            (Opcode.JMP, "JMP", "🔄 Pure FFN (PC manipulation)"),
            (Opcode.BZ, "BZ", "🔄 Pure FFN (conditional PC)"),
            (Opcode.BNZ, "BNZ", "🔄 Pure FFN (conditional PC)"),
            (Opcode.ADJ, "ADJ", "🔄 Pure FFN (SP adjustment)"),
        ]
    }

    # Heap management - Can be autoregressive
    heap = {
        "Heap": [
            (Opcode.MALC, "MALC", "🔄 Pure FFN (bump allocator)"),
            (Opcode.FREE, "FREE", "🔄 Pure FFN (no-op)"),
        ]
    }

    # Syscalls - NOT autoregressive (external I/O)
    syscalls = {
        "File I/O": [
            (Opcode.OPEN, "OPEN", "❌ External syscall"),
            (Opcode.READ, "READ", "❌ External syscall"),
            (Opcode.CLOS, "CLOS", "❌ External syscall"),
        ],
        "Output": [
            (Opcode.PRTF, "PRTF", "❌ External syscall"),
            (Opcode.PUTCHAR, "PUTCHAR", "❌ External I/O"),
        ],
        "Input": [
            (Opcode.GETCHAR, "GETCHAR", "❌ External I/O"),
        ],
        "Program Control": [
            (Opcode.EXIT, "EXIT", "❌ External exit"),
        ],
    }

    # Memory utilities - Could be autoregressive with loops
    mem_utils = {
        "Memory Utils": [
            (Opcode.MSET, "MSET", "⚠️ Loop needed (multi-step)"),
            (Opcode.MCMP, "MCMP", "⚠️ Loop needed (multi-step)"),
        ]
    }

    print("="*80)
    print("AUTOREGRESSIVE STATUS & WEIGHT GENERATION ANALYSIS")
    print("="*80)

    total_count = 0
    autoregressive_count = 0
    weight_gen_done = 0
    weight_gen_possible = 0

    # Pure FFN
    print("\n" + "="*80)
    print("1. PURE FFN OPERATIONS (100% Autoregressive, Weight Gen: DONE)")
    print("="*80)
    for category, ops in pure_ffn.items():
        print(f"\n  {category}:")
        for opcode, name, status in ops:
            print(f"    {opcode:2d} {name:8s} - {status}")
            total_count += 1
            autoregressive_count += 1
            weight_gen_done += 1
            weight_gen_possible += 1

    # Multi-layer
    print("\n" + "="*80)
    print("2. MULTI-LAYER OPERATIONS (100% Autoregressive, Weight Gen: NEEDED)")
    print("="*80)
    print("   Uses autoregressive attention for memory/stack operations")
    for category, ops in multi_layer.items():
        print(f"\n  {category}:")
        for opcode, name, status in ops:
            print(f"    {opcode:2d} {name:8s} - {status}")
            total_count += 1
            autoregressive_count += 1
            weight_gen_possible += 1

    # Control flow
    print("\n" + "="*80)
    print("3. CONTROL FLOW (100% Autoregressive, Weight Gen: POSSIBLE)")
    print("="*80)
    for category, ops in control_flow.items():
        print(f"\n  {category}:")
        for opcode, name, status in ops:
            print(f"    {opcode:2d} {name:8s} - {status}")
            total_count += 1
            autoregressive_count += 1
            weight_gen_possible += 1

    # Heap
    print("\n" + "="*80)
    print("4. HEAP MANAGEMENT (100% Autoregressive, Weight Gen: POSSIBLE)")
    print("="*80)
    for category, ops in heap.items():
        print(f"\n  {category}:")
        for opcode, name, status in ops:
            print(f"    {opcode:2d} {name:8s} - {status}")
            total_count += 1
            autoregressive_count += 1
            weight_gen_possible += 1

    # Memory utilities
    print("\n" + "="*80)
    print("5. MEMORY UTILITIES (Partially Autoregressive)")
    print("="*80)
    print("   Need iterative multi-step execution")
    for category, ops in mem_utils.items():
        print(f"\n  {category}:")
        for opcode, name, status in ops:
            print(f"    {opcode:2d} {name:8s} - {status}")
            total_count += 1
            autoregressive_count += 1  # Can be done with multi-step
            weight_gen_possible += 1

    # Syscalls
    print("\n" + "="*80)
    print("6. SYSCALLS & I/O (NOT Autoregressive - External Dependencies)")
    print("="*80)
    print("   Require external tool calls / environment interaction")
    for category, ops in syscalls.items():
        print(f"\n  {category}:")
        for opcode, name, status in ops:
            print(f"    {opcode:2d} {name:8s} - {status}")
            total_count += 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal opcodes: {total_count}/39")
    print(f"\n✅ 100% Autoregressive: {autoregressive_count}/{total_count} ({100*autoregressive_count/total_count:.1f}%)")
    print(f"   - Pure FFN: 18")
    print(f"   - Multi-layer (FFN+Attention): 8")
    print(f"   - Control flow: 4")
    print(f"   - Heap management: 2")
    print(f"   - Memory utils: 2")

    print(f"\n❌ NOT Autoregressive: {total_count - autoregressive_count}/{total_count} ({100*(total_count-autoregressive_count)/total_count:.1f}%)")
    print(f"   - External syscalls & I/O: {total_count - autoregressive_count}")

    print(f"\n🏗️ Weight Generation Status:")
    print(f"   - Complete: {weight_gen_done}/{total_count} ({100*weight_gen_done/total_count:.1f}%)")
    print(f"   - Possible (with implementation): {weight_gen_possible}/{total_count} ({100*weight_gen_possible/total_count:.1f}%)")
    print(f"   - Not possible (external): {total_count - weight_gen_possible}/{total_count} ({100*(total_count-weight_gen_possible)/total_count:.1f}%)")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. 34/39 opcodes (87%) can be 100% autoregressive")
    print("   - No external dependencies")
    print("   - Pure transformer forward pass")
    print("   - Compilable to neural weights")

    print("\n2. 5/39 opcodes (13%) require external I/O")
    print("   - OPEN, READ, CLOS, PRTF, PUTCHAR, GETCHAR, EXIT")
    print("   - Must break autoregressiveness for environment interaction")
    print("   - Can still be managed with tool-use boundary pattern")

    print("\n3. Weight generation: 34/39 possible (87%)")
    print("   - 18 already done (pure FFN)")
    print("   - 16 need implementation (multi-layer, control, heap, memops)")
    print("   - 5 need external handlers (syscalls)")

    print("\n" + "="*80)

    return autoregressive_count, weight_gen_possible, total_count


if __name__ == "__main__":
    autoregressive_count, weight_gen_possible, total = analyze_opcodes()

    print("\n" + "="*80)
    print("ANSWER TO YOUR QUESTIONS")
    print("="*80)

    print(f"\n❓ Are they all 100% autoregressive?")
    print(f"   ➜ NO: {autoregressive_count}/{total} are autoregressive ({100*autoregressive_count/total:.1f}%)")
    print(f"   ➜ {total - autoregressive_count} opcodes need external I/O (syscalls)")

    print(f"\n❓ Can we get them all working with weight generation?")
    print(f"   ➜ MOSTLY: {weight_gen_possible}/{total} possible ({100*weight_gen_possible/total:.1f}%)")
    print(f"   ➜ Current: 18/{total} done")
    print(f"   ➜ Remaining: {weight_gen_possible - 18} need implementation")
    print(f"   ➜ Impossible: {total - weight_gen_possible} (external syscalls)")

    print("\n" + "="*80)
