#!/usr/bin/env python3
"""
Test All FFN-Compilable Opcodes

Tests weight generation for all opcodes marked as FFN_DIRECT:
- Arithmetic: ADD, SUB, MUL, DIV, MOD
- Comparison: EQ, NE, LT, GT, LE, GE
- Bitwise: OR, XOR, AND
- Shift: SHL, SHR
- Register: LEA, IMM
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.embedding import Opcode


def test_all_ffn_direct_opcodes():
    """Test: Compile all FFN_DIRECT opcodes to weights."""
    print("="*70)
    print("Testing All FFN_DIRECT Opcode Compilation")
    print("="*70)

    compiler = OpcodeNibbleCompiler()

    # All FFN_DIRECT opcodes from opcode_nibble_integration.py
    test_opcodes = [
        # Arithmetic
        (Opcode.ADD, "ADD", "Arithmetic"),
        (Opcode.SUB, "SUB", "Arithmetic"),
        (Opcode.MUL, "MUL", "Arithmetic"),
        (Opcode.DIV, "DIV", "Arithmetic"),
        (Opcode.MOD, "MOD", "Arithmetic"),

        # Comparison
        (Opcode.EQ, "EQ", "Comparison"),
        (Opcode.NE, "NE", "Comparison"),
        (Opcode.LT, "LT", "Comparison"),
        (Opcode.GT, "GT", "Comparison"),
        (Opcode.LE, "LE", "Comparison"),
        (Opcode.GE, "GE", "Comparison"),

        # Bitwise
        (Opcode.OR, "OR", "Bitwise"),
        (Opcode.XOR, "XOR", "Bitwise"),
        (Opcode.AND, "AND", "Bitwise"),

        # Shift
        (Opcode.SHL, "SHL", "Shift"),
        (Opcode.SHR, "SHR", "Shift"),

        # Register
        (Opcode.LEA, "LEA", "Register"),
        (Opcode.IMM, "IMM", "Register"),
    ]

    # Group by category
    categories = {}
    for opcode, name, category in test_opcodes:
        if category not in categories:
            categories[category] = []
        categories[category].append((opcode, name))

    results = []
    total_params = 0
    total_nonzero = 0

    for category in sorted(categories.keys()):
        print(f"\n{category} Operations:")
        for opcode, name in categories[category]:
            try:
                # Check if compilable
                if not compiler.is_compilable(opcode):
                    print(f"  ⚠️  {name:6s}: Marked as not compilable")
                    results.append((name, False, 0, 0))
                    continue

                # Generate weights
                weights = compiler.compile_opcode(opcode)

                # Calculate sparsity
                total = sum(w.numel() for w in weights.values())
                nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
                sparsity = 100 * (1 - nonzero / total)

                total_params += total
                total_nonzero += nonzero

                print(f"  ✅ {name:6s} ({opcode:2d}): {nonzero:,} / {total:,} params ({sparsity:.2f}% sparse)")
                results.append((name, True, nonzero, sparsity))

            except NotImplementedError as e:
                print(f"  ❌ {name:6s}: {e}")
                results.append((name, False, 0, 0))
            except Exception as e:
                print(f"  ❌ {name:6s}: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False, 0, 0))

    # Summary
    passed = sum(1 for _, p, _, _ in results if p)
    total = len(results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if passed > 0:
        avg_sparsity = sum(s for _, p, _, s in results if p) / passed
        print(f"\n  Compiled: {passed}/{total} opcodes ({100*passed/total:.1f}%)")
        print(f"  Total non-zero params: {total_nonzero:,}")
        print(f"  Average sparsity: {avg_sparsity:.2f}%")

    if passed == total:
        print("\n  🎉 ALL FFN_DIRECT OPCODES COMPILE SUCCESSFULLY!")
        print("\n  This confirms:")
        print("    ✅ All arithmetic operations (ADD, SUB, MUL, DIV, MOD)")
        print("    ✅ All comparison operations (EQ, NE, LT, GT, LE, GE)")
        print("    ✅ All bitwise operations (OR, XOR, AND)")
        print("    ✅ All shift operations (SHL, SHR)")
        print("    ✅ Register operations (LEA, IMM)")
        return True
    else:
        print(f"\n  ⚠️  {total - passed} opcode(s) failed to compile")
        return False


if __name__ == "__main__":
    success = test_all_ffn_direct_opcodes()
    sys.exit(0 if success else 1)
