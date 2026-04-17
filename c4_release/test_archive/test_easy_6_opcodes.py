#!/usr/bin/env python3
"""
Test if the 6 "easy" opcodes can compile with existing emitters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.embedding import Opcode


def test_easy_opcodes():
    """Test if easy opcodes compile with existing infrastructure."""
    print("="*70)
    print("Testing 6 'Easy' Opcodes with Existing Emitters")
    print("="*70)

    compiler = OpcodeNibbleCompiler()

    easy_opcodes = [
        (Opcode.JMP, "JMP", "Should work if uses MOVE/CONST"),
        (Opcode.BZ, "BZ", "Should work if uses CMP + PC_CONDITIONAL"),
        (Opcode.BNZ, "BNZ", "Should work if uses CMP + PC_CONDITIONAL"),
        (Opcode.ADJ, "ADJ", "Should work if uses ADD"),
        (Opcode.MALC, "MALC", "Should work if uses ADD + CMP + SELECT"),
        (Opcode.FREE, "FREE", "Should work (no-op)"),
    ]

    results = []
    for opcode, name, expectation in easy_opcodes:
        print(f"\n{name} ({opcode}): {expectation}")

        # Check if marked as compilable
        is_compilable = compiler.is_compilable(opcode)
        support = compiler.get_support_level(opcode)

        print(f"  Support level: {support}")
        print(f"  Is compilable: {is_compilable}")

        if not is_compilable:
            print(f"  ❌ Not marked as FFN_DIRECT")
            results.append((name, False, "not_ffn_direct"))
            continue

        # Try to compile
        try:
            weights = compiler.compile_opcode(opcode)
            total = sum(w.numel() for w in weights.values())
            nonzero = sum((w.abs() > 1e-9).sum().item() for w in weights.values())
            sparsity = 100 * (1 - nonzero / total)

            print(f"  ✅ Compiled successfully!")
            print(f"     Non-zero: {nonzero:,} / {total:,}")
            print(f"     Sparsity: {sparsity:.2f}%")
            results.append((name, True, nonzero))

        except NotImplementedError as e:
            print(f"  ⚠️  Not implemented: {e}")
            results.append((name, False, "not_implemented"))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, False, str(e)[:50]))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, p, _ in results if p)
    print(f"\nCompiled: {passed}/6")

    if passed > 0:
        print("\n✅ Working opcodes:")
        for name, p, info in results:
            if p:
                print(f"   {name}: {info} params")

    if passed < 6:
        print("\n❌ Failed opcodes:")
        for name, p, info in results:
            if not p:
                print(f"   {name}: {info}")

    return passed == 6


if __name__ == "__main__":
    success = test_easy_opcodes()
    sys.exit(0 if success else 1)
