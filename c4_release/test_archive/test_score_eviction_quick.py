#!/usr/bin/env python3
"""
Quick test of score-based vs legacy eviction on subset of test programs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer_vm import C4TransformerVM, C4Config
from src.compiler import compile_c


def quick_test():
    """Run quick tests with both eviction strategies."""
    print("=" * 60)
    print("QUICK SCORE-BASED EVICTION TEST")
    print("=" * 60)

    test_cases = [
        ("Simple return", "int main() { return 42; }", 42),
        ("Addition", "int main() { return 10 + 5; }", 15),
        ("Variable", "int main() { int x; x = 123; return x; }", 123),
        ("Multiply", "int main() { return 6 * 7; }", 42),
        ("Overwrite", "int main() { int x; x = 10; x = 20; return x; }", 20),
    ]

    print("\n🔧 Initializing VMs...")

    # Initialize both VMs
    print("  - Creating legacy VM...")
    vm_legacy = C4TransformerVM(C4Config(use_score_based_eviction=False))

    print("  - Creating score-based VM...")
    vm_score = C4TransformerVM(C4Config(use_score_based_eviction=True))

    print("✓ VMs initialized\n")

    results = []

    for i, (name, source, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}: {name}")
        print("-" * 40)

        # Compile
        bytecode, data = compile_c(source)
        print(f"  Compiled: {len(bytecode)} instructions")

        # Test legacy
        print(f"  Running legacy eviction...", end=" ", flush=True)
        vm_legacy.reset()
        vm_legacy.load_bytecode(bytecode, data)
        result_legacy = vm_legacy.run(max_steps=10000)
        print(f"result={result_legacy}")

        # Test score-based
        print(f"  Running score-based eviction...", end=" ", flush=True)
        vm_score.reset()
        vm_score.load_bytecode(bytecode, data)
        result_score = vm_score.run(max_steps=10000)
        print(f"result={result_score}")

        # Check
        legacy_ok = (result_legacy == expected)
        score_ok = (result_score == expected)
        match = (result_legacy == result_score)

        if not match:
            print(f"  ❌ MISMATCH: legacy={result_legacy}, score={result_score}")
            results.append(False)
        elif not legacy_ok or not score_ok:
            print(f"  ❌ WRONG: expected={expected}")
            results.append(False)
        else:
            print(f"  ✓ PASS: both={result_legacy}, expected={expected}")
            results.append(True)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for i, passed_test in enumerate(results, 1):
        name = test_cases[i-1][0]
        status = "✓" if passed_test else "✗"
        print(f"  {status} Test {i}: {name}")

    print()
    print(f"Result: {passed}/{total} tests passed")

    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nScore-based eviction produces identical results to legacy eviction.")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
