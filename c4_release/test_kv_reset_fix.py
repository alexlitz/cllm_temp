#!/usr/bin/env python3
"""
Quick test to verify KV cache reset fix.
Tests that different-sized batches don't cause tensor size mismatches.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


def test_kv_reset():
    """Test that KV cache reset prevents size mismatches between batches."""
    print("=" * 70)
    print("KV CACHE RESET FIX VERIFICATION")
    print("=" * 70)
    print()

    # Create batches of different sizes to trigger the bug if reset doesn't work
    batch1_programs = [
        ("int main() { return 1; }", 1),
        ("int main() { return 2; }", 2),
        ("int main() { return 3; }", 3),
    ]

    batch2_programs = [
        ("int main() { return 10; }", 10),
        ("int main() { return 20; }", 20),
    ]

    batch3_programs = [
        ("int main() { return 100; }", 100),
        ("int main() { return 200; }", 200),
        ("int main() { return 300; }", 300),
        ("int main() { return 400; }", 400),
    ]

    # Compile all programs
    print("Compiling programs...")
    batches = []
    for batch_idx, programs in enumerate([batch1_programs, batch2_programs, batch3_programs], 1):
        bytecodes = []
        data_list = []
        expected = []
        for source, exp in programs:
            bc, data = compile_c(source)
            bytecodes.append(bc)
            data_list.append(data)
            expected.append(exp)
        batches.append((bytecodes, data_list, expected))
        print(f"  Batch {batch_idx}: {len(programs)} programs")

    print()

    # Create runner with KV cache
    print("Creating batched runner with KV cache enabled...")
    runner = BatchedSpeculativeRunner(
        batch_size=10,
        use_kv_cache=True,
        kv_cache_max_tokens=128,
        use_sparse=True,
    )
    print()

    # Run each batch
    print("Running batches...")
    print("(Without the fix, this would fail with tensor size mismatch errors)")
    print()

    all_passed = True
    for batch_idx, (bytecodes, data_list, expected) in enumerate(batches, 1):
        print(f"Batch {batch_idx} ({len(bytecodes)} programs)...", end=" ", flush=True)

        # Reset KV cache before each batch
        if runner.kv_cache is not None:
            runner.kv_cache.reset()

        try:
            results = runner.run_batch(bytecodes, data_list, max_steps=1000)

            # Check results
            passed = 0
            for i, ((out, result), exp) in enumerate(zip(results, expected)):
                if result == exp:
                    passed += 1

            if passed == len(expected):
                print(f"✓ All {passed} tests passed")
            else:
                print(f"✗ Only {passed}/{len(expected)} passed")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {str(e)[:100]}")
            all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("✓ SUCCESS: KV cache reset fix verified!")
        print("  Different batch sizes work correctly without tensor mismatches.")
        return 0
    else:
        print("✗ FAILURE: Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(test_kv_reset())
