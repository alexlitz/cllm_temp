#!/usr/bin/env python3
"""
Run all Neural VM tests.

Usage:
    python -m neural_vm.tests.run_all_tests
    python neural_vm/tests/run_all_tests.py
"""

import sys
import os
import time
import subprocess

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_test_module(module_path: str, name: str) -> tuple:
    """Run a test module and return (passed, total, time)."""
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, module_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start

        # Parse output for pass/fail counts
        output = result.stdout + result.stderr

        # Look for common patterns
        if "All tests passed" in output or "PASSED" in output.upper():
            return True, output, elapsed
        elif result.returncode == 0:
            return True, output, elapsed
        else:
            return False, output, elapsed
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def main():
    print("=" * 70)
    print("NEURAL VM COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    neural_vm_dir = os.path.dirname(test_dir)
    root_dir = os.path.dirname(neural_vm_dir)

    # Define test modules in order of importance
    test_modules = [
        ("C4 ALU Operations", f"{test_dir}/test_c4_ops.py"),
        ("32-bit Operations", f"{test_dir}/test_32bit_ops.py"),
        ("Overflow & Edge Cases", f"{test_dir}/test_overflow.py"),
        ("I/O Operations", f"{test_dir}/test_io_ops.py"),
        ("Memory Subroutines", f"{test_dir}/test_memory_subs.py"),
        ("Neural I/O Comprehensive", f"{test_dir}/test_neural_io_comprehensive.py"),
        ("Sparse Layers", f"{neural_vm_dir}/sparse_layers.py"),
    ]

    # Also run the main 1000+ test suite if available
    root_tests = [
        ("1000+ Test Suite", f"{root_dir}/tests/run_1000_tests.py --quick"),
    ]

    results = []
    total_time = 0

    for name, path in test_modules:
        if not os.path.exists(path.split()[0]):
            print(f"  SKIP: {name} (not found)")
            continue

        print(f"Running: {name}...")
        passed, output, elapsed = run_test_module(path, name)
        total_time += elapsed

        status = "PASS" if passed else "FAIL"
        results.append((name, passed, elapsed))
        print(f"  {status} ({elapsed:.2f}s)")

        if not passed and len(output) < 500:
            print(f"  Output: {output[:500]}")

    # Run root tests
    print()
    print("Running root-level tests...")
    for name, cmd in root_tests:
        print(f"Running: {name}...")
        parts = cmd.split()
        path = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if not os.path.exists(path):
            print(f"  SKIP: {name} (not found)")
            continue

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, path] + args,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=root_dir
            )
            elapsed = time.time() - start
            passed = result.returncode == 0
            results.append((name, passed, elapsed))
            total_time += elapsed
            status = "PASS" if passed else "FAIL"
            print(f"  {status} ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False, 0))

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)

    for name, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} ({elapsed:.2f}s)")

    print()
    print(f"Total: {passed_count}/{total_count} test suites passed")
    print(f"Time: {total_time:.2f}s")
    print()

    if passed_count == total_count:
        print("ALL TEST SUITES PASSED!")
        return 0
    else:
        print(f"FAILED: {total_count - passed_count} test suites")
        return 1


if __name__ == '__main__':
    sys.exit(main())
