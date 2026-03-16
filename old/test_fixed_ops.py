"""
Test the fixed MUL and DIV operations in the MoE VM.

MUL: Uses quarter-squares identity a*b = [(a+b)² - (a-b)²]/4
DIV: Uses log2 via attention + exp via softmax ratio (exact)
"""

import torch
from c4_moe_vm import ArithmeticExperts

def test_mul():
    """Test multiplication via silu pairs."""
    print("=" * 60)
    print("TESTING MUL VIA SILU PAIRS")
    print("=" * 60)
    print("Formula: a * b = silu(a) * b + silu(-a) * (-b)")
    print("Proof: = a*b*(sigmoid(a) + sigmoid(-a)) = a*b*1 = a*b")
    print()

    arith = ArithmeticExperts()

    tests = [
        (6, 7, 42),
        (3, 5, 15),
        (10, 10, 100),
        (12, 8, 96),
        (1, 100, 100),
        (0, 50, 0),
        (7, 11, 77),
        (13, 17, 221),
        (100, 5, 500),
        (-3, 4, -12),
        (-5, -6, 30),
    ]

    passed = 0
    for a, b, expected in tests:
        result = int(arith.mul(torch.tensor(float(a)), torch.tensor(float(b))).item())
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {a:4d} * {b:4d} = {result:6d} (expected {expected:6d}) {status}")

    print(f"\nMUL via silu pairs: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_div():
    """Test division via log attention + exp softmax."""
    print("\n" + "=" * 60)
    print("TESTING DIV VIA LOG-ATTENTION + EXP-SOFTMAX")
    print("=" * 60)
    print("Formula: a/b = exp(log2(a) - log2(b)) * ln(2)")
    print("         log2 via attention, exp via softmax ratio (exact)")
    print()

    arith = ArithmeticExperts()

    tests = [
        (42, 7, 6),
        (100, 5, 20),
        (20, 4, 5),
        (100, 10, 10),
        (81, 9, 9),
        (144, 12, 12),
        (17, 5, 3),
        (100, 7, 14),
        (1000, 10, 100),
        (120, 4, 30),
        (120, 3, 40),
        (1, 1, 1),
        (50, 50, 1),
        (999, 111, 9),
    ]

    passed = 0
    for a, b, expected in tests:
        result = int(arith.div(torch.tensor(float(a)), torch.tensor(float(b))).item())
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {a:4d} / {b:4d} = {result:4d} (expected {expected:4d}) {status}")

    print(f"\nDIV: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_comprehensive_div():
    """Comprehensive division test."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DIVISION TEST")
    print("=" * 60)

    arith = ArithmeticExperts()

    passed = 0
    total = 0
    failures = []

    for a in range(1, 200):
        for b in range(1, 50):
            expected = a // b
            result = int(arith.div(torch.tensor(float(a)), torch.tensor(float(b))).item())
            total += 1
            if result == expected:
                passed += 1
            else:
                failures.append((a, b, expected, result))

    print(f"  {passed}/{total} passed ({100*passed/total:.1f}%)")

    if failures:
        print(f"\n  First 10 failures:")
        for a, b, exp, got in failures[:10]:
            print(f"    {a}/{b} = {got} (expected {exp})")

    return passed == total


def test_exp_via_softmax():
    """Test exp via softmax ratio."""
    print("\n" + "=" * 60)
    print("TESTING EXP VIA SOFTMAX RATIO")
    print("=" * 60)
    print("Formula: exp(x) = softmax([x, 0])[0] / softmax([x, 0])[1]")
    print()

    arith = ArithmeticExperts()

    import math

    tests = [0.0, 1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0]

    all_close = True
    for x in tests:
        expected = math.exp(x)
        result = arith.exp_via_softmax(torch.tensor(x)).item()
        error = abs(result - expected)
        status = "✓" if error < 1e-6 else "✗"
        if error >= 1e-6:
            all_close = False
        print(f"  exp({x:5.1f}) = {result:10.6f} (expected {expected:10.6f}, err={error:.2e}) {status}")

    print(f"\n  Exp via softmax: {'EXACT' if all_close else 'INEXACT'}")
    return all_close


if __name__ == "__main__":
    print("TESTING FIXED ARITHMETIC OPERATIONS")
    print("=" * 60)
    print("All operations use only transformer primitives:")
    print("  - SwiGLU for sharp gates")
    print("  - Attention for lookup tables")
    print("  - Softmax ratio for exact exp")
    print()

    r1 = test_exp_via_softmax()
    r2 = test_mul()
    r3 = test_div()
    r4 = test_comprehensive_div()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exp via softmax: {'PASS' if r1 else 'FAIL'}")
    print(f"  MUL via silu pairs: {'PASS' if r2 else 'FAIL'}")
    print(f"  DIV basic: {'PASS' if r3 else 'FAIL'}")
    print(f"  DIV comprehensive: {'PASS' if r4 else 'FAIL'}")
    print(f"\n  ALL PASS: {r1 and r2 and r3 and r4}")
