#!/usr/bin/env python3
"""
Test standard ONNX models exported by export_vm_onnx.py.

Loads each .onnx model with onnxruntime, runs test vectors, and
verifies against known correct results.

Usage:
    python test_onnx_sparse.py models/sparse_onnx/
"""

import os
import sys
import argparse
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime required. Install with: pip install onnxruntime")
    sys.exit(1)


def onehot(val, size):
    """Create a one-hot float32 vector."""
    v = np.zeros(size, dtype=np.float32)
    v[val] = 1.0
    return v


def argmax_check(result, expected_idx, label=""):
    """Check that argmax of result matches expected index."""
    got = int(np.argmax(result))
    ok = got == expected_idx
    conf = float(result[got])
    if not ok:
        print(f"  FAIL {label}: expected {expected_idx}, got {got} (conf={conf:.4f})")
    return ok


def test_b2n(model_dir):
    """Test byte-to-nibble: b2n(8) -> hi=0, lo=8"""
    path = os.path.join(model_dir, "b2n.onnx")
    if not os.path.exists(path):
        print(f"  SKIP b2n: {path} not found")
        return 0, 0

    sess = ort.InferenceSession(path)
    passed = 0
    total = 0

    test_cases = [
        (8, 0, 8),       # 0x08 -> hi=0, lo=8
        (0xAB, 0xA, 0xB),  # 0xAB -> hi=A, lo=B
        (0xFF, 0xF, 0xF),  # 0xFF -> hi=F, lo=F
        (0, 0, 0),         # 0x00 -> hi=0, lo=0
        (0x42, 4, 2),      # 0x42 -> hi=4, lo=2
    ]

    for byte_val, exp_hi, exp_lo in test_cases:
        result = sess.run(None, {"byte": onehot(byte_val, 256)})
        hi, lo = result
        ok_hi = argmax_check(hi, exp_hi, f"b2n({byte_val:#x}).hi")
        ok_lo = argmax_check(lo, exp_lo, f"b2n({byte_val:#x}).lo")
        total += 2
        passed += ok_hi + ok_lo

    return passed, total


def test_n2b(model_dir):
    """Test nibble-to-byte: n2b(0, 8) -> 0x08"""
    path = os.path.join(model_dir, "n2b.onnx")
    if not os.path.exists(path):
        print(f"  SKIP n2b: {path} not found")
        return 0, 0

    sess = ort.InferenceSession(path)
    passed = 0
    total = 0

    test_cases = [
        (0, 8, 0x08),
        (0xA, 0xB, 0xAB),
        (0xF, 0xF, 0xFF),
        (0, 0, 0x00),
        (4, 2, 0x42),
    ]

    for hi_val, lo_val, exp_byte in test_cases:
        result = sess.run(None, {"hi": onehot(hi_val, 16), "lo": onehot(lo_val, 16)})
        byte_out = result[0]
        ok = argmax_check(byte_out, exp_byte, f"n2b({hi_val:#x},{lo_val:#x})")
        total += 1
        passed += ok

    return passed, total


def test_nib_add(model_dir):
    """Test nibble add with carry."""
    path = os.path.join(model_dir, "nib_add.onnx")
    if not os.path.exists(path):
        print(f"  SKIP nib_add: {path} not found")
        return 0, 0

    sess = ort.InferenceSession(path)
    passed = 0
    total = 0

    # (a, b, cin) -> (sum, cout)
    test_cases = [
        (8, 8, 0, 0, 1),    # 8+8+0=16 -> sum=0, cout=1
        (3, 4, 0, 7, 0),    # 3+4+0=7  -> sum=7, cout=0
        (0xF, 1, 0, 0, 1),  # 15+1+0=16 -> sum=0, cout=1
        (0xF, 0xF, 0, 0xE, 1),  # 15+15=30 -> sum=14, cout=1
        (0xF, 0xF, 1, 0xF, 1),  # 15+15+1=31 -> sum=15, cout=1
        (0, 0, 0, 0, 0),    # 0+0+0=0 -> sum=0, cout=0
        (0, 0, 1, 1, 0),    # 0+0+1=1 -> sum=1, cout=0
    ]

    for a, b, cin, exp_sum, exp_cout in test_cases:
        result = sess.run(None, {
            "a": onehot(a, 16),
            "b": onehot(b, 16),
            "cin": onehot(cin, 2),
        })
        sum_out, cout_out = result
        ok_s = argmax_check(sum_out, exp_sum, f"nib_add({a},{b},{cin}).sum")
        ok_c = argmax_check(cout_out, exp_cout, f"nib_add({a},{b},{cin}).cout")
        total += 2
        passed += ok_s + ok_c

    return passed, total


def test_nib_op(model_dir, name, op_fn, in1="a", in2="b", out="r"):
    """Test a 2-input nibble op (and/or/xor)."""
    path = os.path.join(model_dir, f"{name}.onnx")
    if not os.path.exists(path):
        print(f"  SKIP {name}: {path} not found")
        return 0, 0

    sess = ort.InferenceSession(path)
    passed = 0
    total = 0

    # Test all 16x16 combinations
    for a in range(16):
        for b in range(16):
            expected = op_fn(a, b) & 0xF
            result = sess.run(None, {
                in1: onehot(a, 16),
                in2: onehot(b, 16),
            })
            got = int(np.argmax(result[0]))
            total += 1
            if got == expected:
                passed += 1
            else:
                print(f"  FAIL {name}({a},{b}): expected {expected}, got {got}")

    return passed, total


def test_nib_mul(model_dir):
    """Test nibble multiply."""
    path = os.path.join(model_dir, "nib_mul.onnx")
    if not os.path.exists(path):
        print(f"  SKIP nib_mul: {path} not found")
        return 0, 0

    sess = ort.InferenceSession(path)
    passed = 0
    total = 0

    for a in range(16):
        for b in range(16):
            product = a * b
            exp_lo = product & 0xF
            exp_hi = (product >> 4) & 0xF
            result = sess.run(None, {
                "a": onehot(a, 16),
                "b": onehot(b, 16),
            })
            lo_out, hi_out = result
            got_lo = int(np.argmax(lo_out))
            got_hi = int(np.argmax(hi_out))
            total += 2
            if got_lo == exp_lo:
                passed += 1
            else:
                print(f"  FAIL nib_mul({a},{b}).lo: expected {exp_lo}, got {got_lo}")
            if got_hi == exp_hi:
                passed += 1
            else:
                print(f"  FAIL nib_mul({a},{b}).hi: expected {exp_hi}, got {got_hi}")

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Test ONNX sparse models")
    parser.add_argument("model_dir", help="Directory containing .onnx files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    total_passed = 0
    total_tests = 0

    tests = [
        ("b2n", lambda: test_b2n(args.model_dir)),
        ("n2b", lambda: test_n2b(args.model_dir)),
        ("nib_add", lambda: test_nib_add(args.model_dir)),
        ("nib_and", lambda: test_nib_op(args.model_dir, "nib_and", lambda a, b: a & b)),
        ("nib_or", lambda: test_nib_op(args.model_dir, "nib_or", lambda a, b: a | b)),
        ("nib_xor", lambda: test_nib_op(args.model_dir, "nib_xor", lambda a, b: a ^ b)),
        ("nib_mul", lambda: test_nib_mul(args.model_dir)),
    ]

    for name, test_fn in tests:
        p, t = test_fn()
        status = "PASS" if p == t else "FAIL"
        print(f"  {name}: {p}/{t} {status}")
        total_passed += p
        total_tests += t

    print(f"\nTotal: {total_passed}/{total_tests}")
    if total_passed == total_tests:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
