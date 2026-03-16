"""
Complex Test Programs for C4 Transformer VM

Demonstrates the VM running real programs:
- Fibonacci sequence
- Factorial
- Prime checking
- Mandelbrot set rendering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List, Tuple

from src.transformer_vm import C4TransformerVM
from src.speculator import FastLogicalVM, SpeculativeVM
from src.compiler import compile_c


def run_with_timing(name: str, source: str, vm: C4TransformerVM) -> Tuple[int, float]:
    """Compile and run with timing."""
    bytecode, data = compile_c(source)
    vm.reset()
    vm.load_bytecode(bytecode, data)

    start = time.time()
    result = vm.run(max_steps=1000000)
    elapsed = time.time() - start

    return result, elapsed


def test_fibonacci():
    """Test Fibonacci sequence."""
    print("=" * 60)
    print("FIBONACCI TEST")
    print("=" * 60)

    source = """
    int fib(int n) {
        if (n < 2) return n;
        return fib(n-1) + fib(n-2);
    }
    int main() { return fib(10); }
    """

    vm = C4TransformerVM()

    # Test various values
    expected_fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    print("\nRecursive Fibonacci:")
    for n in [0, 1, 5, 10]:
        src = f"""
        int fib(int n) {{
            if (n < 2) return n;
            return fib(n-1) + fib(n-2);
        }}
        int main() {{ return fib({n}); }}
        """
        result, elapsed = run_with_timing(f"fib({n})", src, vm)
        expected = expected_fibs[n]
        status = "PASS" if result == expected else "FAIL"
        print(f"  fib({n:2d}) = {result:5d} (expected {expected:5d}) [{elapsed*1000:.1f}ms] {status}")

    # Iterative version (faster)
    print("\nIterative Fibonacci:")
    iter_source = """
    int fib(int n) {
        int a, b, i, tmp;
        if (n < 2) return n;
        a = 0; b = 1;
        i = 2;
        while (i <= n) {
            tmp = a + b;
            a = b;
            b = tmp;
            i = i + 1;
        }
        return b;
    }
    int main() { return fib(20); }
    """
    result, elapsed = run_with_timing("fib(20) iterative", iter_source, vm)
    expected = 6765
    print(f"  fib(20) = {result} (expected {expected}) [{elapsed*1000:.1f}ms]")


def test_factorial():
    """Test factorial calculation."""
    print("\n" + "=" * 60)
    print("FACTORIAL TEST")
    print("=" * 60)

    source_template = """
    int fact(int n) {{
        if (n <= 1) return 1;
        return n * fact(n - 1);
    }}
    int main() {{ return fact({n}); }}
    """

    vm = C4TransformerVM()
    expected_facts = {0: 1, 1: 1, 5: 120, 7: 5040, 10: 3628800}

    for n, expected in expected_facts.items():
        src = source_template.format(n=n)
        result, elapsed = run_with_timing(f"fact({n})", src, vm)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {n}! = {result:10d} (expected {expected:10d}) [{elapsed*1000:.1f}ms] {status}")


def test_primes():
    """Test prime number checking."""
    print("\n" + "=" * 60)
    print("PRIME TEST")
    print("=" * 60)

    source_template = """
    int is_prime(int n) {{
        int i;
        if (n < 2) return 0;
        if (n == 2) return 1;
        if (n % 2 == 0) return 0;
        i = 3;
        while (i * i <= n) {{
            if (n % i == 0) return 0;
            i = i + 2;
        }}
        return 1;
    }}
    int main() {{ return is_prime({n}); }}
    """

    vm = C4TransformerVM()
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    non_primes = [0, 1, 4, 6, 8, 9, 10, 12, 15, 20, 25, 100]

    print("\nPrime numbers:")
    for n in [2, 7, 17, 97]:
        src = source_template.format(n=n)
        result, elapsed = run_with_timing(f"is_prime({n})", src, vm)
        expected = 1 if n in primes else 0
        status = "PASS" if result == expected else "FAIL"
        print(f"  is_prime({n:3d}) = {result} (expected {expected}) [{elapsed*1000:.1f}ms] {status}")

    print("\nNon-prime numbers:")
    for n in [4, 15, 100]:
        src = source_template.format(n=n)
        result, elapsed = run_with_timing(f"is_prime({n})", src, vm)
        expected = 0
        status = "PASS" if result == expected else "FAIL"
        print(f"  is_prime({n:3d}) = {result} (expected {expected}) [{elapsed*1000:.1f}ms] {status}")

    # Count primes up to N
    print("\nCounting primes:")
    count_source = """
    int is_prime(int n) {
        int i;
        if (n < 2) return 0;
        if (n == 2) return 1;
        if (n % 2 == 0) return 0;
        i = 3;
        while (i * i <= n) {
            if (n % i == 0) return 0;
            i = i + 2;
        }
        return 1;
    }
    int main() {
        int n, count;
        n = 2; count = 0;
        while (n <= 50) {
            if (is_prime(n)) count = count + 1;
            n = n + 1;
        }
        return count;
    }
    """
    result, elapsed = run_with_timing("count primes <= 50", count_source, vm)
    expected = 15  # primes: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47
    status = "PASS" if result == expected else "FAIL"
    print(f"  Primes <= 50: {result} (expected {expected}) [{elapsed*1000:.1f}ms] {status}")


def test_gcd():
    """Test GCD calculation."""
    print("\n" + "=" * 60)
    print("GCD TEST")
    print("=" * 60)

    source_template = """
    int gcd(int a, int b) {{
        int tmp;
        while (b != 0) {{
            tmp = b;
            b = a % b;
            a = tmp;
        }}
        return a;
    }}
    int main() {{ return gcd({a}, {b}); }}
    """

    vm = C4TransformerVM()
    test_cases = [
        (48, 18, 6),
        (100, 75, 25),
        (17, 13, 1),
        (1071, 462, 21),
    ]

    for a, b, expected in test_cases:
        src = source_template.format(a=a, b=b)
        result, elapsed = run_with_timing(f"gcd({a}, {b})", src, vm)
        status = "PASS" if result == expected else "FAIL"
        print(f"  gcd({a:4d}, {b:4d}) = {result:4d} (expected {expected:4d}) [{elapsed*1000:.1f}ms] {status}")


def test_mandelbrot():
    """Test Mandelbrot set rendering."""
    print("\n" + "=" * 60)
    print("MANDELBROT TEST")
    print("=" * 60)

    vm = C4TransformerVM()
    spec_vm = SpeculativeVM(transformer_vm=None, validate_ratio=0.0)

    width = 40
    height = 20
    scale = 1024

    print(f"\nRendering {width}x{height} Mandelbrot set...")
    print("(Using speculative execution for speed)")

    output = []
    start_time = time.time()

    for row in range(height):
        cy = -1024 + (row * 2048) // height

        row_chars = []
        for chunk in range(0, width, 20):
            chunk_width = min(20, width - chunk)
            cx_offset = chunk * (3072 // width)

            source = f'''
int main() {{
    int scale, cx_base, dx, cy;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = 1024;
    maxiter = 30;
    cx_base = -2048 + {cx_offset};
    dx = {3072 // width};
    cy = {cy};

    bitmap = 0;
    px = 0;
    while (px < {chunk_width}) {{
        cx = cx_base + px * dx;
        zx = 0;
        zy = 0;
        iter = 0;

        while (iter < maxiter) {{
            zx2 = (zx * zx) / scale;
            zy2 = (zy * zy) / scale;

            if (zx2 + zy2 > 4 * scale) {{
                iter = maxiter + 10;
            }}

            if (iter < maxiter) {{
                tmp = zx2 - zy2 + cx;
                zy = 2 * zx * zy / scale + cy;
                zx = tmp;
                iter = iter + 1;
            }}
        }}

        bitmap = bitmap * 2;
        if (iter == maxiter) {{
            bitmap = bitmap + 1;
        }}

        px = px + 1;
    }}

    return bitmap;
}}
'''
            bytecode, data = compile_c(source)
            bitmap = spec_vm.run(bytecode, data)

            for i in range(chunk_width):
                bit = (bitmap >> (chunk_width - 1 - i)) & 1
                row_chars.append("*" if bit else " ")

        output.append("".join(row_chars))

        # Progress
        elapsed = time.time() - start_time
        print(f"\r  Row {row+1}/{height} ({elapsed:.1f}s)", end="", flush=True)

    print()

    # Display
    print()
    print("+" + "-" * width + "+")
    for line in output:
        print("|" + line + "|")
    print("+" + "-" * width + "+")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.2f}s")
    print(f"  Pixels: {width * height}")


def test_speculator_validation():
    """Validate speculator against transformer VM."""
    print("\n" + "=" * 60)
    print("SPECULATOR VALIDATION")
    print("=" * 60)

    trans_vm = C4TransformerVM()
    spec_vm = SpeculativeVM(transformer_vm=trans_vm, validate_ratio=1.0)

    test_cases = [
        ("6 * 7", "int main() { return 6 * 7; }", 42),
        ("100 / 7", "int main() { return 100 / 7; }", 14),
        ("fib(10)", """
            int fib(int n) {
                if (n < 2) return n;
                return fib(n-1) + fib(n-2);
            }
            int main() { return fib(10); }
        """, 55),
        ("5!", """
            int fact(int n) {
                if (n <= 1) return 1;
                return n * fact(n - 1);
            }
            int main() { return fact(5); }
        """, 120),
    ]

    for name, source, expected in test_cases:
        bytecode, data = compile_c(source)
        result = spec_vm.run(bytecode, data, validate=True)
        status = "PASS" if result == expected and spec_vm.mismatches == 0 else "FAIL"
        print(f"  {name:12} = {result:6d} (expected {expected:6d}) {status}")

    stats = spec_vm.get_stats()
    print(f"\n  Validations: {stats['validations']}")
    print(f"  Mismatches: {stats['mismatches']}")
    print(f"  Match rate: {stats['match_rate']*100:.1f}%")


def run_all_tests():
    """Run all test programs."""
    print()
    print("*" * 60)
    print("* C4 TRANSFORMER VM - COMPLEX PROGRAM TESTS")
    print("*" * 60)
    print()

    test_fibonacci()
    test_factorial()
    test_primes()
    test_gcd()
    test_speculator_validation()
    test_mandelbrot()

    print()
    print("*" * 60)
    print("* ALL TESTS COMPLETE")
    print("*" * 60)


if __name__ == "__main__":
    run_all_tests()
