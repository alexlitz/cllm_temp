"""
1000+ Test Programs for C4 Transformer VM

This module generates a comprehensive test suite covering:
- Basic arithmetic (add, sub, mul, div, mod)
- Variables and assignments
- Comparisons and conditionals
- Loops (while)
- Functions and recursion
- Complex expressions
- Edge cases
"""

import random
import math
from typing import List, Tuple

# Set seed for reproducibility
random.seed(42)


def generate_test_programs() -> List[Tuple[str, int, str]]:
    """
    Generate 1000+ test programs with expected results.

    Returns:
        List of (source_code, expected_result, description)
    """
    tests = []

    # ==========================================================================
    # Category 1: Basic Arithmetic (200 tests)
    # ==========================================================================

    # Addition (50 tests)
    for i in range(50):
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        tests.append((
            f'int main() {{ return {a} + {b}; }}',
            a + b,
            f'add_{i}: {a} + {b}'
        ))

    # Subtraction (50 tests)
    for i in range(50):
        a = random.randint(100, 2000)
        b = random.randint(0, 100)
        tests.append((
            f'int main() {{ return {a} - {b}; }}',
            a - b,
            f'sub_{i}: {a} - {b}'
        ))

    # Multiplication (50 tests)
    for i in range(50):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        tests.append((
            f'int main() {{ return {a} * {b}; }}',
            a * b,
            f'mul_{i}: {a} * {b}'
        ))

    # Division (50 tests)
    for i in range(50):
        b = random.randint(1, 50)
        a = b * random.randint(1, 50) + random.randint(0, b-1)
        tests.append((
            f'int main() {{ return {a} / {b}; }}',
            a // b,
            f'div_{i}: {a} / {b}'
        ))

    # ==========================================================================
    # Category 2: Modulo (50 tests)
    # ==========================================================================

    for i in range(50):
        a = random.randint(10, 500)
        b = random.randint(2, 20)
        tests.append((
            f'int main() {{ return {a} % {b}; }}',
            a % b,
            f'mod_{i}: {a} % {b}'
        ))

    # ==========================================================================
    # Category 3: Variables (100 tests)
    # ==========================================================================

    # Simple variable assignment
    for i in range(25):
        val = random.randint(0, 1000)
        tests.append((
            f'int main() {{ int x; x = {val}; return x; }}',
            val,
            f'var_simple_{i}: x = {val}'
        ))

    # Variable arithmetic
    for i in range(25):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        tests.append((
            f'int main() {{ int a; int b; a = {a}; b = {b}; return a * b; }}',
            a * b,
            f'var_mul_{i}: a={a}, b={b}, a*b'
        ))

    # Multiple variables
    for i in range(25):
        a = random.randint(1, 30)
        b = random.randint(1, 30)
        c = random.randint(1, 30)
        tests.append((
            f'int main() {{ int a; int b; int c; a = {a}; b = {b}; c = {c}; return a + b + c; }}',
            a + b + c,
            f'var_three_{i}: {a}+{b}+{c}'
        ))

    # Variable update
    for i in range(25):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        tests.append((
            f'int main() {{ int x; x = {a}; x = x + {b}; return x; }}',
            a + b,
            f'var_update_{i}: x={a}, x=x+{b}'
        ))

    # ==========================================================================
    # Category 4: Conditionals (100 tests)
    # ==========================================================================

    # If-else with greater than
    for i in range(25):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        result = 1 if a > b else 0
        tests.append((
            f'int main() {{ if ({a} > {b}) return 1; return 0; }}',
            result,
            f'if_gt_{i}: {a} > {b}'
        ))

    # If-else with less than
    for i in range(25):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        result = 1 if a < b else 0
        tests.append((
            f'int main() {{ if ({a} < {b}) return 1; return 0; }}',
            result,
            f'if_lt_{i}: {a} < {b}'
        ))

    # If-else with equality
    for i in range(25):
        a = random.randint(1, 50)
        b = random.choice([a, random.randint(1, 50)])
        result = 1 if a == b else 0
        tests.append((
            f'int main() {{ if ({a} == {b}) return 1; return 0; }}',
            result,
            f'if_eq_{i}: {a} == {b}'
        ))

    # If with variable
    for i in range(25):
        val = random.randint(0, 100)
        threshold = random.randint(0, 100)
        result = 1 if val > threshold else 0
        tests.append((
            f'int main() {{ int x; x = {val}; if (x > {threshold}) return 1; return 0; }}',
            result,
            f'if_var_{i}: x={val}, x>{threshold}'
        ))

    # ==========================================================================
    # Category 5: While Loops (100 tests)
    # ==========================================================================

    # Sum 1 to N
    for i in range(25):
        n = random.randint(1, 20)
        expected = n * (n + 1) // 2
        tests.append((
            f'''int main() {{
                int i; int sum;
                i = 1; sum = 0;
                while (i <= {n}) {{
                    sum = sum + i;
                    i = i + 1;
                }}
                return sum;
            }}''',
            expected,
            f'loop_sum_{i}: sum(1..{n})'
        ))

    # Countdown
    for i in range(25):
        n = random.randint(1, 30)
        tests.append((
            f'''int main() {{
                int x;
                x = {n};
                while (x > 0) {{
                    x = x - 1;
                }}
                return x;
            }}''',
            0,
            f'loop_countdown_{i}: countdown from {n}'
        ))

    # Multiply by addition
    for i in range(25):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        tests.append((
            f'''int main() {{
                int result; int i;
                result = 0; i = 0;
                while (i < {b}) {{
                    result = result + {a};
                    i = i + 1;
                }}
                return result;
            }}''',
            a * b,
            f'loop_mul_{i}: {a}*{b} by addition'
        ))

    # Power of 2
    for i in range(25):
        exp = random.randint(0, 10)
        tests.append((
            f'''int main() {{
                int result; int i;
                result = 1; i = 0;
                while (i < {exp}) {{
                    result = result * 2;
                    i = i + 1;
                }}
                return result;
            }}''',
            2 ** exp,
            f'loop_pow2_{i}: 2^{exp}'
        ))

    # ==========================================================================
    # Category 6: Functions (150 tests)
    # ==========================================================================

    # Simple function call
    for i in range(25):
        val = random.randint(1, 100)
        tests.append((
            f'''int identity(int x) {{ return x; }}
            int main() {{ return identity({val}); }}''',
            val,
            f'func_identity_{i}: identity({val})'
        ))

    # Add function
    for i in range(25):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        tests.append((
            f'''int add(int a, int b) {{ return a + b; }}
            int main() {{ return add({a}, {b}); }}''',
            a + b,
            f'func_add_{i}: add({a}, {b})'
        ))

    # Multiply function
    for i in range(25):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        tests.append((
            f'''int mul(int a, int b) {{ return a * b; }}
            int main() {{ return mul({a}, {b}); }}''',
            a * b,
            f'func_mul_{i}: mul({a}, {b})'
        ))

    # Square function
    for i in range(25):
        x = random.randint(1, 50)
        tests.append((
            f'''int square(int x) {{ return x * x; }}
            int main() {{ return square({x}); }}''',
            x * x,
            f'func_square_{i}: square({x})'
        ))

    # Max function
    for i in range(25):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        tests.append((
            f'''int max(int a, int b) {{ if (a > b) return a; return b; }}
            int main() {{ return max({a}, {b}); }}''',
            max(a, b),
            f'func_max_{i}: max({a}, {b})'
        ))

    # Min function
    for i in range(25):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        tests.append((
            f'''int min(int a, int b) {{ if (a < b) return a; return b; }}
            int main() {{ return min({a}, {b}); }}''',
            min(a, b),
            f'func_min_{i}: min({a}, {b})'
        ))

    # ==========================================================================
    # Category 7: Recursion (100 tests)
    # ==========================================================================

    # Factorial
    for i in range(25):
        n = random.randint(0, 10)
        fact = 1
        for j in range(1, n + 1):
            fact *= j
        tests.append((
            f'''int factorial(int n) {{
                if (n < 2) return 1;
                return n * factorial(n - 1);
            }}
            int main() {{ return factorial({n}); }}''',
            fact,
            f'rec_factorial_{i}: {n}!'
        ))

    # Fibonacci
    for i in range(25):
        n = random.randint(0, 12)
        # Calculate fibonacci
        if n <= 1:
            fib = n
        else:
            a, b = 0, 1
            for _ in range(n - 1):
                a, b = b, a + b
            fib = b
        tests.append((
            f'''int fib(int n) {{
                if (n < 2) return n;
                return fib(n-1) + fib(n-2);
            }}
            int main() {{ return fib({n}); }}''',
            fib,
            f'rec_fib_{i}: fib({n})'
        ))

    # Sum recursive
    for i in range(25):
        n = random.randint(1, 15)
        expected = n * (n + 1) // 2
        tests.append((
            f'''int sum(int n) {{
                if (n <= 0) return 0;
                return n + sum(n - 1);
            }}
            int main() {{ return sum({n}); }}''',
            expected,
            f'rec_sum_{i}: sum(1..{n})'
        ))

    # Power recursive
    for i in range(25):
        base = random.randint(2, 5)
        exp = random.randint(0, 6)
        tests.append((
            f'''int power(int b, int e) {{
                if (e == 0) return 1;
                return b * power(b, e - 1);
            }}
            int main() {{ return power({base}, {exp}); }}''',
            base ** exp,
            f'rec_power_{i}: {base}^{exp}'
        ))

    # ==========================================================================
    # Category 8: Complex Expressions (100 tests)
    # ==========================================================================

    # Three operand expressions
    for i in range(25):
        a = random.randint(1, 30)
        b = random.randint(1, 30)
        c = random.randint(1, 30)
        tests.append((
            f'int main() {{ return {a} + {b} * {c}; }}',
            a + b * c,
            f'expr_add_mul_{i}: {a}+{b}*{c}'
        ))

    # Parenthesized expressions
    for i in range(25):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        c = random.randint(1, 20)
        tests.append((
            f'int main() {{ return ({a} + {b}) * {c}; }}',
            (a + b) * c,
            f'expr_paren_{i}: ({a}+{b})*{c}'
        ))

    # Division and multiplication
    for i in range(25):
        c = random.randint(1, 10)
        b = c * random.randint(1, 10)
        a = random.randint(1, 30)
        tests.append((
            f'int main() {{ return {a} * {b} / {c}; }}',
            a * b // c,
            f'expr_mul_div_{i}: {a}*{b}/{c}'
        ))

    # Modulo in expression
    for i in range(25):
        a = random.randint(10, 100)
        b = random.randint(2, 10)
        c = random.randint(1, 10)
        tests.append((
            f'int main() {{ return {a} % {b} + {c}; }}',
            a % b + c,
            f'expr_mod_{i}: {a}%{b}+{c}'
        ))

    # ==========================================================================
    # Category 9: GCD Algorithm (50 tests)
    # ==========================================================================

    for i in range(50):
        a = random.randint(10, 200)
        b = random.randint(10, 200)
        tests.append((
            f'''int gcd(int a, int b) {{
                int temp;
                while (b != 0) {{
                    temp = b;
                    b = a % b;
                    a = temp;
                }}
                return a;
            }}
            int main() {{ return gcd({a}, {b}); }}''',
            math.gcd(a, b),
            f'gcd_{i}: gcd({a}, {b})'
        ))

    # ==========================================================================
    # Category 10: Nested Functions (50 tests)
    # ==========================================================================

    for i in range(25):
        x = random.randint(1, 10)
        tests.append((
            f'''int double_it(int x) {{ return x * 2; }}
            int quad(int x) {{ return double_it(double_it(x)); }}
            int main() {{ return quad({x}); }}''',
            x * 4,
            f'nested_quad_{i}: quad({x})'
        ))

    for i in range(25):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        tests.append((
            f'''int square(int x) {{ return x * x; }}
            int sum_squares(int a, int b) {{ return square(a) + square(b); }}
            int main() {{ return sum_squares({a}, {b}); }}''',
            a*a + b*b,
            f'nested_sumsq_{i}: {a}^2+{b}^2'
        ))

    # ==========================================================================
    # Category 11: Edge Cases (50 tests)
    # ==========================================================================

    # Zero operations
    tests.extend([
        ('int main() { return 0 + 0; }', 0, 'edge_zero_add'),
        ('int main() { return 0 * 100; }', 0, 'edge_zero_mul'),
        ('int main() { return 100 * 0; }', 0, 'edge_mul_zero'),
        ('int main() { return 0 / 5; }', 0, 'edge_zero_div'),
        ('int main() { return 0 % 5; }', 0, 'edge_zero_mod'),
    ])

    # Identity operations
    tests.extend([
        ('int main() { return 42 + 0; }', 42, 'edge_add_zero'),
        ('int main() { return 42 * 1; }', 42, 'edge_mul_one'),
        ('int main() { return 42 / 1; }', 42, 'edge_div_one'),
    ])

    # Small values
    tests.extend([
        ('int main() { return 1 + 1; }', 2, 'edge_1plus1'),
        ('int main() { return 2 - 1; }', 1, 'edge_2minus1'),
        ('int main() { return 1 * 1; }', 1, 'edge_1times1'),
        ('int main() { return 1 / 1; }', 1, 'edge_1div1'),
    ])

    # Powers of 2
    for i in range(10):
        val = 2 ** i
        tests.append((
            f'int main() {{ return {val}; }}',
            val,
            f'edge_pow2_{i}: 2^{i}'
        ))

    # Specific values
    tests.extend([
        ('int main() { return 255; }', 255, 'edge_255'),
        ('int main() { return 256; }', 256, 'edge_256'),
        ('int main() { return 1000; }', 1000, 'edge_1000'),
        ('int main() { return 9999; }', 9999, 'edge_9999'),
    ])

    # Conditional edge cases
    tests.extend([
        ('int main() { if (0) return 1; return 0; }', 0, 'edge_if_zero'),
        ('int main() { if (1) return 1; return 0; }', 1, 'edge_if_one'),
        ('int main() { if (100) return 1; return 0; }', 1, 'edge_if_hundred'),
    ])

    # Loop edge cases
    tests.extend([
        ('int main() { int i; i = 0; while (i < 0) { i = i + 1; } return i; }', 0, 'edge_loop_never'),
        ('int main() { int i; i = 1; while (i > 1) { i = i - 1; } return i; }', 1, 'edge_loop_once_skip'),
    ])

    # Additional edge cases to reach 50
    for i in range(15):
        val = random.randint(100, 10000)
        tests.append((
            f'int main() {{ return {val}; }}',
            val,
            f'edge_literal_{i}: {val}'
        ))

    # ==========================================================================
    # Category 12: Absolute Value (25 tests)
    # ==========================================================================

    for i in range(25):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        result = x - y if x > y else y - x
        tests.append((
            f'''int abs_diff(int a, int b) {{
                if (a > b) return a - b;
                return b - a;
            }}
            int main() {{ return abs_diff({x}, {y}); }}''',
            result,
            f'absdiff_{i}: |{x}-{y}|'
        ))

    # ==========================================================================
    # Category 13: Boolean Logic (25 tests)
    # ==========================================================================

    for i in range(25):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        c = random.randint(0, 100)
        # (a > b) && (b > c) implemented as nested if
        result = 1 if (a > b and b > c) else 0
        tests.append((
            f'''int main() {{
                if ({a} > {b}) {{
                    if ({b} > {c}) return 1;
                }}
                return 0;
            }}''',
            result,
            f'bool_and_{i}: {a}>{b} && {b}>{c}'
        ))

    return tests


def get_quick_tests() -> List[Tuple[str, int, str]]:
    """Get a smaller subset of tests for quick validation."""
    all_tests = generate_test_programs()
    # Return first 100 tests
    return all_tests[:100]


def get_all_tests() -> List[Tuple[str, int, str]]:
    """Get all 1000+ tests."""
    return generate_test_programs()


# Test counts by category
CATEGORY_COUNTS = {
    'arithmetic': 200,
    'modulo': 50,
    'variables': 100,
    'conditionals': 100,
    'loops': 100,
    'functions': 150,
    'recursion': 100,
    'expressions': 100,
    'gcd': 50,
    'nested_functions': 50,
    'edge_cases': 50,
    'abs_diff': 25,
    'boolean_logic': 25,
}


if __name__ == '__main__':
    tests = generate_test_programs()
    print(f'Generated {len(tests)} test programs')
    print()
    print('Category breakdown:')
    for cat, count in CATEGORY_COUNTS.items():
        print(f'  {cat}: {count}')
    print(f'  TOTAL: {sum(CATEGORY_COUNTS.values())}')
    print()

    # Show a few examples
    print('Example tests:')
    for src, expected, desc in tests[:5]:
        print(f'  {desc}: expected={expected}')
        print(f'    {src[:60]}...')
