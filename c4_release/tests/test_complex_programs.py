#!/usr/bin/env python3
"""
Complex Program Tests.

Tests for full, realistic C programs demonstrating:
- Recursive functions (fibonacci, factorial, ackermann)
- Nested function calls (3+ levels deep)
- Array operations (sorting, searching)
- Pointer arithmetic
- String manipulation
- Control flow combinations (nested loops, if/else chains)
- Data structures (stack, queue simulations)
- Mathematical algorithms (GCD, primality, power)

These tests verify the neural VM can handle real-world programming patterns.

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner


class TestRecursion:
    """Test recursive function execution."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_factorial_5(self, runner):
        """factorial(5) = 120"""
        source = '''
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        int main() { return factorial(5); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=500)
        assert result == 120, f"Expected 120, got {result}"

    def test_factorial_1(self, runner):
        """factorial(1) = 1 (base case)"""
        source = '''
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        int main() { return factorial(1); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 1

    def test_fibonacci_7(self, runner):
        """fibonacci(7) = 13"""
        source = '''
        int fib(int n) {
            if (n <= 1) return n;
            return fib(n-1) + fib(n-2);
        }
        int main() { return fib(7); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=2000)
        assert result == 13, f"Expected 13, got {result}"

    def test_ackermann_2_2(self, runner):
        """ackermann(2, 2) = 7 (tests deep recursion)"""
        source = '''
        int ack(int m, int n) {
            if (m == 0) return n + 1;
            if (n == 0) return ack(m - 1, 1);
            return ack(m - 1, ack(m, n - 1));
        }
        int main() { return ack(2, 2); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=1500)
        assert result == 7, f"Expected 7, got {result}"


class TestNestedFunctionCalls:
    """Test deeply nested function calls."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_three_level_nesting(self, runner):
        """main -> func1 -> func2 -> func3"""
        source = '''
        int func3(int x) { return x * 2; }
        int func2(int x) { return func3(x) + 1; }
        int func1(int x) { return func2(x) + 10; }
        int main() { return func1(5); }
        '''
        # func3(5) = 10, func2(5) = 11, func1(5) = 21
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        assert result == 21, f"Expected 21, got {result}"

    def test_four_level_nesting(self, runner):
        """main -> a -> b -> c -> d"""
        source = '''
        int d(int x) { return x + 1; }
        int c(int x) { return d(x) + 1; }
        int b(int x) { return c(x) + 1; }
        int a(int x) { return b(x) + 1; }
        int main() { return a(0); }
        '''
        # Each adds 1, so result = 4
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=400)
        assert result == 4, f"Expected 4, got {result}"

    def test_mutual_recursion(self, runner):
        """is_even and is_odd call each other."""
        source = '''
        int is_odd(int n);
        int is_even(int n) {
            if (n == 0) return 1;
            return is_odd(n - 1);
        }
        int is_odd(int n) {
            if (n == 0) return 0;
            return is_even(n - 1);
        }
        int main() { return is_even(10); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=500)
        assert result == 1, f"Expected 1 (10 is even), got {result}"


class TestArrayOperations:
    """Test array manipulation."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_array_sum(self, runner):
        """Sum array elements."""
        source = '''
        int main() {
            int arr[5];
            int sum;
            int i;
            sum = 0;
            arr[0] = 1; arr[1] = 2; arr[2] = 3; arr[3] = 4; arr[4] = 5;
            i = 0;
            while (i < 5) {
                sum = sum + arr[i];
                i = i + 1;
            }
            return sum;
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=800)
        assert result == 15, f"Expected 15, got {result}"

    def test_array_max(self, runner):
        """Find maximum element in array."""
        source = '''
        int main() {
            int arr[5];
            int max;
            int i;
            arr[0] = 3; arr[1] = 7; arr[2] = 2; arr[3] = 9; arr[4] = 4;
            max = arr[0];
            i = 1;
            while (i < 5) {
                if (arr[i] > max) max = arr[i];
                i = i + 1;
            }
            return max;
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=800)
        assert result == 9, f"Expected 9, got {result}"

    def test_array_reverse_check(self, runner):
        """Verify array elements can be read in reverse order."""
        source = '''
        int main() {
            int arr[3];
            arr[0] = 1; arr[1] = 2; arr[2] = 3;
            return arr[2] * 100 + arr[1] * 10 + arr[0];
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=400)
        assert result == 321, f"Expected 321, got {result}"


class TestPointerArithmetic:
    """Test pointer operations."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_pointer_increment(self, runner):
        """Traverse array via pointer increment."""
        source = '''
        int main() {
            int arr[3];
            int *p;
            int sum;
            arr[0] = 10; arr[1] = 20; arr[2] = 30;
            p = arr;
            sum = *p;
            p = p + 1;
            sum = sum + *p;
            p = p + 1;
            sum = sum + *p;
            return sum;
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=600)
        assert result == 60, f"Expected 60, got {result}"

    def test_pointer_dereference(self, runner):
        """Pointer dereference and assignment."""
        source = '''
        int main() {
            int x;
            int *p;
            x = 42;
            p = &x;
            return *p;
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        assert result == 42, f"Expected 42, got {result}"


class TestStringOperations:
    """Test string manipulation."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_strlen(self, runner):
        """Compute string length."""
        source = '''
        int strlen(char *s) {
            int len;
            len = 0;
            while (*s) {
                len = len + 1;
                s = s + 1;
            }
            return len;
        }
        int main() {
            return strlen("hello");
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=600)
        assert result == 5, f"Expected 5, got {result}"

    def test_string_first_char(self, runner):
        """Get first character of string."""
        source = '''
        int main() {
            char *s;
            s = "ABC";
            return *s;
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 65, f"Expected 65 ('A'), got {result}"


class TestControlFlowCombinations:
    """Test complex control flow patterns."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_nested_loops(self, runner):
        """Nested for loops."""
        source = '''
        int main() {
            int i;
            int j;
            int count;
            count = 0;
            i = 0;
            while (i < 3) {
                j = 0;
                while (j < 3) {
                    count = count + 1;
                    j = j + 1;
                }
                i = i + 1;
            }
            return count;
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=1000)
        assert result == 9, f"Expected 9 (3x3), got {result}"

    def test_nested_if_else(self, runner):
        """Nested if-else chain."""
        source = '''
        int classify(int n) {
            if (n < 0) return 0;
            if (n == 0) return 1;
            if (n < 10) return 2;
            return 3;
        }
        int main() {
            return classify(5) * 10 + classify(0);
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=400)
        # classify(5) = 2, classify(0) = 1, result = 21
        assert result == 21, f"Expected 21, got {result}"

    def test_early_return(self, runner):
        """Function with early return."""
        source = '''
        int find_first_positive(int a, int b, int c) {
            if (a > 0) return a;
            if (b > 0) return b;
            if (c > 0) return c;
            return 0;
        }
        int main() {
            return find_first_positive(-1, -2, 7);
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        assert result == 7, f"Expected 7, got {result}"


class TestDataStructures:
    """Test data structure patterns."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_stack_push_pop(self, runner):
        """Simulate stack with array."""
        source = '''
        int stack[10];
        int sp;

        void push(int x) { stack[sp] = x; sp = sp + 1; }
        int pop() { sp = sp - 1; return stack[sp]; }

        int main() {
            sp = 0;
            push(10);
            push(20);
            push(30);
            return pop() + pop();
        }
        '''
        # pop() returns 30, then 20, sum = 50
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=600)
        assert result == 50, f"Expected 50, got {result}"

    def test_accumulator_pattern(self, runner):
        """Accumulator pattern for sum."""
        source = '''
        int sum_to_n(int n) {
            int sum;
            int i;
            sum = 0;
            i = 1;
            while (i <= n) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
        int main() {
            return sum_to_n(10);
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=800)
        # sum(1..10) = 55
        assert result == 55, f"Expected 55, got {result}"


class TestMathematicalAlgorithms:
    """Test mathematical computations."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_gcd(self, runner):
        """Greatest common divisor (Euclidean algorithm)."""
        source = '''
        int gcd(int a, int b) {
            int temp;
            while (b != 0) {
                temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }
        int main() { return gcd(48, 18); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=500)
        assert result == 6, f"Expected 6, got {result}"

    def test_is_prime_17(self, runner):
        """Primality test for 17 (prime)."""
        source = '''
        int is_prime(int n) {
            int i;
            if (n < 2) return 0;
            i = 2;
            while (i * i <= n) {
                if (n % i == 0) return 0;
                i = i + 1;
            }
            return 1;
        }
        int main() { return is_prime(17); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=600)
        assert result == 1, f"Expected 1 (17 is prime), got {result}"

    def test_is_prime_18(self, runner):
        """Primality test for 18 (not prime)."""
        source = '''
        int is_prime(int n) {
            int i;
            if (n < 2) return 0;
            i = 2;
            while (i * i <= n) {
                if (n % i == 0) return 0;
                i = i + 1;
            }
            return 1;
        }
        int main() { return is_prime(18); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=600)
        assert result == 0, f"Expected 0 (18 is not prime), got {result}"

    def test_power_2_8(self, runner):
        """Integer exponentiation: 2^8 = 256."""
        source = '''
        int power(int base, int exp) {
            int result;
            result = 1;
            while (exp > 0) {
                result = result * base;
                exp = exp - 1;
            }
            return result;
        }
        int main() { return power(2, 8); }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=600)
        assert result == 256, f"Expected 256, got {result}"

    def test_abs(self, runner):
        """Absolute value."""
        source = '''
        int abs(int x) {
            if (x < 0) return 0 - x;
            return x;
        }
        int main() {
            return abs(5) + abs(0 - 7);
        }
        '''
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        # abs(5) + abs(-7) = 5 + 7 = 12
        assert result == 12, f"Expected 12, got {result}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
