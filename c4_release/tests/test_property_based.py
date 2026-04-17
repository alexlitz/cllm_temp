"""
Property-Based Tests for Neural VM.

Uses Hypothesis to generate test cases automatically.

Run with:
    pytest tests/test_property_based.py -v
"""

import pytest

# Try to import hypothesis
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.run_vm import AutoregressiveVMRunner
from src.c4 import compile_c


@pytest.fixture
def runner():
    """Create a runner for property tests."""
    return AutoregressiveVMRunner()


@pytest.mark.property
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestArithmeticProperties:
    """Property-based tests for arithmetic operations."""

    @given(st.integers(0, 127))
    @settings(max_examples=20)  # Limit to keep tests fast
    def test_return_constant(self, runner, n):
        """Any small constant can be returned."""
        source = f"int main() {{ return {n}; }}"
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == n

    @given(st.integers(0, 50), st.integers(0, 50))
    @settings(max_examples=20)
    def test_add_commutative(self, runner, a, b):
        """Addition is commutative: a + b == b + a."""
        source1 = f"int main() {{ return {a} + {b}; }}"
        source2 = f"int main() {{ return {b} + {a}; }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)

        assert result1 == result2

    @given(st.integers(0, 30), st.integers(0, 30), st.integers(0, 30))
    @settings(max_examples=15)
    def test_add_associative(self, runner, a, b, c):
        """Addition is associative: (a + b) + c == a + (b + c)."""
        source1 = f"int main() {{ return ({a} + {b}) + {c}; }}"
        source2 = f"int main() {{ return {a} + ({b} + {c}); }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=300)
        _, result2 = runner.run(bytecode2, data2, max_steps=300)

        assert result1 == result2

    @given(st.integers(0, 50), st.integers(0, 50))
    @settings(max_examples=20)
    def test_mul_commutative(self, runner, a, b):
        """Multiplication is commutative: a * b == b * a."""
        source1 = f"int main() {{ return {a} * {b}; }}"
        source2 = f"int main() {{ return {b} * {a}; }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)

        assert result1 == result2

    @given(st.integers(50, 200), st.integers(1, 50))
    @settings(max_examples=20)
    def test_sub_positive_result(self, runner, a, b):
        """Subtraction: a - b where a > b gives positive result."""
        assume(a > b)
        source = f"int main() {{ return {a} - {b}; }}"

        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)

        assert result == a - b


@pytest.mark.property
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestComparisonProperties:
    """Property-based tests for comparison operations."""

    @given(st.integers(0, 100), st.integers(0, 100))
    @settings(max_examples=20)
    def test_eq_reflexive(self, runner, a, b):
        """EQ: (a == a) is always 1, (a == b) matches Python."""
        # Test reflexive
        source1 = f"int main() {{ return {a} == {a}; }}"
        bytecode1, data1 = compile_c(source1)
        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        assert result1 == 1

        # Test general
        source2 = f"int main() {{ return {a} == {b}; }}"
        bytecode2, data2 = compile_c(source2)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)
        assert result2 == (1 if a == b else 0)

    @given(st.integers(0, 100), st.integers(0, 100))
    @settings(max_examples=20)
    def test_lt_consistent(self, runner, a, b):
        """LT: (a < b) matches Python behavior."""
        source = f"int main() {{ return {a} < {b}; }}"
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == (1 if a < b else 0)

    @given(st.integers(0, 100), st.integers(0, 100))
    @settings(max_examples=20)
    def test_lt_gt_inverse(self, runner, a, b):
        """LT and GT are inverses: (a < b) == (b > a)."""
        assume(a != b)

        source1 = f"int main() {{ return {a} < {b}; }}"
        source2 = f"int main() {{ return {b} > {a}; }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)

        assert result1 == result2


@pytest.mark.property
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestBitwiseProperties:
    """Property-based tests for bitwise operations."""

    @given(st.integers(0, 255), st.integers(0, 255))
    @settings(max_examples=20)
    def test_and_commutative(self, runner, a, b):
        """AND is commutative: a & b == b & a."""
        source1 = f"int main() {{ return {a} & {b}; }}"
        source2 = f"int main() {{ return {b} & {a}; }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)

        assert result1 == result2

    @given(st.integers(0, 255), st.integers(0, 255))
    @settings(max_examples=20)
    def test_or_commutative(self, runner, a, b):
        """OR is commutative: a | b == b | a."""
        source1 = f"int main() {{ return {a} | {b}; }}"
        source2 = f"int main() {{ return {b} | {a}; }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)

        assert result1 == result2

    @given(st.integers(0, 255), st.integers(0, 255))
    @settings(max_examples=20)
    def test_xor_commutative(self, runner, a, b):
        """XOR is commutative: a ^ b == b ^ a."""
        source1 = f"int main() {{ return {a} ^ {b}; }}"
        source2 = f"int main() {{ return {b} ^ {a}; }}"

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=200)
        _, result2 = runner.run(bytecode2, data2, max_steps=200)

        assert result1 == result2

    @given(st.integers(0, 255))
    @settings(max_examples=20)
    def test_xor_self_is_zero(self, runner, a):
        """XOR with self: a ^ a == 0."""
        source = f"int main() {{ return {a} ^ {a}; }}"
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 0


@pytest.mark.property
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestDivisionProperties:
    """Property-based tests for division and modulo."""

    @given(st.integers(1, 100), st.integers(1, 50))
    @settings(max_examples=20)
    def test_div_mod_identity(self, runner, a, b):
        """Division identity: (a / b) * b + (a % b) == a."""
        div_source = f"int main() {{ return {a} / {b}; }}"
        mod_source = f"int main() {{ return {a} % {b}; }}"

        bytecode_div, data_div = compile_c(div_source)
        bytecode_mod, data_mod = compile_c(mod_source)

        _, div_result = runner.run(bytecode_div, data_div, max_steps=200)
        _, mod_result = runner.run(bytecode_mod, data_mod, max_steps=200)

        # Verify identity
        assert div_result * b + mod_result == a

    @given(st.integers(1, 100))
    @settings(max_examples=20)
    def test_div_by_one(self, runner, a):
        """Division by 1: a / 1 == a."""
        source = f"int main() {{ return {a} / 1; }}"
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == a

    @given(st.integers(1, 100))
    @settings(max_examples=20)
    def test_mod_by_one(self, runner, a):
        """Modulo by 1: a % 1 == 0."""
        source = f"int main() {{ return {a} % 1; }}"
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 0


@pytest.mark.property
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestFunctionProperties:
    """Property-based tests for function calls."""

    @given(st.integers(0, 50), st.integers(0, 50))
    @settings(max_examples=15)
    def test_identity_function(self, runner, a, b):
        """Identity function: f(x) = x."""
        source = f"""
        int identity(int x) {{ return x; }}
        int main() {{ return identity({a}) + identity({b}); }}
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=400)
        assert result == a + b

    @given(st.integers(0, 30), st.integers(0, 30))
    @settings(max_examples=15)
    def test_add_function_commutative(self, runner, a, b):
        """Function preserves commutativity."""
        source1 = f"""
        int add(int x, int y) {{ return x + y; }}
        int main() {{ return add({a}, {b}); }}
        """
        source2 = f"""
        int add(int x, int y) {{ return x + y; }}
        int main() {{ return add({b}, {a}); }}
        """

        bytecode1, data1 = compile_c(source1)
        bytecode2, data2 = compile_c(source2)

        _, result1 = runner.run(bytecode1, data1, max_steps=400)
        _, result2 = runner.run(bytecode2, data2, max_steps=400)

        assert result1 == result2
