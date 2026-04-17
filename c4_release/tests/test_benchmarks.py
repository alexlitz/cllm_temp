"""
Performance Benchmark Tests for Neural VM.

Uses pytest-benchmark to measure execution performance.

Run with:
    pytest tests/test_benchmarks.py --benchmark-only
    pytest tests/test_benchmarks.py --benchmark-json=results.json
"""

import pytest

# Try to import benchmark fixture
try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.run_vm import AutoregressiveVMRunner
from src.c4 import compile_c


@pytest.fixture
def runner():
    """Create a runner for benchmarks."""
    return AutoregressiveVMRunner()


@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestRunnerBenchmarks:
    """Benchmark VM execution throughput."""

    def test_simple_return(self, benchmark, runner):
        """Benchmark: return constant."""
        source = "int main() { return 42; }"
        bytecode, data = compile_c(source)

        def run_vm():
            return runner.run(bytecode, data, max_steps=100)

        result = benchmark(run_vm)
        assert result[1] == 42

    def test_arithmetic_sequence(self, benchmark, runner):
        """Benchmark: simple arithmetic."""
        source = """
        int main() {
            int a = 10;
            int b = 20;
            int c = a + b;
            return c;
        }
        """
        bytecode, data = compile_c(source)

        def run_vm():
            return runner.run(bytecode, data, max_steps=200)

        result = benchmark(run_vm)
        assert result[1] == 30

    def test_loop_10_iterations(self, benchmark, runner):
        """Benchmark: small loop."""
        source = """
        int main() {
            int sum = 0;
            int i;
            for (i = 0; i < 10; i = i + 1) {
                sum = sum + i;
            }
            return sum;
        }
        """
        bytecode, data = compile_c(source)

        def run_vm():
            return runner.run(bytecode, data, max_steps=500)

        result = benchmark(run_vm)
        assert result[1] == 45  # 0+1+2+...+9 = 45

    def test_function_call(self, benchmark, runner):
        """Benchmark: single function call."""
        source = """
        int add(int a, int b) { return a + b; }
        int main() { return add(10, 20); }
        """
        bytecode, data = compile_c(source)

        def run_vm():
            return runner.run(bytecode, data, max_steps=300)

        result = benchmark(run_vm)
        assert result[1] == 30

    def test_recursive_factorial_5(self, benchmark, runner):
        """Benchmark: recursive factorial(5)."""
        source = """
        int factorial(int n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        int main() { return factorial(5); }
        """
        bytecode, data = compile_c(source)

        def run_vm():
            return runner.run(bytecode, data, max_steps=500)

        result = benchmark(run_vm)
        assert result[1] == 120


@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestCompilationBenchmarks:
    """Benchmark compilation performance."""

    def test_compile_simple(self, benchmark):
        """Benchmark: compile simple program."""
        source = "int main() { return 42; }"

        result = benchmark(compile_c, source)
        assert len(result[0]) > 0  # bytecode

    def test_compile_with_functions(self, benchmark):
        """Benchmark: compile program with functions."""
        source = """
        int add(int a, int b) { return a + b; }
        int sub(int a, int b) { return a - b; }
        int mul(int a, int b) { return a * b; }
        int main() { return add(1, mul(2, sub(10, 5))); }
        """

        result = benchmark(compile_c, source)
        assert len(result[0]) > 0


@pytest.mark.benchmark
@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestThroughputBenchmarks:
    """Measure tokens/second and steps/second."""

    def test_steps_per_second(self, benchmark, runner):
        """Measure execution steps per second."""
        source = """
        int main() {
            int i;
            int sum = 0;
            for (i = 0; i < 100; i = i + 1) {
                sum = sum + i;
            }
            return sum;
        }
        """
        bytecode, data = compile_c(source)

        # Run multiple iterations to get stable measurement
        def run_vm():
            return runner.run(bytecode, data, max_steps=2000)

        result = benchmark.pedantic(run_vm, iterations=5, rounds=3)
        # Verify correctness
        assert result[1] == 4950  # sum 0..99


# Simplified benchmarks that work without pytest-benchmark
class TestSimpleBenchmarks:
    """Basic timing tests that don't require pytest-benchmark."""

    def test_execution_completes_quickly(self, runner):
        """Verify simple programs complete in reasonable time."""
        import time

        source = "int main() { return 42; }"
        bytecode, data = compile_c(source)

        start = time.perf_counter()
        _, result = runner.run(bytecode, data, max_steps=100)
        elapsed = time.perf_counter() - start

        assert result == 42
        # Should complete in under 1 second on any hardware
        assert elapsed < 1.0, f"Execution took {elapsed:.2f}s, expected < 1.0s"

    def test_loop_completes_reasonably(self, runner):
        """Verify loop programs complete in reasonable time."""
        import time

        source = """
        int main() {
            int i, sum = 0;
            for (i = 0; i < 10; i = i + 1) sum = sum + i;
            return sum;
        }
        """
        bytecode, data = compile_c(source)

        start = time.perf_counter()
        _, result = runner.run(bytecode, data, max_steps=500)
        elapsed = time.perf_counter() - start

        assert result == 45
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Execution took {elapsed:.2f}s, expected < 5.0s"
