#!/usr/bin/env python3
"""
Pytest Wrapper for 1096 Comprehensive Tests.

Wraps the test_suite_1000.py test generator in pytest format so all 1096 tests
appear as individual pytest test cases.

Testing Checklist Coverage:
- Requirement #1: All of the 1000+ comprehensive tests work

Usage:
    # Run all 1096 tests (slow)
    pytest tests/test_suite_1096_pytest.py -v

    # Run first 100 tests (quick)
    pytest tests/test_suite_1096_pytest.py -v -k "test_program_0"

    # Run specific category
    pytest tests/test_suite_1096_pytest.py -v -k "add_"

    # Run with parallel execution
    pytest tests/test_suite_1096_pytest.py -v -n auto

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner


# Generate all test programs once at module load
ALL_TESTS = generate_test_programs()


def make_test_id(test_tuple):
    """Create a readable test ID from test tuple."""
    source, expected, description = test_tuple
    # Clean up description for test ID
    desc = description.replace(" ", "_").replace(":", "")
    return desc[:50]  # Limit length


# Create parametrized test IDs
TEST_IDS = [make_test_id(t) for t in ALL_TESTS]


class TestSuite1096:
    """Run all 1096 tests from the comprehensive test suite."""

    @pytest.fixture(scope="class")
    def runner(self):
        """Shared runner for all tests in class."""
        return AutoregressiveVMRunner()

    @pytest.mark.parametrize(
        "source,expected,description",
        ALL_TESTS,
        ids=TEST_IDS
    )
    def test_program(self, runner, source, expected, description):
        """Run a single test program and verify result."""
        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=2000)

        assert result == expected, \
            f"{description}: expected {expected}, got {result}"


class TestSuite1096Quick:
    """Quick subset of tests (first 100)."""

    @pytest.fixture(scope="class")
    def runner(self):
        """Shared runner for quick tests."""
        return AutoregressiveVMRunner()

    @pytest.mark.parametrize(
        "source,expected,description",
        ALL_TESTS[:100],
        ids=TEST_IDS[:100]
    )
    def test_quick_program(self, runner, source, expected, description):
        """Run a quick test program."""
        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=1000)

        assert result == expected, \
            f"{description}: expected {expected}, got {result}"


class TestSuiteCategories:
    """Test specific categories from the suite."""

    @pytest.fixture(scope="class")
    def runner(self):
        """Shared runner."""
        return AutoregressiveVMRunner()

    def _filter_tests(self, keyword):
        """Filter tests by keyword in description."""
        return [(s, e, d) for s, e, d in ALL_TESTS if keyword in d.lower()]

    @pytest.mark.slow
    def test_all_addition(self, runner):
        """All addition tests pass."""
        tests = self._filter_tests("add_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=500)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"

    @pytest.mark.slow
    def test_all_subtraction(self, runner):
        """All subtraction tests pass."""
        tests = self._filter_tests("sub_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=500)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"

    @pytest.mark.slow
    def test_all_multiplication(self, runner):
        """All multiplication tests pass."""
        tests = self._filter_tests("mul_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=500)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"

    @pytest.mark.slow
    def test_all_division(self, runner):
        """All division tests pass."""
        tests = self._filter_tests("div_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=500)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"

    @pytest.mark.slow
    def test_all_modulo(self, runner):
        """All modulo tests pass."""
        tests = self._filter_tests("mod_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=500)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"

    @pytest.mark.slow
    def test_all_functions(self, runner):
        """All function tests pass."""
        tests = self._filter_tests("func_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=1000)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"

    @pytest.mark.slow
    def test_all_loops(self, runner):
        """All loop tests pass."""
        tests = self._filter_tests("loop_")
        passed = 0
        failed = []

        for source, expected, desc in tests:
            try:
                bytecode, data = compile_c(source)
                _, result = runner.run(bytecode, data, max_steps=2000)
                if result == expected:
                    passed += 1
                else:
                    failed.append((desc, expected, result))
            except Exception as e:
                failed.append((desc, expected, str(e)))

        assert len(failed) == 0, f"Failed {len(failed)}/{len(tests)}: {failed[:5]}"


class TestSuiteStatistics:
    """Tests to verify suite statistics."""

    def test_suite_has_1096_tests(self):
        """Suite generates exactly 1096 tests."""
        assert len(ALL_TESTS) == 1096, f"Expected 1096 tests, got {len(ALL_TESTS)}"

    def test_all_tests_have_valid_structure(self):
        """All tests have (source, expected, description) structure."""
        for i, test in enumerate(ALL_TESTS):
            assert len(test) == 3, f"Test {i} has wrong structure: {test}"
            source, expected, desc = test
            assert isinstance(source, str), f"Test {i}: source not string"
            assert isinstance(expected, int), f"Test {i}: expected not int"
            assert isinstance(desc, str), f"Test {i}: description not string"

    def test_all_sources_compilable(self):
        """All test sources can be compiled."""
        failed = []
        for i, (source, expected, desc) in enumerate(ALL_TESTS[:100]):  # Quick check first 100
            try:
                bytecode, data = compile_c(source)
                assert len(bytecode) > 0
            except Exception as e:
                failed.append((i, desc, str(e)))

        assert len(failed) == 0, f"Failed to compile {len(failed)} tests: {failed[:5]}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
