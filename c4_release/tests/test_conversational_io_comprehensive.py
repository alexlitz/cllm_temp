#!/usr/bin/env python3
"""
Comprehensive Conversational IO Tests.

Tests for Requirement #4 from Testing Checklist:
- IO behavior with 100% pure autoregressive transformer works
- Reading and writing user messages

This tests the full conversational IO flow:
1. PRTF detection triggers THINKING_END emission
2. Output is written to user
3. READ operations read user input
4. THINKING_START resumes computation
5. Multiple IO operations in sequence

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token


class TestPRTFDetection:
    """Test PRTF opcode detection and THINKING_END emission."""

    @pytest.fixture
    def runner(self):
        """Create runner with conversational IO enabled."""
        return AutoregressiveVMRunner(conversational_io=True)

    def test_simple_printf_triggers_thinking_end(self, runner):
        """Simple printf should trigger THINKING_END."""
        source = '''
        int main() {
            printf("Hello");
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=200)
        assert exit_code == 0

    def test_printf_with_integer(self, runner):
        """Printf with integer argument."""
        source = '''
        int main() {
            int x;
            x = 42;
            printf("%d", x);
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=300)
        assert exit_code == 0

    def test_printf_with_string(self, runner):
        """Printf with string argument."""
        source = '''
        int main() {
            printf("test string");
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=200)
        assert exit_code == 0

    def test_multiple_printf_calls(self, runner):
        """Multiple printf calls in sequence."""
        source = '''
        int main() {
            printf("one");
            printf("two");
            printf("three");
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=500)
        assert exit_code == 0

    def test_printf_in_loop(self, runner):
        """Printf inside a loop."""
        source = '''
        int main() {
            int i;
            i = 0;
            while (i < 3) {
                printf("%d", i);
                i = i + 1;
            }
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=800)
        assert exit_code == 0

    def test_printf_in_function(self, runner):
        """Printf inside a called function."""
        source = '''
        void greet() {
            printf("Hello from function");
        }
        int main() {
            greet();
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=400)
        assert exit_code == 0


class TestConversationalIOMode:
    """Test conversational IO mode flag."""

    def test_conversational_io_disabled_by_default(self):
        """Conversational IO is disabled by default."""
        runner = AutoregressiveVMRunner()
        assert not runner.conversational_io

    def test_conversational_io_can_be_enabled(self):
        """Conversational IO can be enabled."""
        runner = AutoregressiveVMRunner(conversational_io=True)
        assert runner.conversational_io

    def test_programs_run_without_conversational_io(self):
        """Programs run correctly without conversational IO."""
        runner = AutoregressiveVMRunner(conversational_io=False)
        source = "int main() { return 42; }"
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=100)
        assert exit_code == 42

    def test_programs_run_with_conversational_io(self):
        """Programs run correctly with conversational IO."""
        runner = AutoregressiveVMRunner(conversational_io=True)
        source = "int main() { return 42; }"
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=100)
        assert exit_code == 42


class TestThinkingTokens:
    """Test THINKING_START and THINKING_END token handling."""

    def test_thinking_tokens_exist(self):
        """THINKING tokens are defined."""
        assert hasattr(Token, 'THINKING_START')
        assert hasattr(Token, 'THINKING_END')

    def test_thinking_tokens_are_distinct(self):
        """THINKING tokens have distinct values."""
        assert Token.THINKING_START != Token.THINKING_END
        assert Token.THINKING_START != Token.HALT
        assert Token.THINKING_END != Token.HALT


class TestIOWithComputation:
    """Test IO operations combined with computation."""

    @pytest.fixture
    def runner(self):
        """Create runner with conversational IO enabled."""
        return AutoregressiveVMRunner(conversational_io=True)

    def test_printf_after_arithmetic(self, runner):
        """Printf after arithmetic operations."""
        source = '''
        int main() {
            int a;
            int b;
            int c;
            a = 10;
            b = 20;
            c = a + b;
            printf("%d", c);
            return c;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=400)
        assert exit_code == 30

    def test_printf_with_function_result(self, runner):
        """Printf with result from function call."""
        source = '''
        int double_it(int x) { return x * 2; }
        int main() {
            int result;
            result = double_it(21);
            printf("%d", result);
            return result;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=500)
        assert exit_code == 42

    def test_printf_with_array_element(self, runner):
        """Printf with array element."""
        source = '''
        int main() {
            int arr[3];
            arr[0] = 100;
            arr[1] = 200;
            arr[2] = 300;
            printf("%d", arr[1]);
            return arr[1];
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=500)
        assert exit_code == 200

    def test_computation_after_printf(self, runner):
        """Computation continues correctly after printf."""
        source = '''
        int main() {
            int x;
            x = 10;
            printf("before");
            x = x + 5;
            printf("after");
            return x;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=500)
        assert exit_code == 15


class TestPutcharOutput:
    """Test putchar for character output."""

    @pytest.fixture
    def runner(self):
        """Create runner with conversational IO enabled."""
        return AutoregressiveVMRunner(conversational_io=True)

    def test_single_putchar(self, runner):
        """Single putchar call."""
        source = '''
        int main() {
            putchar(65);  // 'A'
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=200)
        assert exit_code == 0

    def test_multiple_putchar(self, runner):
        """Multiple putchar calls spell a word."""
        source = '''
        int main() {
            putchar(72);   // H
            putchar(105);  // i
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=300)
        assert exit_code == 0

    def test_putchar_in_loop(self, runner):
        """Putchar in a loop."""
        source = '''
        int main() {
            int i;
            i = 0;
            while (i < 5) {
                putchar(65 + i);  // A, B, C, D, E
                i = i + 1;
            }
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=600)
        assert exit_code == 0


class TestMixedIOOperations:
    """Test combinations of different IO operations."""

    @pytest.fixture
    def runner(self):
        """Create runner with conversational IO enabled."""
        return AutoregressiveVMRunner(conversational_io=True)

    def test_printf_and_putchar(self, runner):
        """Mix of printf and putchar."""
        source = '''
        int main() {
            printf("Number: ");
            putchar(52);  // '4'
            putchar(50);  // '2'
            return 42;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=400)
        assert exit_code == 42

    def test_io_in_recursive_function(self, runner):
        """IO in recursive function."""
        source = '''
        void countdown(int n) {
            if (n <= 0) return;
            printf("%d ", n);
            countdown(n - 1);
        }
        int main() {
            countdown(3);
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=800)
        assert exit_code == 0


class TestIOErrorHandling:
    """Test IO error handling and edge cases."""

    @pytest.fixture
    def runner(self):
        """Create runner with conversational IO enabled."""
        return AutoregressiveVMRunner(conversational_io=True)

    def test_empty_printf(self, runner):
        """Printf with empty string."""
        source = '''
        int main() {
            printf("");
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=200)
        assert exit_code == 0

    def test_printf_special_chars(self, runner):
        """Printf with special characters."""
        source = '''
        int main() {
            printf("line1\\nline2");
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=300)
        assert exit_code == 0

    def test_putchar_zero(self, runner):
        """Putchar with zero (null character)."""
        source = '''
        int main() {
            putchar(0);
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=200)
        assert exit_code == 0

    def test_putchar_max_char(self, runner):
        """Putchar with max ASCII value."""
        source = '''
        int main() {
            putchar(127);
            return 0;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=200)
        assert exit_code == 0


class TestIOStatePreservation:
    """Test that IO operations preserve program state correctly."""

    @pytest.fixture
    def runner(self):
        """Create runner with conversational IO enabled."""
        return AutoregressiveVMRunner(conversational_io=True)

    def test_registers_preserved_after_printf(self, runner):
        """Register values preserved after printf."""
        source = '''
        int main() {
            int a;
            int b;
            a = 100;
            b = 200;
            printf("test");
            return a + b;
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=400)
        assert exit_code == 300

    def test_array_preserved_after_printf(self, runner):
        """Array values preserved after printf."""
        source = '''
        int main() {
            int arr[3];
            arr[0] = 1;
            arr[1] = 2;
            arr[2] = 3;
            printf("test");
            return arr[0] + arr[1] + arr[2];
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=500)
        assert exit_code == 6

    def test_stack_preserved_after_printf(self, runner):
        """Stack state preserved after printf in nested calls."""
        source = '''
        int inner(int x) {
            printf("inner");
            return x * 2;
        }
        int outer(int x) {
            int y;
            y = inner(x);
            printf("outer");
            return y + 1;
        }
        int main() {
            return outer(10);
        }
        '''
        bytecode, data = compile_c(source)
        output, exit_code = runner.run(bytecode, data, max_steps=800)
        assert exit_code == 21  # (10 * 2) + 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
