#!/usr/bin/env python3
"""
Memory Operations Neural Tests.

Tests for LI, LC, SI, SC, LEA memory operations.

Status (2026-04-17):
- LI: Load int from stack - handler required
- LC: Load char from stack - handler required
- SI: Store int to stack - handler required
- SC: Store char to stack - handler required
- LEA: Load effective address - neural (L14/L15 limited with 2+ locals)

Known Limitation:
- L14/L15 memory mechanism outputs all-zero for MEM sections when
  there are 2+ local variables. Handler fallback used for correctness.

Date: 2026-04-17
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


class TestLIBasic:
    """Test LI (load int from stack) instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_li_single_local(self, runner):
        """LI with single local variable."""
        source = """
        int main() {
            int a;
            a = 42;
            return a;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=100)
        assert result == 42, f"Expected a=42, got {result}"

    def test_li_multiple_locals(self, runner):
        """LI with multiple local variables."""
        source = """
        int main() {
            int a;
            int b;
            int c;
            a = 10;
            b = 20;
            c = 30;
            return a + b + c;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 60, f"Expected sum=60, got {result}"

    @pytest.mark.xfail(reason="L14/L15 MEM sections all-zero with 2+ locals - handler needed")
    def test_li_neural_only(self):
        """Document: LI requires handler with multiple locals."""
        runner = AutoregressiveVMRunner()
        # Remove LI handler
        if Opcode.LI in runner._syscall_handlers:
            del runner._syscall_handlers[Opcode.LI]

        source = """
        int main() {
            int a;
            int b;
            a = 10;
            b = 20;
            return a + b;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 30


class TestSIBasic:
    """Test SI (store int to stack) instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_si_basic(self, runner):
        """SI basic store."""
        source = """
        int main() {
            int arr[2];
            arr[0] = 42;
            arr[1] = 7;
            return arr[0] + arr[1];
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        assert result == 49, f"Expected 49, got {result}"

    def test_si_overwrite(self, runner):
        """SI overwrites previous value."""
        source = """
        int main() {
            int x;
            x = 10;
            x = 20;
            x = 30;
            return x;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 30, f"Expected 30 after overwrite, got {result}"


class TestLEABasic:
    """Test LEA (load effective address) instruction."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_lea_local_address(self, runner):
        """LEA gets address of local variable."""
        source = """
        int main() {
            int a;
            int *p;
            a = 42;
            p = &a;
            return *p;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 42, f"Expected *p=42, got {result}"

    def test_lea_array_element(self, runner):
        """LEA for array element access."""
        source = """
        int main() {
            int arr[5];
            int *p;
            arr[2] = 99;
            p = &arr[2];
            return *p;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        assert result == 99, f"Expected arr[2]=99, got {result}"


class TestMemoryPatterns:
    """Test common memory access patterns."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_array_sum(self, runner):
        """Sum array elements."""
        source = """
        int main() {
            int arr[5];
            int sum;
            int i;
            arr[0] = 1;
            arr[1] = 2;
            arr[2] = 3;
            arr[3] = 4;
            arr[4] = 5;
            sum = 0;
            i = 0;
            while (i < 5) {
                sum = sum + arr[i];
                i = i + 1;
            }
            return sum;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=1000)
        assert result == 15, f"Expected sum=15, got {result}"

    def test_swap_variables(self, runner):
        """Swap two variables using temp."""
        source = """
        int main() {
            int a;
            int b;
            int temp;
            a = 10;
            b = 20;
            temp = a;
            a = b;
            b = temp;
            return a * 10 + b;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        # After swap: a=20, b=10, result = 20*10 + 10 = 210
        assert result == 210, f"Expected 210 after swap, got {result}"

    def test_pointer_arithmetic(self, runner):
        """Pointer arithmetic access."""
        source = """
        int main() {
            int arr[3];
            int *p;
            arr[0] = 1;
            arr[1] = 2;
            arr[2] = 3;
            p = arr;
            return *p + *(p+1) + *(p+2);
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=400)
        assert result == 6, f"Expected 1+2+3=6, got {result}"


class TestCharOperations:
    """Test LC/SC (load/store char) instructions."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_char_store_load(self, runner):
        """Store and load char."""
        source = """
        int main() {
            char c;
            c = 65;
            return c;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=100)
        assert result == 65, f"Expected char 65 ('A'), got {result}"

    def test_char_array(self, runner):
        """Char array operations."""
        source = """
        int main() {
            char buf[4];
            buf[0] = 1;
            buf[1] = 2;
            buf[2] = 3;
            buf[3] = 4;
            return buf[0] + buf[1] + buf[2] + buf[3];
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=300)
        assert result == 10, f"Expected sum=10, got {result}"


class TestNeuralMemoryLimitations:
    """Document neural memory limitations with xfail markers."""

    @pytest.mark.xfail(reason="L14/L15 MEM sections all-zero with 2+ locals")
    def test_li_2_locals_neural(self):
        """Document: LI broken neurally with 2+ locals."""
        runner = AutoregressiveVMRunner()
        # Remove all memory handlers
        for op in [Opcode.LI, Opcode.LC, Opcode.SI, Opcode.SC]:
            if op in runner._syscall_handlers:
                del runner._syscall_handlers[op]

        source = """
        int main() {
            int a;
            int b;
            a = 10;
            b = 20;
            return a + b;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=200)
        assert result == 30

    @pytest.mark.xfail(reason="SI neural path needs handler for shadow memory")
    def test_si_neural_only(self):
        """Document: SI requires handler."""
        runner = AutoregressiveVMRunner()
        if Opcode.SI in runner._syscall_handlers:
            del runner._syscall_handlers[Opcode.SI]

        source = """
        int main() {
            int x;
            x = 42;
            return x;
        }
        """
        bytecode, data = compile_c(source)
        _, result = runner.run(bytecode, data, max_steps=100)
        assert result == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
