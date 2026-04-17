"""
Tests for data section access and code/data interactions.

This test suite verifies that the VM correctly handles data section operations,
including:
- Reading/writing global variables
- Accessing global arrays
- Modifying data section values
- Multiple function calls accessing shared data

This ensures the VM properly handles the boundary between code and data sections.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.compiler import compile_c
from src.speculator import FastLogicalVM
from neural_vm.speculative import DraftVM


class TestDataSectionAccess(unittest.TestCase):
    """Test data section read/write operations."""

    def run_program(self, source: str, max_steps: int = 1000) -> int:
        """Compile and run a C program."""
        bytecode, data = compile_c(source)
        vm = FastLogicalVM()
        vm.load(bytecode, data)
        return vm.run(max_steps=max_steps)

    def run_program_draft(self, source: str, max_steps: int = 1000) -> int:
        """Compile and run using DraftVM."""
        bytecode, data = compile_c(source)
        vm = DraftVM(bytecode)
        if data:
            vm.load_data(data)
        steps = 0
        while not vm.halted and steps < max_steps:
            vm.step()
            steps += 1
        return vm.ax

    def test_simple_global_variable(self):
        """Test reading a simple global variable."""
        code = """
        int global_value = 42;

        int main() {
            return global_value;
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 42, "Should read global variable correctly")

    def test_modify_global_variable(self):
        """Test modifying a global variable."""
        code = """
        int counter = 0;

        int increment() {
            counter = counter + 1;
            return counter;
        }

        int main() {
            int a;
            int b;
            a = increment();
            b = increment();
            return a + b;
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 3, "Should modify global variable correctly (1 + 2 = 3)")

    def test_global_array_access(self):
        """Test accessing a global array."""
        code = """
        int values[3];

        int main() {
            values[0] = 10;
            values[1] = 20;
            values[2] = 30;

            return values[0] + values[1] + values[2];
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 60, "Should access global array correctly")

    def test_global_array_in_loop(self):
        """Test accessing global array in a loop."""
        code = """
        int numbers[4];

        int main() {
            int i;
            int sum;

            numbers[0] = 1;
            numbers[1] = 2;
            numbers[2] = 3;
            numbers[3] = 4;

            sum = 0;
            i = 0;
            while (i < 4) {
                sum = sum + numbers[i];
                i = i + 1;
            }

            return sum;
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 10, "Should sum array elements correctly")

    def test_multiple_globals(self):
        """Test multiple global variables."""
        code = """
        int x = 10;
        int y = 20;
        int z = 30;

        int main() {
            return x + y + z;
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 60, "Should access multiple globals correctly")

    def test_global_state_across_calls(self):
        """Test that global state persists across function calls."""
        code = """
        int state = 100;

        int modify_state(int delta) {
            state = state + delta;
            return state;
        }

        int main() {
            int a;
            int b;
            int c;

            a = modify_state(10);
            b = modify_state(20);
            c = modify_state(30);

            return c;
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 160, "Global state should persist (100 + 10 + 20 + 30 = 160)")

    def test_conditional_global_write(self):
        """Test conditional writing to global variable."""
        code = """
        int result = 0;

        int set_result(int value, int condition) {
            if (condition) {
                result = value;
            }
            return result;
        }

        int main() {
            set_result(10, 0);
            set_result(20, 1);
            return result;
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 20, "Should conditionally write to global")

    def test_array_as_buffer(self):
        """Test using global array as a buffer."""
        code = """
        int buffer[5];

        int fill_buffer() {
            int i;
            i = 0;
            while (i < 5) {
                buffer[i] = i * 10;
                i = i + 1;
            }
            return 0;
        }

        int sum_buffer() {
            int i;
            int total;
            total = 0;
            i = 0;
            while (i < 5) {
                total = total + buffer[i];
                i = i + 1;
            }
            return total;
        }

        int main() {
            fill_buffer();
            return sum_buffer();
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 100, "Should use array as buffer (0+10+20+30+40=100)")

    def test_draft_vs_fast_consistency(self):
        """Verify DraftVM and FastLogicalVM produce same results for data access."""
        code = """
        int data_val = 5;

        int compute() {
            data_val = data_val * 2;
            return data_val;
        }

        int main() {
            int a;
            int b;
            a = compute();
            b = compute();
            return a + b;
        }
        """

        fast_result = self.run_program(code)
        draft_result = self.run_program_draft(code)

        self.assertEqual(fast_result, 30, "FastLogicalVM should compute correctly (10 + 20)")
        self.assertEqual(draft_result, 30, "DraftVM should compute correctly")
        self.assertEqual(fast_result, draft_result, "Both VMs should produce same result")


class TestComplexDataPatterns(unittest.TestCase):
    """Test more complex data section patterns."""

    def run_program(self, source: str, max_steps: int = 1000) -> int:
        """Compile and run a C program."""
        bytecode, data = compile_c(source)
        vm = FastLogicalVM()
        vm.load(bytecode, data)
        return vm.run(max_steps=max_steps)

    def test_nested_array_access(self):
        """Test accessing multi-dimensional array patterns."""
        code = """
        int matrix[9];

        int main() {
            int i;

            i = 0;
            while (i < 9) {
                matrix[i] = i + 1;
                i = i + 1;
            }

            return matrix[0] + matrix[4] + matrix[8];
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 14, "Should access matrix correctly (1 + 5 + 9 = 14)")

    def test_lookup_table_pattern(self):
        """Test using global array as lookup table."""
        code = """
        int squares[10];

        int init_squares() {
            int i;
            i = 0;
            while (i < 10) {
                squares[i] = i * i;
                i = i + 1;
            }
            return 0;
        }

        int get_square(int n) {
            return squares[n];
        }

        int main() {
            init_squares();
            return get_square(5) + get_square(3);
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 34, "Should use lookup table (25 + 9 = 34)")

    def test_max_finder_with_global_array(self):
        """Test finding max value in global array."""
        code = """
        int values[5];

        int find_max() {
            int i;
            int max;

            max = values[0];
            i = 1;

            while (i < 5) {
                if (values[i] > max) {
                    max = values[i];
                }
                i = i + 1;
            }

            return max;
        }

        int main() {
            values[0] = 10;
            values[1] = 50;
            values[2] = 30;
            values[3] = 20;
            values[4] = 40;

            return find_max();
        }
        """
        result = self.run_program(code)
        self.assertEqual(result, 50, "Should find maximum value correctly")


if __name__ == '__main__':
    unittest.main(verbosity=2)
