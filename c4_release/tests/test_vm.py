"""
Test Suite for C4 Transformer VM

Tests all components:
- Byte/nibble conversion
- Arithmetic operations (SwiGLU multiply, FFN divide)
- Bitwise operations
- Full VM execution
- Compiler integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from typing import Tuple

from src.transformer_vm import (
    C4TransformerVM,
    ByteEncoder,
    ByteDecoder,
    SwiGLUMul,
    DivisionFFN,
    ByteAddFFN,
    ByteToNibbleFFN,
    NibbleToByteFFN,
)
from src.speculator import FastLogicalVM, SpeculativeVM
from src.compiler import compile_c


class TestByteNibbleConversion(unittest.TestCase):
    """Test byte to nibble and back conversion."""

    def setUp(self):
        self.b2n = ByteToNibbleFFN()
        self.n2b = NibbleToByteFFN()

    def test_all_bytes(self):
        """Test conversion for all 256 byte values."""
        for byte_val in range(256):
            byte_onehot = torch.zeros(256)
            byte_onehot[byte_val] = 1.0

            # Split
            high, low = self.b2n(byte_onehot)
            h_val = torch.argmax(high).item()
            l_val = torch.argmax(low).item()

            # Expected
            expected_h = (byte_val >> 4) & 0xF
            expected_l = byte_val & 0xF

            self.assertEqual(h_val, expected_h, f"High nibble mismatch for 0x{byte_val:02X}")
            self.assertEqual(l_val, expected_l, f"Low nibble mismatch for 0x{byte_val:02X}")

            # Recombine
            recovered = self.n2b(high, low)
            r_val = torch.argmax(recovered).item()
            self.assertEqual(r_val, byte_val, f"Recovery failed for 0x{byte_val:02X}")


class TestSwiGLUMultiply(unittest.TestCase):
    """Test SwiGLU-based multiplication."""

    def setUp(self):
        self.mul = SwiGLUMul()

    def test_basic_multiply(self):
        """Test basic multiplication cases."""
        test_cases = [
            (6, 7, 42),
            (0, 100, 0),
            (1, 123, 123),
            (10, 10, 100),
            (123, 456, 56088),
        ]

        for a, b, expected in test_cases:
            result = self.mul(torch.tensor(float(a)), torch.tensor(float(b)))
            self.assertEqual(int(round(result.item())), expected, f"{a} * {b}")

    def test_commutative(self):
        """Test that multiplication is commutative."""
        for a in [3, 7, 15, 42]:
            for b in [2, 5, 11, 33]:
                r1 = self.mul(torch.tensor(float(a)), torch.tensor(float(b)))
                r2 = self.mul(torch.tensor(float(b)), torch.tensor(float(a)))
                self.assertEqual(
                    int(round(r1.item())),
                    int(round(r2.item())),
                    f"{a} * {b} != {b} * {a}"
                )


class TestDivision(unittest.TestCase):
    """Test FFN-based division."""

    def setUp(self):
        self.div = DivisionFFN()

    def test_basic_division(self):
        """Test basic integer division."""
        test_cases = [
            (42, 6, 7),
            (100, 7, 14),
            (17, 5, 3),
            (99, 9, 11),
            (1000, 33, 30),
            (255, 16, 15),
        ]

        for a, b, expected in test_cases:
            result = self.div(a, b)
            self.assertEqual(result, expected, f"{a} // {b}")

    def test_division_by_one(self):
        """Test division by 1."""
        for n in [1, 42, 100, 12345]:
            self.assertEqual(self.div(n, 1), n)

    def test_division_by_zero(self):
        """Test division by 0 returns 0."""
        self.assertEqual(self.div(42, 0), 0)

    def test_small_dividend(self):
        """Test when dividend < divisor."""
        self.assertEqual(self.div(5, 10), 0)
        self.assertEqual(self.div(1, 100), 0)


class TestAddition(unittest.TestCase):
    """Test FFN-based addition."""

    def setUp(self):
        self.add = ByteAddFFN()
        self.enc = ByteEncoder()
        self.dec = ByteDecoder()

    def test_basic_addition(self):
        """Test basic addition."""
        test_cases = [
            (100, 50, 150),
            (255, 1, 256),
            (0, 0, 0),
            (0x1234, 0x5678, 0x68AC),
            (123, 456, 579),
        ]

        for a, b, expected in test_cases:
            a_emb = self.enc(a)
            b_emb = self.enc(b)
            r_emb = self.add(a_emb, b_emb)
            result = self.dec(r_emb)
            expected_masked = expected & 0xFFFFFFFF
            self.assertEqual(result, expected_masked, f"{a} + {b}")


class TestVM(unittest.TestCase):
    """Test full VM execution."""

    def setUp(self):
        self.vm = C4TransformerVM()

    def test_immediate_and_exit(self):
        """Test IMM and EXIT instructions."""
        self.vm.reset()
        # IMM 42, EXIT
        self.vm.load([(1, 42), (38, 0)])
        result = self.vm.run()
        self.assertEqual(result, 42)

    def test_arithmetic_ops(self):
        """Test arithmetic operations."""
        # 6 * 7
        self.vm.reset()
        self.vm.load([(1, 6), (13, 0), (1, 7), (27, 0), (38, 0)])
        self.assertEqual(self.vm.run(), 42)

        # 100 + 50
        self.vm.reset()
        self.vm.load([(1, 100), (13, 0), (1, 50), (25, 0), (38, 0)])
        self.assertEqual(self.vm.run(), 150)

        # 100 - 30
        self.vm.reset()
        self.vm.load([(1, 100), (13, 0), (1, 30), (26, 0), (38, 0)])
        self.assertEqual(self.vm.run(), 70)

        # 100 / 7
        self.vm.reset()
        self.vm.load([(1, 100), (13, 0), (1, 7), (28, 0), (38, 0)])
        self.assertEqual(self.vm.run(), 14)

        # 17 % 5
        self.vm.reset()
        self.vm.load([(1, 17), (13, 0), (1, 5), (29, 0), (38, 0)])
        self.assertEqual(self.vm.run(), 2)


class TestCompilerIntegration(unittest.TestCase):
    """Test compiler with VM execution."""

    def setUp(self):
        self.vm = C4TransformerVM()

    def run_c(self, source: str) -> int:
        """Compile and run C source."""
        bytecode, data = compile_c(source)
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run()

    def test_simple_return(self):
        """Test simple return statement."""
        self.assertEqual(self.run_c("int main() { return 42; }"), 42)

    def test_arithmetic(self):
        """Test arithmetic expressions."""
        self.assertEqual(self.run_c("int main() { return 3 + 4; }"), 7)
        self.assertEqual(self.run_c("int main() { return 6 * 7; }"), 42)
        self.assertEqual(self.run_c("int main() { return 3 + 4 * 2; }"), 11)
        self.assertEqual(self.run_c("int main() { return (3 + 4) * 2; }"), 14)

    def test_variables(self):
        """Test variable declarations and assignment."""
        code = """
        int main() {
            int a; int b;
            a = 10;
            b = 20;
            return a + b;
        }
        """
        self.assertEqual(self.run_c(code), 30)

    def test_if_statement(self):
        """Test if statement."""
        code = """
        int main() {
            int x;
            x = 10;
            if (x > 5) {
                return 1;
            }
            return 0;
        }
        """
        self.assertEqual(self.run_c(code), 1)

    def test_while_loop(self):
        """Test while loop."""
        code = """
        int main() {
            int i; int sum;
            i = 0; sum = 0;
            while (i < 5) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
        """
        self.assertEqual(self.run_c(code), 10)  # 0+1+2+3+4

    def test_function_call(self):
        """Test function calls."""
        code = """
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            return add(10, 20);
        }
        """
        self.assertEqual(self.run_c(code), 30)

    def test_recursion(self):
        """Test recursive function calls."""
        code = """
        int fib(int n) {
            if (n < 2) return n;
            return fib(n-1) + fib(n-2);
        }
        int main() { return fib(10); }
        """
        self.assertEqual(self.run_c(code), 55)


class TestSpeculator(unittest.TestCase):
    """Test speculative execution."""

    def test_fast_vm(self):
        """Test FastLogicalVM matches transformer VM."""
        source = """
        int fib(int n) {
            if (n < 2) return n;
            return fib(n-1) + fib(n-2);
        }
        int main() { return fib(10); }
        """
        bytecode, data = compile_c(source)

        # Fast VM
        fast_vm = FastLogicalVM()
        fast_vm.load(bytecode, data)
        fast_result = fast_vm.run()

        # Transformer VM
        trans_vm = C4TransformerVM()
        trans_vm.reset()
        trans_vm.load_bytecode(bytecode, data)
        trans_result = trans_vm.run()

        self.assertEqual(fast_result, trans_result)
        self.assertEqual(fast_result, 55)

    def test_speculative_vm(self):
        """Test SpeculativeVM with validation."""
        trans_vm = C4TransformerVM()
        spec_vm = SpeculativeVM(transformer_vm=trans_vm, validate_ratio=1.0)

        bytecode, data = compile_c("int main() { return 6 * 7; }")
        result = spec_vm.run(bytecode, data, validate=True)

        self.assertEqual(result, 42)
        self.assertEqual(spec_vm.mismatches, 0)


class TestBitwise(unittest.TestCase):
    """Test bitwise operations."""

    def setUp(self):
        self.vm = C4TransformerVM()

    def run_c(self, source: str) -> int:
        bytecode, data = compile_c(source)
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run()

    def test_and(self):
        """Test AND operation."""
        self.assertEqual(self.run_c("int main() { return 0xF0 & 0xAA; }"), 0xA0)
        self.assertEqual(self.run_c("int main() { return 0xFF & 0x0F; }"), 0x0F)

    def test_or(self):
        """Test OR operation."""
        self.assertEqual(self.run_c("int main() { return 0xF0 | 0x0F; }"), 0xFF)

    def test_xor(self):
        """Test XOR operation."""
        self.assertEqual(self.run_c("int main() { return 0xFF ^ 0xAA; }"), 0x55)

    def test_shift(self):
        """Test shift operations."""
        self.assertEqual(self.run_c("int main() { return 1 << 4; }"), 16)
        self.assertEqual(self.run_c("int main() { return 16 >> 2; }"), 4)


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("C4 TRANSFORMER VM TEST SUITE")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestByteNibbleConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestSwiGLUMultiply))
    suite.addTests(loader.loadTestsFromTestCase(TestDivision))
    suite.addTests(loader.loadTestsFromTestCase(TestAddition))
    suite.addTests(loader.loadTestsFromTestCase(TestVM))
    suite.addTests(loader.loadTestsFromTestCase(TestCompilerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSpeculator))
    suite.addTests(loader.loadTestsFromTestCase(TestBitwise))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
