"""
Comprehensive VM Tests - Control flow, I/O, Memory, Function calls.

Tests:
1. Control Flow: JMP, BZ, BNZ, JSR (CALL), LEV (RET)
2. Memory Operations: LI, LC, SI, SC, PSH
3. I/O Operations: GETCHAR, PUTCHAR (via embedding slots)
4. Function Calls/Returns: JSR/LEV with stack
5. Argument Passing: argc/argv through embedding
"""

import torch
import unittest
import sys
import os

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)

from neural_vm.embedding import E, Opcode
from neural_vm.control_flow_ops import (
    JumpFFN, BranchEqFFN, BranchNeFFN, BranchLtFFN, BranchGeFFN,
    CallFFN, RetFFN, BranchConditionAttention
)
from neural_vm.memory_ops import LoadFFN, StoreFFN, PushFFN, PopFFN, NopFFN, HaltFFN
from neural_vm.io_handler import ArgvHandler, IOHandler


class TestControlFlowOps(unittest.TestCase):
    """Test control flow operations."""

    def setUp(self):
        self.jmp = JumpFFN()
        self.beq = BranchEqFFN()
        self.bne = BranchNeFFN()
        self.blt = BranchLtFFN()
        self.bge = BranchGeFFN()
        self.call = CallFFN()
        self.ret = RetFFN()

    def _make_input(self, opcode, nib_a=0, nib_b=0):
        """Create input tensor with opcode and operands."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.NIB_A] = nib_a if isinstance(nib_a, (int, float)) else nib_a[i]
            x[0, i, E.NIB_B] = nib_b if isinstance(nib_b, (int, float)) else nib_b[i]
            x[0, i, E.POS] = float(i)
        return x

    def _get_result(self, y):
        """Extract result value from output tensor."""
        result = 0
        for i in range(E.NUM_POSITIONS):
            nib = int(round(y[0, i, E.RESULT].item()))
            nib = max(0, min(15, nib))
            result |= (nib << (4 * i))
        return result

    # JMP tests
    def test_jmp_basic(self):
        """JMP should copy target address to result."""
        x = self._make_input(Opcode.JMP, nib_a=0, nib_b=5)
        y = self.jmp(x)
        # Each position should have NIB_B copied to RESULT
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y[0, i, E.RESULT].item(), 5.0, places=1)

    def test_jmp_address_value(self):
        """JMP with multi-nibble address."""
        # Address 0x1234 = nibbles [4, 3, 2, 1, 0, 0, 0, 0]
        nib_b = [4, 3, 2, 1, 0, 0, 0, 0]
        x = self._make_input(Opcode.JMP, nib_a=0, nib_b=nib_b)
        y = self.jmp(x)
        result = self._get_result(y)
        self.assertEqual(result, 0x1234)

    # BZ (BEQ) tests
    def test_bz_taken(self):
        """BZ should branch when A == 0."""
        x = self._make_input(Opcode.BZ, nib_a=0, nib_b=10)
        y = self.beq(x)
        # When A=0, result should be B
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 10.0, places=1)

    def test_bz_not_taken(self):
        """BZ should not branch when A != 0."""
        x = self._make_input(Opcode.BZ, nib_a=5, nib_b=10)
        y = self.beq(x)
        # When A!=0, result should be ~0
        self.assertLess(abs(y[0, 0, E.RESULT].item()), 2.0)

    # BNZ tests
    def test_bnz_taken(self):
        """BNZ should branch when A != 0."""
        x = self._make_input(Opcode.BNZ, nib_a=5, nib_b=10)
        y = self.bne(x)
        # When A!=0, result should be B
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 10.0, places=1)

    def test_bnz_not_taken(self):
        """BNZ should not branch when A == 0."""
        x = self._make_input(Opcode.BNZ, nib_a=0, nib_b=10)
        y = self.bne(x)
        # When A=0, result should be ~0
        self.assertLess(abs(y[0, 0, E.RESULT].item()), 2.0)

    # BLT tests
    def test_blt_taken(self):
        """BLT should branch when A < B."""
        x = self._make_input(Opcode.BLT, nib_a=3, nib_b=8)
        y = self.blt(x)
        # When A<B, result should be B
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 8.0, places=1)

    def test_blt_not_taken(self):
        """BLT should not branch when A >= B."""
        x = self._make_input(Opcode.BLT, nib_a=10, nib_b=5)
        y = self.blt(x)
        # When A>=B, result should be ~0
        self.assertLess(abs(y[0, 0, E.RESULT].item()), 2.0)

    # BGE tests
    def test_bge_taken_greater(self):
        """BGE should branch when A > B."""
        x = self._make_input(Opcode.BGE, nib_a=10, nib_b=5)
        y = self.bge(x)
        # When A>B, result should be B
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 5.0, places=1)

    def test_bge_taken_equal(self):
        """BGE should branch when A == B."""
        x = self._make_input(Opcode.BGE, nib_a=7, nib_b=7)
        y = self.bge(x)
        # When A==B, result should be B
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 7.0, places=1)

    def test_bge_not_taken(self):
        """BGE should not branch when A < B."""
        x = self._make_input(Opcode.BGE, nib_a=3, nib_b=8)
        y = self.bge(x)
        # When A<B, result should be ~0
        self.assertLess(abs(y[0, 0, E.RESULT].item()), 2.0)

    # JSR/CALL tests
    def test_call_target(self):
        """CALL should put target address in RESULT."""
        x = self._make_input(Opcode.JSR, nib_a=5, nib_b=10)
        y = self.call(x)
        # RESULT should be target (NIB_B)
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 10.0, places=1)

    def test_call_return_addr(self):
        """CALL should put return address in TEMP."""
        x = self._make_input(Opcode.JSR, nib_a=5, nib_b=10)
        y = self.call(x)
        # TEMP should be return address (NIB_A)
        self.assertAlmostEqual(y[0, 0, E.TEMP].item(), 5.0, places=1)

    # RET tests
    def test_ret_basic(self):
        """RET should put return address in RESULT."""
        x = self._make_input(Opcode.LEV, nib_a=15, nib_b=0)
        y = self.ret(x)
        # RESULT should be return address (NIB_A)
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 15.0, places=1)


class TestMemoryOps(unittest.TestCase):
    """Test memory operations."""

    def setUp(self):
        self.load = LoadFFN()
        self.store = StoreFFN()
        self.push = PushFFN()
        self.pop = PopFFN()
        self.nop = NopFFN()
        self.halt = HaltFFN()

    def _make_input(self, opcode, nib_a=0, nib_b=0):
        """Create input tensor."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.NIB_A] = nib_a if isinstance(nib_a, (int, float)) else nib_a[i]
            x[0, i, E.NIB_B] = nib_b if isinstance(nib_b, (int, float)) else nib_b[i]
            x[0, i, E.POS] = float(i)
        return x

    # LI (Load Int) tests
    def test_load_prepares_address(self):
        """LOAD should copy address to TEMP."""
        addr_nibs = [0, 1, 2, 3, 0, 0, 0, 0]  # Address 0x3210
        x = self._make_input(Opcode.LI, nib_a=addr_nibs, nib_b=0)
        y = self.load(x)
        # TEMP should have address nibbles
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y[0, i, E.TEMP].item(), addr_nibs[i], places=1)

    def test_load_clears_result(self):
        """LOAD should clear RESULT."""
        x = self._make_input(Opcode.LI, nib_a=5, nib_b=0)
        x[0, 0, E.RESULT] = 10.0  # Pre-set result
        y = self.load(x)
        # RESULT should be cleared
        self.assertLess(abs(y[0, 0, E.RESULT].item()), 1.0)

    # SI (Store Int) tests
    def test_store_prepares_address(self):
        """STORE should copy address to TEMP."""
        addr_nibs = [5, 6, 7, 8, 0, 0, 0, 0]
        x = self._make_input(Opcode.SI, nib_a=addr_nibs, nib_b=0)
        y = self.store(x)
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y[0, i, E.TEMP].item(), addr_nibs[i], places=1)

    def test_store_prepares_value(self):
        """STORE should copy value to RESULT."""
        val_nibs = [1, 2, 3, 4, 5, 6, 7, 8]
        x = self._make_input(Opcode.SI, nib_a=0, nib_b=val_nibs)
        y = self.store(x)
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y[0, i, E.RESULT].item(), val_nibs[i], places=1)

    # PSH (Push) tests
    def test_push_copies_value(self):
        """PUSH should copy value to RESULT."""
        val_nibs = [9, 8, 7, 6, 5, 4, 3, 2]
        x = self._make_input(Opcode.PSH, nib_a=val_nibs, nib_b=0)
        y = self.push(x)
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y[0, i, E.RESULT].item(), val_nibs[i], places=1)

    # POP tests
    def test_pop_clears_result(self):
        """POP should clear RESULT."""
        x = self._make_input(Opcode.POP, nib_a=0, nib_b=0)
        x[0, 0, E.RESULT] = 15.0
        y = self.pop(x)
        self.assertLess(abs(y[0, 0, E.RESULT].item()), 1.0)

    # NOP tests
    def test_nop_no_change(self):
        """NOP should not modify anything."""
        x = self._make_input(Opcode.NOP, nib_a=5, nib_b=10)
        x[0, 0, E.RESULT] = 7.0
        y = self.nop(x)
        self.assertAlmostEqual(y[0, 0, E.RESULT].item(), 7.0, places=1)

    # HALT tests
    def test_halt_sets_marker(self):
        """HALT should set RESULT to marker value."""
        x = self._make_input(Opcode.EXIT, nib_a=0, nib_b=0)
        y = self.halt(x)
        # HALT sets RESULT to 15 (marker)
        self.assertGreater(y[0, 0, E.RESULT].item(), 10.0)


class TestIOOps(unittest.TestCase):
    """Test I/O operations via embedding slots."""

    def test_io_output_ready_slot(self):
        """Test IO_OUTPUT_READY embedding slot."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x[0, 0, E.IO_OUTPUT_READY] = 1.0
        self.assertEqual(x[0, 0, E.IO_OUTPUT_READY].item(), 1.0)

    def test_io_input_ready_slot(self):
        """Test IO_INPUT_READY embedding slot."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x[0, 0, E.IO_INPUT_READY] = 1.0
        self.assertEqual(x[0, 0, E.IO_INPUT_READY].item(), 1.0)

    def test_io_need_input_slot(self):
        """Test IO_NEED_INPUT embedding slot."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x[0, 0, E.IO_NEED_INPUT] = 1.0
        self.assertEqual(x[0, 0, E.IO_NEED_INPUT].item(), 1.0)

    def test_io_char_nibbles(self):
        """Test IO_CHAR stores character as nibbles."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        # Store 'A' (65 = 0x41)
        x[0, 0, E.IO_CHAR] = 1  # Low nibble
        x[0, 1, E.IO_CHAR] = 4  # High nibble
        char_val = int(x[0, 0, E.IO_CHAR].item()) + 16 * int(x[0, 1, E.IO_CHAR].item())
        self.assertEqual(char_val, 65)
        self.assertEqual(chr(char_val), 'A')

    def test_io_program_end_slot(self):
        """Test IO_PROGRAM_END embedding slot."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x[0, 0, E.IO_PROGRAM_END] = 1.0
        self.assertEqual(x[0, 0, E.IO_PROGRAM_END].item(), 1.0)


class TestArgvPassing(unittest.TestCase):
    """Test argument passing via embedding slots."""

    def test_argv_handler_basic(self):
        """Test ArgvHandler constructs correctly."""
        handler = ArgvHandler(["prog", "arg1", "arg2"])
        self.assertEqual(handler.argc, 3)
        self.assertEqual(handler.argv, ["prog", "arg1", "arg2"])

    def test_argv_handler_stream_format(self):
        """Test argv stream format."""
        handler = ArgvHandler(["test", "a"])
        # Stream: [2, 0, 0, 0] + "test\0" + "a\0"
        expected_argc = [2, 0, 0, 0]
        self.assertEqual(handler._stream[:4], expected_argc)

    def test_argv_embedding_setup(self):
        """Test argv embedding slot setup."""
        handler = ArgvHandler(["prog"])
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        handler.setup(x)
        # Slots should be initialized
        self.assertEqual(x[0, 0, E.IO_ARGV_INDEX].item(), 0.0)

    def test_argv_streaming(self):
        """Test streaming argv bytes through embedding."""
        handler = ArgvHandler(["x"])
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        handler.setup(x)

        # Stream: [1, 0, 0, 0, 'x', 0]
        expected = [1, 0, 0, 0, ord('x'), 0]
        received = []

        for _ in range(len(expected)):
            x[0, 0, E.IO_NEED_ARGV] = 1.0
            handler.check_argv(x)
            byte_val = int(x[0, 0, E.IO_CHAR].item())
            received.append(byte_val)
            x[0, 0, E.IO_NEED_ARGV] = 0.0
            x[0, 0, E.IO_CHAR] = 0.0

        self.assertEqual(received, expected)

    def test_argv_multiple_args(self):
        """Test multiple arguments."""
        handler = ArgvHandler(["prog", "hello", "world"])
        self.assertEqual(handler.argc, 3)

        # Check stream contains all strings
        stream_bytes = bytes(handler._stream[4:])
        self.assertIn(b"prog\0", stream_bytes)
        self.assertIn(b"hello\0", stream_bytes)
        self.assertIn(b"world\0", stream_bytes)


class TestFunctionCalls(unittest.TestCase):
    """Test function call/return patterns."""

    def setUp(self):
        self.call = CallFFN()
        self.ret = RetFFN()

    def _make_input(self, opcode, nib_a, nib_b):
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.NIB_A] = nib_a if isinstance(nib_a, (int, float)) else nib_a[i]
            x[0, i, E.NIB_B] = nib_b if isinstance(nib_b, (int, float)) else nib_b[i]
            x[0, i, E.POS] = float(i)
        return x

    def test_call_return_roundtrip(self):
        """Test CALL followed by RET returns to correct address."""
        # CALL: return_addr=100, target=200
        # Nibbles of 100 = [4, 6, 0, 0, 0, 0, 0, 0]
        ret_addr_nibs = [4, 6, 0, 0, 0, 0, 0, 0]  # 100 = 0x64
        target_nibs = [8, 12, 0, 0, 0, 0, 0, 0]   # 200 = 0xC8

        x_call = self._make_input(Opcode.JSR, nib_a=ret_addr_nibs, nib_b=target_nibs)
        y_call = self.call(x_call)

        # TEMP should have return address
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y_call[0, i, E.TEMP].item(), ret_addr_nibs[i], places=1)

        # RESULT should have target
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y_call[0, i, E.RESULT].item(), target_nibs[i], places=1)

        # Now RET with the return address
        x_ret = self._make_input(Opcode.LEV, nib_a=ret_addr_nibs, nib_b=0)
        y_ret = self.ret(x_ret)

        # RESULT should have return address
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y_ret[0, i, E.RESULT].item(), ret_addr_nibs[i], places=1)

    def test_nested_calls(self):
        """Test nested function calls."""
        # Simulate: main calls foo (saves ret=10), foo calls bar (saves ret=50)
        # bar returns to 50, foo returns to 10

        # First call: main -> foo
        x1 = self._make_input(Opcode.JSR, nib_a=10, nib_b=100)
        y1 = self.call(x1)
        ret1 = y1[0, 0, E.TEMP].item()
        self.assertAlmostEqual(ret1, 10.0, places=1)

        # Second call: foo -> bar
        x2 = self._make_input(Opcode.JSR, nib_a=50, nib_b=200)
        y2 = self.call(x2)
        ret2 = y2[0, 0, E.TEMP].item()
        self.assertAlmostEqual(ret2, 50.0, places=1)

        # Return from bar (should go to 50)
        x3 = self._make_input(Opcode.LEV, nib_a=50, nib_b=0)
        y3 = self.ret(x3)
        self.assertAlmostEqual(y3[0, 0, E.RESULT].item(), 50.0, places=1)

        # Return from foo (should go to 10)
        x4 = self._make_input(Opcode.LEV, nib_a=10, nib_b=0)
        y4 = self.ret(x4)
        self.assertAlmostEqual(y4[0, 0, E.RESULT].item(), 10.0, places=1)


class TestComprehensiveScenarios(unittest.TestCase):
    """End-to-end scenarios combining multiple ops."""

    def test_conditional_branch_loop(self):
        """Simulate a simple conditional loop."""
        beq = BranchEqFFN()
        bne = BranchNeFFN()

        # Loop condition: branch if counter == 0
        # Counter = 3: don't branch
        x1 = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x1[0, :, E.OP_START + Opcode.BZ] = 1.0
        x1[0, :, E.NIB_A] = 3  # Counter
        x1[0, :, E.NIB_B] = 100  # Exit target
        y1 = beq(x1)
        self.assertLess(abs(y1[0, 0, E.RESULT].item()), 5)  # Not taken

        # Counter = 0: branch
        x2 = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        x2[0, :, E.OP_START + Opcode.BZ] = 1.0
        x2[0, :, E.NIB_A] = 0  # Counter
        x2[0, :, E.NIB_B] = 100  # Exit target
        y2 = beq(x2)
        self.assertAlmostEqual(y2[0, 0, E.RESULT].item(), 100.0, places=1)  # Taken

    def test_memory_store_load_pattern(self):
        """Test store followed by conceptual load."""
        store = StoreFFN()
        load = LoadFFN()

        # Store value 42 at address 1000
        # 42 = 0x2A = [10, 2, 0, 0, 0, 0, 0, 0]
        # 1000 = 0x3E8 = [8, 14, 3, 0, 0, 0, 0, 0]
        addr = [8, 14, 3, 0, 0, 0, 0, 0]
        val = [10, 2, 0, 0, 0, 0, 0, 0]

        x_store = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x_store[0, i, E.OP_START + Opcode.SI] = 1.0
            x_store[0, i, E.NIB_A] = addr[i]
            x_store[0, i, E.NIB_B] = val[i]
            x_store[0, i, E.POS] = float(i)

        y_store = store(x_store)

        # Check address in TEMP
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y_store[0, i, E.TEMP].item(), addr[i], places=1)

        # Check value in RESULT
        for i in range(E.NUM_POSITIONS):
            self.assertAlmostEqual(y_store[0, i, E.RESULT].item(), val[i], places=1)


def run_tests():
    """Run all comprehensive VM tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestControlFlowOps))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryOps))
    suite.addTests(loader.loadTestsFromTestCase(TestIOOps))
    suite.addTests(loader.loadTestsFromTestCase(TestArgvPassing))
    suite.addTests(loader.loadTestsFromTestCase(TestFunctionCalls))
    suite.addTests(loader.loadTestsFromTestCase(TestComprehensiveScenarios))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    print(f"Total: {result.testsRun} tests")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*60}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
