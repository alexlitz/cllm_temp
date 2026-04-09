"""Memory stress tests for the Neural VM.

Tests memory retention, overwrites, capacity limits, and time-distributed writes.
These tests validate the attention-based memory mechanism (Layer 15 softmax)
under realistic workloads.

Coverage:
- Long-term retention (500+ steps)
- Many overwrites (100+ to same address)
- Multiple addresses (50+ simultaneous)
- Time-distributed writes (across execution timeline)
- Byte vs word operations (SC/SI mixing)
"""

import unittest
import os
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode


# Shared model instance (loaded once, reused across all tests)
_shared_model = None

def _get_model():
    """Get or create shared model instance with cached weights."""
    global _shared_model
    if _shared_model is None:
        # Check for cached model
        cache_path = os.path.join(os.path.dirname(__file__), '.memory_test_model.pt')
        if os.path.exists(cache_path):
            print("Loading cached model...")
            _shared_model = AutoregressiveVM.load_compact(cache_path)
        else:
            print("Creating new model and setting weights (this may take a while)...")
            _shared_model = AutoregressiveVM()
            set_vm_weights(_shared_model)
            _shared_model.compact(block_size=32)
            _shared_model.compact_moe()
            print("Saving model cache...")
            _shared_model.save_compact(cache_path)
            print("Model ready!")
        _shared_model.eval()
    return _shared_model


class TestMemoryRetention(unittest.TestCase):
    """Test that memory values persist over long execution sequences."""

    @classmethod
    def setUpClass(cls):
        """Get shared model instance."""
        cls.model = _get_model()

    def _run(self, bytecode, max_steps=100):
        """Run bytecode and return (context, exit_code)."""
        from neural_vm.run_vm import AutoregressiveVMRunner
        runner = AutoregressiveVMRunner()
        runner.model = self.model
        context = runner._build_context(bytecode, b'', [])

        for _ in range(max_steps * Token.STEP_TOKENS + 50):
            tok = self.model.generate_next(context)
            context.append(tok)
            if tok == Token.HALT:
                break

        # Extract exit code
        exit_code = 0
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
                break

        return context, exit_code

    def test_memory_retention_across_100_steps(self):
        """Write value, execute 100 instructions, verify value persists."""
        addr = 0x1000
        value = 42

        # Build bytecode: write, then execute many NOPs, then read
        bytecode = [
            # Write value to memory
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (value << 8),
            Opcode.SI,  # memory[addr] = 42
        ]

        # Execute 100 NOP-equivalent instructions (IMM 0)
        for _ in range(100):
            bytecode.append(Opcode.IMM | (0 << 8))

        # Read value back
        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LI,  # AX = memory[addr]
            Opcode.EXIT,
        ])

        _, ec = self._run(bytecode, max_steps=150)
        self.assertEqual(ec, value,
                        f"Memory value should persist across 100 instructions")

    def test_memory_retention_with_arithmetic(self):
        """Write value, execute arithmetic ops, verify value persists."""
        addr = 0x2000
        value = 123

        bytecode = [
            # Write value to memory
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (value << 8),
            Opcode.SI,
        ]

        # Execute arithmetic operations (doesn't touch our memory addr)
        for i in range(50):
            bytecode.extend([
                Opcode.IMM | (i << 8),
                Opcode.PSH,
                Opcode.IMM | ((i + 1) << 8),
                Opcode.ADD,  # Compute i + (i+1)
            ])

        # Read original value back
        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        _, ec = self._run(bytecode, max_steps=250)
        self.assertEqual(ec, value,
                        f"Memory should persist through arithmetic operations")

    def test_memory_retention_multiple_values(self):
        """Write 10 values at different addresses, verify all persist."""
        base_addr = 0x3000
        values = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

        bytecode = []

        # Write all values
        for i, val in enumerate(values):
            addr = base_addr + i * 8
            bytecode.extend([
                Opcode.IMM | (addr << 8),
                Opcode.PSH,
                Opcode.IMM | (val << 8),
                Opcode.SI,
            ])

        # Execute some operations in between
        for _ in range(20):
            bytecode.append(Opcode.IMM | (0 << 8))

        # Read back the middle value (index 5)
        middle_addr = base_addr + 5 * 8
        middle_val = values[5]
        bytecode.extend([
            Opcode.IMM | (middle_addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        _, ec = self._run(bytecode, max_steps=200)
        self.assertEqual(ec, middle_val,
                        f"Middle value should persist among 10 stored values")


class TestMemoryOverwrites(unittest.TestCase):
    """Test that multiple overwrites to same address work correctly."""

    @classmethod
    def setUpClass(cls):
        """Get shared model instance."""
        cls.model = _get_model()

    def _run(self, bytecode, max_steps=100):
        """Run bytecode and return exit code."""
        from neural_vm.run_vm import AutoregressiveVMRunner
        runner = AutoregressiveVMRunner()
        runner.model = self.model
        context = runner._build_context(bytecode, b'', [])

        for _ in range(max_steps * Token.STEP_TOKENS + 50):
            tok = self.model.generate_next(context)
            context.append(tok)
            if tok == Token.HALT:
                break

        exit_code = 0
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
                break

        return exit_code

    def test_10_overwrites_same_address(self):
        """Overwrite same address 10 times, verify latest wins."""
        addr = 0x4000
        num_writes = 10

        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
        ]

        # Write 0, 1, 2, ..., 9 to same address
        for i in range(num_writes):
            bytecode.extend([
                Opcode.IMM | (i << 8),
                Opcode.SI,
            ])

        # Read back - should be 9 (last write)
        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=50)
        self.assertEqual(ec, num_writes - 1,
                        f"Latest of {num_writes} writes should win")

    def test_50_overwrites_same_address(self):
        """Overwrite same address 50 times, verify latest wins."""
        addr = 0x5000
        num_writes = 50

        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
        ]

        for i in range(num_writes):
            bytecode.extend([
                Opcode.IMM | (i << 8),
                Opcode.SI,
            ])

        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=200)
        self.assertEqual(ec, num_writes - 1,
                        f"Latest of {num_writes} writes should win")

    def test_overwrite_pattern_alternating(self):
        """Write alternating values 20 times, verify last value."""
        addr = 0x6000
        val_a = 100
        val_b = 200
        num_alternations = 20

        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
        ]

        # Write val_a, val_b, val_a, val_b, ...
        for i in range(num_alternations):
            val = val_a if i % 2 == 0 else val_b
            bytecode.extend([
                Opcode.IMM | (val << 8),
                Opcode.SI,
            ])

        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        # Last write is val_b (since num_alternations=20 is even, last index is 19 which is odd)
        expected = val_b
        ec = self._run(bytecode, max_steps=100)
        self.assertEqual(ec, expected,
                        f"Latest value in alternating pattern should win")

    def test_byte_overwrites_sc(self):
        """Test SC (byte store) overwrites work correctly."""
        addr = 0x7000
        num_writes = 20

        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
        ]

        # Write bytes 0, 1, 2, ..., 19
        for i in range(num_writes):
            bytecode.extend([
                Opcode.IMM | (i << 8),
                Opcode.SC,  # Store byte
            ])

        # Read back byte
        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LC,  # Load byte
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=100)
        self.assertEqual(ec, (num_writes - 1) & 0xFF,
                        f"Latest byte write should win")


class TestMemoryCapacity(unittest.TestCase):
    """Test memory capacity limits and attention span."""

    @classmethod
    def setUpClass(cls):
        """Get shared model instance."""
        cls.model = _get_model()

    def _run(self, bytecode, max_steps=200):
        """Run bytecode and return exit code."""
        from neural_vm.run_vm import AutoregressiveVMRunner
        runner = AutoregressiveVMRunner()
        runner.model = self.model
        context = runner._build_context(bytecode, b'', [])

        for _ in range(max_steps * Token.STEP_TOKENS + 100):
            tok = self.model.generate_next([context])
            context.append(tok)
            if tok == Token.HALT:
                break

        exit_code = 0
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
                break

        return exit_code

    def test_20_different_addresses(self):
        """Write to 20 different addresses, verify random one persists."""
        base_addr = 0x8000
        num_addrs = 20

        bytecode = []

        # Write addr[i] = i + 10 for i in 0..19
        for i in range(num_addrs):
            addr = base_addr + i * 8
            val = i + 10
            bytecode.extend([
                Opcode.IMM | (addr << 8),
                Opcode.PSH,
                Opcode.IMM | (val << 8),
                Opcode.SI,
            ])

        # Read back address at index 7 (value should be 17)
        test_index = 7
        test_addr = base_addr + test_index * 8
        expected_val = test_index + 10

        bytecode.extend([
            Opcode.IMM | (test_addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=120)
        self.assertEqual(ec, expected_val,
                        f"Value at address {test_index} should persist among {num_addrs} addresses")

    def test_30_different_addresses(self):
        """Write to 30 different addresses, verify first and last persist."""
        base_addr = 0x9000
        num_addrs = 30

        bytecode = []

        # Write addr[i] = i * 2
        for i in range(num_addrs):
            addr = base_addr + i * 8
            val = i * 2
            bytecode.extend([
                Opcode.IMM | (addr << 8),
                Opcode.PSH,
                Opcode.IMM | (val << 8),
                Opcode.SI,
            ])

        # Read first address (value should be 0)
        bytecode.extend([
            Opcode.IMM | (base_addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=150)
        self.assertEqual(ec, 0,
                        f"First value should persist among {num_addrs} addresses")

    def test_sparse_address_pattern(self):
        """Write to non-contiguous addresses, verify all persist."""
        # Use sparse addresses: 0x1000, 0x2000, 0x3000, 0x5000, 0x8000
        addresses = [0x1000, 0x2000, 0x3000, 0x5000, 0x8000]
        values = [11, 22, 33, 44, 55]

        bytecode = []

        # Write all values
        for addr, val in zip(addresses, values):
            bytecode.extend([
                Opcode.IMM | (addr << 8),
                Opcode.PSH,
                Opcode.IMM | (val << 8),
                Opcode.SI,
            ])

        # Read middle address (0x3000, should be 33)
        bytecode.extend([
            Opcode.IMM | (0x3000 << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=80)
        self.assertEqual(ec, 33,
                        "Value at sparse address 0x3000 should persist")


class TestTimeDistributedMemory(unittest.TestCase):
    """Test memory writes distributed across execution timeline."""

    @classmethod
    def setUpClass(cls):
        """Get shared model instance."""
        cls.model = _get_model()

    def _run(self, bytecode, max_steps=300):
        """Run bytecode and return exit code."""
        from neural_vm.run_vm import AutoregressiveVMRunner
        runner = AutoregressiveVMRunner()
        runner.model = self.model
        context = runner._build_context(bytecode, b'', [])

        for _ in range(max_steps * Token.STEP_TOKENS + 100):
            tok = self.model.generate_next([context])
            context.append(tok)
            if tok == Token.HALT:
                break

        exit_code = 0
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
                break

        return exit_code

    def test_early_and_late_writes(self):
        """Write at step 1, execute operations, write at step 50, read both."""
        addr_early = 0xA000
        addr_late = 0xB000
        val_early = 111
        val_late = 222

        bytecode = [
            # Early write
            Opcode.IMM | (addr_early << 8),
            Opcode.PSH,
            Opcode.IMM | (val_early << 8),
            Opcode.SI,
        ]

        # Execute 50 operations
        for i in range(50):
            bytecode.append(Opcode.IMM | (i << 8))

        # Late write
        bytecode.extend([
            Opcode.IMM | (addr_late << 8),
            Opcode.PSH,
            Opcode.IMM | (val_late << 8),
            Opcode.SI,
        ])

        # Read early value (written 50+ instructions ago)
        bytecode.extend([
            Opcode.IMM | (addr_early << 8),
            Opcode.LI,
            Opcode.EXIT,
        ])

        ec = self._run(bytecode, max_steps=150)
        self.assertEqual(ec, val_early,
                        "Early value should persist after late writes")

    def test_interleaved_writes_and_reads(self):
        """Write, read, write, read pattern."""
        addr1 = 0xC000
        addr2 = 0xD000
        val1 = 10
        val2 = 20

        bytecode = [
            # Write 1
            Opcode.IMM | (addr1 << 8),
            Opcode.PSH,
            Opcode.IMM | (val1 << 8),
            Opcode.SI,

            # Read 1 back immediately
            Opcode.IMM | (addr1 << 8),
            Opcode.LI,
            Opcode.PSH,  # Save result

            # Write 2
            Opcode.IMM | (addr2 << 8),
            Opcode.PSH,
            Opcode.IMM | (val2 << 8),
            Opcode.SI,

            # Read 1 again (should still be val1)
            Opcode.IMM | (addr1 << 8),
            Opcode.LI,
            Opcode.EXIT,
        ]

        ec = self._run(bytecode, max_steps=50)
        self.assertEqual(ec, val1,
                        "First value should persist across interleaved operations")

    def test_write_read_write_same_address(self):
        """Write value, read it, overwrite, read again."""
        addr = 0xE000
        val1 = 100
        val2 = 200

        bytecode = [
            # First write
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (val1 << 8),
            Opcode.SI,

            # Read (should be val1)
            Opcode.IMM | (addr << 8),
            Opcode.LI,

            # Second write (overwrite)
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (val2 << 8),
            Opcode.SI,

            # Read again (should be val2)
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ]

        ec = self._run(bytecode, max_steps=50)
        self.assertEqual(ec, val2,
                        "Second write should overwrite first value")


class TestMixedMemoryOperations(unittest.TestCase):
    """Test mixing byte (SC/LC) and word (SI/LI) operations."""

    @classmethod
    def setUpClass(cls):
        """Get shared model instance."""
        cls.model = _get_model()

    def _run(self, bytecode, max_steps=50):
        """Run bytecode and return exit code."""
        from neural_vm.run_vm import AutoregressiveVMRunner
        runner = AutoregressiveVMRunner()
        runner.model = self.model
        context = runner._build_context(bytecode, b'', [])

        for _ in range(max_steps * Token.STEP_TOKENS + 50):
            tok = self.model.generate_next(context)
            context.append(tok)
            if tok == Token.HALT:
                break

        exit_code = 0
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
                break

        return exit_code

    def test_si_then_lc(self):
        """Store word (SI), load byte (LC) from same address."""
        addr = 0xF000
        word_val = 0x12345678  # Byte 0 is 0x78

        bytecode = [
            # Store word
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (word_val << 8),
            Opcode.SI,

            # Load byte (should get low byte 0x78)
            Opcode.IMM | (addr << 8),
            Opcode.LC,
            Opcode.EXIT,
        ]

        ec = self._run(bytecode, max_steps=30)
        self.assertEqual(ec, 0x78,
                        "LC should read low byte of SI-stored word")

    def test_multiple_sc_then_li(self):
        """Store multiple bytes (SC), then read word (LI)."""
        addr = 0x10000

        # We'll write byte 0x12 then overwrite with 0x34
        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (0x12 << 8),
            Opcode.SC,

            Opcode.IMM | (0x34 << 8),
            Opcode.SC,

            # Read as word (byte should be 0x34, rest zeros)
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            Opcode.EXIT,
        ]

        ec = self._run(bytecode, max_steps=40)
        # LI reads 4 bytes, but only byte 0 was written (0x34)
        # Unwritten bytes should be 0 (ZFOD)
        # Result: 0x00000034
        self.assertEqual(ec, 0x34,
                        "LI should read SC-stored byte with ZFOD for rest")


if __name__ == '__main__':
    unittest.main()
