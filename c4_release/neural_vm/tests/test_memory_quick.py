"""Quick memory validation test (minimal version of memory stress tests).

This is a lightweight version to quickly validate memory operations work.
For comprehensive testing, use test_memory_stress.py
"""

import unittest
import torch
from neural_vm.embedding import Opcode


class TestMemoryQuick(unittest.TestCase):
    """Quick memory validation tests using the runner."""

    def test_basic_memory_write_read(self):
        """Write value to memory, read it back."""
        from neural_vm.run_vm import AutoregressiveVMRunner

        addr = 0x1000
        value = 42

        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (value << 8),
            Opcode.SI,  # memory[addr] = 42
            Opcode.IMM | (addr << 8),
            Opcode.LI,  # AX = memory[addr]
            Opcode.EXIT,
        ]

        runner = AutoregressiveVMRunner()
        # Move model to GPU if available
        if torch.cuda.is_available():
            runner.model = runner.model.cuda()
        output, exit_code = runner.run(bytecode, max_steps=10)

        self.assertEqual(exit_code, value,
                        "Memory write-read should preserve value")

    def test_memory_overwrite(self):
        """Two writes to same address, verify latest wins."""
        from neural_vm.run_vm import AutoregressiveVMRunner

        addr = 0x1000

        bytecode = [
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (10 << 8),
            Opcode.SI,  # memory[addr] = 10
            Opcode.IMM | (20 << 8),
            Opcode.SI,  # memory[addr] = 20 (overwrite)
            Opcode.IMM | (addr << 8),
            Opcode.LI,  # AX = memory[addr]
            Opcode.EXIT,
        ]

        runner = AutoregressiveVMRunner()
        if torch.cuda.is_available():
            runner.model = runner.model.cuda()
        output, exit_code = runner.run(bytecode, max_steps=12)

        self.assertEqual(exit_code, 20,
                        "Latest write should win")

    def test_multiple_addresses(self):
        """Write to 3 addresses, read back middle one."""
        from neural_vm.run_vm import AutoregressiveVMRunner

        bytecode = [
            # Write 3 addresses
            Opcode.IMM | (0x1000 << 8),
            Opcode.PSH,
            Opcode.IMM | (11 << 8),
            Opcode.SI,

            Opcode.IMM | (0x2000 << 8),
            Opcode.PSH,
            Opcode.IMM | (22 << 8),
            Opcode.SI,

            Opcode.IMM | (0x3000 << 8),
            Opcode.PSH,
            Opcode.IMM | (33 << 8),
            Opcode.SI,

            # Read middle address
            Opcode.IMM | (0x2000 << 8),
            Opcode.LI,
            Opcode.EXIT,
        ]

        runner = AutoregressiveVMRunner()
        if torch.cuda.is_available():
            runner.model = runner.model.cuda()
        output, exit_code = runner.run(bytecode, max_steps=20)

        self.assertEqual(exit_code, 22,
                        "Middle address should retain value")


if __name__ == '__main__':
    unittest.main()
