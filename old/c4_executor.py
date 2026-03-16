"""
C4 Executor with minimal state and memory tools.

State: Just 4 registers (PC, SP, BP, AX)
Memory: Accessed via read/write tools
"""

import torch
import struct
from dataclasses import dataclass
from typing import Tuple
from c4_vm import Op, CODE_BASE, DATA_BASE, STACK_BASE, STACK_SIZE, MEMORY_SIZE


@dataclass
class Registers:
    """Minimal C4 state - just the 4 registers."""
    pc: int  # program counter
    sp: int  # stack pointer
    bp: int  # base pointer
    ax: int  # accumulator

    def to_tensor(self, device='cpu') -> torch.Tensor:
        """Pack registers into tensor (4 int64s = 32 bytes)."""
        return torch.tensor([self.pc, self.sp, self.bp, self.ax],
                          dtype=torch.long, device=device)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> 'Registers':
        return Registers(pc=t[0].item(), sp=t[1].item(),
                        bp=t[2].item(), ax=t[3].item())


class Memory:
    """Memory with read/write tools."""

    def __init__(self):
        self.data = bytearray(MEMORY_SIZE)

    def read_byte(self, addr: int) -> int:
        """Tool: Read 1 byte from memory."""
        if 0 <= addr < MEMORY_SIZE:
            return self.data[addr]
        return 0

    def read_int(self, addr: int) -> int:
        """Tool: Read 8 bytes (int64) from memory."""
        if 0 <= addr < MEMORY_SIZE - 7:
            return struct.unpack_from('<q', self.data, addr)[0]
        return 0

    def write_byte(self, addr: int, value: int):
        """Tool: Write 1 byte to memory."""
        if 0 <= addr < MEMORY_SIZE:
            self.data[addr] = value & 0xFF

    def write_int(self, addr: int, value: int):
        """Tool: Write 8 bytes (int64) to memory."""
        if 0 <= addr < MEMORY_SIZE - 7:
            struct.pack_into('<q', self.data, addr, value)

    def load_code(self, instructions: list):
        """Load bytecode into code segment."""
        for i, instr in enumerate(instructions):
            struct.pack_into('<q', self.data, CODE_BASE + i * 8, instr)


class C4Executor:
    """
    Execute C4 opcodes using registers + memory tools.

    The executor has:
    - State: 4 registers (PC, SP, BP, AX)
    - Tools: read_byte, read_int, write_byte, write_int

    Each step:
    1. Fetch instruction using read_int(PC)
    2. Decode opcode and immediate
    3. Execute using register ops + memory tools
    4. Update registers
    """

    def __init__(self):
        self.regs = Registers(
            pc=CODE_BASE,
            sp=STACK_BASE + STACK_SIZE,
            bp=STACK_BASE + STACK_SIZE,
            ax=0
        )
        self.mem = Memory()
        self.halted = False

    def reset(self):
        self.regs = Registers(
            pc=CODE_BASE,
            sp=STACK_BASE + STACK_SIZE,
            bp=STACK_BASE + STACK_SIZE,
            ax=0
        )
        self.mem = Memory()
        self.halted = False

    def load(self, code: list):
        """Load bytecode."""
        self.reset()
        self.mem.load_code(code)

    # =========================================================================
    # Memory Tools (what the transformer would call)
    # =========================================================================

    def tool_read_int(self, addr: int) -> int:
        """Read int64 from memory."""
        return self.mem.read_int(addr)

    def tool_write_int(self, addr: int, value: int):
        """Write int64 to memory."""
        self.mem.write_int(addr, value)

    def tool_read_byte(self, addr: int) -> int:
        """Read byte from memory."""
        return self.mem.read_byte(addr)

    def tool_write_byte(self, addr: int, value: int):
        """Write byte to memory."""
        self.mem.write_byte(addr, value)

    # =========================================================================
    # Stack operations (use memory tools internally)
    # =========================================================================

    def push(self, value: int):
        """Push onto stack: SP -= 8, write value."""
        self.regs.sp -= 8
        self.tool_write_int(self.regs.sp, value)

    def pop(self) -> int:
        """Pop from stack: read value, SP += 8."""
        value = self.tool_read_int(self.regs.sp)
        self.regs.sp += 8
        return value

    # =========================================================================
    # Step execution
    # =========================================================================

    def step(self) -> bool:
        """Execute one instruction. Returns False if halted."""
        if self.halted:
            return False

        # Fetch: use read tool
        instruction = self.tool_read_int(self.regs.pc)
        opcode = instruction & 0xFF
        imm = instruction >> 8

        # Default: advance PC
        self.regs.pc += 8

        # Execute based on opcode
        if opcode == Op.LEA:
            self.regs.ax = self.regs.bp + imm

        elif opcode == Op.IMM:
            self.regs.ax = imm

        elif opcode == Op.JMP:
            self.regs.pc = imm

        elif opcode == Op.JSR:
            self.push(self.regs.pc)
            self.regs.pc = imm

        elif opcode == Op.BZ:
            if self.regs.ax == 0:
                self.regs.pc = imm

        elif opcode == Op.BNZ:
            if self.regs.ax != 0:
                self.regs.pc = imm

        elif opcode == Op.ENT:
            self.push(self.regs.bp)
            self.regs.bp = self.regs.sp
            self.regs.sp -= imm

        elif opcode == Op.ADJ:
            self.regs.sp += imm

        elif opcode == Op.LEV:
            self.regs.sp = self.regs.bp
            self.regs.bp = self.pop()
            self.regs.pc = self.pop()

        elif opcode == Op.LI:
            self.regs.ax = self.tool_read_int(self.regs.ax)

        elif opcode == Op.LC:
            self.regs.ax = self.tool_read_byte(self.regs.ax)

        elif opcode == Op.SI:
            addr = self.pop()
            self.tool_write_int(addr, self.regs.ax)

        elif opcode == Op.SC:
            addr = self.pop()
            self.tool_write_byte(addr, self.regs.ax)

        elif opcode == Op.PSH:
            self.push(self.regs.ax)

        elif opcode == Op.ADD:
            self.regs.ax = self.pop() + self.regs.ax

        elif opcode == Op.SUB:
            self.regs.ax = self.pop() - self.regs.ax

        elif opcode == Op.MUL:
            self.regs.ax = self.pop() * self.regs.ax

        elif opcode == Op.DIV:
            if self.regs.ax != 0:
                self.regs.ax = self.pop() // self.regs.ax
            else:
                self.pop()

        elif opcode == Op.MOD:
            if self.regs.ax != 0:
                self.regs.ax = self.pop() % self.regs.ax
            else:
                self.pop()

        elif opcode == Op.OR:
            self.regs.ax = self.pop() | self.regs.ax

        elif opcode == Op.XOR:
            self.regs.ax = self.pop() ^ self.regs.ax

        elif opcode == Op.AND:
            self.regs.ax = self.pop() & self.regs.ax

        elif opcode == Op.SHL:
            self.regs.ax = self.pop() << self.regs.ax

        elif opcode == Op.SHR:
            self.regs.ax = self.pop() >> self.regs.ax

        elif opcode == Op.EQ:
            self.regs.ax = 1 if self.pop() == self.regs.ax else 0

        elif opcode == Op.NE:
            self.regs.ax = 1 if self.pop() != self.regs.ax else 0

        elif opcode == Op.LT:
            self.regs.ax = 1 if self.pop() < self.regs.ax else 0

        elif opcode == Op.GT:
            self.regs.ax = 1 if self.pop() > self.regs.ax else 0

        elif opcode == Op.LE:
            self.regs.ax = 1 if self.pop() <= self.regs.ax else 0

        elif opcode == Op.GE:
            self.regs.ax = 1 if self.pop() >= self.regs.ax else 0

        elif opcode == Op.EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps=10000) -> int:
        """Run until EXIT or max_steps."""
        for _ in range(max_steps):
            if not self.step():
                break
        return self.regs.ax


def test_executor():
    """Test the simplified executor."""
    from c4_vm import program_simple_arithmetic, program_sum

    print("=" * 60)
    print("C4 EXECUTOR TEST")
    print("=" * 60)
    print()
    print("State: 4 registers (PC, SP, BP, AX) = 32 bytes")
    print("Tools: read_int, write_int, read_byte, write_byte")
    print()

    tests = [
        ("(3+4)*5", program_simple_arithmetic, 35),
        ("sum(1)", lambda: program_sum(1), 1),
        ("sum(10)", lambda: program_sum(10), 55),
        ("sum(100)", lambda: program_sum(100), 5050),
    ]

    executor = C4Executor()

    for name, prog_fn, expected in tests:
        executor.load(prog_fn())
        result = executor.run()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name:15s} = {result}")

    print()
    print("Registers after last run:")
    print(f"  PC = {executor.regs.pc:#x}")
    print(f"  SP = {executor.regs.sp:#x}")
    print(f"  BP = {executor.regs.bp:#x}")
    print(f"  AX = {executor.regs.ax}")


if __name__ == "__main__":
    test_executor()
