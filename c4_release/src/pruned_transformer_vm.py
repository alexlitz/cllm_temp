"""
Transformer VM with Memory Pruning

Optimizations:
1. Prune overwritten memory - only keep latest value per address
2. Prune expired register states - only keep current state
3. Sparse KV cache - track only live memory locations

This dramatically reduces memory for long-running programs.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def silu(x):
    """SiLU activation."""
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    """Exact multiplication using SwiGLU identity."""
    return silu(a) * b + silu(-a) * (-b)


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    address: int
    value: int
    write_step: int  # When was this written
    last_read_step: int  # When was this last read


class PrunedMemory:
    """
    Memory with automatic pruning of overwritten values.

    Maintains a sparse dictionary of live memory locations.
    When memory is overwritten, the old entry is removed.
    """

    def __init__(self, max_entries: int = 65536):
        self.entries: Dict[int, MemoryEntry] = {}
        self.max_entries = max_entries
        self.current_step = 0

        # Stats
        self.total_writes = 0
        self.pruned_overwrites = 0

    def write(self, addr: int, value: int):
        """Write value to address, pruning old value if exists."""
        self.total_writes += 1

        if addr in self.entries:
            self.pruned_overwrites += 1

        self.entries[addr] = MemoryEntry(
            address=addr,
            value=value,
            write_step=self.current_step,
            last_read_step=self.current_step
        )

        # If we exceed max entries, prune least recently used
        if len(self.entries) > self.max_entries:
            self._prune_lru()

    def read(self, addr: int) -> int:
        """Read value from address."""
        if addr in self.entries:
            self.entries[addr].last_read_step = self.current_step
            return self.entries[addr].value
        return 0

    def _prune_lru(self):
        """Prune least recently used entries."""
        # Sort by last_read_step, remove oldest 10%
        entries_list = list(self.entries.items())
        entries_list.sort(key=lambda x: x[1].last_read_step)

        prune_count = len(entries_list) // 10
        for addr, _ in entries_list[:prune_count]:
            del self.entries[addr]

    def step(self):
        """Advance the step counter."""
        self.current_step += 1

    def stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'live_entries': len(self.entries),
            'total_writes': self.total_writes,
            'pruned_overwrites': self.pruned_overwrites,
            'prune_ratio': self.pruned_overwrites / max(1, self.total_writes)
        }


class PrunedRegisterFile:
    """
    Register file that only keeps current state.

    Unlike a full trace that keeps all register states,
    this only tracks the current values.
    """

    def __init__(self):
        self.pc = 0
        self.sp = 0x30000  # Stack pointer
        self.bp = 0x30000  # Base pointer
        self.ax = 0        # Accumulator

        # History for debugging (optional, limited size)
        self.history_size = 100
        self.history: List[Tuple[int, int, int, int]] = []

    def snapshot(self):
        """Record current state (limited history)."""
        if len(self.history) >= self.history_size:
            self.history.pop(0)  # Remove oldest
        self.history.append((self.pc, self.sp, self.bp, self.ax))

    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current register state."""
        return (self.pc, self.sp, self.bp, self.ax)


class NeuralALU:
    """Neural ALU using SwiGLU for multiplication."""

    def __init__(self):
        # Reciprocal table for division (256 entries)
        self.recip_table = torch.zeros(256)
        for i in range(256):
            x = 0.5 + i / 256.0  # [0.5, 1.0)
            self.recip_table[i] = 1.0 / x

    def mul(self, a: int, b: int) -> int:
        """Multiply using SwiGLU."""
        a_t = torch.tensor(float(a))
        b_t = torch.tensor(float(b))
        result = swiglu_mul(a_t, b_t)
        return int(round(result.item())) & 0xFFFFFFFF

    def div(self, a: int, b: int) -> int:
        """Integer division using Newton-Raphson."""
        if b == 0:
            return 0

        sign = 1
        if a < 0:
            sign = -sign
            a = -a
        if b < 0:
            sign = -sign
            b = -b

        # Normalize b to [0.5, 1.0)
        exp = 0
        temp = float(b)
        while temp >= 1.0:
            temp *= 0.5
            exp += 1
        while temp < 0.5:
            temp *= 2.0
            exp -= 1

        # Table lookup
        idx = int((temp - 0.5) * 256)
        idx = max(0, min(255, idx))
        y = float(self.recip_table[idx])

        # Newton iterations
        y = y * (2.0 - temp * y)
        y = y * (2.0 - temp * y)

        # Scale back
        y = y * (2.0 ** (-exp))

        result = int(a * y * sign)
        return result


class PrunedTransformerVM:
    """
    Transformer VM with memory and register pruning.

    Key optimizations:
    - Memory is sparse (only live locations stored)
    - Overwritten memory is immediately pruned
    - Only current register state is maintained
    - Neural operations (MUL/DIV) use SwiGLU
    """

    def __init__(self):
        self.memory = PrunedMemory()
        self.regs = PrunedRegisterFile()
        self.alu = NeuralALU()
        self.code: List[Tuple[int, int]] = []
        self.halted = False

        # I/O
        self.stdin_data = ""
        self.stdin_pos = 0
        self.stdout_data: List[int] = []

        # Operation counts
        self.op_counts: Dict[int, int] = {}
        self.neural_ops = {'mul': 0, 'div': 0}

    def load(self, bytecode: List[int], data: Optional[List[int]] = None):
        """Load bytecode and data."""
        self.code = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

        if data:
            for i, b in enumerate(data):
                self.memory.write(0x10000 + i, b & 0xFF)

    def set_stdin(self, data: str):
        """Set stdin data."""
        self.stdin_data = data
        self.stdin_pos = 0

    def step(self) -> bool:
        """Execute one instruction. Returns True if should continue."""
        if self.halted:
            return False

        idx = self.regs.pc // 8
        if idx >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[idx]
        self.regs.pc += 8
        self.memory.step()

        # Count operations
        self.op_counts[op] = self.op_counts.get(op, 0) + 1

        # I/O
        if op == 64:  # GETCHAR
            if self.stdin_pos < len(self.stdin_data):
                self.regs.ax = ord(self.stdin_data[self.stdin_pos])
                self.stdin_pos += 1
            else:
                self.regs.ax = 0xFFFFFFFF

        elif op == 65:  # PUTCHAR
            c = self.memory.read(self.regs.sp)
            self.stdout_data.append(c & 0xFF)
            self.regs.ax = c

        # Control flow
        elif op == 0:  # LEA
            self.regs.ax = self.regs.bp + imm
        elif op == 1:  # IMM
            self.regs.ax = imm
        elif op == 2:  # JMP
            self.regs.pc = imm
        elif op == 3:  # JSR
            self.regs.sp -= 8
            self.memory.write(self.regs.sp, self.regs.pc)
            self.regs.pc = imm
        elif op == 4:  # BZ
            if self.regs.ax == 0:
                self.regs.pc = imm
        elif op == 5:  # BNZ
            if self.regs.ax != 0:
                self.regs.pc = imm
        elif op == 6:  # ENT
            self.regs.sp -= 8
            self.memory.write(self.regs.sp, self.regs.bp)
            self.regs.bp = self.regs.sp
            self.regs.sp -= imm
        elif op == 7:  # ADJ
            self.regs.sp += imm
        elif op == 8:  # LEV
            self.regs.sp = self.regs.bp
            self.regs.bp = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.pc = self.memory.read(self.regs.sp)
            self.regs.sp += 8

        # Memory
        elif op == 9:  # LI (load int)
            self.regs.ax = self.memory.read(self.regs.ax)
        elif op == 10:  # LC (load char)
            self.regs.ax = self.memory.read(self.regs.ax) & 0xFF
        elif op == 11:  # SI (store int)
            addr = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.memory.write(addr, self.regs.ax)
        elif op == 12:  # SC (store char)
            addr = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.memory.write(addr, self.regs.ax & 0xFF)
        elif op == 13:  # PSH
            self.regs.sp -= 8
            self.memory.write(self.regs.sp, self.regs.ax)

        # Bitwise
        elif op == 14:  # OR
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = a | self.regs.ax
        elif op == 15:  # XOR
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = a ^ self.regs.ax
        elif op == 16:  # AND
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = a & self.regs.ax

        # Comparison
        elif op == 17:  # EQ
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = 1 if a == self.regs.ax else 0
        elif op == 18:  # NE
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = 1 if a != self.regs.ax else 0
        elif op == 19:  # LT
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = 1 if a < self.regs.ax else 0
        elif op == 20:  # GT
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = 1 if a > self.regs.ax else 0
        elif op == 21:  # LE
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = 1 if a <= self.regs.ax else 0
        elif op == 22:  # GE
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = 1 if a >= self.regs.ax else 0

        # Shift
        elif op == 23:  # SHL
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = (a << self.regs.ax) & 0xFFFFFFFF
        elif op == 24:  # SHR
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = a >> self.regs.ax

        # Arithmetic
        elif op == 25:  # ADD
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = (a + self.regs.ax) & 0xFFFFFFFF
        elif op == 26:  # SUB
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = (a - self.regs.ax) & 0xFFFFFFFF
        elif op == 27:  # MUL - Neural SwiGLU
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = self.alu.mul(a, self.regs.ax)
            self.neural_ops['mul'] += 1
        elif op == 28:  # DIV - Neural Newton-Raphson
            a = self.memory.read(self.regs.sp)
            self.regs.sp += 8
            self.regs.ax = self.alu.div(a, self.regs.ax)
            self.neural_ops['div'] += 1
        elif op == 29:  # MOD
            a = self.memory.read(self.regs.sp)
            b = self.regs.ax
            self.regs.sp += 8
            if b != 0:
                self.regs.ax = a % b
            else:
                self.regs.ax = 0

        # Exit
        elif op == 38:  # EXIT
            self.halted = True
            return False

        return True

    def run(self, max_steps: int = 10000000) -> int:
        """Run until halt or max steps."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.regs.ax

    def get_stdout(self) -> str:
        """Get stdout as string."""
        return ''.join(chr(b) for b in self.stdout_data)

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'memory': self.memory.stats(),
            'neural_ops': self.neural_ops,
            'op_counts': self.op_counts,
            'total_ops': sum(self.op_counts.values())
        }


def run_program(source: str, stdin: str = "") -> Tuple[str, Dict]:
    """Run C source through pruned transformer VM."""
    from src.compiler import compile_c

    bytecode, data = compile_c(source)

    vm = PrunedTransformerVM()
    vm.load(bytecode, data)
    vm.set_stdin(stdin)

    vm.run()

    return vm.get_stdout(), vm.get_stats()


if __name__ == "__main__":
    # Test with ELIZA
    with open("eliza_simple.c") as f:
        source = f.read()

    stdin = "hello\ni feel sad\nbye\n"

    output, stats = run_program(source, stdin)

    print("OUTPUT:")
    print(output)
    print()
    print("STATS:")
    print(f"  Memory entries: {stats['memory']['live_entries']}")
    print(f"  Memory writes: {stats['memory']['total_writes']}")
    print(f"  Pruned (overwrites): {stats['memory']['pruned_overwrites']}")
    print(f"  Prune ratio: {stats['memory']['prune_ratio']:.1%}")
    print()
    print(f"  Neural MUL: {stats['neural_ops']['mul']}")
    print(f"  Neural DIV: {stats['neural_ops']['div']}")
    print(f"  Total ops: {stats['total_ops']}")
