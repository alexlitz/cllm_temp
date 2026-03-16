"""
Transformer VM with Principled Attention-Based Memory Pruning

Instead of tracking addresses explicitly, we use attention mechanics:
- Keys encode memory locations
- Values store data
- Contributions determine what to keep

Pruning is based on attention weights, not address matching.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    return silu(a) * b + silu(-a) * (-b)


class AttentionMemory:
    """
    Memory using attention mechanics with contribution-based pruning.

    Each memory location is a (key, value) pair.
    Keys are binary-encoded addresses.
    Pruning removes entries with negligible attention contribution.
    """

    def __init__(self, num_bits: int = 20, scale: float = 100.0,
                 prune_threshold: float = 1e-8, max_entries: int = 100000):
        self.num_bits = num_bits
        self.scale = scale
        self.prune_threshold = prune_threshold
        self.max_entries = max_entries

        # Storage as tensors for batched operations
        self.keys = torch.zeros(0, num_bits, dtype=torch.float32)
        self.values = torch.zeros(0, dtype=torch.float32)
        self.addresses = []  # Track original addresses for debugging

        # Stats
        self.total_writes = 0
        self.total_reads = 0
        self.prune_events = 0
        self.entries_pruned = 0

    def _encode_address(self, addr: int) -> torch.Tensor:
        """Encode address as binary vector with ±scale."""
        bits = []
        for b in range(self.num_bits):
            bit = (addr >> b) & 1
            bits.append(self.scale if bit else -self.scale)
        return torch.tensor(bits, dtype=torch.float32)

    def write(self, addr: int, value: int):
        """Write value to address."""
        self.total_writes += 1
        key = self._encode_address(addr)

        # Check if address exists (via attention)
        if len(self.keys) > 0:
            scores = torch.matmul(self.keys, key)
            weights = F.softmax(scores / 10.0, dim=0)
            max_weight = weights.max().item()

            # If existing entry has high weight, update it
            if max_weight > 0.99:
                idx = weights.argmax().item()
                self.values[idx] = float(value)
                return

        # Add new entry
        self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)
        self.values = torch.cat([self.values, torch.tensor([float(value)])])
        self.addresses.append(addr)

        # Prune if needed
        if len(self.keys) > self.max_entries:
            self._prune_by_contribution()

    def read(self, addr: int) -> int:
        """Read value from address using attention."""
        self.total_reads += 1

        if len(self.keys) == 0:
            return 0

        query = self._encode_address(addr)
        scores = torch.matmul(self.keys, query)
        weights = F.softmax(scores / 10.0, dim=0)

        # Weighted sum of values
        result = torch.sum(weights * self.values)
        return int(round(result.item()))

    def _prune_by_contribution(self):
        """Prune entries with low potential contribution."""
        if len(self.keys) < 100:
            return

        self.prune_events += 1

        # Compute self-attention scores for each entry
        # High self-score = unique address, keep it
        # Low self-score = overshadowed by similar entry, prune it

        # For each key, compute its max attention from any other key
        scores = torch.matmul(self.keys, self.keys.T) / 10.0

        # Zero out diagonal (self-attention)
        scores.fill_diagonal_(-float('inf'))

        # Max attention each entry receives from others
        max_received = scores.max(dim=0).values

        # Entries that receive high attention from others are redundant
        # (another entry at same/similar address overshadows them)
        keep_mask = max_received < 5.0  # Threshold for "overshadowed"

        # Always keep at least some entries
        if keep_mask.sum() < 100:
            # Keep top 100 by recency (last entries)
            keep_mask[-100:] = True

        before = len(self.keys)
        self.keys = self.keys[keep_mask]
        self.values = self.values[keep_mask]
        self.addresses = [a for a, k in zip(self.addresses, keep_mask.tolist()) if k]
        self.entries_pruned += before - len(self.keys)

    def stats(self) -> Dict:
        return {
            'live_entries': len(self.keys),
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'prune_events': self.prune_events,
            'entries_pruned': self.entries_pruned,
            'prune_ratio': self.entries_pruned / max(1, self.total_writes)
        }


class NeuralALU:
    """Neural ALU with SwiGLU multiply and Newton-Raphson divide."""

    def __init__(self):
        self.recip_table = torch.zeros(256)
        for i in range(256):
            x = 0.5 + i / 256.0
            self.recip_table[i] = 1.0 / x

    def mul(self, a: int, b: int) -> int:
        result = swiglu_mul(torch.tensor(float(a)), torch.tensor(float(b)))
        return int(round(result.item())) & 0xFFFFFFFF

    def div(self, a: int, b: int) -> int:
        if b == 0:
            return 0
        sign = 1
        if a < 0:
            sign, a = -sign, -a
        if b < 0:
            sign, b = -sign, -b

        # Normalize and lookup
        exp = 0
        temp = float(b)
        while temp >= 1.0:
            temp *= 0.5
            exp += 1
        while temp < 0.5:
            temp *= 2.0
            exp -= 1

        idx = min(255, max(0, int((temp - 0.5) * 256)))
        y = float(self.recip_table[idx])
        y = y * (2.0 - temp * y)
        y = y * (2.0 - temp * y)
        y = y * (2.0 ** (-exp))

        return int(a * y * sign)


class AttentionMemoryVM:
    """
    VM using attention-based memory with principled pruning.
    """

    def __init__(self):
        self.memory = AttentionMemory()
        self.alu = NeuralALU()

        # Registers (not using attention for these - too frequent)
        self.pc = 0
        self.sp = 0x30000
        self.bp = 0x30000
        self.ax = 0

        self.code: List[Tuple[int, int]] = []
        self.halted = False

        # I/O
        self.stdin = ""
        self.stdin_pos = 0
        self.stdout: List[int] = []

        # Counts
        self.op_counts: Dict[int, int] = {}
        self.neural_ops = {'mul': 0, 'div': 0}

    def load(self, bytecode: List[int], data: Optional[List[int]] = None):
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
        self.stdin = data
        self.stdin_pos = 0

    def step(self) -> bool:
        if self.halted or self.pc // 8 >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[self.pc // 8]
        self.pc += 8
        self.op_counts[op] = self.op_counts.get(op, 0) + 1

        # I/O
        if op == 64:  # GETCHAR
            if self.stdin_pos < len(self.stdin):
                self.ax = ord(self.stdin[self.stdin_pos])
                self.stdin_pos += 1
            else:
                self.ax = 0xFFFFFFFF
        elif op == 65:  # PUTCHAR
            c = self.memory.read(self.sp)
            self.stdout.append(c & 0xFF)
            self.ax = c

        # Control
        elif op == 0:  self.ax = self.bp + imm
        elif op == 1:  self.ax = imm
        elif op == 2:  self.pc = imm
        elif op == 3:
            self.sp -= 8
            self.memory.write(self.sp, self.pc)
            self.pc = imm
        elif op == 4:
            if self.ax == 0: self.pc = imm
        elif op == 5:
            if self.ax != 0: self.pc = imm
        elif op == 6:
            self.sp -= 8
            self.memory.write(self.sp, self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == 7:  self.sp += imm
        elif op == 8:
            self.sp = self.bp
            self.bp = self.memory.read(self.sp)
            self.sp += 8
            self.pc = self.memory.read(self.sp)
            self.sp += 8

        # Memory
        elif op == 9:  self.ax = self.memory.read(self.ax)
        elif op == 10: self.ax = self.memory.read(self.ax) & 0xFF
        elif op == 11:
            addr = self.memory.read(self.sp)
            self.sp += 8
            self.memory.write(addr, self.ax)
        elif op == 12:
            addr = self.memory.read(self.sp)
            self.sp += 8
            self.memory.write(addr, self.ax & 0xFF)
        elif op == 13:
            self.sp -= 8
            self.memory.write(self.sp, self.ax)

        # Bitwise/Compare
        elif op == 14:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = a | self.ax
        elif op == 15:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = a ^ self.ax
        elif op == 16:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = a & self.ax
        elif op == 17:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = 1 if a == self.ax else 0
        elif op == 18:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = 1 if a != self.ax else 0
        elif op == 19:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = 1 if a < self.ax else 0
        elif op == 20:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = 1 if a > self.ax else 0
        elif op == 21:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = 1 if a <= self.ax else 0
        elif op == 22:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = 1 if a >= self.ax else 0
        elif op == 23:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = (a << self.ax) & 0xFFFFFFFF
        elif op == 24:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = a >> self.ax

        # Arithmetic
        elif op == 25:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = (a + self.ax) & 0xFFFFFFFF
        elif op == 26:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = (a - self.ax) & 0xFFFFFFFF
        elif op == 27:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = self.alu.mul(a, self.ax)
            self.neural_ops['mul'] += 1
        elif op == 28:
            a = self.memory.read(self.sp); self.sp += 8
            self.ax = self.alu.div(a, self.ax)
            self.neural_ops['div'] += 1
        elif op == 29:
            a = self.memory.read(self.sp)
            b = self.ax
            self.sp += 8
            self.ax = a % b if b else 0

        elif op == 38:
            self.halted = True
            return False

        return True

    def run(self, max_steps: int = 10000000) -> int:
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.ax

    def get_stdout(self) -> str:
        return ''.join(chr(b) for b in self.stdout)

    def stats(self) -> Dict:
        return {
            'memory': self.memory.stats(),
            'neural_ops': self.neural_ops,
            'op_counts': self.op_counts
        }


if __name__ == "__main__":
    from src.compiler import compile_c
    import time

    with open("mandelbrot_putchar_c4.c") as f:
        source = f.read()

    # Test 32x32
    src = source.replace('width = 32;', 'width = 32;')
    src = src.replace('height = 32;', 'height = 32;')

    bytecode, data = compile_c(src)
    print(f"Compiled: {len(bytecode)} instructions")

    vm = AttentionMemoryVM()
    vm.load(bytecode, data)

    start = time.time()
    vm.run(max_steps=50000000)
    elapsed = time.time() - start

    stats = vm.stats()
    mem = stats['memory']

    print(f"\n32x32 Mandelbrot with Attention Memory:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Live entries: {mem['live_entries']:,}")
    print(f"  Total writes: {mem['total_writes']:,}")
    print(f"  Entries pruned: {mem['entries_pruned']:,}")
    print(f"  Prune ratio: {mem['prune_ratio']:.6f} ({mem['prune_ratio']*100:.4f}%)")
    print(f"  Neural MUL: {stats['neural_ops']['mul']:,}")
    print(f"  Neural DIV: {stats['neural_ops']['div']:,}")
    print(f"  Output: {len(vm.stdout)} bytes")
    print(f"  PNG valid: {vm.stdout[:4] == [137, 80, 78, 71]}")
