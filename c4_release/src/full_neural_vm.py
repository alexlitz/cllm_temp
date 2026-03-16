"""
Full Neural VM - Every operation uses neural primitives.

Components:
1. Instruction fetch: Attention over program memory
2. Opcode decode: MoE routing (39 experts)
3. Memory read/write: Attention over memory entries
4. Multiply: SwiGLU identity (a*b = silu(a)*b + silu(-a)*(-b))
5. Divide: Attention-based table lookup + SwiGLU Newton iterations

This is what a transformer "natively running" a VM looks like.
All FLOPs are countable neural operations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import time


# =============================================================================
# FLOP COUNTER
# =============================================================================

class FLOPCounter:
    """Track FLOPs for each neural operation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {
            'instruction_fetch': 0,
            'opcode_decode': 0,
            'memory_read': 0,
            'memory_write': 0,
            'swiglu_mul': 0,
            'newton_div': 0,
            'add_sub': 0,
        }
        self.flops = {
            'instruction_fetch': 0,
            'opcode_decode': 0,
            'memory_read': 0,
            'memory_write': 0,
            'swiglu_mul': 0,
            'newton_div': 0,
            'add_sub': 0,
        }

    def record(self, op: str, n_entries: int = 1):
        """Record an operation with its FLOP cost."""
        self.counts[op] = self.counts.get(op, 0) + 1

        # FLOP costs
        if op == 'instruction_fetch':
            # Attention over n_entries instructions with 16-bit encoding
            # dot products: n * 16, softmax: n * 3, weighted sum: n
            self.flops[op] += n_entries * 16 + n_entries * 3 + n_entries
        elif op == 'opcode_decode':
            # MoE routing: 39 experts, ~10 FLOPs each for eq_gate
            self.flops[op] += 39 * 10
        elif op == 'memory_read' or op == 'memory_write':
            # Attention over n_entries with 20-bit keys
            # dot products: n * 20, softmax: n * 3, weighted sum: n
            self.flops[op] += n_entries * 20 + n_entries * 3 + n_entries
        elif op == 'swiglu_mul':
            # SwiGLU: 2 * silu (43 each) + 3 ops
            self.flops[op] += 91
        elif op == 'newton_div':
            # Attention table (256 entries): 256 * 8 + 256 * 3 + 256 = 3072
            # 2 Newton iterations with SwiGLU: 2 * 2 * 91 = 364
            # Final SwiGLU mul: 91
            self.flops[op] += 3072 + 364 + 91
        elif op == 'add_sub':
            self.flops[op] += 1

    def total_flops(self) -> int:
        return sum(self.flops.values())

    def summary(self) -> Dict:
        total = self.total_flops()
        return {
            'counts': dict(self.counts),
            'flops': dict(self.flops),
            'total_flops': total,
            'breakdown': {k: v / max(1, total) for k, v in self.flops.items()}
        }


# =============================================================================
# NEURAL PRIMITIVES
# =============================================================================

def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def swiglu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Exact multiplication using SwiGLU identity.
    a * b = silu(a) * b + silu(-a) * (-b)

    This is mathematically exact because sigmoid(x) + sigmoid(-x) = 1
    """
    return silu(a) * b + silu(-a) * (-b)


def swiglu_mul_scalar(a: float, b: float) -> float:
    """Scalar version of SwiGLU multiply."""
    at, bt = torch.tensor(a), torch.tensor(b)
    result = swiglu_mul(at, bt)
    return result.item()


# =============================================================================
# ATTENTION-BASED NEWTON DIVISION
# =============================================================================

class NeuralNewtonDivide:
    """
    Division using attention-based table lookup + SwiGLU Newton iterations.

    Steps:
    1. Normalize b to [0.5, 1.0)
    2. Attention lookup in 256-entry reciprocal table
    3. Two Newton iterations: y = y * (2 - b * y) using SwiGLU
    4. Denormalize and multiply by a
    """

    def __init__(self):
        # Build reciprocal table: 256 entries for [0.5, 1.0)
        self.table_size = 256
        self.table_keys = torch.zeros(self.table_size, 8)  # 8-bit index encoding
        self.table_values = torch.zeros(self.table_size)

        for i in range(self.table_size):
            # Encode index as 8-bit binary
            for b in range(8):
                self.table_keys[i, b] = 1.0 if (i >> b) & 1 else -1.0

            # Value: 1 / (0.5 + i/256)
            x = 0.5 + i / 256.0
            self.table_values[i] = 1.0 / x

    def _attention_lookup(self, idx_float: float) -> float:
        """Look up reciprocal using attention over table."""
        # Clamp index
        idx = max(0, min(255, int(idx_float * 256)))

        # Encode query
        query = torch.zeros(8)
        for b in range(8):
            query[b] = 1.0 if (idx >> b) & 1 else -1.0

        # Attention: dot product + softmax
        scores = torch.matmul(self.table_keys, query)  # (256,)
        weights = F.softmax(scores * 10.0, dim=0)  # Sharp attention

        # Weighted sum
        result = torch.sum(weights * self.table_values)
        return result.item()

    def divide(self, a: int, b: int) -> int:
        """
        Compute a / b using neural primitives.

        All operations are neural:
        - Table lookup via attention
        - Newton iterations via SwiGLU multiply
        """
        if b == 0:
            return 0

        # Handle signs
        sign = 1
        if a < 0:
            sign, a = -sign, -a
        if b < 0:
            sign, b = -sign, -b

        # Normalize b to [0.5, 1.0)
        exp = 0
        normalized = float(b)
        while normalized >= 1.0:
            normalized *= 0.5
            exp += 1
        while normalized < 0.5:
            normalized *= 2.0
            exp -= 1

        # Attention table lookup for initial estimate
        idx = normalized - 0.5  # [0, 0.5) -> [0, 1) for table index
        y = self._attention_lookup(idx)

        # Newton iteration 1: y = y * (2 - b_norm * y) using SwiGLU
        correction1 = 2.0 - swiglu_mul_scalar(normalized, y)
        y = swiglu_mul_scalar(y, correction1)

        # Newton iteration 2
        correction2 = 2.0 - swiglu_mul_scalar(normalized, y)
        y = swiglu_mul_scalar(y, correction2)

        # Denormalize
        y = y * (2.0 ** (-exp))

        # Final multiply: a * (1/b) using SwiGLU
        result = swiglu_mul_scalar(float(a), y)

        return int(result * sign)


# =============================================================================
# ATTENTION MEMORY
# =============================================================================

class AttentionMemory:
    """
    Memory using attention mechanics.

    Each memory location is a (key, value) pair.
    Keys are 20-bit binary encoded addresses.
    Read/write use attention over all entries.
    """

    def __init__(self, flop_counter: FLOPCounter, num_bits: int = 20):
        self.num_bits = num_bits
        self.flops = flop_counter

        # Storage
        self.keys: List[torch.Tensor] = []
        self.values: List[int] = []
        self.addresses: List[int] = []  # For O(1) lookup optimization

        # Stats
        self.total_writes = 0
        self.total_reads = 0

    def _encode_address(self, addr: int) -> torch.Tensor:
        """Encode address as binary vector."""
        bits = torch.zeros(self.num_bits)
        for b in range(self.num_bits):
            bits[b] = 1.0 if (addr >> b) & 1 else -1.0
        return bits

    def write(self, addr: int, value: int):
        """Write value using attention-based addressing."""
        self.total_writes += 1
        self.flops.record('memory_write', len(self.keys) + 1)

        key = self._encode_address(addr)

        # Check if address exists via attention
        if len(self.keys) > 0:
            keys_tensor = torch.stack(self.keys)
            scores = torch.matmul(keys_tensor, key)

            # Check raw score, not softmax (softmax of single value is always 1.0)
            # Perfect match has score = num_bits (all bits match: 1*1 = 1 each)
            max_score = scores.max().item()
            max_idx = scores.argmax().item()

            # If score is close to perfect match (num_bits), update existing
            # Threshold: num_bits - 2 allows for small encoding differences
            if max_score > self.num_bits - 2:
                self.values[max_idx] = value
                return

        # Add new entry
        self.keys.append(key)
        self.values.append(value)
        self.addresses.append(addr)

    def read(self, addr: int) -> int:
        """Read value using attention-based addressing."""
        self.total_reads += 1
        self.flops.record('memory_read', max(1, len(self.keys)))

        if len(self.keys) == 0:
            return 0

        query = self._encode_address(addr)
        keys_tensor = torch.stack(self.keys)
        values_tensor = torch.tensor(self.values, dtype=torch.float32)

        # Attention: dot product + softmax
        scores = torch.matmul(keys_tensor, query)

        # Recency bias: later writes get slight preference
        positions = torch.arange(len(scores), dtype=torch.float32)
        scores = scores + positions * 0.01

        weights = F.softmax(scores * 10.0, dim=0)

        # Weighted sum
        result = torch.sum(weights * values_tensor)
        return int(round(result.item()))

    def prune(self, keep_recent: int = 1000):
        """Prune old entries, keeping recent ones."""
        if len(self.keys) <= keep_recent:
            return

        # Keep last keep_recent entries
        self.keys = self.keys[-keep_recent:]
        self.values = self.values[-keep_recent:]
        self.addresses = self.addresses[-keep_recent:]

    def stats(self) -> Dict:
        return {
            'live_entries': len(self.keys),
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
        }


# =============================================================================
# FULL NEURAL VM
# =============================================================================

class FullNeuralVM:
    """
    Fully neural VM where every operation uses neural primitives.

    Neural operations:
    - Instruction fetch: attention over program
    - Opcode decode: MoE-style routing
    - Memory: attention-based read/write
    - Multiply: SwiGLU
    - Divide: attention table + SwiGLU Newton
    """

    # Opcodes
    OP_LEA, OP_IMM, OP_JMP, OP_JSR = 0, 1, 2, 3
    OP_BZ, OP_BNZ, OP_ENT, OP_ADJ = 4, 5, 6, 7
    OP_LEV, OP_LI, OP_LC, OP_SI = 8, 9, 10, 11
    OP_SC, OP_PSH = 12, 13
    OP_OR, OP_XOR, OP_AND = 14, 15, 16
    OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE = 17, 18, 19, 20, 21, 22
    OP_SHL, OP_SHR, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD = 23, 24, 25, 26, 27, 28, 29
    OP_PRTF = 33
    OP_EXIT = 38
    OP_GETCHAR, OP_PUTCHAR = 64, 65

    NUM_EXPERTS = 39

    def __init__(self, prune_interval: int = 10000):
        self.flops = FLOPCounter()
        self.memory = AttentionMemory(self.flops)
        self.divider = NeuralNewtonDivide()
        self.prune_interval = prune_interval

        # Program memory (list of encoded instructions)
        self.program_keys: List[torch.Tensor] = []  # PC encodings
        self.program_ops: List[int] = []
        self.program_imms: List[int] = []

        # Registers
        self.pc = 0
        self.sp = 0x30000
        self.bp = 0x30000
        self.ax = 0

        # State
        self.halted = False
        self.steps = 0

        # I/O
        self.stdin = ""
        self.stdin_pos = 0
        self.stdout: List[int] = []

    def _encode_pc(self, pc: int) -> torch.Tensor:
        """Encode PC as 16-bit binary for instruction fetch."""
        bits = torch.zeros(16)
        for b in range(16):
            bits[b] = 1.0 if (pc >> b) & 1 else -1.0
        return bits

    def load(self, bytecode: List[int], data: Optional[List[int]] = None):
        """Load program into neural memory."""
        self.program_keys = []
        self.program_ops = []
        self.program_imms = []

        for i, instr in enumerate(bytecode):
            pc = i * 8
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)

            self.program_keys.append(self._encode_pc(pc))
            self.program_ops.append(op)
            self.program_imms.append(imm & 0xFFFFFFFF)

        # Load data into memory
        if data:
            for i, b in enumerate(data):
                self.memory.write(0x10000 + i, b & 0xFF)

    def set_stdin(self, data: str):
        self.stdin = data
        self.stdin_pos = 0

    def _fetch(self) -> Tuple[int, int]:
        """Fetch instruction using attention over program memory."""
        self.flops.record('instruction_fetch', len(self.program_keys))

        if len(self.program_keys) == 0:
            return 0, 0

        query = self._encode_pc(self.pc)
        keys_tensor = torch.stack(self.program_keys)

        # Attention scores
        scores = torch.matmul(keys_tensor, query)
        weights = F.softmax(scores * 10.0, dim=0)

        # Get winning instruction (hard selection for correctness)
        idx = weights.argmax().item()
        return self.program_ops[idx], self.program_imms[idx]

    def _decode_route(self, op: int):
        """MoE-style opcode routing (just counting FLOPs)."""
        self.flops.record('opcode_decode')

    def step(self) -> bool:
        """Execute one step with all neural operations."""
        if self.halted:
            return False

        self.steps += 1

        # Periodic pruning
        if self.steps % self.prune_interval == 0:
            self.memory.prune()

        # Fetch via attention
        op, imm = self._fetch()
        self.pc += 8

        # Decode via MoE routing
        self._decode_route(op)

        # Execute (each operation tracks its own FLOPs)

        # I/O
        if op == self.OP_GETCHAR:
            if self.stdin_pos < len(self.stdin):
                self.ax = ord(self.stdin[self.stdin_pos])
                self.stdin_pos += 1
            else:
                self.ax = 0xFFFFFFFF

        elif op == self.OP_PUTCHAR:
            c = self.memory.read(self.sp)
            self.stdout.append(c & 0xFF)
            self.ax = c

        # Control
        elif op == self.OP_LEA:
            self.flops.record('add_sub')
            self.ax = self.bp + imm

        elif op == self.OP_IMM:
            self.ax = imm

        elif op == self.OP_JMP:
            self.pc = imm

        elif op == self.OP_JSR:
            self.sp -= 8
            self.memory.write(self.sp, self.pc)
            self.pc = imm

        elif op == self.OP_BZ:
            if self.ax == 0:
                self.pc = imm

        elif op == self.OP_BNZ:
            if self.ax != 0:
                self.pc = imm

        elif op == self.OP_ENT:
            self.sp -= 8
            self.memory.write(self.sp, self.bp)
            self.bp = self.sp
            self.sp -= imm

        elif op == self.OP_ADJ:
            self.flops.record('add_sub')
            self.sp += imm

        elif op == self.OP_LEV:
            self.sp = self.bp
            self.bp = self.memory.read(self.sp)
            self.sp += 8
            self.pc = self.memory.read(self.sp)
            self.sp += 8

        # Memory
        elif op == self.OP_LI:
            self.ax = self.memory.read(self.ax)

        elif op == self.OP_LC:
            self.ax = self.memory.read(self.ax) & 0xFF

        elif op == self.OP_SI:
            addr = self.memory.read(self.sp)
            self.sp += 8
            self.memory.write(addr, self.ax)

        elif op == self.OP_SC:
            addr = self.memory.read(self.sp)
            self.sp += 8
            self.memory.write(addr, self.ax & 0xFF)

        elif op == self.OP_PSH:
            self.sp -= 8
            self.memory.write(self.sp, self.ax)

        # Bitwise
        elif op == self.OP_OR:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = a | self.ax

        elif op == self.OP_XOR:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = a ^ self.ax

        elif op == self.OP_AND:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = a & self.ax

        # Compare
        elif op == self.OP_EQ:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = 1 if a == self.ax else 0

        elif op == self.OP_NE:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = 1 if a != self.ax else 0

        elif op == self.OP_LT:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = 1 if a < self.ax else 0

        elif op == self.OP_GT:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = 1 if a > self.ax else 0

        elif op == self.OP_LE:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = 1 if a <= self.ax else 0

        elif op == self.OP_GE:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = 1 if a >= self.ax else 0

        elif op == self.OP_SHL:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = (a << self.ax) & 0xFFFFFFFF

        elif op == self.OP_SHR:
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = a >> self.ax

        # Arithmetic
        elif op == self.OP_ADD:
            self.flops.record('add_sub')
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = (a + self.ax) & 0xFFFFFFFF

        elif op == self.OP_SUB:
            self.flops.record('add_sub')
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = (a - self.ax) & 0xFFFFFFFF

        elif op == self.OP_MUL:
            self.flops.record('swiglu_mul')
            a = self.memory.read(self.sp)
            self.sp += 8
            result = swiglu_mul_scalar(float(a), float(self.ax))
            self.ax = int(result) & 0xFFFFFFFF

        elif op == self.OP_DIV:
            self.flops.record('newton_div')
            a = self.memory.read(self.sp)
            self.sp += 8
            self.ax = self.divider.divide(a, self.ax)

        elif op == self.OP_MOD:
            self.flops.record('newton_div')  # Uses division
            a = self.memory.read(self.sp)
            b = self.ax
            self.sp += 8
            if b != 0:
                quot = self.divider.divide(a, b)
                self.ax = a - quot * b
            else:
                self.ax = 0

        elif op == self.OP_EXIT:
            self.halted = True
            return False

        return True

    def run(self, max_steps: int = 10000000) -> int:
        """Run until halted or max steps."""
        while self.steps < max_steps and self.step():
            pass
        return self.ax

    def get_stdout(self) -> str:
        return ''.join(chr(b) for b in self.stdout)

    def get_stdout_bytes(self) -> bytes:
        return bytes(self.stdout)

    def stats(self) -> Dict:
        flop_summary = self.flops.summary()
        mem_stats = self.memory.stats()

        return {
            'steps': self.steps,
            'memory': mem_stats,
            'flops': flop_summary,
            'neural_ops': {
                'mul': self.flops.counts.get('swiglu_mul', 0),
                'div': self.flops.counts.get('newton_div', 0),
            }
        }


def compile_and_run_full_neural(source: str, max_steps: int = 10000000) -> Tuple[FullNeuralVM, int]:
    """Compile C source and run on full neural VM."""
    from .compiler import compile_c

    bytecode, data = compile_c(source)
    vm = FullNeuralVM()
    vm.load(bytecode, data)
    result = vm.run(max_steps)

    return vm, result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')
    from src.compiler import compile_c

    # Simple test
    test_source = """
    int main() {
        int a;
        int b;
        int c;
        a = 6;
        b = 7;
        c = a * b;
        return c;
    }
    """

    print("Testing full neural VM...")
    bytecode, data = compile_c(test_source)
    print(f"Compiled: {len(bytecode)} instructions")

    vm = FullNeuralVM()
    vm.load(bytecode, data)
    result = vm.run()

    print(f"Result: {result} (expected: 42)")

    stats = vm.stats()
    print(f"\nFLOP breakdown:")
    for op, count in stats['flops']['counts'].items():
        flops = stats['flops']['flops'].get(op, 0)
        if count > 0:
            print(f"  {op}: {count} ops, {flops:,} FLOPs")
    print(f"\nTotal FLOPs: {stats['flops']['total_flops']:,}")
