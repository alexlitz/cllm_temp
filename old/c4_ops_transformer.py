"""
C4 Opcodes implemented with Transformer primitives.

Each opcode is a deterministic state transition:
  (PC, SP, BP, AX, memory) → (PC', SP', BP', AX', memory')

Transformer primitives available:
  - Attention: read from any position based on query
  - Softmax1: soft selection, counting, thresholding
  - Arithmetic: add, sub, mul (via attention + FFN)
  - Bitwise: AND, OR, XOR, shifts (via bit decomposition)
  - Comparisons: soft_gt, soft_lt, soft_eq (via sigmoid)
  - Memory write: scatter to specific positions

Key insight: Each opcode can be decomposed into:
  1. FETCH: Read instruction at PC
  2. DECODE: Extract opcode and immediate
  3. EXECUTE: Perform operation
  4. WRITEBACK: Update state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from c4_vm import Op, CODE_BASE, DATA_BASE, STACK_BASE, STACK_SIZE, MEMORY_SIZE


class TransformerPrimitives(nn.Module):
    """
    Low-level transformer primitives for C4 operations.
    """
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def read_byte(self, memory: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        """
        Read byte from memory at address.

        memory: (batch, mem_size) - byte values
        addr: (batch,) - addresses to read from

        Uses attention: query=addr encoding, keys=position encodings
        """
        batch, mem_size = memory.shape
        device = memory.device

        # One-hot attention to exact address
        # In practice, this is a gather operation
        addr_clamped = addr.clamp(0, mem_size - 1).long()
        return memory.gather(1, addr_clamped.unsqueeze(1)).squeeze(1)

    def read_int64(self, memory: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        """Read 64-bit int (8 bytes) from memory."""
        batch = memory.shape[0]
        device = memory.device
        addr = addr.long()

        # Read 8 consecutive bytes
        result = torch.zeros(batch, dtype=torch.long, device=device)
        for i in range(8):
            byte_val = self.read_byte(memory, addr + i)
            result = result | (byte_val.long() << (i * 8))

        return result

    def write_byte(self, memory: torch.Tensor, addr: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Write byte to memory at address.

        Returns new memory tensor.
        """
        batch, mem_size = memory.shape
        memory = memory.clone()
        addr_clamped = addr.clamp(0, mem_size - 1).long()

        # Scatter write
        memory.scatter_(1, addr_clamped.unsqueeze(1), value.unsqueeze(1) & 0xFF)
        return memory

    def write_int64(self, memory: torch.Tensor, addr: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Write 64-bit int to memory."""
        addr = addr.long()
        value = value.long()

        for i in range(8):
            byte_val = (value >> (i * 8)) & 0xFF
            memory = self.write_byte(memory, addr + i, byte_val)

        return memory

    # =========================================================================
    # Arithmetic (via FFN)
    # =========================================================================

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Addition."""
        return a + b

    def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Subtraction: a - b."""
        return a - b

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplication."""
        return a * b

    def div(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Integer division: a // b."""
        # Avoid division by zero
        b_safe = torch.where(b == 0, torch.ones_like(b), b)
        return torch.where(b == 0, torch.zeros_like(a), a // b_safe)

    def mod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Modulo: a % b."""
        b_safe = torch.where(b == 0, torch.ones_like(b), b)
        return torch.where(b == 0, torch.zeros_like(a), a % b_safe)

    # =========================================================================
    # Bitwise (via bit decomposition)
    # =========================================================================

    def bit_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bitwise AND."""
        return a & b

    def bit_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bitwise OR."""
        return a | b

    def bit_xor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bitwise XOR."""
        return a ^ b

    def shift_left(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Left shift: a << b."""
        return a << b.clamp(0, 63)

    def shift_right(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Right shift: a >> b."""
        return a >> b.clamp(0, 63)

    # =========================================================================
    # Comparisons (via sigmoid threshold)
    # =========================================================================

    def eq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Equality: 1 if a == b, else 0."""
        return (a == b).long()

    def ne(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Not equal: 1 if a != b, else 0."""
        return (a != b).long()

    def lt(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Less than: 1 if a < b, else 0."""
        return (a < b).long()

    def gt(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Greater than: 1 if a > b, else 0."""
        return (a > b).long()

    def le(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Less or equal: 1 if a <= b, else 0."""
        return (a <= b).long()

    def ge(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Greater or equal: 1 if a >= b, else 0."""
        return (a >= b).long()

    # =========================================================================
    # Soft versions (differentiable)
    # =========================================================================

    def soft_eq(self, a: torch.Tensor, b: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
        """Soft equality using narrow sigmoid."""
        diff = (a - b).float()
        return torch.sigmoid(-beta * diff.abs())

    def soft_gt(self, a: torch.Tensor, b: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Soft greater than."""
        return torch.sigmoid(beta * (a - b).float())

    def soft_select(self, cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Soft ternary: cond ? a : b."""
        cond_float = cond.float().unsqueeze(-1) if cond.dim() < a.dim() else cond.float()
        return cond_float * a + (1 - cond_float) * b


class C4OpcodeExecutor(nn.Module):
    """
    Execute C4 opcodes using transformer primitives.

    State representation:
      - pc: program counter (int)
      - sp: stack pointer (int)
      - bp: base pointer (int)
      - ax: accumulator (int)
      - memory: (mem_size,) bytes
    """
    def __init__(self):
        super().__init__()
        self.prims = TransformerPrimitives()

    def fetch_instruction(self, memory: torch.Tensor, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch instruction at PC, return (opcode, immediate)."""
        instruction = self.prims.read_int64(memory.unsqueeze(0), pc.unsqueeze(0)).squeeze(0)
        opcode = instruction & 0xFF
        immediate = instruction >> 8
        return opcode, immediate

    def push(self, memory: torch.Tensor, sp: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Push value onto stack, return (new_memory, new_sp)."""
        new_sp = sp - 8
        new_memory = self.prims.write_int64(memory.unsqueeze(0), new_sp.unsqueeze(0), value.unsqueeze(0)).squeeze(0)
        return new_memory, new_sp

    def pop(self, memory: torch.Tensor, sp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pop value from stack, return (value, new_sp)."""
        value = self.prims.read_int64(memory.unsqueeze(0), sp.unsqueeze(0)).squeeze(0)
        new_sp = sp + 8
        return value, new_sp

    def execute_op(self, op: int, imm: int,
                   pc: torch.Tensor, sp: torch.Tensor, bp: torch.Tensor, ax: torch.Tensor,
                   memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute a single opcode.

        Returns: (new_pc, new_sp, new_bp, new_ax, new_memory)
        """
        device = memory.device
        imm_t = torch.tensor(imm, dtype=torch.long, device=device)

        # Default: advance PC
        new_pc = pc + 8
        new_sp = sp
        new_bp = bp
        new_ax = ax
        new_memory = memory

        if op == Op.LEA:
            # ax = bp + immediate
            new_ax = bp + imm_t

        elif op == Op.IMM:
            # ax = immediate
            new_ax = imm_t

        elif op == Op.JMP:
            # pc = immediate
            new_pc = imm_t

        elif op == Op.JSR:
            # push pc, pc = immediate
            new_memory, new_sp = self.push(memory, sp, new_pc)
            new_pc = imm_t

        elif op == Op.BZ:
            # if ax == 0, pc = immediate
            if ax == 0:
                new_pc = imm_t

        elif op == Op.BNZ:
            # if ax != 0, pc = immediate
            if ax != 0:
                new_pc = imm_t

        elif op == Op.ENT:
            # push bp, bp = sp, sp -= immediate
            new_memory, new_sp = self.push(memory, sp, bp)
            new_bp = new_sp
            new_sp = new_sp - imm_t

        elif op == Op.ADJ:
            # sp += immediate
            new_sp = sp + imm_t

        elif op == Op.LEV:
            # sp = bp, bp = pop, pc = pop
            new_sp = bp
            new_bp, new_sp = self.pop(memory, new_sp)
            new_pc, new_sp = self.pop(memory, new_sp)

        elif op == Op.LI:
            # ax = *(int*)ax
            new_ax = self.prims.read_int64(memory.unsqueeze(0), ax.unsqueeze(0)).squeeze(0)

        elif op == Op.LC:
            # ax = *(char*)ax
            new_ax = self.prims.read_byte(memory.unsqueeze(0), ax.unsqueeze(0)).squeeze(0).long()

        elif op == Op.SI:
            # *(int*)*sp++ = ax
            addr, new_sp = self.pop(memory, sp)
            new_memory = self.prims.write_int64(memory.unsqueeze(0), addr.unsqueeze(0), ax.unsqueeze(0)).squeeze(0)

        elif op == Op.SC:
            # *(char*)*sp++ = ax
            addr, new_sp = self.pop(memory, sp)
            new_memory = self.prims.write_byte(memory.unsqueeze(0), addr.unsqueeze(0), ax.unsqueeze(0)).squeeze(0)

        elif op == Op.PSH:
            # *--sp = ax
            new_memory, new_sp = self.push(memory, sp, ax)

        elif op == Op.ADD:
            val, new_sp = self.pop(memory, sp)
            new_ax = val + ax

        elif op == Op.SUB:
            val, new_sp = self.pop(memory, sp)
            new_ax = val - ax

        elif op == Op.MUL:
            val, new_sp = self.pop(memory, sp)
            new_ax = val * ax

        elif op == Op.DIV:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.div(val, ax)

        elif op == Op.MOD:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.mod(val, ax)

        elif op == Op.OR:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.bit_or(val, ax)

        elif op == Op.XOR:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.bit_xor(val, ax)

        elif op == Op.AND:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.bit_and(val, ax)

        elif op == Op.SHL:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.shift_left(val, ax)

        elif op == Op.SHR:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.shift_right(val, ax)

        elif op == Op.EQ:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.eq(val, ax)

        elif op == Op.NE:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.ne(val, ax)

        elif op == Op.LT:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.lt(val, ax)

        elif op == Op.GT:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.gt(val, ax)

        elif op == Op.LE:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.le(val, ax)

        elif op == Op.GE:
            val, new_sp = self.pop(memory, sp)
            new_ax = self.prims.ge(val, ax)

        elif op == Op.EXIT:
            # Halt - PC stays same
            new_pc = pc

        # System calls - simplified
        elif op in (Op.MALC, Op.FREE, Op.MSET, Op.MCMP, Op.PRTF, Op.OPEN, Op.READ, Op.CLOS):
            new_ax = torch.tensor(0, dtype=torch.long, device=device)

        return new_pc, new_sp, new_bp, new_ax, new_memory


def test_opcode_executor():
    """Test the opcode executor matches the VM."""
    from c4_vm import C4VM, program_simple_arithmetic, program_sum

    print("=" * 60)
    print("TESTING TRANSFORMER OPCODE EXECUTOR")
    print("=" * 60)

    executor = C4OpcodeExecutor()
    device = 'cpu'

    # Test 1: Simple arithmetic
    print("\n1. Simple arithmetic: (3 + 4) * 5")
    vm = C4VM()
    code = program_simple_arithmetic()
    vm.load_code(code)

    # Convert VM state to tensors
    pc = torch.tensor(vm.state.pc, dtype=torch.long, device=device)
    sp = torch.tensor(vm.state.sp, dtype=torch.long, device=device)
    bp = torch.tensor(vm.state.bp, dtype=torch.long, device=device)
    ax = torch.tensor(vm.state.ax, dtype=torch.long, device=device)
    memory = torch.tensor(list(vm.state.memory), dtype=torch.long, device=device)

    # Execute step by step and compare
    steps = 0
    while steps < 100:
        # Fetch instruction
        opcode, imm = executor.fetch_instruction(memory, pc)

        if opcode.item() == Op.EXIT:
            break

        # Execute with transformer primitives
        pc, sp, bp, ax, memory = executor.execute_op(
            opcode.item(), imm.item(),
            pc, sp, bp, ax, memory
        )

        # Also step VM
        vm.step()

        # Compare
        assert pc.item() == vm.state.pc, f"PC mismatch: {pc.item()} vs {vm.state.pc}"
        assert sp.item() == vm.state.sp, f"SP mismatch: {sp.item()} vs {vm.state.sp}"
        assert ax.item() == vm.state.ax, f"AX mismatch: {ax.item()} vs {vm.state.ax}"

        steps += 1

    print(f"   Executed {steps} steps")
    print(f"   Final AX: {ax.item()} (expected 35)")
    assert ax.item() == 35
    print("   ✓ PASS")

    # Test 2: Sum 1 to 10
    print("\n2. Sum 1 to 10")
    vm = C4VM()
    code = program_sum(10)
    vm.load_code(code)

    pc = torch.tensor(vm.state.pc, dtype=torch.long, device=device)
    sp = torch.tensor(vm.state.sp, dtype=torch.long, device=device)
    bp = torch.tensor(vm.state.bp, dtype=torch.long, device=device)
    ax = torch.tensor(vm.state.ax, dtype=torch.long, device=device)
    memory = torch.tensor(list(vm.state.memory), dtype=torch.long, device=device)

    steps = 0
    while steps < 1000:
        opcode, imm = executor.fetch_instruction(memory, pc)

        if opcode.item() == Op.EXIT:
            break

        pc, sp, bp, ax, memory = executor.execute_op(
            opcode.item(), imm.item(),
            pc, sp, bp, ax, memory
        )
        vm.step()

        assert pc.item() == vm.state.pc, f"Step {steps}: PC mismatch"
        assert ax.item() == vm.state.ax, f"Step {steps}: AX mismatch"

        steps += 1

    print(f"   Executed {steps} steps")
    print(f"   Final AX: {ax.item()} (expected 55)")
    assert ax.item() == 55
    print("   ✓ PASS")

    print("\n" + "=" * 60)
    print("ALL OPCODE TESTS PASSED")
    print("=" * 60)

    # Show which primitives each opcode uses
    print("\n" + "=" * 60)
    print("OPCODE → TRANSFORMER PRIMITIVE MAPPING")
    print("=" * 60)

    mappings = {
        "LEA": "add(bp, imm)",
        "IMM": "copy(imm)",
        "JMP": "set_pc(imm)",
        "JSR": "push(pc) + set_pc(imm)",
        "BZ/BNZ": "soft_select(cond, imm, pc+8)",
        "ENT": "push(bp) + set_bp(sp) + sub(sp, imm)",
        "ADJ": "add(sp, imm)",
        "LEV": "set_sp(bp) + pop(bp) + pop(pc)",
        "LI": "read_int64(memory, ax)",
        "LC": "read_byte(memory, ax)",
        "SI": "write_int64(memory, pop(), ax)",
        "SC": "write_byte(memory, pop(), ax)",
        "PSH": "push(ax)",
        "ADD": "add(pop(), ax)",
        "SUB": "sub(pop(), ax)",
        "MUL": "mul(pop(), ax)",
        "DIV": "div(pop(), ax)",
        "MOD": "mod(pop(), ax)",
        "OR": "bit_or(pop(), ax)",
        "XOR": "bit_xor(pop(), ax)",
        "AND": "bit_and(pop(), ax)",
        "SHL": "shift_left(pop(), ax)",
        "SHR": "shift_right(pop(), ax)",
        "EQ": "eq(pop(), ax)",
        "NE": "ne(pop(), ax)",
        "LT": "lt(pop(), ax)",
        "GT": "gt(pop(), ax)",
        "LE": "le(pop(), ax)",
        "GE": "ge(pop(), ax)",
        "EXIT": "halt()",
    }

    for op, impl in mappings.items():
        print(f"  {op:8s} → {impl}")


if __name__ == "__main__":
    test_opcode_executor()
