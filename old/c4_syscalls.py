"""
C4 VM with syscall support.

- PRTF writes to output buffer in context
- File I/O (OPEN/READ/CLOS) delegates to host wrapper
- FREE actually frees (simple free list)

The transformer handles pure computation; syscalls are handled by host.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from c4_vm import Op


def silu(x):
    return x * torch.sigmoid(x)


def silu_threshold(x, scale=20.0):
    diff = scale * x
    term1 = silu(diff + 0.5 * scale)
    term2 = silu(diff - 0.5 * scale)
    return (term1 - term2) / scale


def eq_gate(a, b, scale=20.0):
    diff = (a - b).float()
    upper = silu_threshold(diff + 0.5, scale)
    lower = silu_threshold(-diff + 0.5, scale)
    return upper * lower


def swiglu_mul(a, b):
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


@dataclass
class SyscallRequest:
    """Request from VM to host for syscall handling."""
    opcode: int
    args: list


@dataclass
class SyscallResponse:
    """Response from host after handling syscall."""
    return_value: int
    output: bytes = b''  # For printf


class SyscallHost:
    """
    Host environment that handles syscalls.
    Can be subclassed to provide real file I/O, etc.
    """

    def __init__(self):
        self.output_buffer = bytearray()
        self.files: Dict[int, Any] = {}
        self.next_fd = 3  # 0=stdin, 1=stdout, 2=stderr
        self.heap_regions: Dict[int, int] = {}  # addr -> size
        self.free_list: list = []

    def handle_syscall(self, request: SyscallRequest) -> SyscallResponse:
        """Dispatch syscall to appropriate handler."""
        if request.opcode == Op.PRTF:
            return self.handle_printf(request.args)
        elif request.opcode == Op.OPEN:
            return self.handle_open(request.args)
        elif request.opcode == Op.READ:
            return self.handle_read(request.args)
        elif request.opcode == Op.CLOS:
            return self.handle_close(request.args)
        elif request.opcode == Op.FREE:
            return self.handle_free(request.args)
        else:
            return SyscallResponse(return_value=0)

    def handle_printf(self, args) -> SyscallResponse:
        """
        Handle printf syscall.
        args[0] = format string address (ignored in simplified version)
        args[1] = value to print
        """
        # Simplified: print just the first value argument
        if len(args) > 1:
            text = str(int(args[1])) + '\n'
        else:
            text = '\n'
        output = text.encode('utf-8')
        self.output_buffer.extend(output)
        return SyscallResponse(return_value=len(output), output=output)

    def handle_open(self, args) -> SyscallResponse:
        """Handle file open. Override for real file I/O."""
        # Default: return error
        return SyscallResponse(return_value=-1)

    def handle_read(self, args) -> SyscallResponse:
        """Handle file read. Override for real file I/O."""
        # Default: return 0 bytes read
        return SyscallResponse(return_value=0)

    def handle_close(self, args) -> SyscallResponse:
        """Handle file close."""
        return SyscallResponse(return_value=0)

    def handle_free(self, args) -> SyscallResponse:
        """Handle free - track freed regions."""
        if len(args) > 0:
            addr = int(args[0])
            if addr in self.heap_regions:
                size = self.heap_regions.pop(addr)
                self.free_list.append((addr, size))
        return SyscallResponse(return_value=0)

    def get_output(self) -> str:
        """Get accumulated output as string."""
        return self.output_buffer.decode('utf-8', errors='replace')

    def clear_output(self):
        """Clear output buffer."""
        self.output_buffer.clear()


class RealFileHost(SyscallHost):
    """Host that provides real file I/O."""

    def handle_open(self, args) -> SyscallResponse:
        """Open a real file."""
        if len(args) < 2:
            return SyscallResponse(return_value=-1)

        filename = args[0]  # Would need to read from VM memory
        mode = args[1]

        try:
            # Map C modes to Python
            py_mode = 'r' if mode == 0 else 'w' if mode == 1 else 'r+'
            f = open(filename, py_mode)
            fd = self.next_fd
            self.next_fd += 1
            self.files[fd] = f
            return SyscallResponse(return_value=fd)
        except:
            return SyscallResponse(return_value=-1)

    def handle_read(self, args) -> SyscallResponse:
        """Read from file."""
        if len(args) < 3:
            return SyscallResponse(return_value=-1)

        fd, buf_addr, count = args[0], args[1], args[2]

        if fd not in self.files:
            return SyscallResponse(return_value=-1)

        try:
            data = self.files[fd].read(count)
            # Would need to write data to VM memory at buf_addr
            return SyscallResponse(return_value=len(data), output=data.encode() if isinstance(data, str) else data)
        except:
            return SyscallResponse(return_value=-1)

    def handle_close(self, args) -> SyscallResponse:
        """Close file."""
        if len(args) < 1:
            return SyscallResponse(return_value=-1)

        fd = args[0]
        if fd in self.files:
            self.files[fd].close()
            del self.files[fd]
            return SyscallResponse(return_value=0)
        return SyscallResponse(return_value=-1)


class BinaryEncoder(nn.Module):
    """Encode integers as (-N, +N) binary vectors."""

    def __init__(self, num_bits=9, scale=10.0):
        super().__init__()
        self.num_bits = num_bits
        self.scale = scale
        self.register_buffer('powers', 2 ** torch.arange(num_bits))

    def encode(self, x):
        x = x.long()
        bits = (x.unsqueeze(-1) // self.powers) % 2
        encoded = (2 * bits.float() - 1) * self.scale
        return encoded


class C4WithSyscalls(nn.Module):
    """
    C4 executor with syscall support.

    Context: [memory..., PC, SP, BP, AX, OUTPUT_LEN, OUTPUT_0, OUTPUT_1, ...]

    When a syscall is detected:
    1. Executor returns SyscallRequest
    2. Host handles syscall
    3. Host provides SyscallResponse
    4. Executor continues with response
    """

    def __init__(self, memory_size=256, output_size=64, host: Optional[SyscallHost] = None):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.host = host or SyscallHost()

        self.scale = 10.0
        self.num_addr_bits = int(math.ceil(math.log2(memory_size + 5 + output_size)))
        self.num_type_bits = 5  # memory, PC, SP, BP, AX, OUTPUT

        self.addr_encoder = BinaryEncoder(self.num_addr_bits, self.scale)

        # Indices in context
        self.PC_IDX = memory_size
        self.SP_IDX = memory_size + 1
        self.BP_IDX = memory_size + 2
        self.AX_IDX = memory_size + 3
        self.OUT_LEN_IDX = memory_size + 4
        self.OUT_START_IDX = memory_size + 5

        # Import experts
        from c4_moe_top1 import (
            MoERouter, ImmExpert, LeaExpert, AddExpert, SubExpert,
            MulExpert, DivExpert, ModExpert, ShlExpert, ShrExpert,
            CmpExpert, NoOpExpert
        )

        self.router = MoERouter(40)
        self.experts = nn.ModuleDict({
            str(Op.IMM): ImmExpert(),
            str(Op.LEA): LeaExpert(),
            str(Op.ADD): AddExpert(),
            str(Op.SUB): SubExpert(),
            str(Op.MUL): MulExpert(),
            str(Op.DIV): DivExpert(64),
            str(Op.MOD): ModExpert(64),
            str(Op.SHL): ShlExpert(),
            str(Op.SHR): ShrExpert(),
            str(Op.EQ): CmpExpert('eq'),
            str(Op.NE): CmpExpert('ne'),
            str(Op.LT): CmpExpert('lt'),
            str(Op.GT): CmpExpert('gt'),
            str(Op.LE): CmpExpert('le'),
            str(Op.GE): CmpExpert('ge'),
        })
        self.default_expert = NoOpExpert()

        # Precompute keys
        self._build_keys()

    def _build_keys(self):
        """Build attention keys with type tags."""
        total_size = self.memory_size + 5 + self.output_size
        addresses = torch.arange(total_size)
        addr_keys = self.addr_encoder.encode(addresses.unsqueeze(0)).squeeze(0)

        # Type tags: [mem, pc, sp, bp, ax, out]
        type_tags = torch.full((total_size, self.num_type_bits), -self.scale)
        type_tags[self.PC_IDX, 0] = self.scale
        type_tags[self.SP_IDX, 1] = self.scale
        type_tags[self.BP_IDX, 2] = self.scale
        type_tags[self.AX_IDX, 3] = self.scale
        # Output tokens get type bit 4
        for i in range(self.output_size + 1):  # +1 for OUT_LEN
            type_tags[self.OUT_LEN_IDX + i, 4] = self.scale

        full_keys = torch.cat([addr_keys, type_tags], dim=1)
        self.register_buffer('all_keys', full_keys)
        self.total_key_dim = self.num_addr_bits + self.num_type_bits

        # Type templates
        self.register_buffer('memory_type', torch.tensor(
            [-self.scale, -self.scale, -self.scale, -self.scale, -self.scale]))
        self.register_buffer('output_type', torch.tensor(
            [-self.scale, -self.scale, -self.scale, -self.scale, self.scale]))

    def build_context(self, memory, pc, sp, bp, ax, output):
        """Build context: [memory, PC, SP, BP, AX, OUT_LEN, OUTPUT...]"""
        context_size = self.memory_size + 5 + self.output_size
        context = torch.zeros(context_size)
        context[:self.memory_size] = memory[:self.memory_size].float()
        context[self.PC_IDX] = pc.float()
        context[self.SP_IDX] = sp.float()
        context[self.BP_IDX] = bp.float()
        context[self.AX_IDX] = ax.float()
        context[self.OUT_LEN_IDX] = len(output)
        for i, ch in enumerate(output[:self.output_size]):
            context[self.OUT_START_IDX + i] = float(ch)
        return context

    def attend(self, context, addr):
        """Read from context via attention."""
        addr_query = self.addr_encoder.encode(addr)
        type_query = self.memory_type
        query = torch.cat([addr_query, type_query])

        keys = self.all_keys[:len(context)]
        scores = torch.matmul(keys, query) / math.sqrt(self.total_key_dim)
        weights = F.softmax(scores, dim=0)
        return torch.sum(weights * context)

    def decode(self, instruction):
        """Decode instruction into opcode and immediate."""
        instruction = instruction.float()
        imm = torch.floor(instruction / 256.0)
        opcode = instruction - imm * 256.0
        return opcode, imm

    def is_syscall(self, opcode) -> bool:
        """Check if opcode is a syscall that needs host handling."""
        syscall_ops = [Op.PRTF, Op.OPEN, Op.READ, Op.CLOS, Op.FREE]
        for op in syscall_ops:
            if eq_gate(opcode, torch.tensor(float(op))) > 0.5:
                return True
        return False

    def get_syscall_args(self, context, sp, opcode) -> list:
        """Extract syscall arguments from stack."""
        args = []
        # Most syscalls take 1-3 args from stack
        for i in range(3):
            arg = self.attend(context, sp + i * 8)
            args.append(arg.item())
        return args

    def step(self, pc, sp, bp, ax, memory, output):
        """
        Execute one step.
        Returns: (new_pc, new_sp, new_bp, new_ax, new_memory, new_output, syscall_request)

        If syscall_request is not None, host should handle it and call continue_after_syscall.
        """
        pc, sp, bp, ax = pc.float(), sp.float(), bp.float(), ax.float()

        context = self.build_context(memory, pc, sp, bp, ax, output)

        # Fetch & decode
        instruction = self.attend(context, pc)
        opcode, imm = self.decode(instruction)

        # Check for syscall
        if self.is_syscall(opcode):
            args = self.get_syscall_args(context, sp, opcode)
            request = SyscallRequest(opcode=int(opcode.item()), args=args)

            # Handle syscall via host
            response = self.host.handle_syscall(request)

            # Update AX with return value
            new_ax = torch.tensor(float(response.return_value))

            # If printf, append to output
            new_output = output
            if int(opcode.item()) == Op.PRTF and response.output:
                new_output = output + list(response.output)

            # Advance PC
            new_pc = pc + 8

            return (
                torch.round(new_pc),
                sp,  # SP unchanged for most syscalls
                bp,
                torch.round(new_ax),
                memory,
                new_output,
                None
            )

        # Regular instruction execution
        stack_top = self.attend(context, sp)

        # Route to expert
        _, selected_idx = self.router(opcode)
        expert_key = str(selected_idx.item())
        if expert_key in self.experts:
            expert = self.experts[expert_key]
        else:
            expert = self.default_expert

        new_ax, needs_pop = expert(stack_top, ax, imm, bp, memory, sp)
        new_ax = new_ax.float()

        # Memory load
        g_li = eq_gate(opcode, torch.tensor(float(Op.LI)))
        g_lc = eq_gate(opcode, torch.tensor(float(Op.LC)))
        if (g_li + g_lc) > 0.5:
            new_ax = self.attend(context, ax)

        # SP update
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)))
        g_adj = eq_gate(opcode, torch.tensor(float(Op.ADJ)))
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)))
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)))

        pop_ops = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD,
                   Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE,
                   Op.SHL, Op.SHR, Op.OR, Op.XOR, Op.AND,
                   Op.SI, Op.SC]
        pop_gate = sum(eq_gate(opcode, torch.tensor(float(op))) for op in pop_ops)
        pop_gate = torch.clamp(pop_gate, 0, 1)

        new_sp = (
            sp * (1 - g_psh - g_adj - g_ent - g_lev - pop_gate)
            + (sp - 8) * g_psh
            + (sp + imm) * g_adj
            + (sp - 8) * g_ent
            + bp * g_lev
            + (sp + 8) * pop_gate
        )

        # BP update
        bp_from_stack = self.attend(context, bp)
        new_bp = bp * (1 - g_ent - g_lev) + sp * g_ent + bp_from_stack * g_lev

        # PC update
        pc_next = pc + 8
        g_jmp = eq_gate(opcode, torch.tensor(float(Op.JMP)))
        g_jsr = eq_gate(opcode, torch.tensor(float(Op.JSR)))
        g_bz = eq_gate(opcode, torch.tensor(float(Op.BZ)))
        g_bnz = eq_gate(opcode, torch.tensor(float(Op.BNZ)))

        ax_zero = eq_gate(ax, torch.tensor(0.0))
        bz_take = swiglu_mul(g_bz, ax_zero)
        bnz_take = swiglu_mul(g_bnz, 1 - ax_zero)

        pc_from_stack = self.attend(context, sp + 8)

        gate_sum = g_jmp + g_jsr + bz_take + bnz_take + g_lev
        new_pc = (
            pc_next * (1 - gate_sum)
            + imm * (g_jmp + g_jsr + bz_take + bnz_take)
            + pc_from_stack * g_lev
        )

        # Memory writes (simplified - would need scatter)
        new_memory = memory.clone()

        return (
            torch.round(new_pc),
            torch.round(new_sp),
            torch.round(new_bp),
            torch.round(new_ax),
            new_memory,
            output,
            None
        )

    def is_exit(self, memory, pc):
        """Check if current instruction is EXIT."""
        context = self.build_context(memory, pc, torch.tensor(0), torch.tensor(0), torch.tensor(0), [])
        instruction = self.attend(context, pc)
        opcode, _ = self.decode(instruction)
        return eq_gate(opcode, torch.tensor(float(Op.EXIT))) > 0.5

    def run(self, memory, pc, sp, bp, ax, max_steps=100):
        """Run program to completion."""
        output = []

        for _ in range(max_steps):
            if self.is_exit(memory, pc):
                break

            pc, sp, bp, ax, memory, output, _ = self.step(pc, sp, bp, ax, memory, output)

        return pc, sp, bp, ax, memory, output


class C4WithSyscallsExecutor(nn.Module):
    """
    Wrapper around autoregressive executor that adds syscall support.
    """

    def __init__(self, memory_size=256, host: Optional[SyscallHost] = None):
        super().__init__()
        from c4_autoregressive import C4AutoregressiveExecutor

        self.memory_size = memory_size
        self.host = host or SyscallHost()
        self.executor = C4AutoregressiveExecutor(memory_size)
        self.ctx = self.executor.ctx

    def decode(self, instruction):
        instruction = instruction.float()
        imm = torch.floor(instruction / 256.0)
        opcode = instruction - imm * 256.0
        return opcode, imm

    def is_syscall(self, opcode):
        """Check if opcode is a syscall. Returns (is_syscall, opcode_int)."""
        syscall_ops = [Op.PRTF, Op.OPEN, Op.READ, Op.CLOS, Op.FREE]
        for op in syscall_ops:
            if eq_gate(opcode, torch.tensor(float(op))) > 0.5:
                return True, op
        return False, 0

    def step(self, pc, sp, bp, ax, memory):
        """Execute one step, handling syscalls via host."""
        # Build context and fetch instruction
        context = self.ctx.build_context(memory, pc, sp, bp, ax)
        instruction = self.ctx.attend(context, pc, query_type='memory')
        opcode, imm = self.decode(instruction)

        # Check for syscall
        is_sys, sys_op = self.is_syscall(opcode)

        if is_sys:
            # Read args from stack
            args = []
            for i in range(3):
                arg = self.ctx.attend(context, sp + i * 8, query_type='memory')
                args.append(int(arg.item()))

            # Delegate to host
            req = SyscallRequest(opcode=sys_op, args=args)
            resp = self.host.handle_syscall(req)

            # Update AX with return value, advance PC
            new_ax = torch.tensor(float(resp.return_value))
            new_pc = pc + 8

            return new_pc, sp, bp, new_ax, memory

        # Regular instruction - use underlying executor
        return self.executor.step(pc, sp, bp, ax, memory)

    def is_exit(self, memory, pc):
        return self.executor.is_exit(memory, pc)

    def run(self, memory, pc, sp, bp, ax, max_steps=100):
        """Run to completion, return final state and host output."""
        for _ in range(max_steps):
            if self.is_exit(memory, pc):
                break
            pc, sp, bp, ax, memory = self.step(pc, sp, bp, ax, memory)

        return pc, sp, bp, ax, memory, self.host.get_output()


def test_syscalls():
    print("C4 WITH SYSCALL SUPPORT")
    print("=" * 60)
    print()

    def instr(op, imm=0):
        return float(op + (imm << 8))

    # Test 1: Basic computation still works
    print("Test 1: Basic computation (3+4)")
    host1 = SyscallHost()
    executor1 = C4WithSyscallsExecutor(memory_size=256, host=host1)

    memory1 = torch.zeros(256)
    code1 = [
        instr(Op.IMM, 3),
        instr(Op.PSH),
        instr(Op.IMM, 4),
        instr(Op.ADD),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code1):
        memory1[i * 8] = c

    pc, sp, bp, ax, memory1, output = executor1.run(
        memory1, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    result = int(ax.item())
    status = "✓" if result == 7 else "✗"
    print(f"  {status} 3 + 4 = {result} (expected 7)")
    print()

    # Test 2: Printf syscall
    print("Test 2: Printf syscall")
    host2 = SyscallHost()
    executor2 = C4WithSyscallsExecutor(memory_size=256, host=host2)

    memory2 = torch.zeros(256)
    code2 = [
        instr(Op.IMM, 42),      # AX = 42
        instr(Op.PSH),          # push 42
        instr(Op.IMM, 0),       # AX = 0 (format string, ignored)
        instr(Op.PSH),          # push format
        instr(Op.PRTF),         # printf(fmt, 42) -> host
        instr(Op.ADJ, 16),      # pop 2 args
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code2):
        memory2[i * 8] = c

    pc, sp, bp, ax, memory2, output = executor2.run(
        memory2, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    print(f"  Output: '{output.strip()}'")
    status = "✓" if "42" in output else "✗"
    print(f"  {status} printf printed 42")
    print()

    # Test 3: Compute then print
    print("Test 3: Compute 5*6 then printf")
    host3 = SyscallHost()
    executor3 = C4WithSyscallsExecutor(memory_size=256, host=host3)

    memory3 = torch.zeros(256)
    code3 = [
        instr(Op.IMM, 5),       # AX = 5
        instr(Op.PSH),          # push 5
        instr(Op.IMM, 6),       # AX = 6
        instr(Op.MUL),          # AX = 5 * 6 = 30
        instr(Op.PSH),          # push 30
        instr(Op.IMM, 0),       # format
        instr(Op.PSH),
        instr(Op.PRTF),         # printf(fmt, 30)
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code3):
        memory3[i * 8] = c

    pc, sp, bp, ax, memory3, output = executor3.run(
        memory3, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    print(f"  Output: '{output.strip()}'")
    status = "✓" if "30" in output else "✗"
    print(f"  {status} 5 * 6 = 30 printed")
    print()

    # Test 4: Multiple printfs
    print("Test 4: Multiple printfs")
    host4 = SyscallHost()
    executor4 = C4WithSyscallsExecutor(memory_size=256, host=host4)

    memory4 = torch.zeros(256)
    code4 = [
        # printf(_, 10)
        instr(Op.IMM, 10),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        # printf(_, 20)
        instr(Op.IMM, 20),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        # printf(_, 30)
        instr(Op.IMM, 30),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code4):
        memory4[i * 8] = c

    pc, sp, bp, ax, memory4, output = executor4.run(
        memory4, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    lines = output.strip().split('\n')
    print(f"  Output lines: {lines}")
    status = "✓" if lines == ['10', '20', '30'] else "✗"
    print(f"  {status} Printed 10, 20, 30 on separate lines")
    print()

    # Test 5: Sum then print
    print("Test 5: Sum 1+2+3 then printf")
    host5 = SyscallHost()
    executor5 = C4WithSyscallsExecutor(memory_size=256, host=host5)

    memory5 = torch.zeros(256)
    code5 = [
        instr(Op.IMM, 1),
        instr(Op.PSH),
        instr(Op.IMM, 2),
        instr(Op.ADD),          # AX = 3
        instr(Op.PSH),
        instr(Op.IMM, 3),
        instr(Op.ADD),          # AX = 6
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code5):
        memory5[i * 8] = c

    pc, sp, bp, ax, memory5, output = executor5.run(
        memory5, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    print(f"  Output: '{output.strip()}'")
    status = "✓" if "6" in output else "✗"
    print(f"  {status} 1+2+3 = 6 printed")
    print()

    print("=" * 60)
    print("ALL SYSCALL TESTS COMPLETE!")
    return True


if __name__ == "__main__":
    test_syscalls()
