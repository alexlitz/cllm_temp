"""
C4 VM with Mixture of Experts routing.

Instead of computing all 39 opcode outcomes and gating,
use MoE to route to only the needed expert.

Router: opcode -> one-hot selection
Experts: one per opcode (or grouped by type)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from c4_vm import Op


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


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


def ge_gate(a, b, scale=20.0):
    return silu_threshold((a - b).float() + 0.5, scale)


def gt_gate(a, b, scale=20.0):
    return silu_threshold((a - b).float() - 0.5, scale)


class OpcodeRouter(nn.Module):
    """
    MoE Router: routes to the correct opcode expert.

    Uses soft one-hot via eq_gate for each opcode.
    Returns routing weights and selected expert indices.
    """

    def __init__(self, num_opcodes=40, scale=20.0, top_k=1):
        super().__init__()
        self.num_opcodes = num_opcodes
        self.scale = scale
        self.top_k = top_k

    def forward(self, opcode):
        """
        Returns routing weights for each opcode.
        In practice, only one should be ~1.
        """
        opcode = opcode.float()
        weights = torch.zeros(self.num_opcodes)

        for i in range(self.num_opcodes):
            weights[i] = eq_gate(opcode, torch.tensor(float(i)), self.scale)

        # Normalize (should already sum to ~1)
        weights = weights / (weights.sum() + 1e-8)

        return weights

    def get_top_k(self, opcode):
        """Get top-k expert indices (for sparse computation)."""
        weights = self.forward(opcode)
        _, indices = torch.topk(weights, self.top_k)
        return indices, weights[indices]


class ArithmeticExpert(nn.Module):
    """Expert for arithmetic ops: ADD, SUB, MUL, DIV, MOD"""

    def __init__(self, max_quotient=64):
        super().__init__()
        self.max_quotient = max_quotient

    def forward(self, op, stack_top, ax):
        """Compute arithmetic result based on op."""
        stack_top, ax = stack_top.float(), ax.float()

        # Compute all arithmetic results
        results = {
            Op.ADD: stack_top + ax,
            Op.SUB: stack_top - ax,
            Op.MUL: swiglu_mul(stack_top, ax),
        }

        # Division with zero protection
        ax_safe = ax + eq_gate(ax, torch.tensor(0.0)) * 1e-8

        # Simple division for small quotients
        div_result = torch.zeros_like(stack_top)
        for q in range(self.max_quotient):
            lower = q * ax_safe
            upper = (q + 1) * ax_safe
            gate = ge_gate(stack_top, lower) * gt_gate(upper, stack_top)
            div_result = div_result + gate * q

        results[Op.DIV] = div_result
        results[Op.MOD] = stack_top - swiglu_mul(div_result, ax_safe)

        # Select based on op
        result = torch.zeros_like(stack_top)
        for op_code, value in results.items():
            gate = eq_gate(torch.tensor(float(op)), torch.tensor(float(op_code)))
            result = result + gate * value

        return result


class ComparisonExpert(nn.Module):
    """Expert for comparison ops: EQ, NE, LT, GT, LE, GE"""

    def forward(self, op, stack_top, ax):
        stack_top, ax = stack_top.float(), ax.float()

        results = {
            Op.EQ: eq_gate(stack_top, ax),
            Op.NE: 1 - eq_gate(stack_top, ax),
            Op.LT: gt_gate(ax, stack_top),
            Op.GT: gt_gate(stack_top, ax),
            Op.LE: ge_gate(ax, stack_top),
            Op.GE: ge_gate(stack_top, ax),
        }

        result = torch.zeros_like(stack_top)
        for op_code, value in results.items():
            gate = eq_gate(torch.tensor(float(op)), torch.tensor(float(op_code)))
            result = result + gate * value

        return result


class ShiftExpert(nn.Module):
    """Expert for shift ops: SHL, SHR"""

    def __init__(self, max_shift=32, scale=20.0):
        super().__init__()
        self.max_shift = max_shift
        self.scale = scale
        self.register_buffer('powers', 2.0 ** torch.arange(max_shift))

    def _pulse(self, x, center):
        s = self.scale
        diff = x - center
        t1 = silu_threshold(diff + 0.5, s)
        t2 = silu_threshold(-diff + 0.5, s)
        return swiglu_mul(t1, t2)

    def forward(self, op, stack_top, ax):
        stack_top, ax = stack_top.float(), ax.float()

        # Left shift
        shl_result = torch.zeros_like(stack_top)
        for i in range(self.max_shift):
            gate = self._pulse(ax, float(i))
            shl_result = shl_result + swiglu_mul(stack_top, self.powers[i]) * gate

        # Right shift
        shr_result = torch.zeros_like(stack_top)
        for i in range(self.max_shift):
            gate = self._pulse(ax, float(i))
            shr_result = shr_result + torch.floor(stack_top / self.powers[i]) * gate

        # Select
        g_shl = eq_gate(torch.tensor(float(op)), torch.tensor(float(Op.SHL)))
        g_shr = eq_gate(torch.tensor(float(op)), torch.tensor(float(Op.SHR)))

        return shl_result * g_shl + shr_result * g_shr


class BitwiseExpert(nn.Module):
    """Expert for bitwise ops: AND, OR, XOR (simplified for single bits)"""

    def forward(self, op, stack_top, ax):
        stack_top, ax = stack_top.float(), ax.float()

        # Simplified bitwise (works for 0/1, approximates for larger)
        and_result = swiglu_mul(stack_top, ax)
        or_result = stack_top + ax - swiglu_mul(stack_top, ax)
        xor_result = stack_top + ax - 2 * swiglu_mul(stack_top, ax)

        g_and = eq_gate(torch.tensor(float(op)), torch.tensor(float(Op.AND)))
        g_or = eq_gate(torch.tensor(float(op)), torch.tensor(float(Op.OR)))
        g_xor = eq_gate(torch.tensor(float(op)), torch.tensor(float(Op.XOR)))

        return and_result * g_and + or_result * g_or + xor_result * g_xor


class C4MoEExecutor(nn.Module):
    """
    C4 Executor using Mixture of Experts.

    Expert groups:
    1. Immediate/LEA (no computation)
    2. Arithmetic (ADD, SUB, MUL, DIV, MOD)
    3. Comparison (EQ, NE, LT, GT, LE, GE)
    4. Shift (SHL, SHR)
    5. Bitwise (AND, OR, XOR)
    6. Memory (LI, LC, SI, SC)
    7. Stack (PSH, ENT, ADJ, LEV)
    8. Control (JMP, JSR, BZ, BNZ)
    """

    def __init__(self, memory_size=512, max_quotient=64):
        super().__init__()
        self.memory_size = memory_size
        self.scale = 20.0

        # Expert modules
        self.arith_expert = ArithmeticExpert(max_quotient)
        self.cmp_expert = ComparisonExpert()
        self.shift_expert = ShiftExpert()
        self.bitwise_expert = BitwiseExpert()

        # Opcode groups for routing
        self.arith_ops = {Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD}
        self.cmp_ops = {Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE}
        self.shift_ops = {Op.SHL, Op.SHR}
        self.bitwise_ops = {Op.AND, Op.OR, Op.XOR}

    def read_memory(self, memory, addr):
        addr = addr.float()
        result = torch.tensor(0.0)
        for i in range(min(len(memory), 512)):
            gate = eq_gate(addr, torch.tensor(float(i)), self.scale)
            result = result + gate * memory[i].float()
        return result

    def write_memory(self, memory, addr, value, write_gate=1.0):
        addr, value = addr.float(), value.float()
        new_memory = memory.float().clone()
        for i in range(min(len(memory), 512)):
            gate = eq_gate(addr, torch.tensor(float(i)), self.scale) * write_gate
            new_memory[i] = new_memory[i] * (1 - gate) + value * gate
        return new_memory

    def route_and_compute_ax(self, opcode, imm, stack_top, ax, bp):
        """
        MoE routing: determine which expert to use and compute result.
        """
        opcode_val = int(opcode.item())

        # Check which expert group (using Python for routing efficiency)
        # In a pure transformer, this would use gating
        if opcode_val in self.arith_ops:
            return self.arith_expert(opcode, stack_top, ax)
        elif opcode_val in self.cmp_ops:
            return self.cmp_expert(opcode, stack_top, ax)
        elif opcode_val in self.shift_ops:
            return self.shift_expert(opcode, stack_top, ax)
        elif opcode_val in self.bitwise_ops:
            return self.bitwise_expert(opcode, stack_top, ax)
        elif opcode_val == Op.IMM:
            return imm
        elif opcode_val == Op.LEA:
            return bp + imm
        elif opcode_val == Op.LI or opcode_val == Op.LC:
            # Memory load handled separately
            return ax
        else:
            return ax

    def step(self, pc, sp, bp, ax, memory):
        """Execute one instruction with MoE routing."""
        pc, sp, bp, ax = pc.float(), sp.float(), bp.float(), ax.float()

        # Fetch
        instruction = self.read_memory(memory, pc)
        opcode = torch.remainder(instruction, 256)
        imm = torch.floor(instruction / 256)

        # Stack top
        stack_top = self.read_memory(memory, sp)

        # Route to expert and compute new AX
        new_ax = self.route_and_compute_ax(opcode, imm, stack_top, ax, bp)

        # Memory load (special handling)
        g_li = eq_gate(opcode, torch.tensor(float(Op.LI)), self.scale)
        g_lc = eq_gate(opcode, torch.tensor(float(Op.LC)), self.scale)
        if g_li > 0.5 or g_lc > 0.5:
            new_ax = self.read_memory(memory, ax)

        # Stack operations
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)), self.scale)
        g_adj = eq_gate(opcode, torch.tensor(float(Op.ADJ)), self.scale)
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)), self.scale)
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)), self.scale)

        # Binary ops that pop
        pops = sum(eq_gate(opcode, torch.tensor(float(op)), self.scale)
                   for op in list(self.arith_ops) + list(self.cmp_ops) +
                   list(self.shift_ops) + list(self.bitwise_ops))

        new_sp = (
            sp * (1 - g_psh - g_adj - g_ent - g_lev - pops)
            + (sp - 8) * g_psh
            + (sp + imm) * g_adj
            + (sp - imm) * g_ent
            + bp * g_lev
            + (sp + 8) * pops
        )

        # Base pointer
        bp_from_stack = self.read_memory(memory, bp)
        new_bp = (
            bp * (1 - g_ent - g_lev)
            + sp * g_ent
            + bp_from_stack * g_lev
        )

        # Program counter
        pc_next = pc + 8
        g_jmp = eq_gate(opcode, torch.tensor(float(Op.JMP)), self.scale)
        g_jsr = eq_gate(opcode, torch.tensor(float(Op.JSR)), self.scale)
        g_bz = eq_gate(opcode, torch.tensor(float(Op.BZ)), self.scale)
        g_bnz = eq_gate(opcode, torch.tensor(float(Op.BNZ)), self.scale)

        bz_take = swiglu_mul(g_bz, eq_gate(ax, torch.tensor(0.0), self.scale))
        bnz_take = swiglu_mul(g_bnz, 1 - eq_gate(ax, torch.tensor(0.0), self.scale))

        pc_from_stack = self.read_memory(memory, sp + 8)

        new_pc = (
            pc_next * (1 - g_jmp - g_jsr - bz_take - bnz_take - g_lev)
            + imm * g_jmp
            + imm * g_jsr
            + imm * bz_take
            + imm * bnz_take
            + pc_from_stack * g_lev
        )

        # Memory writes
        new_memory = memory.float().clone()

        # PSH
        new_memory = self.write_memory(new_memory, sp - 8, ax, g_psh)
        # JSR
        new_memory = self.write_memory(new_memory, sp - 8, pc_next, g_jsr)
        # ENT
        new_memory = self.write_memory(new_memory, sp - 8, bp, g_ent)
        # SI
        g_si = eq_gate(opcode, torch.tensor(float(Op.SI)), self.scale)
        new_memory = self.write_memory(new_memory, stack_top, ax, g_si)
        # SC
        g_sc = eq_gate(opcode, torch.tensor(float(Op.SC)), self.scale)
        new_memory = self.write_memory(new_memory, stack_top, ax, g_sc)

        return (
            torch.round(new_pc),
            torch.round(new_sp),
            torch.round(new_bp),
            torch.round(new_ax),
            torch.round(new_memory)
        )

    def is_exit(self, memory, pc):
        instruction = self.read_memory(memory, pc)
        opcode = torch.remainder(instruction, 256)
        return eq_gate(opcode, torch.tensor(float(Op.EXIT)), self.scale) > 0.5


def test_moe():
    """Test C4 with MoE routing."""
    print("C4 WITH MOE ROUTING")
    print("=" * 60)
    print()
    print("Experts: Arithmetic, Comparison, Shift, Bitwise")
    print("Router selects expert based on opcode group")
    print()

    executor = C4MoEExecutor(memory_size=512, max_quotient=64)

    def make_instr(opcode, imm=0):
        return opcode | (imm << 8)

    def run_program(name, bytecode, expected):
        memory = torch.zeros(512)
        for i, instr in enumerate(bytecode):
            memory[i * 8] = float(instr)

        pc = torch.tensor(0.0)
        sp = torch.tensor(256.0)
        bp = torch.tensor(256.0)
        ax = torch.tensor(0.0)

        for step in range(100):
            if executor.is_exit(memory, pc):
                break
            pc, sp, bp, ax, memory = executor.step(pc, sp, bp, ax, memory)

        result = int(ax.item())
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name}: {result} (expected {expected})")
        return result == expected

    tests = [
        ("(3+4)*5", [
            make_instr(Op.IMM, 3),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 4),
            make_instr(Op.ADD),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 5),
            make_instr(Op.MUL),
            make_instr(Op.EXIT),
        ], 35),
        ("10-3", [
            make_instr(Op.IMM, 10),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 3),
            make_instr(Op.SUB),
            make_instr(Op.EXIT),
        ], 7),
        ("20/4", [
            make_instr(Op.IMM, 20),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 4),
            make_instr(Op.DIV),
            make_instr(Op.EXIT),
        ], 5),
        ("5>3", [
            make_instr(Op.IMM, 5),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 3),
            make_instr(Op.GT),
            make_instr(Op.EXIT),
        ], 1),
        ("3<<2", [
            make_instr(Op.IMM, 3),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 2),
            make_instr(Op.SHL),
            make_instr(Op.EXIT),
        ], 12),
    ]

    passed = all(run_program(name, code, exp) for name, code, exp in tests)

    print()
    if passed:
        print("SUCCESS: C4 MoE routing works!")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    test_moe()
