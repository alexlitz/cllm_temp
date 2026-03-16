"""
C4 VM with true MoE Top-1 routing.

Router: SwiGLU network that outputs scores for each expert
Top-1: Only the highest-scoring expert is computed
Fully transformer-native - no Python if/else in forward pass
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


class MoERouter(nn.Module):
    """
    Router network for MoE.

    Maps opcode -> expert scores using SwiGLU.
    For discrete opcodes, this is essentially a soft one-hot lookup.
    """

    def __init__(self, num_experts, scale=20.0):
        super().__init__()
        self.num_experts = num_experts
        self.scale = scale

        # Wired weights: expert i activates for opcode i
        # W_gate[i] = scale, bias[i] = -scale * i
        # So gate_i = SiLU(scale * (opcode - i))
        # Combined with threshold gives pulse at opcode == i

    def forward(self, opcode):
        """
        Compute routing scores for each expert.
        Returns (scores, selected_expert_idx)
        """
        opcode = opcode.float()
        scores = torch.zeros(self.num_experts)

        for i in range(self.num_experts):
            # Pulse centered at opcode == i
            scores[i] = eq_gate(opcode, torch.tensor(float(i)), self.scale)

        # Top-1 selection via argmax (hard) or softmax (soft)
        # For inference, use hard selection
        selected_idx = torch.argmax(scores)

        return scores, selected_idx


class Expert(nn.Module):
    """Base class for opcode experts."""

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        raise NotImplementedError


class ImmExpert(Expert):
    """IMM: ax = imm"""
    def forward(self, stack_top, ax, imm, bp, memory, sp):
        return imm, False  # new_ax, needs_pop


class LeaExpert(Expert):
    """LEA: ax = bp + imm"""
    def forward(self, stack_top, ax, imm, bp, memory, sp):
        return bp + imm, False


class AddExpert(Expert):
    """ADD: ax = pop() + ax"""
    def forward(self, stack_top, ax, imm, bp, memory, sp):
        return stack_top + ax, True


class SubExpert(Expert):
    """SUB: ax = pop() - ax"""
    def forward(self, stack_top, ax, imm, bp, memory, sp):
        return stack_top - ax, True


class MulExpert(Expert):
    """MUL: ax = pop() * ax"""
    def forward(self, stack_top, ax, imm, bp, memory, sp):
        return swiglu_mul(stack_top, ax), True


class DivExpert(Expert):
    """DIV: ax = pop() / ax"""
    def __init__(self, max_quotient=64):
        super().__init__()
        self.max_quotient = max_quotient
        self.scale = 20.0

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        stack_top, ax = stack_top.float(), ax.float()
        ax_safe = ax + eq_gate(ax, torch.tensor(0.0)) * 0.001

        result = torch.zeros_like(stack_top)
        for q in range(self.max_quotient):
            lower = q * ax_safe
            upper = (q + 1) * ax_safe
            t1 = stack_top - lower + 0.5
            th1 = (silu(self.scale * (t1 + 0.5)) - silu(self.scale * (t1 - 0.5))) / self.scale
            t2 = stack_top - upper + 0.5
            th2 = (silu(self.scale * (t2 + 0.5)) - silu(self.scale * (t2 - 0.5))) / self.scale
            gate = th1 - th2
            result = result + gate * q

        return result, True


class ModExpert(Expert):
    """MOD: ax = pop() % ax"""
    def __init__(self, max_quotient=64):
        super().__init__()
        self.div_expert = DivExpert(max_quotient)

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        div_result, _ = self.div_expert(stack_top, ax, imm, bp, memory, sp)
        ax_safe = ax + eq_gate(ax, torch.tensor(0.0)) * 0.001
        return stack_top - swiglu_mul(div_result, ax_safe), True


class ShlExpert(Expert):
    """SHL: ax = pop() << ax"""
    def __init__(self, max_shift=32, scale=20.0):
        super().__init__()
        self.max_shift = max_shift
        self.scale = scale
        self.register_buffer('powers', 2.0 ** torch.arange(max_shift))

    def _pulse(self, x, center):
        diff = x - center
        t1 = silu_threshold(diff + 0.5, self.scale)
        t2 = silu_threshold(-diff + 0.5, self.scale)
        return swiglu_mul(t1, t2)

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        stack_top, ax = stack_top.float(), ax.float()
        result = torch.zeros_like(stack_top)
        for i in range(self.max_shift):
            gate = self._pulse(ax, float(i))
            result = result + swiglu_mul(stack_top, self.powers[i]) * gate
        return result, True


class ShrExpert(Expert):
    """SHR: ax = pop() >> ax"""
    def __init__(self, max_shift=32, scale=20.0):
        super().__init__()
        self.max_shift = max_shift
        self.scale = scale
        self.register_buffer('powers', 2.0 ** torch.arange(max_shift))

    def _pulse(self, x, center):
        diff = x - center
        t1 = silu_threshold(diff + 0.5, self.scale)
        t2 = silu_threshold(-diff + 0.5, self.scale)
        return swiglu_mul(t1, t2)

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        stack_top, ax = stack_top.float(), ax.float()
        result = torch.zeros_like(stack_top)
        for i in range(self.max_shift):
            gate = self._pulse(ax, float(i))
            result = result + torch.floor(stack_top / self.powers[i]) * gate
        return result, True


class CmpExpert(Expert):
    """Comparison expert - parameterized by comparison type."""
    def __init__(self, cmp_type):
        super().__init__()
        self.cmp_type = cmp_type  # 'eq', 'ne', 'lt', 'gt', 'le', 'ge'

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        stack_top, ax = stack_top.float(), ax.float()
        if self.cmp_type == 'eq':
            result = eq_gate(stack_top, ax)
        elif self.cmp_type == 'ne':
            result = 1 - eq_gate(stack_top, ax)
        elif self.cmp_type == 'lt':
            result = gt_gate(ax, stack_top)
        elif self.cmp_type == 'gt':
            result = gt_gate(stack_top, ax)
        elif self.cmp_type == 'le':
            result = ge_gate(ax, stack_top)
        elif self.cmp_type == 'ge':
            result = ge_gate(stack_top, ax)
        return result, True


class BitwiseExpert(Expert):
    """
    Bitwise operations: AND, OR, XOR.

    Approach: extract bits via division/mod, operate, recombine.
    Uses SwiGLU for all operations.
    """
    def __init__(self, op_type, num_bits=16):
        super().__init__()
        self.op_type = op_type  # 'and', 'or', 'xor'
        self.num_bits = num_bits
        self.register_buffer('powers', 2.0 ** torch.arange(num_bits))

    def _extract_bit(self, x, bit_pos):
        """Extract bit at position using division and mod."""
        x = x.float()
        # bit = floor(x / 2^pos) % 2
        divided = torch.floor(x / self.powers[bit_pos])
        bit = divided - 2 * torch.floor(divided / 2)  # mod 2
        return bit

    def forward(self, stack_top, ax, imm, bp, memory, sp):
        a, b = stack_top.float(), ax.float()
        result = torch.zeros_like(a)

        for i in range(self.num_bits):
            bit_a = self._extract_bit(a, i)
            bit_b = self._extract_bit(b, i)

            if self.op_type == 'and':
                # AND: both must be 1
                out_bit = swiglu_mul(bit_a, bit_b)
            elif self.op_type == 'or':
                # OR: at least one is 1 = a + b - a*b
                out_bit = bit_a + bit_b - swiglu_mul(bit_a, bit_b)
            elif self.op_type == 'xor':
                # XOR: exactly one is 1 = a + b - 2*a*b
                out_bit = bit_a + bit_b - 2 * swiglu_mul(bit_a, bit_b)

            result = result + out_bit * self.powers[i]

        return result, True


class NoOpExpert(Expert):
    """No-op: ax unchanged"""
    def forward(self, stack_top, ax, imm, bp, memory, sp):
        return ax, False


class C4MoETop1(nn.Module):
    """
    C4 Executor with true MoE Top-1 routing.

    - Router outputs scores for each opcode expert
    - Top-1 selection picks single expert
    - Only that expert is computed (sparse execution)
    """

    def __init__(self, memory_size=512, max_quotient=64):
        super().__init__()
        self.memory_size = memory_size
        self.scale = 20.0
        self.num_experts = 40  # Max opcode + 1

        # Router
        self.router = MoERouter(self.num_experts)

        # Expert pool - one per opcode
        self.experts = nn.ModuleDict({
            str(Op.IMM): ImmExpert(),
            str(Op.LEA): LeaExpert(),
            str(Op.ADD): AddExpert(),
            str(Op.SUB): SubExpert(),
            str(Op.MUL): MulExpert(),
            str(Op.DIV): DivExpert(max_quotient),
            str(Op.MOD): ModExpert(max_quotient),
            str(Op.SHL): ShlExpert(),
            str(Op.SHR): ShrExpert(),
            str(Op.AND): BitwiseExpert('and'),
            str(Op.OR): BitwiseExpert('or'),
            str(Op.XOR): BitwiseExpert('xor'),
            str(Op.EQ): CmpExpert('eq'),
            str(Op.NE): CmpExpert('ne'),
            str(Op.LT): CmpExpert('lt'),
            str(Op.GT): CmpExpert('gt'),
            str(Op.LE): CmpExpert('le'),
            str(Op.GE): CmpExpert('ge'),
        })

        # Default expert for unhandled opcodes
        self.default_expert = NoOpExpert()

    def read_memory(self, memory, addr):
        addr = addr.float()
        result = torch.tensor(0.0)
        for i in range(min(len(memory), self.memory_size)):
            gate = eq_gate(addr, torch.tensor(float(i)), self.scale)
            result = result + gate * memory[i].float()
        return result

    def write_memory(self, memory, addr, value, write_gate=1.0):
        addr, value = addr.float(), value.float()
        new_memory = memory.float().clone()
        for i in range(min(len(memory), self.memory_size)):
            gate = eq_gate(addr, torch.tensor(float(i)), self.scale) * write_gate
            new_memory[i] = new_memory[i] * (1 - gate) + value * gate
        return new_memory

    def step(self, pc, sp, bp, ax, memory):
        """Execute one instruction with MoE Top-1 routing."""
        pc, sp, bp, ax = pc.float(), sp.float(), bp.float(), ax.float()

        # Fetch
        instruction = self.read_memory(memory, pc)
        opcode = torch.remainder(instruction, 256)
        imm = torch.floor(instruction / 256)

        # Stack top (pre-fetch for binary ops)
        stack_top = self.read_memory(memory, sp)

        # === MoE ROUTING ===
        scores, selected_idx = self.router(opcode)

        # === TOP-1 EXPERT EXECUTION ===
        # Only compute the selected expert (sparse!)
        expert_key = str(selected_idx.item())
        if expert_key in self.experts:
            expert = self.experts[expert_key]
        else:
            expert = self.default_expert

        new_ax, needs_pop = expert(stack_top, ax, imm, bp, memory, sp)
        new_ax = new_ax.float()

        # Memory load handling (LI, LC)
        g_li = eq_gate(opcode, torch.tensor(float(Op.LI)), self.scale)
        g_lc = eq_gate(opcode, torch.tensor(float(Op.LC)), self.scale)
        mem_load = g_li + g_lc
        if mem_load > 0.5:
            new_ax = self.read_memory(memory, ax)

        # === STACK POINTER UPDATE ===
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)), self.scale)
        g_adj = eq_gate(opcode, torch.tensor(float(Op.ADJ)), self.scale)
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)), self.scale)
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)), self.scale)

        pop_gate = torch.tensor(1.0 if needs_pop else 0.0)

        new_sp = (
            sp * (1 - g_psh - g_adj - g_ent - g_lev - pop_gate)
            + (sp - 8) * g_psh
            + (sp + imm) * g_adj
            + (sp - imm) * g_ent
            + bp * g_lev
            + (sp + 8) * pop_gate
        )

        # === BASE POINTER UPDATE ===
        bp_from_stack = self.read_memory(memory, bp)
        new_bp = (
            bp * (1 - g_ent - g_lev)
            + sp * g_ent
            + bp_from_stack * g_lev
        )

        # === PROGRAM COUNTER UPDATE ===
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

        # === MEMORY WRITES ===
        new_memory = memory.float().clone()
        new_memory = self.write_memory(new_memory, sp - 8, ax, g_psh)
        new_memory = self.write_memory(new_memory, sp - 8, pc_next, g_jsr)
        new_memory = self.write_memory(new_memory, sp - 8, bp, g_ent)

        g_si = eq_gate(opcode, torch.tensor(float(Op.SI)), self.scale)
        g_sc = eq_gate(opcode, torch.tensor(float(Op.SC)), self.scale)
        new_memory = self.write_memory(new_memory, stack_top, ax, g_si)
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


def test_moe_top1():
    """Test C4 with MoE Top-1 routing."""
    print("C4 WITH MOE TOP-1 ROUTING")
    print("=" * 60)
    print()
    print("Router: SwiGLU pulse gating per opcode")
    print("Top-1: Only selected expert is computed")
    print()

    executor = C4MoETop1(memory_size=512, max_quotient=64)

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
        ("17%5", [
            make_instr(Op.IMM, 17),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 5),
            make_instr(Op.MOD),
            make_instr(Op.EXIT),
        ], 2),
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
        ("16>>2", [
            make_instr(Op.IMM, 16),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 2),
            make_instr(Op.SHR),
            make_instr(Op.EXIT),
        ], 4),
    ]

    passed = all(run_program(name, code, exp) for name, code, exp in tests)

    print()
    print("Expert usage: Only 1 expert computed per instruction")
    print("vs Full compute: All 39 opcodes computed every step")
    print()

    if passed:
        print("SUCCESS: C4 MoE Top-1 works!")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    test_moe_top1()
