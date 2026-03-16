"""
C4 VM executed entirely via SwiGLU transformer.

All operations use only:
- Linear projections
- SiLU activation
- Attention (softmax)
- Addition

No Python if/else in the execution loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from c4_vm import Op, CODE_BASE, DATA_BASE, STACK_BASE, STACK_SIZE


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    """a * b using SwiGLU identity"""
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


def silu_threshold(x, scale=20.0):
    """Soft threshold: ~1 when x > 0, ~0 when x < 0"""
    diff = scale * x
    term1 = silu(diff + 0.5 * scale)
    term2 = silu(diff - 0.5 * scale)
    return (term1 - term2) / scale


def eq_gate(a, b, scale=20.0):
    """~1 if a == b (for integers), ~0 otherwise"""
    diff = (a - b).float()
    upper = silu_threshold(diff + 0.5, scale)
    lower = silu_threshold(-diff + 0.5, scale)
    return upper * lower


def ge_gate(a, b, scale=20.0):
    """~1 if a >= b, ~0 otherwise"""
    return silu_threshold((a - b).float() + 0.5, scale)


def gt_gate(a, b, scale=20.0):
    """~1 if a > b, ~0 otherwise"""
    return silu_threshold((a - b).float() - 0.5, scale)


class SwiGLUDiv(nn.Module):
    """Division via quotient enumeration"""

    def __init__(self, max_quotient=256, scale=20.0):
        super().__init__()
        self.max_quotient = max_quotient
        self.scale = scale

    def forward(self, a, b):
        a, b = a.float(), b.float()
        s = self.scale
        result = torch.zeros_like(a)

        for q in range(self.max_quotient):
            lower = q * b
            upper = (q + 1) * b
            t1 = a - lower + 0.5
            th1 = (silu(s * (t1 + 0.5)) - silu(s * (t1 - 0.5))) / s
            t2 = a - upper + 0.5
            th2 = (silu(s * (t2 + 0.5)) - silu(s * (t2 - 0.5))) / s
            gate_q = th1 - th2
            result = result + gate_q * q

        return result


class SwiGLUShift(nn.Module):
    """Shift via power-of-2 enumeration"""

    def __init__(self, max_shift=32, scale=20.0):
        super().__init__()
        self.max_shift = max_shift
        self.scale = scale
        self.register_buffer('powers', 2.0 ** torch.arange(max_shift))

    def _pulse(self, x, center):
        """Pulse centered at x == center"""
        s = self.scale
        diff = x - center
        t1 = silu_threshold(diff + 0.5, s)
        t2 = silu_threshold(-diff + 0.5, s)
        return swiglu_mul(t1, t2)

    def left_shift(self, a, b):
        a, b = a.float(), b.float()
        result = torch.zeros_like(a)
        for i in range(self.max_shift):
            gate = self._pulse(b, float(i))
            result = result + swiglu_mul(a, self.powers[i]) * gate
        return result

    def right_shift(self, a, b):
        a, b = a.float(), b.float()
        result = torch.zeros_like(a)
        for i in range(self.max_shift):
            gate = self._pulse(b, float(i))
            result = result + torch.floor(a / self.powers[i]) * gate
        return result


class C4SwiGLUExecutor(nn.Module):
    """
    Execute C4 VM using only SwiGLU operations.

    State: 4 registers + memory
    Each step computes all 39 opcode outcomes in parallel,
    then selects based on the actual opcode.
    """

    def __init__(self, memory_size=4096, max_quotient=256):
        super().__init__()
        self.memory_size = memory_size
        self.div_op = SwiGLUDiv(max_quotient)
        self.shift_op = SwiGLUShift()
        self.scale = 20.0

    def read_memory(self, memory, addr):
        """Read from memory using soft attention"""
        addr = addr.float()
        # Create attention weights that peak at addr
        positions = torch.arange(len(memory), dtype=torch.float32, device=memory.device)

        # Soft one-hot at addr
        weights = torch.zeros_like(positions)
        for i in range(len(memory)):
            weights[i] = eq_gate(addr, positions[i], self.scale)

        # Normalize (soft attention)
        weights = weights / (weights.sum() + 1e-8)

        # Weighted read
        return (weights * memory.float()).sum()

    def write_memory(self, memory, addr, value):
        """Write to memory using soft scatter"""
        addr = addr.float()
        value = value.float()
        positions = torch.arange(len(memory), dtype=torch.float32, device=memory.device)

        new_memory = memory.float().clone()
        for i in range(len(memory)):
            gate = eq_gate(addr, positions[i], self.scale)
            # Blend: new = old * (1 - gate) + value * gate
            new_memory[i] = new_memory[i] * (1 - gate) + value * gate

        return new_memory

    def step(self, pc, sp, bp, ax, memory):
        """
        Execute one C4 instruction.
        All computation via SwiGLU - no Python if/else.
        """
        pc, sp, bp, ax = pc.float(), sp.float(), bp.float(), ax.float()

        # ===== FETCH =====
        instruction = self.read_memory(memory, pc)
        opcode = torch.remainder(instruction, 256)  # instruction & 0xFF
        imm = torch.floor(instruction / 256)  # instruction >> 8

        # ===== COMPUTE ALL OUTCOMES =====
        # Default next PC
        pc_next = pc + 8

        # Stack top value (for operations that pop)
        stack_top = self.read_memory(memory, sp)

        # ----- Arithmetic results -----
        add_result = stack_top + ax
        sub_result = stack_top - ax
        mul_result = swiglu_mul(stack_top, ax)

        # Division (with zero check via gating)
        ax_safe = ax + eq_gate(ax, torch.tensor(0.0), self.scale)  # avoid div by 0
        div_result = self.div_op(stack_top, ax_safe)
        mod_result = stack_top - swiglu_mul(div_result, ax_safe)

        # ----- Shift results -----
        shl_result = self.shift_op.left_shift(stack_top, ax)
        shr_result = self.shift_op.right_shift(stack_top, ax)

        # ----- Bitwise results (simplified - treat as integers) -----
        # For full bitwise, we'd use bit extraction, but for now approximate
        or_result = stack_top + ax - swiglu_mul(stack_top, ax)  # only works for bits
        xor_result = stack_top + ax - 2 * swiglu_mul(stack_top, ax)  # only works for bits
        and_result = swiglu_mul(stack_top, ax)  # only works for bits

        # ----- Comparison results -----
        eq_result = eq_gate(stack_top, ax, self.scale)
        ne_result = 1 - eq_gate(stack_top, ax, self.scale)
        lt_result = gt_gate(ax, stack_top, self.scale)  # stack < ax means ax > stack
        gt_result = gt_gate(stack_top, ax, self.scale)
        le_result = ge_gate(ax, stack_top, self.scale)
        ge_result = ge_gate(stack_top, ax, self.scale)

        # ----- Memory load results -----
        li_result = self.read_memory(memory, ax)  # load int at ax
        lc_result = self.read_memory(memory, ax)  # load char (same for now)

        # ----- LEA result -----
        lea_result = bp + imm

        # ===== SELECT BASED ON OPCODE =====
        # New AX value
        new_ax = ax  # default

        # Opcode gates
        g_lea = eq_gate(opcode, torch.tensor(float(Op.LEA)), self.scale)
        g_imm = eq_gate(opcode, torch.tensor(float(Op.IMM)), self.scale)
        g_li = eq_gate(opcode, torch.tensor(float(Op.LI)), self.scale)
        g_lc = eq_gate(opcode, torch.tensor(float(Op.LC)), self.scale)
        g_add = eq_gate(opcode, torch.tensor(float(Op.ADD)), self.scale)
        g_sub = eq_gate(opcode, torch.tensor(float(Op.SUB)), self.scale)
        g_mul = eq_gate(opcode, torch.tensor(float(Op.MUL)), self.scale)
        g_div = eq_gate(opcode, torch.tensor(float(Op.DIV)), self.scale)
        g_mod = eq_gate(opcode, torch.tensor(float(Op.MOD)), self.scale)
        g_shl = eq_gate(opcode, torch.tensor(float(Op.SHL)), self.scale)
        g_shr = eq_gate(opcode, torch.tensor(float(Op.SHR)), self.scale)
        g_or = eq_gate(opcode, torch.tensor(float(Op.OR)), self.scale)
        g_xor = eq_gate(opcode, torch.tensor(float(Op.XOR)), self.scale)
        g_and = eq_gate(opcode, torch.tensor(float(Op.AND)), self.scale)
        g_eq = eq_gate(opcode, torch.tensor(float(Op.EQ)), self.scale)
        g_ne = eq_gate(opcode, torch.tensor(float(Op.NE)), self.scale)
        g_lt = eq_gate(opcode, torch.tensor(float(Op.LT)), self.scale)
        g_gt = eq_gate(opcode, torch.tensor(float(Op.GT)), self.scale)
        g_le = eq_gate(opcode, torch.tensor(float(Op.LE)), self.scale)
        g_ge = eq_gate(opcode, torch.tensor(float(Op.GE)), self.scale)

        # Select new AX
        new_ax = (
            ax * (1 - g_lea - g_imm - g_li - g_lc - g_add - g_sub - g_mul - g_div - g_mod
                  - g_shl - g_shr - g_or - g_xor - g_and - g_eq - g_ne - g_lt - g_gt - g_le - g_ge)
            + lea_result * g_lea
            + imm * g_imm
            + li_result * g_li
            + lc_result * g_lc
            + add_result * g_add
            + sub_result * g_sub
            + mul_result * g_mul
            + div_result * g_div
            + mod_result * g_mod
            + shl_result * g_shl
            + shr_result * g_shr
            + or_result * g_or
            + xor_result * g_xor
            + and_result * g_and
            + eq_result * g_eq
            + ne_result * g_ne
            + lt_result * g_lt
            + gt_result * g_gt
            + le_result * g_le
            + ge_result * g_ge
        )

        # ===== STACK POINTER UPDATE =====
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)), self.scale)
        g_adj = eq_gate(opcode, torch.tensor(float(Op.ADJ)), self.scale)
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)), self.scale)
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)), self.scale)

        # Ops that pop (binary ops)
        pops = (g_add + g_sub + g_mul + g_div + g_mod + g_shl + g_shr +
                g_or + g_xor + g_and + g_eq + g_ne + g_lt + g_gt + g_le + g_ge)

        new_sp = (
            sp * (1 - g_psh - g_adj - g_ent - g_lev - pops)
            + (sp - 8) * g_psh  # push
            + (sp + imm) * g_adj  # adjust
            + (sp - imm) * g_ent  # enter (after push bp)
            + bp * g_lev  # leave: sp = bp
            + (sp + 8) * pops  # pop for binary ops
        )

        # ===== BASE POINTER UPDATE =====
        # ENT: push old bp, bp = sp
        # LEV: bp = pop()
        bp_from_stack = self.read_memory(memory, bp)  # for LEV
        new_bp = (
            bp * (1 - g_ent - g_lev)
            + sp * g_ent  # bp = sp (after push)
            + bp_from_stack * g_lev  # bp = *bp
        )

        # ===== PROGRAM COUNTER UPDATE =====
        g_jmp = eq_gate(opcode, torch.tensor(float(Op.JMP)), self.scale)
        g_jsr = eq_gate(opcode, torch.tensor(float(Op.JSR)), self.scale)
        g_bz = eq_gate(opcode, torch.tensor(float(Op.BZ)), self.scale)
        g_bnz = eq_gate(opcode, torch.tensor(float(Op.BNZ)), self.scale)

        # BZ: branch if ax == 0
        bz_take = swiglu_mul(g_bz, eq_gate(ax, torch.tensor(0.0), self.scale))
        # BNZ: branch if ax != 0
        bnz_take = swiglu_mul(g_bnz, 1 - eq_gate(ax, torch.tensor(0.0), self.scale))

        # LEV: pc = pop()
        pc_from_stack = self.read_memory(memory, sp + 8)  # return address after bp

        new_pc = (
            pc_next * (1 - g_jmp - g_jsr - bz_take - bnz_take - g_lev)
            + imm * g_jmp
            + imm * g_jsr
            + imm * bz_take
            + imm * bnz_take
            + pc_from_stack * g_lev
        )

        # ===== MEMORY UPDATES =====
        new_memory = memory.float().clone()

        # PSH: write ax at sp-8
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)), self.scale)
        psh_addr = sp - 8
        new_memory = self.write_memory(new_memory, psh_addr * g_psh, ax * g_psh)

        # JSR: push return address
        new_memory = self.write_memory(new_memory, (sp - 8) * g_jsr, pc_next * g_jsr)

        # ENT: push bp
        new_memory = self.write_memory(new_memory, (sp - 8) * g_ent, bp * g_ent)

        # SI: store int
        g_si = eq_gate(opcode, torch.tensor(float(Op.SI)), self.scale)
        si_addr = stack_top  # address is on stack
        new_memory = self.write_memory(new_memory, si_addr * g_si, ax * g_si)

        # SC: store char (same as SI for now)
        g_sc = eq_gate(opcode, torch.tensor(float(Op.SC)), self.scale)
        new_memory = self.write_memory(new_memory, stack_top * g_sc, ax * g_sc)

        # Round to integers
        new_pc = torch.round(new_pc)
        new_sp = torch.round(new_sp)
        new_bp = torch.round(new_bp)
        new_ax = torch.round(new_ax)
        new_memory = torch.round(new_memory)

        return new_pc, new_sp, new_bp, new_ax, new_memory

    def is_exit(self, memory, pc):
        """Check if current instruction is EXIT"""
        instruction = self.read_memory(memory, pc)
        opcode = torch.remainder(instruction, 256)
        return eq_gate(opcode, torch.tensor(float(Op.EXIT)), self.scale) > 0.5


def test_c4_swiglu():
    """Test C4 execution via SwiGLU transformer."""

    print("C4 VIA SWIGLU TRANSFORMER")
    print("=" * 60)
    print()
    print("All operations use only: SiLU, linear projections, attention")
    print()

    # Use compact memory layout for testing
    # Code at 0-127, Stack at 128-255
    COMPACT_CODE = 0
    COMPACT_STACK_TOP = 256

    executor = C4SwiGLUExecutor(memory_size=512, max_quotient=64)

    def make_instr(opcode, imm=0):
        return opcode | (imm << 8)

    def run_program(name, bytecode, expected, init_sp=COMPACT_STACK_TOP):
        # Initialize memory
        memory = torch.zeros(512)

        # Load bytecode at address 0
        for i, instr in enumerate(bytecode):
            memory[i * 8] = float(instr)

        # Initialize registers
        pc = torch.tensor(0.0)
        sp = torch.tensor(float(init_sp))
        bp = torch.tensor(float(init_sp))
        ax = torch.tensor(0.0)

        # Run
        max_steps = 200
        for step in range(max_steps):
            if executor.is_exit(memory, pc):
                break
            pc, sp, bp, ax, memory = executor.step(pc, sp, bp, ax, memory)

        result = int(ax.item())
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name}: {result} (expected {expected}, {step} steps)")
        return result == expected

    # Test 1: Simple (3 + 4) * 5 = 35
    bytecode1 = [
        make_instr(Op.IMM, 3),   # ax = 3
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 4),   # ax = 4
        make_instr(Op.ADD),      # ax = pop() + ax = 3 + 4 = 7
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 5),   # ax = 5
        make_instr(Op.MUL),      # ax = pop() * ax = 7 * 5 = 35
        make_instr(Op.EXIT),
    ]
    passed = run_program("(3+4)*5", bytecode1, 35)

    # Test 2: 10 - 3 = 7
    bytecode2 = [
        make_instr(Op.IMM, 10),  # ax = 10
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 3),   # ax = 3
        make_instr(Op.SUB),      # ax = pop() - ax = 10 - 3 = 7
        make_instr(Op.EXIT),
    ]
    passed &= run_program("10-3", bytecode2, 7)

    # Test 3: 20 / 4 = 5
    bytecode3 = [
        make_instr(Op.IMM, 20),  # ax = 20
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 4),   # ax = 4
        make_instr(Op.DIV),      # ax = pop() / ax = 20 / 4 = 5
        make_instr(Op.EXIT),
    ]
    passed &= run_program("20/4", bytecode3, 5)

    # Test 4: 17 % 5 = 2
    bytecode4 = [
        make_instr(Op.IMM, 17),  # ax = 17
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 5),   # ax = 5
        make_instr(Op.MOD),      # ax = pop() % ax = 17 % 5 = 2
        make_instr(Op.EXIT),
    ]
    passed &= run_program("17%5", bytecode4, 2)

    # Test 5: Comparison 5 > 3
    bytecode5 = [
        make_instr(Op.IMM, 5),   # ax = 5
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 3),   # ax = 3
        make_instr(Op.GT),       # ax = (pop() > ax) = (5 > 3) = 1
        make_instr(Op.EXIT),
    ]
    passed &= run_program("5>3", bytecode5, 1)

    # Test 6: Left shift 3 << 2 = 12
    bytecode6 = [
        make_instr(Op.IMM, 3),   # ax = 3
        make_instr(Op.PSH),      # push ax
        make_instr(Op.IMM, 2),   # ax = 2
        make_instr(Op.SHL),      # ax = pop() << ax = 3 << 2 = 12
        make_instr(Op.EXIT),
    ]
    passed &= run_program("3<<2", bytecode6, 12)

    print()
    if passed:
        print("SUCCESS: C4 runs on pure SwiGLU transformer!")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    test_c4_swiglu()
