"""
C4 VM with autoregressive register generation.

Registers (PC, SP, BP, AX) are tokens in the context, generated in sequence
before each instruction executes. This matches how a real autoregressive
transformer would work.

Context layout:
  [memory_0, memory_1, ..., memory_n, PC, SP, BP, AX]

Each step:
  1. Generate new_PC (attending to full context)
  2. Generate new_SP (attending to context + new_PC)
  3. Generate new_BP (attending to context + new_PC + new_SP)
  4. Generate new_AX (attending to context + new_PC + new_SP + new_BP)
  5. Update memory if needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from c4_vm import Op


def silu(x):
    return x * torch.sigmoid(x)


def swiglu_mul(a, b):
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


def silu_threshold(x, scale=20.0):
    """Soft threshold: ~1 for x >= 0, ~0 for x < 0."""
    diff = scale * x
    term1 = silu(diff + 0.5 * scale)
    term2 = silu(diff - 0.5 * scale)
    return (term1 - term2) / scale


def eq_gate(a, b, scale=20.0):
    """Gate that's ~1 when a≈b, ~0 otherwise."""
    diff = (a - b).float()
    upper = silu_threshold(diff + 0.5, scale)
    lower = silu_threshold(-diff + 0.5, scale)
    return upper * lower


class BinaryEncoder(nn.Module):
    """Encode integers as (-N, +N) binary vectors for sharp attention."""

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


class AutoregressiveContext(nn.Module):
    """
    Manages the context sequence: [memory..., PC, SP, BP, AX]

    Each token is a vector: [value, type_tag_0, type_tag_1, type_tag_2, type_tag_3]

    Type tags (using -N/+N encoding):
      memory: [-N, -N, -N, -N]  (type 0)
      PC:     [+N, -N, -N, -N]  (type 1)
      SP:     [-N, +N, -N, -N]  (type 2)
      BP:     [-N, -N, +N, -N]  (type 3)
      AX:     [-N, -N, -N, +N]  (type 4)
    """

    def __init__(self, memory_size=256, scale=10.0):
        super().__init__()
        self.memory_size = memory_size
        self.scale = scale
        self.num_addr_bits = int(math.ceil(math.log2(memory_size + 4)))
        self.num_type_bits = 4  # 4 register types
        self.total_key_dim = self.num_addr_bits + self.num_type_bits

        self.addr_encoder = BinaryEncoder(self.num_addr_bits, scale)

        # Register position indices (after memory)
        self.PC_IDX = memory_size
        self.SP_IDX = memory_size + 1
        self.BP_IDX = memory_size + 2
        self.AX_IDX = memory_size + 3

        # Precompute address keys for all positions
        addresses = torch.arange(memory_size + 4)
        addr_keys = self.addr_encoder.encode(addresses.unsqueeze(0)).squeeze(0)

        # Build type tags: memory gets [-N,-N,-N,-N], registers get one-hot
        type_tags = torch.full((memory_size + 4, self.num_type_bits), -scale)
        type_tags[self.PC_IDX, 0] = scale   # PC: [+N, -N, -N, -N]
        type_tags[self.SP_IDX, 1] = scale   # SP: [-N, +N, -N, -N]
        type_tags[self.BP_IDX, 2] = scale   # BP: [-N, -N, +N, -N]
        type_tags[self.AX_IDX, 3] = scale   # AX: [-N, -N, -N, +N]

        # Full keys = [addr_encoding, type_tags]
        full_keys = torch.cat([addr_keys, type_tags], dim=1)
        self.register_buffer('all_keys', full_keys)

        # Type tag templates for queries
        self.register_buffer('memory_type', torch.tensor([-scale, -scale, -scale, -scale]))
        self.register_buffer('pc_type', torch.tensor([scale, -scale, -scale, -scale]))
        self.register_buffer('sp_type', torch.tensor([-scale, scale, -scale, -scale]))
        self.register_buffer('bp_type', torch.tensor([-scale, -scale, scale, -scale]))
        self.register_buffer('ax_type', torch.tensor([-scale, -scale, -scale, scale]))

    def build_context(self, memory, pc, sp, bp, ax):
        """Build full context sequence: [memory..., PC, SP, BP, AX]"""
        context = torch.zeros(self.memory_size + 4)
        context[:self.memory_size] = memory[:self.memory_size].float()
        context[self.PC_IDX] = pc.float()
        context[self.SP_IDX] = sp.float()
        context[self.BP_IDX] = bp.float()
        context[self.AX_IDX] = ax.float()
        return context

    def attend(self, context, query_addr, query_type='memory'):
        """
        Read from context using binary-encoded attention.
        query_type: 'memory', 'pc', 'sp', 'bp', or 'ax'
        """
        # Build query: [addr_encoding, type_tag]
        addr_query = self.addr_encoder.encode(query_addr)

        if query_type == 'memory':
            type_query = self.memory_type
        elif query_type == 'pc':
            type_query = self.pc_type
        elif query_type == 'sp':
            type_query = self.sp_type
        elif query_type == 'bp':
            type_query = self.bp_type
        elif query_type == 'ax':
            type_query = self.ax_type
        else:
            type_query = self.memory_type

        query = torch.cat([addr_query, type_query])

        keys = self.all_keys[:len(context)]
        scores = torch.matmul(keys, query) / math.sqrt(self.total_key_dim)
        weights = F.softmax(scores, dim=0)
        return torch.sum(weights * context)

    def attend_by_type(self, context, query_type):
        """Read a register directly by type (no address needed)."""
        if query_type == 'pc':
            type_query = self.pc_type
            dummy_addr = self.PC_IDX
        elif query_type == 'sp':
            type_query = self.sp_type
            dummy_addr = self.SP_IDX
        elif query_type == 'bp':
            type_query = self.bp_type
            dummy_addr = self.BP_IDX
        elif query_type == 'ax':
            type_query = self.ax_type
            dummy_addr = self.AX_IDX
        else:
            raise ValueError(f"Unknown type: {query_type}")

        addr_query = self.addr_encoder.encode(torch.tensor(dummy_addr))
        query = torch.cat([addr_query, type_query])

        keys = self.all_keys[:len(context)]
        scores = torch.matmul(keys, query) / math.sqrt(self.total_key_dim)
        weights = F.softmax(scores, dim=0)
        return torch.sum(weights * context)

    def scatter(self, context, addr, value, gate=1.0, query_type='memory'):
        """Write to context position with gating."""
        addr_query = self.addr_encoder.encode(addr)

        if query_type == 'memory':
            type_query = self.memory_type
        else:
            type_query = self.memory_type

        query = torch.cat([addr_query, type_query])

        keys = self.all_keys[:len(context)]
        scores = torch.matmul(keys, query) / math.sqrt(self.total_key_dim)
        weights = F.softmax(scores, dim=0)
        new_context = context * (1 - gate * weights) + value * gate * weights
        return new_context


class HardwiredDecode(nn.Module):
    """Decode instruction into opcode and immediate."""

    def __init__(self):
        super().__init__()
        self.imm_scale = 1.0 / 256.0

    def forward(self, instruction):
        instruction = instruction.float()
        imm = torch.floor(instruction * self.imm_scale)
        opcode = instruction - imm * 256.0
        return opcode, imm


class RegisterPredictor(nn.Module):
    """
    Predicts next register value by attending to context.

    Each register predictor sees the current context plus any
    previously predicted registers in this step.
    """

    def __init__(self, memory_size=256, scale=10.0):
        super().__init__()
        self.ctx = AutoregressiveContext(memory_size, scale)
        self.decoder = HardwiredDecode()

        # Import experts
        from c4_moe_top1 import (
            MoERouter, ImmExpert, LeaExpert, AddExpert, SubExpert,
            MulExpert, DivExpert, ModExpert, ShlExpert, ShrExpert,
            BitwiseExpert, CmpExpert, NoOpExpert
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
        self.default_expert = NoOpExpert()

    def predict_pc(self, context, pc, sp, bp, ax):
        """
        Predict next PC by attending to context.
        Output: new_PC token
        """
        # Fetch instruction at current PC
        instruction = self.ctx.attend(context, pc)
        opcode, imm = self.decoder(instruction)

        # PC update logic (gated)
        pc_next = pc + 8

        g_jmp = eq_gate(opcode, torch.tensor(float(Op.JMP)))
        g_jsr = eq_gate(opcode, torch.tensor(float(Op.JSR)))
        g_bz = eq_gate(opcode, torch.tensor(float(Op.BZ)))
        g_bnz = eq_gate(opcode, torch.tensor(float(Op.BNZ)))
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)))

        # Conditional branches
        ax_zero = eq_gate(ax, torch.tensor(0.0))
        bz_take = swiglu_mul(g_bz, ax_zero)
        bnz_take = swiglu_mul(g_bnz, 1 - ax_zero)

        # LEV returns to saved PC
        pc_from_stack = self.ctx.attend(context, sp + 8)

        gate_sum = g_jmp + g_jsr + bz_take + bnz_take + g_lev

        new_pc = (
            pc_next * (1 - gate_sum)
            + imm * (g_jmp + g_jsr + bz_take + bnz_take)
            + pc_from_stack * g_lev
        )

        return torch.round(new_pc)

    def predict_sp(self, context, pc, sp, bp, ax, new_pc):
        """
        Predict next SP, can attend to new_pc.
        Output: new_SP token
        """
        instruction = self.ctx.attend(context, pc)
        opcode, imm = self.decoder(instruction)

        # SP update gates
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)))
        g_adj = eq_gate(opcode, torch.tensor(float(Op.ADJ)))
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)))
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)))

        # Pop operations
        pop_ops = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD,
                   Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE,
                   Op.SHL, Op.SHR, Op.OR, Op.XOR, Op.AND]
        pop_gate = sum(eq_gate(opcode, torch.tensor(float(op))) for op in pop_ops)
        pop_gate = torch.clamp(pop_gate, 0, 1)

        # Also pop for SI, SC (store ops)
        g_si = eq_gate(opcode, torch.tensor(float(Op.SI)))
        g_sc = eq_gate(opcode, torch.tensor(float(Op.SC)))
        pop_gate = pop_gate + g_si + g_sc

        new_sp = (
            sp * (1 - g_psh - g_adj - g_ent - g_lev - pop_gate)
            + (sp - 8) * g_psh
            + (sp + imm) * g_adj
            + (sp - 8) * g_ent  # ENT pushes BP then allocates
            + bp * g_lev
            + (sp + 8) * pop_gate
        )

        return torch.round(new_sp)

    def predict_bp(self, context, pc, sp, bp, ax, new_pc, new_sp):
        """
        Predict next BP, can attend to new_pc, new_sp.
        Output: new_BP token
        """
        instruction = self.ctx.attend(context, pc)
        opcode, imm = self.decoder(instruction)

        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)))
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)))

        bp_from_stack = self.ctx.attend(context, bp)

        new_bp = (
            bp * (1 - g_ent - g_lev)
            + sp * g_ent  # BP = SP before allocation
            + bp_from_stack * g_lev
        )

        return torch.round(new_bp)

    def predict_ax(self, context, pc, sp, bp, ax, new_pc, new_sp, new_bp):
        """
        Predict next AX, can attend to all previous outputs.
        Output: new_AX token
        """
        instruction = self.ctx.attend(context, pc)
        opcode, imm = self.decoder(instruction)

        # Stack top for binary ops
        stack_top = self.ctx.attend(context, sp)

        # Route to expert
        _, selected_idx = self.router(opcode)
        expert_key = str(selected_idx.item())
        if expert_key in self.experts:
            expert = self.experts[expert_key]
        else:
            expert = self.default_expert

        new_ax, _ = expert(stack_top, ax, imm, bp, context[:self.ctx.memory_size], sp)
        new_ax = new_ax.float()

        # Memory load operations
        g_li = eq_gate(opcode, torch.tensor(float(Op.LI)))
        g_lc = eq_gate(opcode, torch.tensor(float(Op.LC)))
        if (g_li + g_lc) > 0.5:
            new_ax = self.ctx.attend(context, ax)

        return torch.round(new_ax)


class C4AutoregressiveExecutor(nn.Module):
    """
    Executes C4 by generating register tokens autoregressively.

    Each step:
      1. Attend to context, generate PC
      2. Attend to context + PC, generate SP
      3. Attend to context + PC + SP, generate BP
      4. Attend to context + PC + SP + BP, generate AX
      5. Update memory
    """

    def __init__(self, memory_size=256):
        super().__init__()
        self.memory_size = memory_size
        self.predictor = RegisterPredictor(memory_size)
        self.ctx = self.predictor.ctx
        self.decoder = HardwiredDecode()

    def step(self, pc, sp, bp, ax, memory):
        """
        Execute one step with autoregressive register generation.
        Returns: (new_pc, new_sp, new_bp, new_ax, new_memory)
        """
        pc, sp, bp, ax = pc.float(), sp.float(), bp.float(), ax.float()

        # Build current context
        context = self.ctx.build_context(memory, pc, sp, bp, ax)

        # === AUTOREGRESSIVE REGISTER GENERATION ===
        # Each register can attend to previously generated registers

        # 1. Generate PC
        new_pc = self.predictor.predict_pc(context, pc, sp, bp, ax)

        # 2. Generate SP (can see new_pc in extended context)
        new_sp = self.predictor.predict_sp(context, pc, sp, bp, ax, new_pc)

        # 3. Generate BP (can see new_pc, new_sp)
        new_bp = self.predictor.predict_bp(context, pc, sp, bp, ax, new_pc, new_sp)

        # 4. Generate AX (can see all previous)
        new_ax = self.predictor.predict_ax(context, pc, sp, bp, ax, new_pc, new_sp, new_bp)

        # === MEMORY UPDATES ===
        instruction = self.ctx.attend(context, pc)
        opcode, imm = self.decoder(instruction)

        new_memory = memory.float().clone()

        # PSH: push AX to stack
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)))
        new_memory = self.ctx.scatter(
            torch.cat([new_memory, torch.zeros(4)]),  # pad for register slots
            sp - 8, ax, g_psh
        )[:self.memory_size]

        # JSR: push return address
        g_jsr = eq_gate(opcode, torch.tensor(float(Op.JSR)))
        new_memory = self.ctx.scatter(
            torch.cat([new_memory, torch.zeros(4)]),
            sp - 8, pc + 8, g_jsr
        )[:self.memory_size]

        # ENT: push BP
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)))
        new_memory = self.ctx.scatter(
            torch.cat([new_memory, torch.zeros(4)]),
            sp - 8, bp, g_ent
        )[:self.memory_size]

        # SI/SC: store AX to address on stack
        g_si = eq_gate(opcode, torch.tensor(float(Op.SI)))
        g_sc = eq_gate(opcode, torch.tensor(float(Op.SC)))
        stack_top = self.ctx.attend(context, sp)
        new_memory = self.ctx.scatter(
            torch.cat([new_memory, torch.zeros(4)]),
            stack_top, ax, g_si + g_sc
        )[:self.memory_size]

        return (
            new_pc,
            new_sp,
            new_bp,
            new_ax,
            torch.round(new_memory)
        )

    def is_exit(self, memory, pc):
        context = self.ctx.build_context(memory, pc, torch.tensor(0), torch.tensor(0), torch.tensor(0))
        instruction = self.ctx.attend(context, pc)
        opcode, _ = self.decoder(instruction)
        return eq_gate(opcode, torch.tensor(float(Op.EXIT))) > 0.5


def test_autoregressive():
    print("C4 AUTOREGRESSIVE REGISTER GENERATION")
    print("=" * 60)
    print()
    print("Context: [memory_0, ..., memory_n, PC, SP, BP, AX]")
    print("Each step generates: PC → SP → BP → AX (in sequence)")
    print()

    # Test context building
    print("Context Structure Test:")
    ctx = AutoregressiveContext(memory_size=16)
    memory = torch.arange(16).float()
    context = ctx.build_context(memory, torch.tensor(100), torch.tensor(200),
                                torch.tensor(300), torch.tensor(400))
    print(f"  Memory positions 0-15: {context[:16].tolist()}")
    print(f"  PC (idx 16): {context[16].item()}")
    print(f"  SP (idx 17): {context[17].item()}")
    print(f"  BP (idx 18): {context[18].item()}")
    print(f"  AX (idx 19): {context[19].item()}")
    print()

    # Test type-tagged keys
    print("Type Tag Test:")
    print(f"  Key dim: {ctx.total_key_dim} (addr: {ctx.num_addr_bits}, type: {ctx.num_type_bits})")
    print(f"  Memory key[0] type bits: {ctx.all_keys[0, -4:].tolist()}")
    print(f"  PC key type bits:        {ctx.all_keys[ctx.PC_IDX, -4:].tolist()}")
    print(f"  SP key type bits:        {ctx.all_keys[ctx.SP_IDX, -4:].tolist()}")
    print(f"  BP key type bits:        {ctx.all_keys[ctx.BP_IDX, -4:].tolist()}")
    print(f"  AX key type bits:        {ctx.all_keys[ctx.AX_IDX, -4:].tolist()}")
    print()

    # Test attention read by type
    print("Attention Read Test:")
    val = ctx.attend(context, torch.tensor(5), query_type='memory')
    print(f"  Read memory[5]: {val.item():.2f} (expected 5)")
    val = ctx.attend_by_type(context, 'pc')
    print(f"  Read PC by type: {val.item():.2f} (expected 100)")
    val = ctx.attend_by_type(context, 'sp')
    print(f"  Read SP by type: {val.item():.2f} (expected 200)")
    val = ctx.attend_by_type(context, 'ax')
    print(f"  Read AX by type: {val.item():.2f} (expected 400)")
    print()

    # Full executor test
    print("Full Autoregressive Executor Test:")

    def run_program(name, code, expected):
        executor = C4AutoregressiveExecutor(memory_size=256)

        memory = torch.zeros(256)
        for i, instr in enumerate(code):
            memory[i * 8] = instr

        pc = torch.tensor(0.0)
        sp = torch.tensor(200.0)  # Stack starts at 200
        bp = torch.tensor(200.0)
        ax = torch.tensor(0.0)

        for step in range(50):
            if executor.is_exit(memory, pc):
                break
            pc, sp, bp, ax, memory = executor.step(pc, sp, bp, ax, memory)

        result = int(ax.item())
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name}: {result} (expected {expected})")
        return result == expected

    # Test programs: opcode + (imm << 8)
    def instr(op, imm=0):
        return float(op + (imm << 8))

    tests = [
        ("3+4", [
            instr(Op.IMM, 3),   # AX = 3
            instr(Op.PSH),      # push 3
            instr(Op.IMM, 4),   # AX = 4
            instr(Op.ADD),      # AX = 3 + 4
            instr(Op.EXIT),
        ], 7),

        ("10*5", [
            instr(Op.IMM, 10),
            instr(Op.PSH),
            instr(Op.IMM, 5),
            instr(Op.MUL),
            instr(Op.EXIT),
        ], 50),

        ("20/4", [
            instr(Op.IMM, 20),
            instr(Op.PSH),
            instr(Op.IMM, 4),
            instr(Op.DIV),
            instr(Op.EXIT),
        ], 5),

        ("15-7", [
            instr(Op.IMM, 15),
            instr(Op.PSH),
            instr(Op.IMM, 7),
            instr(Op.SUB),
            instr(Op.EXIT),
        ], 8),
    ]

    passed = all(run_program(name, code, exp) for name, code, exp in tests)
    print()

    if passed:
        print("SUCCESS!")
    else:
        print("FAILED")

    return passed


if __name__ == "__main__":
    test_autoregressive()
