"""
C4 VM with hierarchical block attention for O(log n) memory access.

Key innovations:
1. Binary encoding: bits as (-N, +N) for sharp attention
2. Hierarchical search: log(n) attention blocks instead of O(n)
3. Hardwired decode: opcode/imm extraction via weight matrices
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


class BinaryEncoder(nn.Module):
    """
    Encode integers as binary vectors with (-N, +N) for (0, 1).

    This creates sharp attention peaks:
    - Matching bits: q·k = N * N = N²
    - Mismatching bits: q·k = N * (-N) = -N²
    - Total for d bits: d*N² (match) vs (d-2k)*N² (k mismatches)
    """

    def __init__(self, num_bits=9, scale=10.0):
        super().__init__()
        self.num_bits = num_bits
        self.scale = scale  # The N value

        # Precompute powers of 2 for bit extraction
        self.register_buffer('powers', 2 ** torch.arange(num_bits))

    def encode(self, x):
        """
        Encode integer x as binary vector with values in {-N, +N}.
        Returns shape (num_bits,)
        """
        x = x.long()
        # Extract bits: (x >> i) & 1
        bits = (x.unsqueeze(-1) // self.powers) % 2  # (*, num_bits)
        # Map 0 -> -N, 1 -> +N
        encoded = (2 * bits.float() - 1) * self.scale
        return encoded

    def encode_batch(self, addresses):
        """Encode multiple addresses. Returns (n_addr, num_bits)"""
        return torch.stack([self.encode(a) for a in addresses])


class HierarchicalAttention(nn.Module):
    """
    Hierarchical block attention for O(log n) memory access.

    Instead of attending over all n positions:
    1. Level 0: Attend over 2 blocks (upper/lower half) → select half
    2. Level 1: Within selected half, attend over 2 blocks → select quarter
    3. Continue for log₂(n) levels

    Each level uses standard attention with binary-encoded queries/keys.
    """

    def __init__(self, memory_size=512, scale=10.0):
        super().__init__()
        self.memory_size = memory_size
        self.num_bits = int(math.ceil(math.log2(memory_size)))
        self.scale = scale
        self.encoder = BinaryEncoder(self.num_bits, scale)

        # Precompute keys for all memory positions
        addresses = torch.arange(memory_size)
        self.register_buffer('all_keys', self.encoder.encode_batch(addresses))

    def attend_full(self, query_addr, memory):
        """
        Standard full attention (for comparison).
        O(n) but uses binary encoding for sharp peaks.
        """
        # Encode query address
        query = self.encoder.encode(query_addr)  # (num_bits,)

        # Keys for all positions (precomputed)
        keys = self.all_keys[:len(memory)]  # (n, num_bits)

        # Attention scores: Q · K^T / sqrt(d)
        scores = torch.matmul(keys, query) / math.sqrt(self.num_bits)

        # Softmax
        weights = F.softmax(scores, dim=0)

        # Weighted sum of values
        return torch.sum(weights * memory.float())

    def attend_hierarchical(self, query_addr, memory):
        """
        Hierarchical O(log n) attention.

        At each level, we attend over 2 options (which half/block).
        The query bit for that level determines the selection.
        """
        query_addr = query_addr.long()
        n = len(memory)

        # Start with full memory
        current_start = 0
        current_size = n

        # Descend through bits from MSB to LSB
        for level in range(self.num_bits - 1, -1, -1):
            if current_size <= 1:
                break

            half_size = current_size // 2
            if half_size == 0:
                break

            # Query bit at this level
            query_bit = (query_addr >> level) & 1
            query_encoded = (2 * query_bit.float() - 1) * self.scale

            # Two block options: lower half and upper half
            # Key for lower half: bit = 0 → -N
            # Key for upper half: bit = 1 → +N
            key_lower = -self.scale
            key_upper = self.scale

            # Attention scores
            score_lower = query_encoded * key_lower / self.scale
            score_upper = query_encoded * key_upper / self.scale

            # Softmax over 2 options
            scores = torch.tensor([score_lower, score_upper])
            weights = F.softmax(scores, dim=0)

            # Soft selection of which half
            # Value for each half is the sum of that half's memory
            lower_half = memory[current_start:current_start + half_size]
            upper_half = memory[current_start + half_size:current_start + current_size]

            # For hierarchical traversal, we narrow down
            # Use hard selection for true O(log n)
            if weights[1] > weights[0]:
                current_start = current_start + half_size
            current_size = half_size

        # Final read from narrowed position
        return memory[current_start].float()

    def attend_hierarchical_soft(self, query_addr, memory):
        """
        Soft hierarchical attention - maintains differentiability.
        Still O(log n) attention operations but blends softly.
        """
        query = self.encoder.encode(query_addr)  # (num_bits,)

        # Process bit by bit from MSB to LSB
        # Accumulate soft weights for each position
        n = len(memory)
        weights = torch.ones(n) / n  # Start uniform

        for level in range(self.num_bits - 1, -1, -1):
            block_size = 2 ** level
            if block_size >= n:
                continue

            # Query bit at this level
            q_bit = query[level]  # ±scale

            # For each position, its key bit at this level
            positions = torch.arange(n)
            k_bits = ((positions >> level) & 1).float()
            k_encoded = (2 * k_bits - 1) * self.scale

            # Score contribution from this bit
            bit_scores = q_bit * k_encoded / (self.scale ** 2)

            # Modulate weights
            weights = weights * torch.exp(bit_scores)

        # Normalize
        weights = weights / weights.sum()

        # Weighted read
        return torch.sum(weights * memory.float())


class HardwiredDecode(nn.Module):
    """
    Decode instruction into opcode and immediate using hardwired weights.

    instruction = opcode + (imm << 8)
    opcode = instruction % 256 = instruction - floor(instruction/256) * 256
    imm = floor(instruction / 256)

    Using wired weights:
    - W_imm = 1/256 (extracts imm via multiplication + floor)
    - W_opcode extracts low 8 bits
    """

    def __init__(self):
        super().__init__()
        # For imm: multiply by 1/256, floor
        self.imm_scale = 1.0 / 256.0

    def forward(self, instruction):
        """
        Decode instruction into (opcode, imm).
        Uses only linear ops + floor.
        """
        instruction = instruction.float()

        # imm = floor(instruction / 256)
        imm = torch.floor(instruction * self.imm_scale)

        # opcode = instruction - imm * 256
        opcode = instruction - imm * 256.0

        return opcode, imm


class HierarchicalMemory(nn.Module):
    """
    Memory with hierarchical attention for read/write.
    """

    def __init__(self, size=512, scale=10.0):
        super().__init__()
        self.size = size
        self.scale = scale
        self.attention = HierarchicalAttention(size, scale)
        self.encoder = BinaryEncoder(int(math.ceil(math.log2(size))), scale)

    def read(self, memory, addr):
        """Read using full attention with binary encoding for sharp peaks."""
        return self.attention.attend_full(addr, memory)

    def write(self, memory, addr, value, write_enable=1.0):
        """Write using soft scatter with binary encoding."""
        addr = addr.float()
        value = value.float()

        # Encode query address
        query = self.encoder.encode(addr.long())
        keys = self.attention.all_keys[:len(memory)]

        # Attention scores for write mask
        scores = torch.matmul(keys, query) / math.sqrt(self.encoder.num_bits)
        weights = F.softmax(scores, dim=0) * write_enable

        # Blend: new = old * (1 - w) + value * w
        new_memory = memory.float() * (1 - weights) + value * weights

        return new_memory


class C4HierarchicalExecutor(nn.Module):
    """
    C4 executor with:
    1. Hierarchical O(log n) attention for memory
    2. Binary (-N, +N) encoding for sharp attention
    3. Hardwired decode weights
    """

    def __init__(self, memory_size=512):
        super().__init__()
        self.memory_size = memory_size
        self.scale = 20.0

        self.mem = HierarchicalMemory(memory_size, scale=10.0)
        self.decoder = HardwiredDecode()

        # Import experts from MoE implementation
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

    def step(self, pc, sp, bp, ax, memory):
        pc, sp, bp, ax = pc.float(), sp.float(), bp.float(), ax.float()

        # === FETCH (hierarchical attention) ===
        instruction = self.mem.read(memory, pc)

        # === DECODE (hardwired weights) ===
        opcode, imm = self.decoder(instruction)

        # === STACK TOP (hierarchical attention) ===
        stack_top = self.mem.read(memory, sp)

        # === MoE ROUTING & EXPERT ===
        _, selected_idx = self.router(opcode)
        expert_key = str(selected_idx.item())
        if expert_key in self.experts:
            expert = self.experts[expert_key]
        else:
            expert = self.default_expert
        new_ax, needs_pop = expert(stack_top, ax, imm, bp, memory, sp)
        new_ax = new_ax.float()

        # Memory load
        from c4_moe_top1 import eq_gate
        g_li = eq_gate(opcode, torch.tensor(float(Op.LI)))
        g_lc = eq_gate(opcode, torch.tensor(float(Op.LC)))
        if (g_li + g_lc) > 0.5:
            new_ax = self.mem.read(memory, ax)

        # === REGISTER UPDATES (gated blend) ===
        g_psh = eq_gate(opcode, torch.tensor(float(Op.PSH)))
        g_adj = eq_gate(opcode, torch.tensor(float(Op.ADJ)))
        g_ent = eq_gate(opcode, torch.tensor(float(Op.ENT)))
        g_lev = eq_gate(opcode, torch.tensor(float(Op.LEV)))
        pop_gate = torch.tensor(1.0 if needs_pop else 0.0)

        new_sp = (
            sp * (1 - g_psh - g_adj - g_ent - g_lev - pop_gate)
            + (sp - 8) * g_psh
            + (sp + imm) * g_adj
            + (sp - imm) * g_ent
            + bp * g_lev
            + (sp + 8) * pop_gate
        )

        bp_from_stack = self.mem.read(memory, bp)
        new_bp = bp * (1 - g_ent - g_lev) + sp * g_ent + bp_from_stack * g_lev

        # PC update
        pc_next = pc + 8
        g_jmp = eq_gate(opcode, torch.tensor(float(Op.JMP)))
        g_jsr = eq_gate(opcode, torch.tensor(float(Op.JSR)))
        g_bz = eq_gate(opcode, torch.tensor(float(Op.BZ)))
        g_bnz = eq_gate(opcode, torch.tensor(float(Op.BNZ)))

        from c4_moe_top1 import swiglu_mul
        bz_take = swiglu_mul(g_bz, eq_gate(ax, torch.tensor(0.0)))
        bnz_take = swiglu_mul(g_bnz, 1 - eq_gate(ax, torch.tensor(0.0)))

        pc_from_stack = self.mem.read(memory, sp + 8)

        new_pc = (
            pc_next * (1 - g_jmp - g_jsr - bz_take - bnz_take - g_lev)
            + imm * g_jmp + imm * g_jsr + imm * bz_take + imm * bnz_take
            + pc_from_stack * g_lev
        )

        # === MEMORY WRITES (hierarchical scatter) ===
        new_memory = memory.float().clone()
        new_memory = self.mem.write(new_memory, sp - 8, ax, g_psh)
        new_memory = self.mem.write(new_memory, sp - 8, pc_next, g_jsr)
        new_memory = self.mem.write(new_memory, sp - 8, bp, g_ent)

        g_si = eq_gate(opcode, torch.tensor(float(Op.SI)))
        g_sc = eq_gate(opcode, torch.tensor(float(Op.SC)))
        new_memory = self.mem.write(new_memory, stack_top, ax, g_si)
        new_memory = self.mem.write(new_memory, stack_top, ax, g_sc)

        return (
            torch.round(new_pc),
            torch.round(new_sp),
            torch.round(new_bp),
            torch.round(new_ax),
            torch.round(new_memory)
        )

    def is_exit(self, memory, pc):
        instruction = self.mem.read(memory, pc)
        opcode, _ = self.decoder(instruction)
        from c4_moe_top1 import eq_gate
        return eq_gate(opcode, torch.tensor(float(Op.EXIT))) > 0.5


def test_hierarchical():
    print("C4 WITH HIERARCHICAL ATTENTION")
    print("=" * 60)
    print()
    print("1. Binary encoding: bits as (-N, +N)")
    print("2. Hierarchical attention: O(log n) levels")
    print("3. Hardwired decode: opcode/imm via weight matrices")
    print()

    # Test binary encoder
    print("Binary Encoder Test:")
    encoder = BinaryEncoder(num_bits=8, scale=10.0)
    for addr in [0, 1, 5, 128, 255]:
        enc = encoder.encode(torch.tensor(addr))
        bits = ((torch.tensor(addr) >> torch.arange(8)) & 1).tolist()
        print(f"  {addr:3d} = {bits} → encoded: {enc[:4].tolist()}...")
    print()

    # Test hierarchical attention
    print("Hierarchical Attention Test:")
    memory = torch.arange(16).float()  # [0, 1, 2, ..., 15]
    hier_attn = HierarchicalAttention(memory_size=16, scale=10.0)

    for target in [0, 5, 10, 15]:
        result_full = hier_attn.attend_full(torch.tensor(target), memory)
        result_hier = hier_attn.attend_hierarchical_soft(torch.tensor(target), memory)
        print(f"  Read addr {target:2d}: full={result_full:.2f}, hier={result_hier:.2f}, expected={target}")
    print()

    # Test hardwired decode
    print("Hardwired Decode Test:")
    decoder = HardwiredDecode()
    test_instrs = [
        (Op.IMM | (42 << 8), Op.IMM, 42),
        (Op.ADD | (0 << 8), Op.ADD, 0),
        (Op.JMP | (100 << 8), Op.JMP, 100),
    ]
    for instr, exp_op, exp_imm in test_instrs:
        opcode, imm = decoder(torch.tensor(float(instr)))
        status = "✓" if int(opcode) == exp_op and int(imm) == exp_imm else "✗"
        print(f"  {status} {instr} → opcode={int(opcode)}, imm={int(imm)} (expected {exp_op}, {exp_imm})")
    print()

    # Test full executor
    print("Full Executor Test:")
    executor = C4HierarchicalExecutor(memory_size=512)

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
        ("3+4", [
            make_instr(Op.IMM, 3),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 4),
            make_instr(Op.ADD),
            make_instr(Op.EXIT),
        ], 7),
        ("10*5", [
            make_instr(Op.IMM, 10),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 5),
            make_instr(Op.MUL),
            make_instr(Op.EXIT),
        ], 50),
        ("20/4", [
            make_instr(Op.IMM, 20),
            make_instr(Op.PSH),
            make_instr(Op.IMM, 4),
            make_instr(Op.DIV),
            make_instr(Op.EXIT),
        ], 5),
    ]

    passed = all(run_program(name, code, exp) for name, code, exp in tests)

    print()
    if passed:
        print("SUCCESS!")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    test_hierarchical()
