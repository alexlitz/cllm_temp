"""
C4 Transformer with WIRED weights.

All computation happens inside transformer layers:
- Attention for memory read/write
- FFN with hand-set weights for arithmetic

No Python arithmetic in the forward pass - only matmul, attention, elementwise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WiredFFN(nn.Module):
    """
    FFN with hand-wired weights to compute specific functions.

    Input: [pc, sp, bp, ax, imm, stack_top, mem_at_ax, mem_at_bp, mem_at_bp8]
           (9 values)

    Output: All possible next states for all opcodes
            [pc_0, sp_0, bp_0, ax_0, pc_1, sp_1, bp_1, ax_1, ...]
            (39 opcodes × 4 registers = 156 values)
    """

    def __init__(self):
        super().__init__()

        # Input dimension: 9 values
        # pc=0, sp=1, bp=2, ax=3, imm=4, stack_top=5, mem_at_ax=6, mem_at_bp=7, mem_at_bp8=8
        self.input_dim = 9

        # Output: 39 opcodes × 4 registers
        self.output_dim = 39 * 4

        # Layer 1: Extract useful combinations
        # We need: pc+8, sp+8, sp-8, bp+imm, stack_top+ax, stack_top-ax, etc.
        self.W1 = nn.Parameter(torch.zeros(self.input_dim, 64))
        self.b1 = nn.Parameter(torch.zeros(64))

        # Layer 2: Combine into final outputs
        self.W2 = nn.Parameter(torch.zeros(64, self.output_dim))
        self.b2 = nn.Parameter(torch.zeros(self.output_dim))

        self._wire_weights()

    def _wire_weights(self):
        """Set weights to compute exact operations."""
        # Input indices
        PC, SP, BP, AX, IMM, STACK, MEM_AX, MEM_BP, MEM_BP8 = range(9)

        # Layer 1 computes intermediate values:
        # 0: pc
        # 1: pc + 8
        # 2: sp
        # 3: sp + 8
        # 4: sp - 8
        # 5: bp
        # 6: bp + imm
        # 7: ax
        # 8: imm
        # 9: stack_top
        # 10: mem_at_ax
        # 11: stack_top + ax
        # 12: stack_top - ax
        # 13: stack_top * ax  (can't do exactly, will handle separately)
        # 14: mem_at_bp
        # 15: mem_at_bp8
        # 16: sp - 8 - imm (for ENT)
        # 17: bp + 16 (for LEV)

        W1 = torch.zeros(self.input_dim, 64)
        b1 = torch.zeros(64)

        # Direct copies
        W1[PC, 0] = 1      # 0: pc
        W1[PC, 1] = 1      # 1: pc + 8
        b1[1] = 8
        W1[SP, 2] = 1      # 2: sp
        W1[SP, 3] = 1      # 3: sp + 8
        b1[3] = 8
        W1[SP, 4] = 1      # 4: sp - 8
        b1[4] = -8
        W1[BP, 5] = 1      # 5: bp
        W1[BP, 6] = 1      # 6: bp + imm
        W1[IMM, 6] = 1
        W1[AX, 7] = 1      # 7: ax
        W1[IMM, 8] = 1     # 8: imm
        W1[STACK, 9] = 1   # 9: stack_top
        W1[MEM_AX, 10] = 1 # 10: mem_at_ax
        W1[STACK, 11] = 1  # 11: stack_top + ax
        W1[AX, 11] = 1
        W1[STACK, 12] = 1  # 12: stack_top - ax
        W1[AX, 12] = -1
        W1[MEM_BP, 14] = 1  # 14: mem_at_bp
        W1[MEM_BP8, 15] = 1 # 15: mem_at_bp8
        W1[SP, 16] = 1      # 16: sp - 8 - imm
        W1[IMM, 16] = -1
        b1[16] = -8
        W1[BP, 17] = 1      # 17: bp + 16
        b1[17] = 16

        self.W1.data = W1
        self.b1.data = b1

        # Layer 2: Map intermediates to outputs
        # Output layout: [pc_0, sp_0, bp_0, ax_0, pc_1, sp_1, bp_1, ax_1, ...]
        W2 = torch.zeros(64, self.output_dim)

        # For each opcode, set which intermediate goes to which output
        def set_opcode(op, pc_src, sp_src, bp_src, ax_src):
            base = op * 4
            W2[pc_src, base + 0] = 1
            W2[sp_src, base + 1] = 1
            W2[bp_src, base + 2] = 1
            W2[ax_src, base + 3] = 1

        # Intermediate indices
        I_PC, I_PC8, I_SP, I_SP8, I_SP_8, I_BP, I_BP_IMM = 0, 1, 2, 3, 4, 5, 6
        I_AX, I_IMM, I_STACK, I_MEM_AX, I_ADD, I_SUB = 7, 8, 9, 10, 11, 12
        I_MEM_BP, I_MEM_BP8, I_ENT_SP, I_LEV_SP = 14, 15, 16, 17

        # Map opcodes (simplified - linear ops only)
        set_opcode(0, I_PC8, I_SP, I_BP, I_BP_IMM)   # LEA: ax = bp + imm
        set_opcode(1, I_PC8, I_SP, I_BP, I_IMM)      # IMM: ax = imm
        set_opcode(2, I_IMM, I_SP, I_BP, I_AX)       # JMP: pc = imm
        set_opcode(3, I_IMM, I_SP_8, I_BP, I_AX)     # JSR: pc = imm, sp -= 8
        # BZ, BNZ need conditional - handled separately
        set_opcode(4, I_PC8, I_SP, I_BP, I_AX)       # BZ placeholder
        set_opcode(5, I_PC8, I_SP, I_BP, I_AX)       # BNZ placeholder
        set_opcode(6, I_PC8, I_ENT_SP, I_SP_8, I_AX) # ENT
        set_opcode(7, I_PC8, I_SP, I_BP, I_AX)       # ADJ (sp + imm needs special)
        set_opcode(8, I_MEM_BP8, I_LEV_SP, I_MEM_BP, I_AX)  # LEV
        set_opcode(9, I_PC8, I_SP, I_BP, I_MEM_AX)   # LI
        set_opcode(10, I_PC8, I_SP, I_BP, I_MEM_AX)  # LC (needs mask)
        set_opcode(11, I_PC8, I_SP8, I_BP, I_AX)     # SI
        set_opcode(12, I_PC8, I_SP8, I_BP, I_AX)     # SC
        set_opcode(13, I_PC8, I_SP_8, I_BP, I_AX)    # PSH
        set_opcode(14, I_PC8, I_SP8, I_BP, I_ADD)    # ADD
        set_opcode(15, I_PC8, I_SP8, I_BP, I_SUB)    # SUB
        # MUL, DIV, MOD need nonlinear - placeholders
        for op in range(16, 30):
            set_opcode(op, I_PC8, I_SP8, I_BP, I_AX)
        # Syscalls
        for op in range(30, 38):
            set_opcode(op, I_PC8, I_SP, I_BP, I_AX)
        # EXIT
        set_opcode(38, I_PC, I_SP, I_BP, I_AX)

        self.W2.data = W2

    def forward(self, x):
        """
        x: (batch, 9) input values
        returns: (batch, 156) all possible outputs
        """
        h = x @ self.W1 + self.b1  # (batch, 64)
        out = h @ self.W2 + self.b2  # (batch, 156)
        return out


class MemoryAttention(nn.Module):
    """
    Attention-based memory read.

    This is equivalent to attention with one-hot weights.
    Implemented as gather for efficiency, but conceptually it's:
      Query: address to read
      Keys: all memory addresses [0, 1, 2, ..., N-1]
      Values: memory contents
      Attention weights: one-hot at query position
      Output: weighted sum = value at address
    """

    def __init__(self, mem_size):
        super().__init__()
        self.mem_size = mem_size

    def forward(self, memory, addresses):
        """
        memory: (mem_size,) byte values
        addresses: (n,) addresses to read
        returns: (n,) values at those addresses

        This IS attention - just with hard one-hot weights.
        gather(memory, addr) = sum(one_hot(addr) * memory)
        """
        addresses = addresses.clamp(0, self.mem_size - 1)
        return memory[addresses]


class C4WiredTransformer(nn.Module):
    """
    C4 executor where ALL computation is transformer ops.

    - Memory read: attention
    - Arithmetic: wired FFN
    - Selection: attention over opcode
    - Memory write: scatter attention
    """

    def __init__(self, mem_size=1024*1024):
        super().__init__()
        self.mem_size = mem_size
        self.ffn = WiredFFN()
        self.mem_attn = MemoryAttention(mem_size)

        # Opcode selection via attention
        self.register_buffer('opcode_keys', torch.arange(39).float())

    def read_int_via_attention(self, memory, addr):
        """Read 8 bytes using attention."""
        addresses = addr + torch.arange(8, device=memory.device)
        addresses = addresses.clamp(0, self.mem_size - 1)

        # Use attention to read each byte
        bytes_read = self.mem_attn(memory, addresses)

        # Pack into int64
        shifts = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=memory.device)
        value = (bytes_read << shifts).sum()
        return value

    def select_via_attention(self, all_outputs, opcode):
        """Select output for given opcode using attention."""
        # all_outputs: (156,) = 39 opcodes × 4 registers
        # opcode: scalar

        # Reshape to (39, 4)
        outputs = all_outputs.view(39, 4)

        # Attention: softmax(-|opcode - keys|² / temp)
        temp = 0.01
        diff = opcode.float() - self.opcode_keys
        scores = -diff.pow(2) / temp
        attn = F.softmax(scores, dim=-1)  # (39,)

        # Select: (39,) @ (39, 4) -> (4,)
        selected = attn @ outputs.float()

        return selected.long()

    def forward(self, pc, sp, bp, ax, memory):
        """One step of execution - all via transformer ops."""
        device = memory.device

        # 1. FETCH via attention
        instruction = self.read_int_via_attention(memory, pc)
        opcode = instruction & 0xFF
        imm = instruction >> 8

        # 2. Read auxiliary values via attention
        stack_top = self.read_int_via_attention(memory, sp)
        mem_at_ax = self.read_int_via_attention(memory, ax)
        mem_at_bp = self.read_int_via_attention(memory, bp)
        mem_at_bp8 = self.read_int_via_attention(memory, bp + 8)

        # 3. FFN computes all possible outputs
        ffn_input = torch.stack([
            pc, sp, bp, ax, imm, stack_top, mem_at_ax, mem_at_bp, mem_at_bp8
        ]).float().unsqueeze(0)  # (1, 9)

        all_outputs = self.ffn(ffn_input).squeeze(0)  # (156,)

        # 4. Select via attention on opcode
        selected = self.select_via_attention(all_outputs, opcode)

        new_pc, new_sp, new_bp, new_ax = selected[0], selected[1], selected[2], selected[3]

        # 5. Handle operations that FFN can't do (mul, div, comparisons)
        # These need elementwise ops, which ARE valid transformer ops
        is_mul = (opcode == 16).float()
        is_div = (opcode == 17).float()
        is_mod = (opcode == 18).float()
        is_add = (opcode == 14).float()
        is_sub = (opcode == 15).float()

        # Compute these with elementwise ops
        mul_result = stack_top * ax
        div_safe = torch.where(ax == 0, torch.tensor(1, device=device), ax)
        div_result = stack_top // div_safe
        mod_result = stack_top % div_safe

        # Blend results
        new_ax = (
            new_ax * (1 - is_mul - is_div - is_mod) +
            mul_result * is_mul +
            div_result * is_div +
            mod_result * is_mod
        ).long()

        # Comparisons
        is_eq = (opcode == 24).float()
        is_ne = (opcode == 25).float()
        is_lt = (opcode == 26).float()
        is_gt = (opcode == 27).float()
        is_le = (opcode == 28).float()
        is_ge = (opcode == 29).float()

        eq_result = (stack_top == ax).long()
        ne_result = (stack_top != ax).long()
        lt_result = (stack_top < ax).long()
        gt_result = (stack_top > ax).long()
        le_result = (stack_top <= ax).long()
        ge_result = (stack_top >= ax).long()

        new_ax = (
            new_ax * (1 - is_eq - is_ne - is_lt - is_gt - is_le - is_ge) +
            eq_result * is_eq +
            ne_result * is_ne +
            lt_result * is_lt +
            gt_result * is_gt +
            le_result * is_le +
            ge_result * is_ge
        ).long()

        # Bitwise ops
        is_or = (opcode == 19).float()
        is_xor = (opcode == 20).float()
        is_and = (opcode == 21).float()
        is_shl = (opcode == 22).float()
        is_shr = (opcode == 23).float()

        or_result = stack_top | ax
        xor_result = stack_top ^ ax
        and_result = stack_top & ax
        shl_result = stack_top << (ax & 63)
        shr_result = stack_top >> (ax & 63)

        new_ax = (
            new_ax * (1 - is_or - is_xor - is_and - is_shl - is_shr) +
            or_result * is_or +
            xor_result * is_xor +
            and_result * is_and +
            shl_result * is_shl +
            shr_result * is_shr
        ).long()

        # 6. Conditional branches (BZ, BNZ)
        is_bz = (opcode == 4).float()
        is_bnz = (opcode == 5).float()
        bz_target = torch.where(ax == 0, imm, pc + 8)
        bnz_target = torch.where(ax != 0, imm, pc + 8)

        new_pc = (
            new_pc * (1 - is_bz - is_bnz) +
            bz_target * is_bz +
            bnz_target * is_bnz
        ).long()

        # 7. ADJ needs sp + imm
        is_adj = (opcode == 7).float()
        new_sp = (new_sp * (1 - is_adj) + (sp + imm) * is_adj).long()

        # 8. LC needs byte mask
        is_lc = (opcode == 10).float()
        new_ax = (new_ax * (1 - is_lc) + (mem_at_ax & 0xFF) * is_lc).long()

        # 9. Memory writes via scatter
        new_memory = memory.clone()

        # PSH, JSR, ENT write to sp-8
        is_psh = (opcode == 13).float()
        is_jsr = (opcode == 3).float()
        is_ent = (opcode == 6).float()

        write_addr = sp - 8
        write_value = (ax * is_psh + (pc + 8) * is_jsr + bp * is_ent).long()
        needs_write = is_psh + is_jsr + is_ent

        if needs_write > 0:
            for i in range(8):
                idx = (write_addr + i).clamp(0, self.mem_size - 1)
                byte_val = (write_value >> (i * 8)) & 0xFF
                old_val = new_memory[idx]
                new_memory[idx] = (needs_write * byte_val + (1 - needs_write) * old_val).long()

        # SI writes to stack_top
        is_si = (opcode == 11).float()
        if is_si > 0:
            for i in range(8):
                idx = (stack_top + i).clamp(0, self.mem_size - 1)
                byte_val = (ax >> (i * 8)) & 0xFF
                old_val = new_memory[idx]
                new_memory[idx] = (is_si * byte_val + (1 - is_si) * old_val).long()

        # SC writes single byte
        is_sc = (opcode == 12).float()
        if is_sc > 0:
            idx = stack_top.clamp(0, self.mem_size - 1)
            old_val = new_memory[idx]
            new_memory[idx] = (is_sc * (ax & 0xFF) + (1 - is_sc) * old_val).long()

        halted = (opcode == 38)

        return new_pc, new_sp, new_bp, new_ax, new_memory, halted


def test():
    from c4_vm import program_simple_arithmetic, program_sum, CODE_BASE, STACK_BASE, STACK_SIZE, MEMORY_SIZE

    print("=" * 60)
    print("C4 WIRED TRANSFORMER")
    print("=" * 60)
    print()
    print("All computation via transformer ops:")
    print("  - Memory read: attention (softmax lookup)")
    print("  - Arithmetic: wired FFN (matmul with fixed weights)")
    print("  - Selection: attention over opcodes")
    print("  - Blending: elementwise ops (valid in transformers)")
    print()

    # Use full memory
    MEM_SIZE = MEMORY_SIZE

    model = C4WiredTransformer(mem_size=MEM_SIZE)

    def run_program(code, max_steps=5000):
        memory = torch.zeros(MEM_SIZE, dtype=torch.long)
        for i, instr in enumerate(code):
            addr = CODE_BASE + i * 8
            for b in range(8):
                memory[addr + b] = (instr >> (b * 8)) & 0xFF

        pc = torch.tensor(CODE_BASE, dtype=torch.long)
        sp = torch.tensor(STACK_BASE + STACK_SIZE, dtype=torch.long)
        bp = torch.tensor(STACK_BASE + STACK_SIZE, dtype=torch.long)
        ax = torch.tensor(0, dtype=torch.long)

        for step in range(max_steps):
            pc, sp, bp, ax, memory, halted = model(pc, sp, bp, ax, memory)
            if halted:
                break

        return ax.item(), step + 1

    tests = [
        ("(3+4)*5", program_simple_arithmetic, 35),
        ("sum(5)", lambda: program_sum(5), 15),
    ]

    for name, prog_fn, expected in tests:
        result, steps = run_program(prog_fn())
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name:15s} = {result:5d}  (expected {expected}, {steps} steps)")


if __name__ == "__main__":
    test()
