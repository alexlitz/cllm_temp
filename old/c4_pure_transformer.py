"""
Pure Transformer C4 Executor

No if/else, no Python control flow.
Only: matrix multiply, attention, elementwise ops.

Design:
1. Compute ALL possible next states in parallel (one per opcode)
2. Select the correct one using opcode as index
3. Memory via attention (gather/scatter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class C4PureTransformer(nn.Module):
    """
    Pure transformer that executes C4 opcodes.

    State: [PC, SP, BP, AX] - 4 integers
    Memory: external tensor, accessed via attention

    All operations are parallel matrix ops - no branching.
    """

    def __init__(self):
        super().__init__()
        self.num_opcodes = 39

        # Opcode operation table (wired, not learned)
        # Each row defines: [pc_delta, sp_delta, bp_update, ax_update, mem_read, mem_write]
        # These are operation TYPE selectors, not learned weights

    def fetch(self, memory: torch.Tensor, pc: torch.Tensor) -> tuple:
        """
        Fetch instruction from memory using attention.

        memory: (mem_size,) bytes
        pc: scalar, program counter

        Returns: (opcode, immediate)
        """
        # Read 8 bytes at PC using gather (attention with one-hot query)
        # This is equivalent to: instruction = memory[pc:pc+8]

        indices = pc + torch.arange(8, device=memory.device)
        indices = indices.clamp(0, len(memory) - 1)

        # Gather bytes
        bytes_at_pc = memory[indices]  # (8,)

        # Pack into int64 (little endian)
        shifts = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=memory.device)
        instruction = (bytes_at_pc.long() << shifts).sum()

        opcode = instruction & 0xFF
        immediate = instruction >> 8

        return opcode, immediate

    def read_int(self, memory: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        """Read 8 bytes from memory via attention gather."""
        indices = addr + torch.arange(8, device=memory.device)
        indices = indices.clamp(0, len(memory) - 1)
        bytes_read = memory[indices]
        shifts = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=memory.device)
        return (bytes_read.long() << shifts).sum()

    def write_int(self, memory: torch.Tensor, addr: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Write 8 bytes to memory via scatter."""
        memory = memory.clone()
        shifts = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=memory.device)
        bytes_to_write = (value >> shifts) & 0xFF

        for i in range(8):
            idx = (addr + i).clamp(0, len(memory) - 1)
            memory[idx] = bytes_to_write[i]

        return memory

    def masked_write_int(self, memory: torch.Tensor, addr: torch.Tensor,
                         value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Write 8 bytes to memory, masked by scalar mask (0 or 1)."""
        shifts = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=memory.device)
        bytes_to_write = (value >> shifts) & 0xFF

        for i in range(8):
            idx = (addr + i).clamp(0, len(memory) - 1)
            old_val = memory[idx]
            new_val = bytes_to_write[i]
            # Blend: mask * new + (1-mask) * old
            memory[idx] = mask * new_val + (1 - mask) * old_val

        return memory

    def masked_write_byte(self, memory: torch.Tensor, addr: torch.Tensor,
                          value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Write 1 byte to memory, masked by scalar mask (0 or 1)."""
        idx = addr.clamp(0, len(memory) - 1)
        old_val = memory[idx]
        new_val = value & 0xFF
        memory[idx] = mask * new_val + (1 - mask) * old_val
        return memory

    def step(self, pc: torch.Tensor, sp: torch.Tensor, bp: torch.Tensor, ax: torch.Tensor,
             memory: torch.Tensor) -> tuple:
        """
        Execute one instruction using ONLY parallel ops.

        Strategy: Compute ALL possible outcomes, then select based on opcode.
        """
        device = memory.device

        # 1. FETCH - attention gather at PC
        opcode, imm = self.fetch(memory, pc)

        # 2. PRE-COMPUTE values we might need
        stack_top = self.read_int(memory, sp)          # value at top of stack
        mem_at_ax = self.read_int(memory, ax)          # memory[ax]
        mem_at_bp_plus_imm = self.read_int(memory, bp + imm)  # memory[bp+imm]

        # Stack after pop
        sp_after_pop = sp + 8

        # Stack after push
        sp_after_push = sp - 8

        # 3. COMPUTE all possible next states (one per opcode class)
        # We group opcodes by behavior pattern

        # --- PC updates ---
        pc_plus_8 = pc + 8                    # normal advance
        pc_eq_imm = imm                        # JMP, JSR target
        pc_from_stack = self.read_int(memory, bp + 8)  # LEV return address

        # --- SP updates ---
        sp_unchanged = sp
        sp_minus_8 = sp - 8                   # PSH, JSR
        sp_plus_8 = sp + 8                    # pop operations
        sp_plus_imm = sp + imm                # ADJ
        sp_minus_imm = sp - imm               # ENT local space
        sp_eq_bp = bp                         # LEV

        # --- BP updates ---
        bp_unchanged = bp
        bp_eq_sp = sp                         # ENT (after push)
        bp_from_stack = self.read_int(memory, bp)  # LEV

        # --- AX updates ---
        ax_unchanged = ax
        ax_eq_imm = imm                       # IMM
        ax_eq_bp_plus_imm = bp + imm          # LEA
        ax_eq_mem_at_ax = mem_at_ax           # LI
        ax_eq_mem_byte = mem_at_ax & 0xFF     # LC
        ax_add = stack_top + ax               # ADD
        ax_sub = stack_top - ax               # SUB
        ax_mul = stack_top * ax               # MUL
        # Guard div/mod against zero (replace 0 with 1 for computation, result selected away anyway)
        ax_safe = torch.where(ax == 0, torch.tensor(1, device=device), ax)
        ax_div = stack_top // ax_safe
        ax_mod = stack_top % ax_safe
        ax_or = stack_top | ax
        ax_xor = stack_top ^ ax
        ax_and = stack_top & ax
        ax_shl = stack_top << (ax & 63)
        ax_shr = stack_top >> (ax & 63)
        ax_eq = (stack_top == ax).long()
        ax_ne = (stack_top != ax).long()
        ax_lt = (stack_top < ax).long()
        ax_gt = (stack_top > ax).long()
        ax_le = (stack_top <= ax).long()
        ax_ge = (stack_top >= ax).long()

        # 4. BUILD selection table
        # Opcode -> (new_pc, new_sp, new_bp, new_ax, memory_op)

        # Stack all possible PC values
        all_pc = torch.stack([
            pc_plus_8,    # 0: LEA
            pc_plus_8,    # 1: IMM
            pc_eq_imm,    # 2: JMP
            pc_eq_imm,    # 3: JSR
            torch.where(ax == 0, pc_eq_imm, pc_plus_8),  # 4: BZ
            torch.where(ax != 0, pc_eq_imm, pc_plus_8),  # 5: BNZ
            pc_plus_8,    # 6: ENT
            pc_plus_8,    # 7: ADJ
            self.read_int(memory, bp + 8),  # 8: LEV - return addr at bp+8
            pc_plus_8,    # 9: LI
            pc_plus_8,    # 10: LC
            pc_plus_8,    # 11: SI
            pc_plus_8,    # 12: SC
            pc_plus_8,    # 13: PSH
            pc_plus_8,    # 14: ADD
            pc_plus_8,    # 15: SUB
            pc_plus_8,    # 16: MUL
            pc_plus_8,    # 17: DIV
            pc_plus_8,    # 18: MOD
            pc_plus_8,    # 19: OR
            pc_plus_8,    # 20: XOR
            pc_plus_8,    # 21: AND
            pc_plus_8,    # 22: SHL
            pc_plus_8,    # 23: SHR
            pc_plus_8,    # 24: EQ
            pc_plus_8,    # 25: NE
            pc_plus_8,    # 26: LT
            pc_plus_8,    # 27: GT
            pc_plus_8,    # 28: LE
            pc_plus_8,    # 29: GE
            pc_plus_8,    # 30: OPEN
            pc_plus_8,    # 31: READ
            pc_plus_8,    # 32: CLOS
            pc_plus_8,    # 33: PRTF
            pc_plus_8,    # 34: MALC
            pc_plus_8,    # 35: FREE
            pc_plus_8,    # 36: MSET
            pc_plus_8,    # 37: MCMP
            pc,           # 38: EXIT (halt - PC unchanged)
        ])

        all_sp = torch.stack([
            sp_unchanged,    # 0: LEA
            sp_unchanged,    # 1: IMM
            sp_unchanged,    # 2: JMP
            sp_minus_8,      # 3: JSR (push return addr)
            sp_unchanged,    # 4: BZ
            sp_unchanged,    # 5: BNZ
            sp - 8 - imm,    # 6: ENT (push bp, then subtract imm)
            sp_plus_imm,     # 7: ADJ
            bp + 16,         # 8: LEV (sp = bp, pop bp, pop pc)
            sp_unchanged,    # 9: LI
            sp_unchanged,    # 10: LC
            sp_plus_8,       # 11: SI (pop addr)
            sp_plus_8,       # 12: SC (pop addr)
            sp_minus_8,      # 13: PSH
            sp_plus_8,       # 14: ADD (pop operand)
            sp_plus_8,       # 15: SUB
            sp_plus_8,       # 16: MUL
            sp_plus_8,       # 17: DIV
            sp_plus_8,       # 18: MOD
            sp_plus_8,       # 19: OR
            sp_plus_8,       # 20: XOR
            sp_plus_8,       # 21: AND
            sp_plus_8,       # 22: SHL
            sp_plus_8,       # 23: SHR
            sp_plus_8,       # 24: EQ
            sp_plus_8,       # 25: NE
            sp_plus_8,       # 26: LT
            sp_plus_8,       # 27: GT
            sp_plus_8,       # 28: LE
            sp_plus_8,       # 29: GE
            sp_unchanged,    # 30-37: syscalls
            sp_unchanged,
            sp_unchanged,
            sp_unchanged,
            sp_unchanged,
            sp_unchanged,
            sp_unchanged,
            sp_unchanged,
            sp_unchanged,    # 38: EXIT
        ])

        all_bp = torch.stack([
            bp_unchanged,    # 0: LEA
            bp_unchanged,    # 1: IMM
            bp_unchanged,    # 2: JMP
            bp_unchanged,    # 3: JSR
            bp_unchanged,    # 4: BZ
            bp_unchanged,    # 5: BNZ
            sp - 8,          # 6: ENT (bp = sp after push)
            bp_unchanged,    # 7: ADJ
            self.read_int(memory, bp),  # 8: LEV (pop old bp)
            bp_unchanged,    # 9: LI
            bp_unchanged,    # 10: LC
            bp_unchanged,    # 11: SI
            bp_unchanged,    # 12: SC
            bp_unchanged,    # 13: PSH
            bp_unchanged,    # 14-29: arithmetic/comparison
            bp_unchanged, bp_unchanged, bp_unchanged, bp_unchanged,
            bp_unchanged, bp_unchanged, bp_unchanged, bp_unchanged,
            bp_unchanged, bp_unchanged, bp_unchanged, bp_unchanged,
            bp_unchanged, bp_unchanged, bp_unchanged,
            bp_unchanged, bp_unchanged, bp_unchanged, bp_unchanged,  # 30-33
            bp_unchanged, bp_unchanged, bp_unchanged, bp_unchanged,  # 34-37
            bp_unchanged,    # 38: EXIT
        ])

        all_ax = torch.stack([
            ax_eq_bp_plus_imm,  # 0: LEA
            ax_eq_imm,          # 1: IMM
            ax_unchanged,       # 2: JMP
            ax_unchanged,       # 3: JSR
            ax_unchanged,       # 4: BZ
            ax_unchanged,       # 5: BNZ
            ax_unchanged,       # 6: ENT
            ax_unchanged,       # 7: ADJ
            ax_unchanged,       # 8: LEV
            ax_eq_mem_at_ax,    # 9: LI
            ax_eq_mem_byte,     # 10: LC
            ax_unchanged,       # 11: SI
            ax_unchanged,       # 12: SC
            ax_unchanged,       # 13: PSH
            ax_add,             # 14: ADD
            ax_sub,             # 15: SUB
            ax_mul,             # 16: MUL
            ax_div,             # 17: DIV
            ax_mod,             # 18: MOD
            ax_or,              # 19: OR
            ax_xor,             # 20: XOR
            ax_and,             # 21: AND
            ax_shl,             # 22: SHL
            ax_shr,             # 23: SHR
            ax_eq,              # 24: EQ
            ax_ne,              # 25: NE
            ax_lt,              # 26: LT
            ax_gt,              # 27: GT
            ax_le,              # 28: LE
            ax_ge,              # 29: GE
            torch.tensor(0, device=device),  # 30-37: syscalls return 0
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            ax_unchanged,       # 38: EXIT
        ])

        # 5. SELECT based on opcode (this is attention with one-hot query)
        opcode_idx = opcode.clamp(0, 38).long()

        new_pc = all_pc[opcode_idx]
        new_sp = all_sp[opcode_idx]
        new_bp = all_bp[opcode_idx]
        new_ax = all_ax[opcode_idx]

        # 6. MEMORY WRITES via masked scatter (no if/else)
        new_memory = memory.clone()

        # Opcode masks
        is_psh = (opcode == 13).long()
        is_jsr = (opcode == 3).long()
        is_ent = (opcode == 6).long()
        is_si = (opcode == 11).long()
        is_sc = (opcode == 12).long()

        # Push-type ops (PSH, JSR, ENT) all write to sp-8
        push_addr = sp - 8
        push_value = is_psh * ax + is_jsr * (pc + 8) + is_ent * bp
        needs_push = is_psh + is_jsr + is_ent  # 0 or 1

        # SI writes ax to address from stack
        si_addr = stack_top

        # SC writes byte to address from stack
        sc_addr = stack_top

        # Apply all writes using masked scatter
        # Write to push_addr (masked by needs_push)
        new_memory = self.masked_write_int(new_memory, push_addr, push_value, needs_push)

        # Write for SI
        new_memory = self.masked_write_int(new_memory, si_addr, ax, is_si)

        # Write for SC (single byte)
        new_memory = self.masked_write_byte(new_memory, sc_addr, ax & 0xFF, is_sc)

        halted = (opcode == 38)

        return new_pc, new_sp, new_bp, new_ax, new_memory, halted


def test_pure_transformer():
    """Test the pure transformer executor."""
    from c4_vm import program_simple_arithmetic, program_sum, CODE_BASE, STACK_BASE, STACK_SIZE, MEMORY_SIZE

    print("=" * 60)
    print("PURE TRANSFORMER C4 EXECUTOR")
    print("=" * 60)
    print()
    print("No if/else in step(). All ops computed in parallel,")
    print("then selected via opcode indexing (one-hot attention).")
    print()

    model = C4PureTransformer()

    def run_program(code, max_steps=5000):
        # Initialize state
        memory = torch.zeros(MEMORY_SIZE, dtype=torch.long)

        # Load code
        for i, instr in enumerate(code):
            addr = CODE_BASE + i * 8
            for b in range(8):
                memory[addr + b] = (instr >> (b * 8)) & 0xFF

        pc = torch.tensor(CODE_BASE, dtype=torch.long)
        sp = torch.tensor(STACK_BASE + STACK_SIZE, dtype=torch.long)
        bp = torch.tensor(STACK_BASE + STACK_SIZE, dtype=torch.long)
        ax = torch.tensor(0, dtype=torch.long)

        for step in range(max_steps):
            pc, sp, bp, ax, memory, halted = model.step(pc, sp, bp, ax, memory)
            if halted:
                break

        return ax.item(), step + 1

    tests = [
        ("(3+4)*5", program_simple_arithmetic, 35),
        ("sum(1)", lambda: program_sum(1), 1),
        ("sum(10)", lambda: program_sum(10), 55),
        ("sum(100)", lambda: program_sum(100), 5050),
    ]

    for name, prog_fn, expected in tests:
        result, steps = run_program(prog_fn())
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name:15s} = {result:5d}  ({steps} steps)")

    print()
    print("This is a PURE TRANSFORMER:")
    print("  - Attention gather for memory read")
    print("  - Parallel compute of all 39 opcode results")
    print("  - One-hot selection based on opcode")
    print("  - Attention scatter for memory write")
    print("  - No Python if/else in the hot path")


if __name__ == "__main__":
    test_pure_transformer()
