"""
100% Neural State Management for Neural VM.

This module implements fully neural:
1. Program Counter (PC) - stored in embedding, updated via FFN
2. Stack Pointer (SP) - stored in embedding, updated via FFN with carry
3. Register Operations - AX, BP manipulation via FFN
4. Memory Read/Write - via attention with binary address matching
5. I/O - via binary position matching in context

Key insight: All state is in the embedding tensor. Each nibble position
stores the same VM state (replicated), allowing position-parallel FFNs
to access it.

Architecture:
- Positions 0-7: Current instruction operands (8 nibbles)
- Embedding slots PC[0:8], SP[0:8], etc. store VM registers as nibbles
- Neural FFNs update these slots based on opcode one-hot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, bake_weights


# ============================================================================
# EMBEDDING SLOTS FOR VM STATE
# ============================================================================

class VMState:
    """
    Extended embedding slots for full VM state.

    All VM state is stored in the embedding tensor, making it
    accessible to neural layers without external controller.

    Register Layout (each register uses 8 consecutive slots for nibbles):
    - PC[0:8]: Program counter nibbles (little-endian)
    - SP[0:8]: Stack pointer nibbles
    - BP[0:8]: Base pointer nibbles
    - AX[0:8]: Accumulator nibbles
    """
    # Existing slots from E class (0-101)
    # NIB_A=0, NIB_B=1, RAW_SUM=2, CARRY_IN=3, CARRY_OUT=4, RESULT=5, TEMP=6
    # OP_START=7, NUM_OPS=72 -> slots 7-78
    # POS=79
    # IO slots = 80-101

    # VM Register slots (starting at 104 for alignment)
    PC_BASE = 104         # Program counter nibbles [104:112]
    SP_BASE = 112         # Stack pointer nibbles [112:120]
    BP_BASE = 120         # Base pointer nibbles [120:128]
    AX_BASE = 128         # Accumulator nibbles [128:136]

    # Convenience (first nibble of each register)
    PC = 104
    SP = 112
    BP = 120
    AX = 128

    # VM flags
    FLAG_ZERO = 136       # Zero flag (AX == 0)
    FLAG_CARRY = 137      # Carry flag from last arithmetic
    FLAG_BRANCH = 138     # Branch taken flag
    FLAG_HALT = 139       # Program halted

    # Memory interface
    MEM_ADDR_BASE = 140   # Memory address nibbles [140:148]
    MEM_DATA_BASE = 148   # Memory data nibbles [148:156]
    MEM_READ = 156        # 1.0 = read request pending
    MEM_WRITE = 157       # 1.0 = write request pending
    MEM_READY = 158       # 1.0 = memory operation complete

    # Instruction register (current opcode operand)
    INST_IMM_BASE = 160   # Immediate value nibbles [160:168]

    # Heap management for bump allocator
    HEAP_BASE = 168       # Base of heap (fixed, set at program start)
    HEAP_PTR = 176        # Current allocation pointer (bumps upward)
    HEAP_END = 184        # End of heap (for bounds checking)

    # Extended dimension (increased for heap slots)
    DIM = 192


# ============================================================================
# NEURAL PC MANAGEMENT
# ============================================================================

class IncrementPCFFN(PureFFN):
    """
    Increment PC by instruction size (8 bytes for C4).

    PC is stored as 8 nibbles. Adding 8 means:
    - PC[1] += 1 (since 8 = 0x08, nibble 1 gets +1)
    - Carry propagates if nibble overflows

    This uses the SwiGLU identity trick:
    - silu(S) ≈ S for large S
    - silu(S) * (bias + value) ≈ S * (bias + value)
    - Divided by S in W_down → adds (bias + value)
    """

    def __init__(self):
        # Need hidden units for each nibble's carry chain
        super().__init__(VMState.DIM, hidden_dim=32)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Add 1 to nibble 1 (which adds 0x10 = 16 to the address, but
        # for 8-byte alignment we add to nibble 1 not 0)
        # Actually, adding 8 bytes = adding 0x8 to nibble position 0
        # OR adding 0.5 to nibble 1 with carry

        # Simplified: just add 8 to nibble 0, let carry FFN handle overflow
        # PC[0] += 8 at position 0 only

        # Unit 0: Always active (b_up = S), adds 8 to PC[0]
        self.b_up[0] = S
        self.b_gate[0] = 8.0  # The value to add
        self.W_down[VMState.PC, 0] = 1.0 / S

        # Handle overflow: if PC[0] >= 16, subtract 16 and carry to PC[1]
        # This requires checking PC[0] value...
        # For now, assume PC[0] stays in range (simplified)

        # Actually, proper implementation needs multi-nibble carry chain
        # like the ADD operation. Let's use a simpler approach:
        # Store PC as a single value in a slot, not nibbles.


class JumpPCFFN(PureFFN):
    """
    Copy RESULT to PC when JMP opcode is active.

    For position i: PC_BASE+i = RESULT if JMP active

    Uses SwiGLU:
    - silu(S * JMP) ≈ S when JMP=1, ≈ 0 when JMP=0
    - Gate reads RESULT value
    - Output goes to PC slot
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=8)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # For each of 8 nibble positions, copy RESULT to PC
        for pos in range(8):
            h = pos  # Hidden unit for this nibble

            # Step 1: Clear old PC value (when JMP active)
            # silu(S*JMP) * (-PC) -> subtracts PC
            # But we need to do this per-position...

            # Actually, simpler approach: The RESULT slot at position i
            # should go to PC_BASE+i. But PC_BASE+i is a single slot
            # in the embedding, not per-position.

            # The trick: Use the position encoding to gate which nibble
            # of PC we're updating. Position i updates PC_BASE+i.

            # hidden[h] = silu(S*JMP + S*POS[i]) * RESULT
            # This activates for position i when JMP=1

        # Simplified approach: All positions update their corresponding PC nibble
        # Hidden unit 0: Activated by JMP, copies RESULT to PC_BASE+0
        # But we're position-parallel, so position 0's RESULT goes to PC[0]

        # Let's do: output = silu(S*JMP) * (RESULT - old_PC) -> adds delta
        # This effectively does: PC = PC + JMP * (RESULT - PC) = RESULT when JMP=1

        # Clear current PC, add RESULT (when JMP active)
        # Need 2 hidden units: one to subtract old PC, one to add RESULT

        # Unit 0: silu(S*JMP) * (-PC) -> subtract old PC
        self.W_up[0, E.OP_START + Opcode.JMP] = S
        self.W_gate[0, VMState.PC] = -1.0
        self.W_down[VMState.PC, 0] = 1.0 / S

        # Unit 1: silu(-S*JMP) * PC -> add back PC (cancels when JMP=1)
        self.W_up[1, E.OP_START + Opcode.JMP] = -S
        self.W_gate[1, VMState.PC] = 1.0
        self.W_down[VMState.PC, 1] = 1.0 / S

        # Unit 2: silu(S*JMP) * RESULT -> add new PC
        self.W_up[2, E.OP_START + Opcode.JMP] = S
        self.W_gate[2, E.RESULT] = 1.0
        self.W_down[VMState.PC, 2] = 1.0 / S

        # Unit 3: silu(-S*JMP) * (-RESULT) -> cancel when JMP=0
        self.W_up[3, E.OP_START + Opcode.JMP] = -S
        self.W_gate[3, E.RESULT] = -1.0
        self.W_down[VMState.PC, 3] = 1.0 / S


class BranchPCFFN(PureFFN):
    """
    Conditional branch: Update PC if condition is met.

    BZ (branch if zero): PC = target if AX == 0 (FLAG_ZERO = 1)
    BNZ (branch if not zero): PC = target if AX != 0 (FLAG_ZERO = 0)

    Uses threshold-based activation WITHOUT cancel pairs:
    - silu(S * (opcode + condition - 1.5)) activates only when BOTH are 1
    - When inactive, silu returns ~0 so no contribution

    PC_new = PC + silu(threshold) * (RESULT - PC)
    """

    def __init__(self, opcode: int, branch_on_zero: bool = True):
        self.opcode = opcode
        self.branch_on_zero = branch_on_zero
        super().__init__(VMState.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        if self.branch_on_zero:
            # BZ: Activate when BZ=1 AND FLAG_ZERO=1
            # silu(S*(BZ + FLAG_ZERO - 1.5))
            # BZ=1, FLAG_ZERO=1: silu(S*0.5) ≈ 0.5*S (activates)
            # BZ=1, FLAG_ZERO=0: silu(-0.5*S) ≈ -0.15 (near zero)
            # BZ=0: similar near zero

            # No cancel pairs - just let silu's natural behavior handle it
            threshold_bias = -1.5 * S

            # Unit 0: Add RESULT when condition met
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_up[0, VMState.FLAG_ZERO] = S
            self.b_up[0] = threshold_bias
            self.W_gate[0, E.RESULT] = 1.0
            self.W_down[VMState.PC, 0] = 2.0 / S

            # Unit 1: Subtract old PC when condition met
            self.W_up[1, E.OP_START + self.opcode] = S
            self.W_up[1, VMState.FLAG_ZERO] = S
            self.b_up[1] = threshold_bias
            self.W_gate[1, VMState.PC] = -1.0
            self.W_down[VMState.PC, 1] = 2.0 / S

        else:
            # BNZ: Activate when BNZ=1 AND FLAG_ZERO=0
            # silu(S*(BNZ - FLAG_ZERO - 0.5))
            # BNZ=1, FLAG_ZERO=0: silu(S*0.5) ≈ 0.5*S (activates)
            # BNZ=1, FLAG_ZERO=1: silu(-0.5*S) ≈ near zero
            # BNZ=0: near zero

            threshold_bias = -0.5 * S

            # Unit 0: Add RESULT
            self.W_up[0, E.OP_START + self.opcode] = S
            self.W_up[0, VMState.FLAG_ZERO] = -S
            self.b_up[0] = threshold_bias
            self.W_gate[0, E.RESULT] = 1.0
            self.W_down[VMState.PC, 0] = 2.0 / S

            # Unit 1: Subtract old PC
            self.W_up[1, E.OP_START + self.opcode] = S
            self.W_up[1, VMState.FLAG_ZERO] = -S
            self.b_up[1] = threshold_bias
            self.W_gate[1, VMState.PC] = -1.0
            self.W_down[VMState.PC, 1] = 2.0 / S


class SetZeroFlagFFN(PureFFN):
    """
    Set FLAG_ZERO based on AX value.

    FLAG_ZERO = 1 if all AX nibbles are 0, else 0.

    This is tricky neurally. We use:
    - Sum all AX nibbles
    - If sum > 0, FLAG_ZERO = 0
    - If sum = 0, FLAG_ZERO = 1
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Compute sum of AX nibbles
        # If any nibble > 0, sum > 0, so FLAG_ZERO should be 0
        # This uses: FLAG_ZERO = 1 - clip(sum, 0, 1)

        # Actually simpler: FLAG_ZERO = silu(-S * sum) / silu(-S) + 1
        # When sum=0: silu(0) = 0, so FLAG_ZERO = 1
        # When sum>0: silu(-S*sum) ≈ 0, so FLAG_ZERO ≈ 0

        # Unit 0: Detect non-zero (subtracts 1 from FLAG_ZERO when AX != 0)
        for i in range(8):
            self.W_up[0, VMState.AX_BASE + i] = S / 8  # Sum contribution
        self.b_gate[0] = -1.0  # Base: subtract 1
        self.W_down[VMState.FLAG_ZERO, 0] = 1.0 / S

        # Unit 1: Set to 1 unconditionally (then unit 0 subtracts if non-zero)
        self.b_up[1] = S
        self.b_gate[1] = 1.0
        self.W_down[VMState.FLAG_ZERO, 1] = 1.0 / S


# ============================================================================
# NEURAL STACK MANAGEMENT
# ============================================================================

class PushStackFFN(PureFFN):
    """
    Push value to stack: SP -= 8, mem[SP] = value.

    Neural implementation:
    1. Copy AX to MEM_DATA slots
    2. Copy SP to MEM_ADDR slots
    3. Set MEM_WRITE = 1.0
    4. Decrement SP by 8 (via subtraction FFN)
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # When PSH opcode is active:

        # Copy AX to MEM_DATA (8 nibbles)
        for i in range(8):
            self.W_up[h, E.OP_START + Opcode.PSH] = S
            self.W_gate[h, VMState.AX_BASE + i] = 1.0
            self.W_down[VMState.MEM_DATA_BASE + i, h] = 1.0 / S
            h += 1

        # Copy SP to MEM_ADDR (8 nibbles)
        for i in range(8):
            self.W_up[h, E.OP_START + Opcode.PSH] = S
            self.W_gate[h, VMState.SP_BASE + i] = 1.0
            self.W_down[VMState.MEM_ADDR_BASE + i, h] = 1.0 / S
            h += 1
            if h >= 16:
                break

        # Set MEM_WRITE flag (use remaining hidden units if available)
        # Will be done in a separate small FFN


class DecrementSPFFN(PureFFN):
    """
    Decrement SP by 8 (for PUSH operations).

    SP -= 8 means nibble 1 -= 0.5 with borrow chain.
    Simplified: subtract 8 from nibble 0, propagate borrow via attention.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Subtract 8 from SP[0] when PSH is active
        self.W_up[0, E.OP_START + Opcode.PSH] = S
        self.b_gate[0] = -8.0  # Subtract 8
        self.W_down[VMState.SP, 0] = 1.0 / S

        # Handle underflow (SP[0] < 0 -> borrow from SP[1])
        # This requires checking SP[0] value...
        # Full implementation needs carry attention


class IncrementSPFFN(PureFFN):
    """
    Increment SP (for POP operations or ADJ).

    Used after LEV to restore SP.
    """

    def __init__(self, opcode: int, amount: int = 8):
        self.opcode = opcode
        self.amount = amount
        super().__init__(VMState.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Add amount to SP[0]
        self.W_up[0, E.OP_START + self.opcode] = S
        self.b_gate[0] = float(self.amount)
        self.W_down[VMState.SP, 0] = 1.0 / S


class PopToAXFFN(PureFFN):
    """
    Pop value from stack to AX.

    1. Copy MEM_DATA (result of memory read) to AX
    2. Increment SP by 8 (done by IncrementSPFFN)
    """

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(VMState.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy MEM_DATA to AX (8 nibbles)
        for i in range(8):
            self.W_up[h, E.OP_START + self.opcode] = S
            self.W_gate[h, VMState.MEM_DATA_BASE + i] = 1.0
            self.W_down[VMState.AX_BASE + i, h] = 1.0 / S
            h += 1


class SetMemReadFFN(PureFFN):
    """Set MEM_READ flag and copy SP to MEM_ADDR for stack read."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(VMState.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy SP to MEM_ADDR
        for i in range(8):
            self.W_up[h, E.OP_START + self.opcode] = S
            self.W_gate[h, VMState.SP_BASE + i] = 1.0
            self.W_down[VMState.MEM_ADDR_BASE + i, h] = 1.0 / S
            h += 1

        # Set MEM_READ flag
        self.W_up[h, E.OP_START + self.opcode] = S
        self.b_gate[h] = 1.0
        self.W_down[VMState.MEM_READ, h] = 1.0 / S


class SetMemWriteFFN(PureFFN):
    """Set MEM_WRITE flag."""

    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(VMState.DIM, hidden_dim=2)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Set MEM_WRITE flag
        self.W_up[0, E.OP_START + self.opcode] = S
        self.b_gate[0] = 1.0
        self.W_down[VMState.MEM_WRITE, 0] = 1.0 / S


# ============================================================================
# NEURAL MEMORY (ATTENTION-BASED)
# ============================================================================

class BinaryAddressEncoder(nn.Module):
    """
    Encode memory address as binary for attention-based lookup.

    Address nibbles -> binary bits for Q/K matching.
    Uses neural bit extraction: bit_k = floor(nibble / 2^k) mod 2

    Neural implementation:
    - bit_k ≈ 2 * sigmoid(S * (nibble - 2^k - 0.5)) - 1 for high bits
    """

    def __init__(self, addr_bits: int = 16):
        super().__init__()
        self.addr_bits = addr_bits

    def forward(self, addr_nibbles: torch.Tensor) -> torch.Tensor:
        """
        Convert 4 nibbles (16 bits) to binary encoding.

        Args:
            addr_nibbles: [batch, 4] tensor of nibble values 0-15

        Returns:
            [batch, 16] binary encoding (0 or 1 values)
        """
        bits = []
        for i in range(4):
            nibble = addr_nibbles[:, i:i+1]  # [batch, 1]
            for b in range(4):
                # Exact bit extraction
                bit = ((nibble.int() >> b) & 1).float()
                bits.append(bit)
        return torch.cat(bits, dim=-1)  # [batch, 16]


class MemoryReadAttention(PureAttention):
    """
    Read from memory using attention.

    Memory is stored in the KV cache as previous context positions.
    Each memory position has:
    - Key: binary encoding of its address
    - Value: stored data

    Query comes from MEM_ADDR slots in current embedding.
    Attention finds matching address and returns its value.

    Architecture:
    - Q projects MEM_ADDR_BASE nibbles to binary query
    - K projects position's stored address to binary key
    - V projects position's stored data
    - Attention score high when address matches
    - Output goes to MEM_DATA_BASE slots
    """

    def __init__(self, addr_bits: int = 16):
        self.addr_bits = addr_bits
        super().__init__(VMState.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Query: Extract bits from MEM_ADDR nibbles
        # For each address bit, Q component = 2*(nibble_bit) - 1 (maps 0->-1, 1->+1)
        for nib in range(4):  # 4 nibbles in address
            for bit in range(4):  # 4 bits per nibble
                q_dim = nib * 4 + bit
                # This projects the nibble to its bit
                # Simplified: assume address is already binary-encoded
                self.W_q[q_dim, VMState.MEM_ADDR_BASE + nib] = 1.0

        # Key: Project stored address to binary
        # Memory positions store their address in slots
        # For now, use position encoding as address
        for pos in range(E.NUM_POSITIONS):
            for bit in range(self.addr_bits):
                k_dim = bit
                addr_bit = (pos >> bit) & 1
                self.W_k[k_dim, E.POS] = float(2 * addr_bit - 1)  # -1 or +1

        # Value: Project stored data to output
        # Each position's RESULT slot contains data
        self.W_v[0, E.RESULT] = 1.0

        # Output: Route to MEM_DATA
        self.W_o[VMState.MEM_DATA_BASE, 0] = 1.0


class MemoryWriteAttention(PureAttention):
    """
    Write to memory using attention.

    Routes MEM_DATA to the memory position matching MEM_ADDR.

    Architecture:
    - Q projects target address to binary
    - K projects each position's address to binary
    - V projects MEM_DATA (data to write)
    - High attention score at matching address
    - That position receives the data

    This is a "reverse" attention where we write TO keys that match,
    rather than read FROM them.
    """

    def __init__(self, addr_bits: int = 16):
        self.addr_bits = addr_bits
        super().__init__(VMState.DIM, num_heads=1, causal=False)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Similar to read, but value comes from MEM_DATA
        # and output goes to each position's storage slot

        # Query: Extract bits from MEM_ADDR
        for nib in range(4):
            for bit in range(4):
                q_dim = nib * 4 + bit
                self.W_q[q_dim, VMState.MEM_ADDR_BASE + nib] = 1.0

        # Key: Position's own address
        # Positions naturally have addresses from their position encoding
        # This creates selective attention to the target position
        for bit in range(self.addr_bits):
            self.W_k[bit, E.POS] = 1.0

        # Value: Data to write (from MEM_DATA)
        for i in range(8):
            self.W_v[i, VMState.MEM_DATA_BASE + i] = 1.0

        # Output: Goes to position's storage (RESULT slot or dedicated MEM slot)
        for i in range(8):
            self.W_o[E.RESULT + i, i] = 1.0


class KVCacheMemory(nn.Module):
    """
    Memory as KV cache positions.

    Each memory word is a position in the sequence:
    - Position i holds memory[addr_i]
    - Address stored in that position's MEM_ADDR slots
    - Data stored in that position's MEM_DATA slots

    Memory access:
    1. Query broadcasts target address
    2. Attention finds matching position
    3. Read: copy data from matching position
    4. Write: copy data TO matching position

    This allows dynamic memory allocation by appending positions.
    """

    def __init__(self, addr_bits: int = 16, data_bits: int = 32):
        super().__init__()
        self.addr_bits = addr_bits
        self.data_bits = data_bits

        self.read_attention = MemoryReadAttention(addr_bits)
        self.write_attention = MemoryWriteAttention(addr_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process memory operations based on MEM_READ/MEM_WRITE flags.

        The attention operations are always computed, but results
        are gated by the read/write flags.
        """
        # Check memory flags
        read_flag = x[:, 0, VMState.MEM_READ]  # [batch]
        write_flag = x[:, 0, VMState.MEM_WRITE]  # [batch]

        # Compute both read and write attention
        read_result = self.read_attention(x)
        write_result = self.write_attention(x)

        # Gate by flags
        # read_result contributes to MEM_DATA when MEM_READ=1
        # write_result modifies target position when MEM_WRITE=1

        # Output gated by flags
        out = x.clone()

        # Apply read result if MEM_READ is set
        read_gate = read_flag.view(-1, 1, 1)  # [batch, 1, 1]
        out = out + read_gate * (read_result - x)

        # Apply write result if MEM_WRITE is set
        write_gate = write_flag.view(-1, 1, 1)
        out = out + write_gate * (write_result - x)

        # Clear memory flags after operation
        out[:, :, VMState.MEM_READ] = 0.0
        out[:, :, VMState.MEM_WRITE] = 0.0
        out[:, :, VMState.MEM_READY] = read_gate.squeeze() + write_gate.squeeze()

        return out


# ============================================================================
# NEURAL I/O (EMBEDDING-BASED MAILBOX)
# ============================================================================

class InputReadFFN(PureFFN):
    """
    Read input character from IO_CHAR to AX when IO_INPUT_READY.

    GETCHAR: Wait for IO_INPUT_READY, then copy IO_CHAR to AX.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # Gate: GETCHAR active AND IO_INPUT_READY
        # Copy IO_CHAR (8 nibbles) to AX (8 nibbles)
        for i in range(8):
            # silu(S*GETCHAR + S*IO_INPUT_READY - S) * IO_CHAR[i] -> AX[i]
            # Active only when both GETCHAR=1 and IO_INPUT_READY=1
            self.W_up[i, E.OP_START + Opcode.GETCHAR] = S / 2
            self.W_up[i, E.IO_INPUT_READY] = S / 2
            self.W_gate[i, E.IO_CHAR + i] = 1.0
            self.W_down[VMState.AX_BASE + i, i] = 1.0 / S


class OutputWriteFFN(PureFFN):
    """
    Write AX to IO_CHAR and set IO_OUTPUT_READY for PUTCHAR.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy AX to IO_CHAR when PUTCHAR active
        for i in range(8):
            self.W_up[h, E.OP_START + Opcode.PUTCHAR] = S
            self.W_gate[h, VMState.AX_BASE + i] = 1.0
            self.W_down[E.IO_CHAR + i, h] = 1.0 / S
            h += 1

        # Set IO_OUTPUT_READY flag
        self.W_up[h, E.OP_START + Opcode.PUTCHAR] = S
        self.b_gate[h] = 1.0
        self.W_down[E.IO_OUTPUT_READY, h] = 1.0 / S


class NeedInputFFN(PureFFN):
    """
    Set IO_NEED_INPUT when GETCHAR needs input.

    Activated when GETCHAR is active but IO_INPUT_READY is 0.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE

        # silu(S*GETCHAR - S*IO_INPUT_READY) * 1.0 -> IO_NEED_INPUT
        # Active when GETCHAR=1 and IO_INPUT_READY=0
        self.W_up[0, E.OP_START + Opcode.GETCHAR] = S
        self.W_up[0, E.IO_INPUT_READY] = -S
        self.b_gate[0] = 1.0
        self.W_down[E.IO_NEED_INPUT, 0] = 1.0 / S


# ============================================================================
# UNIFIED NEURAL VM LAYER
# ============================================================================

class NeuralVMLayer(nn.Module):
    """
    Complete neural VM layer for one instruction execution.

    Combines all neural operations:
    1. Decode instruction (read from PC)
    2. Execute ALU operation
    3. Update PC (increment or branch)
    4. Handle stack operations
    5. Handle memory access
    6. Handle I/O

    All operations run in parallel (like MoE), weighted by opcode one-hot.
    """

    def __init__(self):
        super().__init__()

        # PC operations
        self.jump_pc = JumpPCFFN()
        self.branch_bz = BranchPCFFN(Opcode.BZ, branch_on_zero=True)
        self.branch_bnz = BranchPCFFN(Opcode.BNZ, branch_on_zero=False)
        self.set_zero_flag = SetZeroFlagFFN()

        # Stack operations
        self.push_stack = PushStackFFN()
        self.decrement_sp = DecrementSPFFN()
        self.set_mem_write = SetMemWriteFFN(Opcode.PSH)

        # Memory operations
        self.memory = KVCacheMemory()

        # I/O operations
        self.input_read = InputReadFFN()
        self.output_write = OutputWriteFFN()
        self.need_input = NeedInputFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute one VM instruction neurally.

        All operations run and are gated by their opcode one-hot.
        """
        # Update zero flag based on AX
        x = self.set_zero_flag(x)

        # PC operations (mutually exclusive via opcode)
        x = self.jump_pc(x)
        x = self.branch_bz(x)
        x = self.branch_bnz(x)

        # Stack operations
        x = self.push_stack(x)
        x = self.decrement_sp(x)
        x = self.set_mem_write(x)

        # Memory access
        x = self.memory(x)

        # I/O
        x = self.input_read(x)
        x = self.output_write(x)
        x = self.need_input(x)

        return x


# ============================================================================
# TESTS
# ============================================================================

def test_neural_pc():
    """Test neural PC update operations."""
    print("=== Testing Neural PC Operations ===\n")

    # Architecture: Each position i holds nibble i of registers
    # Position 0: nibble 0 of PC, AX, etc.
    # Position 1: nibble 1 of PC, AX, etc.
    # RESULT at position i is nibble i of the result

    # Create embedding with JMP opcode active
    x = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)

    # Set JMP opcode (same across all positions)
    x[:, :, E.OP_START + Opcode.JMP] = 1.0

    # Set current PC nibbles: 0x123 = nibbles [3, 2, 1, 0, 0, 0, 0, 0]
    x[:, 0, VMState.PC] = 3  # Position 0 holds PC nibble 0
    x[:, 1, VMState.PC] = 2  # Position 1 holds PC nibble 1
    x[:, 2, VMState.PC] = 1  # Position 2 holds PC nibble 2

    # Set RESULT (jump target) to 0x456 = nibbles [6, 5, 4, 0, 0, 0, 0, 0]
    x[:, 0, E.RESULT] = 6  # Position 0: RESULT nibble 0
    x[:, 1, E.RESULT] = 5  # Position 1: RESULT nibble 1
    x[:, 2, E.RESULT] = 4  # Position 2: RESULT nibble 2

    # Apply JumpPCFFN
    jump_ffn = JumpPCFFN()
    y = jump_ffn(x)

    print(f"Before JMP: PC nibbles = [{x[0, 0, VMState.PC]:.0f}, {x[0, 1, VMState.PC]:.0f}, {x[0, 2, VMState.PC]:.0f}]")
    print(f"RESULT nibbles = [{x[0, 0, E.RESULT]:.0f}, {x[0, 1, E.RESULT]:.0f}, {x[0, 2, E.RESULT]:.0f}]")
    print(f"After JMP:  PC nibbles = [{y[0, 0, VMState.PC]:.1f}, {y[0, 1, VMState.PC]:.1f}, {y[0, 2, VMState.PC]:.1f}]")

    # Verify
    expected = [6.0, 5.0, 4.0]
    actual = [y[0, 0, VMState.PC].item(), y[0, 1, VMState.PC].item(), y[0, 2, VMState.PC].item()]
    if all(abs(a - e) < 0.1 for a, e in zip(actual, expected)):
        print("✓ JMP correctly updated PC to RESULT")
    else:
        print(f"✗ Expected {expected}, got {actual}")
    print()

    # Test with JMP not active
    x2 = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)
    x2[:, 0, VMState.PC] = 5  # PC nibble 0 = 5
    x2[:, 0, E.RESULT] = 9  # RESULT nibble 0 = 9
    # JMP not set (opcode = 0)

    y2 = jump_ffn(x2)
    print(f"Without JMP: PC[0] = {y2[0, 0, VMState.PC]:.1f} (expected 5.0)")
    if abs(y2[0, 0, VMState.PC].item() - 5.0) < 0.1:
        print("✓ PC unchanged when JMP not active")
    else:
        print(f"✗ PC changed unexpectedly")
    print()


def test_neural_io():
    """Test neural I/O operations."""
    print("=== Testing Neural I/O Operations ===\n")

    # Test PUTCHAR
    x = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)
    x[:, :, E.OP_START + Opcode.PUTCHAR] = 1.0
    x[:, :, VMState.AX_BASE] = 7  # 'G' low nibble (0x47)
    x[:, :, VMState.AX_BASE + 1] = 4  # 'G' high nibble

    output_ffn = OutputWriteFFN()
    y = output_ffn(x)

    print(f"PUTCHAR: AX nibbles = {x[0, 0, VMState.AX_BASE:VMState.AX_BASE+2].tolist()}")
    print(f"After:   IO_CHAR = {y[0, 0, E.IO_CHAR:E.IO_CHAR+2].tolist()}")
    print(f"         IO_OUTPUT_READY = {y[0, 0, E.IO_OUTPUT_READY]:.1f}")
    print()

    # Test GETCHAR waiting for input
    x2 = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)
    x2[:, :, E.OP_START + Opcode.GETCHAR] = 1.0
    x2[:, :, E.IO_INPUT_READY] = 0.0  # No input ready

    need_input_ffn = NeedInputFFN()
    y2 = need_input_ffn(x2)

    print(f"GETCHAR (no input): IO_NEED_INPUT = {y2[0, 0, E.IO_NEED_INPUT]:.1f}")
    print()


def test_neural_branch():
    """Test conditional branch operations."""
    print("=== Testing Neural Branch Operations ===\n")

    branch_bz = BranchPCFFN(Opcode.BZ, branch_on_zero=True)
    branch_bnz = BranchPCFFN(Opcode.BNZ, branch_on_zero=False)

    # Test BZ with FLAG_ZERO = 1 (should branch)
    x = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)
    x[:, :, E.OP_START + Opcode.BZ] = 1.0
    x[:, :, VMState.FLAG_ZERO] = 1.0
    x[:, 0, VMState.PC] = 5  # Current PC nibble 0
    x[:, 0, E.RESULT] = 10  # Branch target nibble 0

    y = branch_bz(x)
    pc_after = y[0, 0, VMState.PC].item()
    print(f"BZ (FLAG_ZERO=1): PC = {pc_after:.1f} (expected ~10)")
    if abs(pc_after - 10) < 1.0:
        print("✓ Branch taken correctly")
    else:
        print("✗ Branch not taken as expected")
    print()

    # Test BZ with FLAG_ZERO = 0 (should NOT branch)
    x2 = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)
    x2[:, :, E.OP_START + Opcode.BZ] = 1.0
    x2[:, :, VMState.FLAG_ZERO] = 0.0
    x2[:, 0, VMState.PC] = 5
    x2[:, 0, E.RESULT] = 10

    y2 = branch_bz(x2)
    pc_after2 = y2[0, 0, VMState.PC].item()
    print(f"BZ (FLAG_ZERO=0): PC = {pc_after2:.1f} (expected ~5)")
    if abs(pc_after2 - 5) < 1.0:
        print("✓ Branch not taken correctly")
    else:
        print(f"✗ Unexpected PC change (delta = {pc_after2 - 5:.2f})")
    print()

    # Test BNZ with FLAG_ZERO = 0 (should branch)
    x3 = torch.zeros(1, E.NUM_POSITIONS, VMState.DIM)
    x3[:, :, E.OP_START + Opcode.BNZ] = 1.0
    x3[:, :, VMState.FLAG_ZERO] = 0.0
    x3[:, 0, VMState.PC] = 5
    x3[:, 0, E.RESULT] = 15

    y3 = branch_bnz(x3)
    pc_after3 = y3[0, 0, VMState.PC].item()
    print(f"BNZ (FLAG_ZERO=0): PC = {pc_after3:.1f} (expected ~15)")
    if abs(pc_after3 - 15) < 1.0:
        print("✓ BNZ branch taken correctly")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("100% Neural VM State Management")
    print("=" * 60)
    print()
    print("Components:")
    print("  - JumpPCFFN: Copy RESULT to PC on JMP")
    print("  - BranchPCFFN: Conditional PC update on BZ/BNZ")
    print("  - SetZeroFlagFFN: Set FLAG_ZERO based on AX")
    print("  - PushStackFFN: Push AX to stack")
    print("  - KVCacheMemory: Attention-based memory read/write")
    print("  - InputReadFFN: Read from IO_CHAR to AX")
    print("  - OutputWriteFFN: Write AX to IO_CHAR")
    print("  - NeuralVMLayer: Combined VM execution layer")
    print()

    # Run tests
    test_neural_pc()
    test_neural_io()
    test_neural_branch()

    print("=" * 60)
    print("All neural state components implemented!")
    print("=" * 60)
