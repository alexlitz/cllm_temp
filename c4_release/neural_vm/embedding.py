"""
Embedding layout and opcode definitions for Neural VM V7.

Full C4 compiler opcode compatibility.
"""


class E:
    """Embedding dimensions - 8 nibble positions, each with features."""

    # Per-nibble features
    NIB_A = 0          # Operand A nibble (0-15 as scalar, for ADD/SUB)
    NIB_B = 1          # Operand B nibble (0-15 as scalar, for ADD/SUB)

    # Binary encoding of nibbles (for exact lookup operations)
    NIB_A_BIT0 = 2     # Operand A bit 0 (LSB)
    NIB_A_BIT1 = 3     # Operand A bit 1
    NIB_A_BIT2 = 4     # Operand A bit 2
    NIB_A_BIT3 = 5     # Operand A bit 3 (MSB)
    NIB_B_BIT0 = 6     # Operand B bit 0 (LSB)
    NIB_B_BIT1 = 7     # Operand B bit 1
    NIB_B_BIT2 = 8     # Operand B bit 2
    NIB_B_BIT3 = 9     # Operand B bit 3 (MSB)

    RAW_SUM = 10       # Raw A + B or A - B (before carry/borrow)
    CARRY_IN = 11      # Carry/borrow from lower nibble
    CARRY_OUT = 12     # Carry/borrow to higher nibble
    RESULT = 13        # Result nibble
    TEMP = 14          # Temporary storage for multi-step ops

    # Opcode encoding (shared across positions)
    OPCODE = 15        # Scalar opcode value (0-71) for continuous gating
    OP_START = 16      # Start of one-hot region (deprecated, keeping for compatibility)
    NUM_OPS = 72       # Extended for C4 compatibility (0-38 + 64-66)

    # Position encoding
    # NOTE: Position should be derived from ALiBi/RoPE positional encoding,
    # not stored directly in the embedding. The POS slot is populated by
    # PositionEncoder layer (see position_encoding.py) which uses:
    #   - ALiBi: attention bias based on distance
    #   - RoPE: Q/K rotation based on position
    # FFN layers then read from POS after position injection.
    # This ensures position comes from positional encoding, not hardcoded values.
    POS = 88           # Shifted by +8 for binary bits, +1 for OPCODE

    # I/O Mailbox Slots (for embedding-based I/O)
    IO_CHAR = 89       # Slots 80-87: Character as nibbles (8 nibbles = 32 bits)
    IO_OUTPUT_READY = 97   # 1.0 when PUTCHAR has a char to emit
    IO_INPUT_READY = 98    # 1.0 when external input is available
    IO_NEED_INPUT = 99     # 1.0 when GETCHAR needs more input
    IO_PROGRAM_END = 100    # 1.0 when EXIT is called
    IO_EXIT_CODE = 101      # Exit code value

    # Tool-Use Mode Slots
    IO_TOOL_CALL_TYPE = 102   # Type of tool call (0=none, 1=getchar, 2=putchar, ...)
    IO_TOOL_CALL_ID = 103     # Unique call ID
    IO_TOOL_RESPONSE = 104    # Response value from handler

    # Argv Mailbox Slots (for plaintext argc/argv passing)
    IO_ARGC = 105             # Argument count (set before program starts)
    IO_ARGV_INDEX = 106       # Current argv index being read (0-based)
    IO_NEED_ARGV = 107        # 1.0 when program needs next argv character
    IO_ARGV_READY = 108       # 1.0 when argv character is available in IO_CHAR
    IO_ARGV_END = 109        # 1.0 when current argv string is complete (\0)
    IO_ALL_ARGV_READ = 110   # 1.0 when all argv strings have been read

    # Heap management for bump allocator
    HEAP_BASE = 113          # Base of heap [104:112] (8 nibbles, fixed at program start)
    HEAP_PTR = 121           # Current allocation pointer [112:120] (bumps upward)
    HEAP_END = 129           # End of heap [120:128] (for bounds checking)

    # AX register for return values (used by malloc)
    AX_BASE = 137            # AX register [128:136] (8 nibbles)

    # Memory interface
    MEM_ADDR_BASE = 145      # Memory address [136:144] (8 nibbles)
    MEM_DATA_BASE = 153      # Memory data [144:152] (8 nibbles)
    MEM_WRITE = 161          # 1.0 = write request pending
    MEM_READ = 162           # 1.0 = read request pending
    MEM_READY = 163          # 1.0 = memory operation complete

    # Shift extraction temp slots (used by unified bit extraction)
    SHIFT_EXTRACT_A = 164    # Temp slot for bit 2 extraction
    SHIFT_EXTRACT_B = 165    # Temp slot for bit 3 extraction

    # Fused schoolbook MUL temp slots (second pair for double-offset)
    TEMP_A2 = 166            # Second a_slot for fused double-offset MUL
    TEMP_B2 = 167            # Second b_slot for fused double-offset MUL

    DIM = 169          # Total per-position dimension (was 168)
    NUM_POSITIONS = 8  # 8 nibbles

    # Scale for SwiGLU identity (higher = tighter approximations)
    SCALE = 100.0

    # Scale for quotient computation in ComputeQuotientNibbleFFN.
    # Must be high enough that silu(-S) is negligible, because the
    # remainder is multiplied by 16 each iteration (8 iterations total),
    # amplifying any silu leakage by up to 16^8 ~ 4.3e9.
    # S=10 gives silu(-10) ~ 4.5e-4, causing catastrophic error amplification.
    # S=50 gives silu(-50) ~ 1e-20, which is safe even after 16^8 amplification.
    # Float32 headroom: max(up) = S * 15 * max_divisor ~ 50 * 15 * 4.3e9 ~ 3.2e12,
    # well within float32 range (3.4e38).
    DIV_Q_SCALE = 50.0

    # Scale for remainder extraction (needs higher value for sharp integer steps)
    DIV_SCALE = 100.0


class IOToolCallType:
    """Tool call types for agentic I/O mode."""
    NONE = 0
    GETCHAR = 1
    PUTCHAR = 2
    EXIT = 3
    OPEN = 4
    READ = 5
    CLOSE = 6
    PRINTF = 7


class Opcode:
    """
    C4 VM Opcodes - Full compatibility with C4 compiler.

    The opcode values match the C4 compiler output exactly.
    """

    # Stack/Address operations (0-8)
    LEA = 0     # Load effective address: AX = BP + imm
    IMM = 1     # Load immediate: AX = imm
    JMP = 2     # Unconditional jump: PC = imm
    JSR = 3     # Jump subroutine: push PC, PC = imm
    BZ = 4      # Branch if zero: if AX == 0 then PC = imm
    BNZ = 5     # Branch if not zero: if AX != 0 then PC = imm
    ENT = 6     # Enter function: push BP, BP = SP, SP -= imm
    ADJ = 7     # Adjust stack: SP += imm
    LEV = 8     # Leave function: SP = BP, BP = pop, PC = pop

    # Memory operations (9-13)
    LI = 9      # Load int: AX = *AX (load 8 bytes from address in AX)
    LC = 10     # Load char: AX = *(char*)AX (load 1 byte)
    SI = 11     # Store int: *pop = AX (store 8 bytes)
    SC = 12     # Store char: *(char*)pop = AX (store 1 byte)
    PSH = 13    # Push: SP -= 8, *SP = AX

    # Bitwise operations (14-16)
    OR = 14     # AX = pop | AX
    XOR = 15    # AX = pop ^ AX
    AND = 16    # AX = pop & AX

    # Comparison operations (17-22)
    EQ = 17     # AX = (pop == AX)
    NE = 18     # AX = (pop != AX)
    LT = 19     # AX = (pop < AX)
    GT = 20     # AX = (pop > AX)
    LE = 21     # AX = (pop <= AX)
    GE = 22     # AX = (pop >= AX)

    # Shift operations (23-24)
    SHL = 23    # AX = pop << AX
    SHR = 24    # AX = pop >> AX

    # Arithmetic operations (25-29)
    ADD = 25    # AX = pop + AX
    SUB = 26    # AX = pop - AX
    MUL = 27    # AX = pop * AX
    DIV = 28    # AX = pop / AX
    MOD = 29    # AX = pop % AX

    # System calls (30-38)
    OPEN = 30   # AX = open(filename, mode)
    READ = 31   # AX = read(fd, buf, count)
    CLOS = 32   # close(fd)
    PRTF = 33   # printf(fmt, ...)
    MALC = 34   # AX = malloc(size)
    FREE = 35   # free(ptr)
    MSET = 36   # memset(ptr, val, size)
    MCMP = 37   # AX = memcmp(p1, p2, size)
    EXIT = 38   # exit(code)

    # I/O operations (64-66)
    GETCHAR = 64   # AX = getchar()
    PUTCHAR = 65   # putchar(AX)
    PRINTF2 = 66   # printf variant

    # Aliases for backward compatibility with neural_vm simplified ISA
    # (These map to the C4 equivalents)
    CALL = JSR     # Function call (alias for JSR)
    RET = LEV      # Return (alias for LEV)
    BEQ = BZ       # Branch if equal/zero (alias for BZ)
    BNE = BNZ      # Branch if not equal/not zero (alias for BNZ)
    BLT = 41       # Branch if less than (neural VM specific)
    BGE = 42       # Branch if greater or equal (neural VM specific)
    LOAD = LI      # Load (alias for LI)
    STORE = SI     # Store (alias for SI)
    PUSH = PSH     # Push (alias for PSH)
    NOP = 39       # No operation (unused C4 slot)
    HALT = EXIT    # Halt (alias for EXIT)
    POP = 40       # Pop from stack (neural VM specific)
