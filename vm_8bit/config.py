"""
8-bit Neural VM configuration.

ISA definition, token vocabulary, memory layout, and instruction format.
Values are 0-255, addresses are 0-255, 256 bytes total memory.
"""


class Op:
    IMM = 1
    JMP = 2
    JSR = 3
    BZ = 4
    BNZ = 5
    ENT = 6
    ADJ = 7
    LEV = 8
    LI = 9
    LC = 10
    SI = 11
    SC = 12
    PSH = 13
    OR = 14
    XOR = 15
    AND = 16
    EQ = 17
    NE = 18
    LT = 19
    GT = 20
    LE = 21
    GE = 22
    SHL = 23
    SHR = 24
    ADD = 25
    SUB = 26
    MUL = 27
    DIV = 28
    MOD = 29
    EXIT = 38
    GETCHAR = 64
    PUTCHAR = 65


ALU_OPS = {Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD,
           Op.AND, Op.OR, Op.XOR, Op.SHL, Op.SHR}
COMPARE_OPS = {Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE}
BINARY_OPS = ALU_OPS | COMPARE_OPS
JUMP_OPS = {Op.JMP, Op.JSR, Op.BZ, Op.BNZ}

INSTR_WIDTH = 2
VALUE_MASK = 0xFF
STACK_INIT = 0xFE
HEAP_START = 0x80

NIBBLE_BITS = 4
NUM_NIBBLES = 2
NIBBLE_BASE = 16
NIBBLE_MASK = 0xF


class Tok:
    CODE_START = 256
    CODE_END = 257
    REG_PC = 258
    REG_AX = 259
    REG_SP = 260
    REG_BP = 261
    STACK0 = 262
    MEM = 263
    STEP_END = 264
    HALT = 265
    VOCAB_SIZE = 266


TOKENS_PER_STEP = 14
TOKENS_PER_REG = 2
TOKENS_FOR_MEM = 3
