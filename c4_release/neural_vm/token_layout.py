"""Token Layout Constants for Neural VM Output.

The Neural VM generates 35 tokens per execution step, representing the VM state
after executing one instruction. This module defines the positions of each field
in the token sequence.

Token Format (35 tokens per step):
    [0]       REG_PC marker (Token.REG_PC = 257)
    [1-4]     PC value (4 bytes, little-endian)
    [5]       REG_AX marker (Token.REG_AX = 258)
    [6-9]     AX value (4 bytes, little-endian)
    [10]      REG_SP marker (Token.REG_SP = 259)
    [11-14]   SP value (4 bytes, little-endian)
    [15]      REG_BP marker (Token.REG_BP = 260)
    [16-19]   BP value (4 bytes, little-endian)
    [20]      STACK0 marker (Token.STACK0 = 261)
    [21-24]   STACK0 value (4 bytes, little-endian)
    [25]      MEM marker (Token.MEM = 262)
    [26-29]   MEM address (4 bytes, little-endian)
    [30-33]   MEM value (4 bytes, little-endian)
    [34]      STEP_END or HALT (Token.STEP_END = 263, Token.HALT = 264)
"""

# Register marker positions
POS_PC_MARKER = 0
POS_AX_MARKER = 5
POS_SP_MARKER = 10
POS_BP_MARKER = 15
POS_STACK0_MARKER = 20
POS_MEM_MARKER = 25
POS_END_MARKER = 34

# Register value byte positions (first byte of 4-byte little-endian value)
POS_PC_BYTE0 = 1
POS_PC_BYTE1 = 2
POS_PC_BYTE2 = 3
POS_PC_BYTE3 = 4

POS_AX_BYTE0 = 6
POS_AX_BYTE1 = 7
POS_AX_BYTE2 = 8
POS_AX_BYTE3 = 9

POS_SP_BYTE0 = 11
POS_SP_BYTE1 = 12
POS_SP_BYTE2 = 13
POS_SP_BYTE3 = 14

POS_BP_BYTE0 = 16
POS_BP_BYTE1 = 17
POS_BP_BYTE2 = 18
POS_BP_BYTE3 = 19

POS_STACK0_BYTE0 = 21
POS_STACK0_BYTE1 = 22
POS_STACK0_BYTE2 = 23
POS_STACK0_BYTE3 = 24

POS_MEM_ADDR_BYTE0 = 26
POS_MEM_ADDR_BYTE1 = 27
POS_MEM_ADDR_BYTE2 = 28
POS_MEM_ADDR_BYTE3 = 29

POS_MEM_VAL_BYTE0 = 30
POS_MEM_VAL_BYTE1 = 31
POS_MEM_VAL_BYTE2 = 32
POS_MEM_VAL_BYTE3 = 33

# Register value ranges (inclusive)
RANGE_PC = (POS_PC_BYTE0, POS_PC_BYTE3)
RANGE_AX = (POS_AX_BYTE0, POS_AX_BYTE3)
RANGE_SP = (POS_SP_BYTE0, POS_SP_BYTE3)
RANGE_BP = (POS_BP_BYTE0, POS_BP_BYTE3)
RANGE_STACK0 = (POS_STACK0_BYTE0, POS_STACK0_BYTE3)
RANGE_MEM_ADDR = (POS_MEM_ADDR_BYTE0, POS_MEM_ADDR_BYTE3)
RANGE_MEM_VAL = (POS_MEM_VAL_BYTE0, POS_MEM_VAL_BYTE3)

# Total tokens per step
TOKENS_PER_STEP = 35

# Helper functions
def get_pc_bytes(tokens):
    """Extract PC value from token sequence."""
    return tokens[POS_PC_BYTE0:POS_PC_BYTE3+1]

def get_ax_bytes(tokens):
    """Extract AX value from token sequence."""
    return tokens[POS_AX_BYTE0:POS_AX_BYTE3+1]

def get_sp_bytes(tokens):
    """Extract SP value from token sequence."""
    return tokens[POS_SP_BYTE0:POS_SP_BYTE3+1]

def get_bp_bytes(tokens):
    """Extract BP value from token sequence."""
    return tokens[POS_BP_BYTE0:POS_BP_BYTE3+1]

def bytes_to_int32(byte_list):
    """Convert 4-byte little-endian list to 32-bit integer."""
    return (byte_list[0] |
            (byte_list[1] << 8) |
            (byte_list[2] << 16) |
            (byte_list[3] << 24))

def get_pc_value(tokens):
    """Extract PC value as integer."""
    return bytes_to_int32(get_pc_bytes(tokens))

def get_ax_value(tokens):
    """Extract AX value as integer."""
    return bytes_to_int32(get_ax_bytes(tokens))

def get_sp_value(tokens):
    """Extract SP value as integer."""
    return bytes_to_int32(get_sp_bytes(tokens))

def get_bp_value(tokens):
    """Extract BP value as integer."""
    return bytes_to_int32(get_bp_bytes(tokens))
