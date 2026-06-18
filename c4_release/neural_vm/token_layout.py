"""Token Layout Constants for Neural VM Output.

The Neural VM generates ``Token.STEP_TOKENS`` tokens per execution step,
representing the VM state after executing one instruction. This module defines
the positions of each field in the token sequence.

Default token format (35 tokens per step):
    [0]       REG_PC marker (Token.REG_PC = 257)
    [1-4]     PC value (4 bytes, little-endian)
    [5]       REG_AX marker (Token.REG_AX = 258)
    [6-9]     AX value (4 bytes, little-endian)
    [10]      REG_SP marker (Token.REG_SP = 259)
    [11-14]   SP value (4 bytes, little-endian)
    [15]      REG_BP marker (Token.REG_BP = 260)
    [16-19]   BP value (4 bytes, little-endian)
    [20]      STACK0 marker (Token.STACK0 = 268)
    [21-24]   STACK0 value (4 bytes, little-endian)
    [25]      MEM marker (Token.MEM = 261)
    [26-29]   MEM address (4 bytes, little-endian)
    [30-33]   MEM value (4 bytes, little-endian)
    [34]      STEP_END or HALT (Token.STEP_END = 262, Token.HALT = 263)

Prototype 30-token format (``C4_NO_STACK0_EMIT=1`` => ``Token.STEP_TOKENS == 30``):
the STACK0 register block (marker + 4 value bytes) is DROPPED, so everything
from the MEM marker on shifts 5 positions earlier. The register markers stay at
{0, 5, 10, 15}; MEM marker 25 -> 20, MEM addr 26..29 -> 21..24, MEM val
30..33 -> 25..28, STEP_END 34 -> 29. ``POS_STACK0_*`` are still exported (set to
``None`` under the flag) so importers that reference them don't ImportError; the
flag-OFF (35-token) values are byte-IDENTICAL to the historical hardcodes.

All positions below are PARAMETRIZED on ``Token.STEP_TOKENS`` — a no-op when it
is 35. ``Token`` is the single authority (it resolves the env flag once at
import); importing it here is safe (``vm_step`` does not import this module, so
no cycle).
"""

from .vm_step import Token

# True when the 30-token (STACK0-dropped) layout is active.
_NO_STACK0_EMIT = Token.STEP_TOKENS == 30

# Register marker positions. PC/AX/SP/BP markers are fixed at {0,5,10,15} in
# both layouts. STACK0 occupies slot 20 ONLY in the 35-token layout; under the
# 30-token flag it is dropped (None) and the MEM/SE markers shift 5 earlier.
POS_PC_MARKER = 0
POS_AX_MARKER = 5
POS_SP_MARKER = 10
POS_BP_MARKER = 15
POS_STACK0_MARKER = None if _NO_STACK0_EMIT else 20
POS_MEM_MARKER = 20 if _NO_STACK0_EMIT else 25
POS_END_MARKER = Token.STEP_TOKENS - 1  # 29 (30-token) / 34 (35-token)

# Register value byte positions (first byte of 4-byte little-endian value).
# Byte ``k`` lives at ``POS_<REG>_BYTE0 + k``.
POS_PC_BYTE0 = POS_PC_MARKER + 1
POS_PC_BYTE1 = POS_PC_BYTE0 + 1
POS_PC_BYTE2 = POS_PC_BYTE0 + 2
POS_PC_BYTE3 = POS_PC_BYTE0 + 3

POS_AX_BYTE0 = POS_AX_MARKER + 1
POS_AX_BYTE1 = POS_AX_BYTE0 + 1
POS_AX_BYTE2 = POS_AX_BYTE0 + 2
POS_AX_BYTE3 = POS_AX_BYTE0 + 3

POS_SP_BYTE0 = POS_SP_MARKER + 1
POS_SP_BYTE1 = POS_SP_BYTE0 + 1
POS_SP_BYTE2 = POS_SP_BYTE0 + 2
POS_SP_BYTE3 = POS_SP_BYTE0 + 3

POS_BP_BYTE0 = POS_BP_MARKER + 1
POS_BP_BYTE1 = POS_BP_BYTE0 + 1
POS_BP_BYTE2 = POS_BP_BYTE0 + 2
POS_BP_BYTE3 = POS_BP_BYTE0 + 3

# STACK0 value bytes — only present in the 35-token layout (None under the flag).
if POS_STACK0_MARKER is None:
    POS_STACK0_BYTE0 = POS_STACK0_BYTE1 = POS_STACK0_BYTE2 = POS_STACK0_BYTE3 = None
else:
    POS_STACK0_BYTE0 = POS_STACK0_MARKER + 1
    POS_STACK0_BYTE1 = POS_STACK0_BYTE0 + 1
    POS_STACK0_BYTE2 = POS_STACK0_BYTE0 + 2
    POS_STACK0_BYTE3 = POS_STACK0_BYTE0 + 3

POS_MEM_ADDR_BYTE0 = POS_MEM_MARKER + 1
POS_MEM_ADDR_BYTE1 = POS_MEM_ADDR_BYTE0 + 1
POS_MEM_ADDR_BYTE2 = POS_MEM_ADDR_BYTE0 + 2
POS_MEM_ADDR_BYTE3 = POS_MEM_ADDR_BYTE0 + 3

POS_MEM_VAL_BYTE0 = POS_MEM_ADDR_BYTE0 + 4
POS_MEM_VAL_BYTE1 = POS_MEM_VAL_BYTE0 + 1
POS_MEM_VAL_BYTE2 = POS_MEM_VAL_BYTE0 + 2
POS_MEM_VAL_BYTE3 = POS_MEM_VAL_BYTE0 + 3

# Register value ranges (inclusive)
RANGE_PC = (POS_PC_BYTE0, POS_PC_BYTE3)
RANGE_AX = (POS_AX_BYTE0, POS_AX_BYTE3)
RANGE_SP = (POS_SP_BYTE0, POS_SP_BYTE3)
RANGE_BP = (POS_BP_BYTE0, POS_BP_BYTE3)
RANGE_STACK0 = (
    None if POS_STACK0_BYTE0 is None else (POS_STACK0_BYTE0, POS_STACK0_BYTE3)
)
RANGE_MEM_ADDR = (POS_MEM_ADDR_BYTE0, POS_MEM_ADDR_BYTE3)
RANGE_MEM_VAL = (POS_MEM_VAL_BYTE0, POS_MEM_VAL_BYTE3)

# Total tokens per step (35 flag-OFF, 30 flag-ON). Mirrors Token.STEP_TOKENS.
TOKENS_PER_STEP = Token.STEP_TOKENS

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
