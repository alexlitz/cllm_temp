"""
Constants for the C4 Neural VM.

This module defines all magic numbers used throughout the neural VM implementation,
making the code more maintainable and self-documenting.
"""

# =============================================================================
# Instruction Format Constants
# =============================================================================

# Instruction layout: 1 opcode byte + 4 immediate bytes = 5 bytes, plus 3 bytes padding
# for 8-byte alignment. Model weights were trained with this scheme.
# Compiler uses * 8 and // 8 for all address calculations.
INSTR_WIDTH = 8          # Total bytes per instruction slot (5 data + 3 padding)
OPCODE_SIZE = 1          # Opcode field is 1 byte
IMMEDIATE_SIZE = 4       # Immediate value field is 4 bytes (32-bit)
PADDING_SIZE = 3         # Padding bytes to align to 8-byte boundary

# PC addressing scheme configuration
# Two conventions supported:
#   PC_OFFSET = 0: Clean convention where PC points to the opcode
#                  Instruction 0 at address 0, instruction 1 at address 8, etc.
#   PC_OFFSET = 2: Legacy convention where PC points to the first immediate byte
#                  Instruction 0 has PC=2, instruction 1 has PC=10, etc.
#
# NOTE: Both DraftVM (speculative.py) and neural VM (vm_step.py) must use
#       the same PC_OFFSET value for consistency.
# IMPORTANT: Model weights were trained with PC_OFFSET = 2
# BUT: Current implementation uses PC_OFFSET = 0 for consistency
PC_OFFSET = 0            # Direct addressing: PC=0 points to first instruction

# Derived addressing constants
def pc_to_idx(pc):
    """Convert PC value to instruction index."""
    return (pc - PC_OFFSET) // INSTR_WIDTH

def idx_to_pc(idx):
    """Convert instruction index to PC value."""
    return idx * INSTR_WIDTH + PC_OFFSET

def opcode_address(pc):
    """Get opcode address from PC value."""
    if PC_OFFSET == 0:
        return pc  # PC points directly to opcode
    else:
        return pc - PC_OFFSET  # PC points to immediate, opcode is PC_OFFSET bytes before

def immediate_address(pc):
    """Get immediate address from PC value."""
    if PC_OFFSET == 0:
        return pc + OPCODE_SIZE  # Immediate follows opcode
    else:
        return pc  # PC points directly to immediate

# =============================================================================
# Token Format Constants (Autoregressive Output)
# =============================================================================

# Each VM step generates 35 tokens in a fixed format
TOKENS_PER_STEP = 35     # Total tokens per VM step

# Token field sizes
TOKENS_PER_REGISTER = 5  # Marker + 4 value bytes (for PC, AX, SP, BP, STACK0)
TOKENS_FOR_MEM = 9       # Marker + 4 addr bytes + 4 value bytes
TOKENS_FOR_TERMINATOR = 1  # STEP_END or HALT

# Validate token count
assert (TOKENS_PER_REGISTER * 5  # PC, AX, SP, BP, STACK0
        + TOKENS_FOR_MEM         # MEM field
        + TOKENS_FOR_TERMINATOR  # Terminator
        ) == TOKENS_PER_STEP, "Token count mismatch"

# =============================================================================
# Data Type Sizes
# =============================================================================

VALUE_BYTES = 4          # 32-bit register values = 4 bytes
ADDR_BYTES = 4           # 32-bit memory addresses = 4 bytes

# =============================================================================
# Bit Manipulation Constants
# =============================================================================

BITS_PER_BYTE = 8
NIBBLE_SIZE = 4          # 4 bits per nibble
NIBBLES_PER_BYTE = 2     # 2 nibbles per byte (low and high)
NIBBLE_RANGE = 16        # 0-15 for 4-bit nibbles (used for one-hot encoding dimensions)

# Masks
BYTE_MASK = 0xFF         # 8-bit mask
WORD_MASK = 0xFFFFFFFF   # 32-bit mask
NIBBLE_MASK = 0x0F       # 4-bit mask

# =============================================================================
# Memory Layout Constants
# =============================================================================

STACK_ALIGNMENT = 8      # Stack pointer grows/shrinks in 8-byte chunks
STACK_INIT = 0x10000     # Initial stack pointer (typical C4 convention)
HEAP_INIT = 0x200000     # Initial heap pointer for malloc

# =============================================================================
# Neural Layer Thresholds and Parameters
# =============================================================================

# FFN activation thresholds
# These control when SwiGLU units activate based on input conditions
OPCODE_THRESHOLD = 4.0   # Separates correct opcode (≈5.0) from false positive (≈0)
PC_INCREMENT_THRESHOLD = 0.5  # Threshold for PC increment logic
CARRY_THRESHOLD = 1.5    # Threshold for multi-input carry operations

# Attention scale parameters
# L = attention scale; higher values make attention more focused
ATTENTION_SCALE_L = 15.0      # General attention strength for relay operations
FETCH_ATTENTION_L = 20.0      # L5 fetch attention (stronger for precise address matching)
RELAY_ATTENTION_L = 15.0      # L4/L6 relay attention strength

# Anti-leakage gates
# Strong negative bias to suppress unintended attention at non-target positions
ANTI_LEAK_GATE = -5000.0      # Standard anti-leakage gate value
ANTI_LEAK_GATE_STRONG = -10000.0  # Extra-strong suppression for critical paths

# Attention head slopes (ALiBi)
# Penalizes attention based on distance: score -= slope * distance
ALIBI_SLOPE_STANDARD = 0.1    # Standard slope for positional bias
ALIBI_SLOPE_STEEP = 5.0       # Steep slope for focused relay (suppress distance)

# =============================================================================
# Dimension Sizes (for documentation)
# =============================================================================

# These are informational constants documenting the model architecture
D_MODEL = 512            # Model hidden dimension
N_HEADS = 8              # Number of attention heads
HEAD_DIM = 64            # Dimension per head (D_MODEL / N_HEADS)
FFN_HIDDEN_DIM = 256     # FFN intermediate dimension (d_ffn in SwiGLU)

N_LAYERS = 16            # Total number of transformer layers

# =============================================================================
# Layer-Specific Constants
# =============================================================================

# Layer indices (for documentation and debugging)
LAYER_L0_EMBED = 0           # Embedding layer
LAYER_L1_POSITIONAL = 1      # Positional encoding
LAYER_L2_MARKER = 2          # Marker detection and HAS_SE flag
LAYER_L3_CARRY_INIT = 3      # Register carry-forward and PC initialization
LAYER_L4_PC_RELAY = 4        # PC relay to AX marker for fetch
LAYER_L5_FETCH = 5           # Instruction fetch (opcode and immediate)
LAYER_L6_ROUTING = 6         # Output routing and PC increment
LAYER_L7_OPERAND_GATHER = 7  # Gather operands for ALU ops
LAYER_L8_ALU_START = 8       # ALU operations start
LAYER_L9_ALU_CONTINUE = 9    # ALU operations continue
LAYER_L10_BYTE_PASS = 10     # Byte passthrough for multi-byte values
LAYER_L11_MEM_GATHER = 11    # Memory address gathering
LAYER_L12_MEM_LOOKUP = 12    # Memory value lookup (placeholder)
LAYER_L13_MEM_ROUTE = 13     # Memory routing to output
LAYER_L14_CLEANUP = 14       # Cleanup and consolidation
LAYER_L15_FINAL = 15         # Final layer
