"""Layer-specific constants for Neural VM architecture.

This module defines layer indices, threshold values, and other layer-specific
constants used throughout the Neural VM weight initialization.
"""

# =============================================================================
# Layer Indices
# =============================================================================

# Total number of transformer layers
NUM_LAYERS = 16

# Specific layer roles
LAYER_EMBEDDING = -1  # Conceptual layer (embedding)
LAYER_L0 = 0          # First threshold detection layer
LAYER_L1 = 1          # Threshold pattern refinement
LAYER_L2 = 2          # Threshold final decision
LAYER_L3 = 3          # PC/SP/BP defaults + PC increment
LAYER_L4 = 4          # PC relay to AX marker
LAYER_L5 = 5          # Fetch immediate + opcode decode
LAYER_L6 = 6          # Function call output routing
LAYER_L7 = 7          # Operand gather (memory, stack)
LAYER_L8 = 8          # ALU operations (lo nibble)
LAYER_L9 = 9          # ALU operations (hi nibble)
LAYER_L10 = 10        # Opcode-specific routing
LAYER_L11 = 11        # Control flow + stack compaction
LAYER_L12 = 12        # Memory operations
LAYER_L13 = 13        # I/O operations
LAYER_L14 = 14        # Final value selection
LAYER_L15 = 15        # Memory address keys + output logits

# =============================================================================
# Threshold Values (for SwiGLU activation)
# =============================================================================

# These thresholds determine when FFN units activate based on input sums.
# Format: threshold T where activation occurs when sum(inputs) > T

# Layer 3 thresholds
THRESHOLD_PC_MARKER = 0.5      # PC marker detection
THRESHOLD_HAS_SE = 0.5         # Has side-effect detection
THRESHOLD_PC_INCREMENT = 1.5   # PC + HAS_SE activation
THRESHOLD_PC_CARRY = 5.5       # PC carry propagation

# Layer 5 thresholds
THRESHOLD_FETCH_MATCH = 1.5    # Opcode byte matching
THRESHOLD_LEA_DECODE = 2.5     # LEA opcode detection

# Layer 6 thresholds
THRESHOLD_JSR_RELAY = 1.5      # JSR flag relay
THRESHOLD_LEA_FIRST_STEP = 1.5 # LEA first-step ALU init

# Layer 8 thresholds
THRESHOLD_ALU_3WAY = 2.5       # 3-way AND for ALU ops
THRESHOLD_LEA_LO = 15.5        # LEA lo nibble (high due to amplification)

# Layer 9 thresholds
THRESHOLD_ALU_NO_CARRY = 2.5   # Hi nibble without carry
THRESHOLD_ALU_WITH_CARRY = 2.9 # Hi nibble with carry
THRESHOLD_LEA_HI = 15.5        # LEA hi nibble (high due to amplification)
THRESHOLD_LEA_HI_CARRY = 15.9  # LEA hi nibble with carry

# =============================================================================
# Attention Constants
# =============================================================================

# Query/Key weights for strong attention patterns
ATTN_WEIGHT_LARGE = 50.0       # Large attention weight (L)
ATTN_WEIGHT_MEDIUM = 15.0      # Medium attention weight
ATTN_WEIGHT_BLOCKING = 500.0   # Anti-leakage blocking weight

# ALiBi slopes for position-based attention
ALIBI_SLOPE_STANDARD = 0.5     # Standard ALiBi slope
ALIBI_SLOPE_STEEP = 5.0        # Steep ALiBi for critical heads

# =============================================================================
# FFN Unit Scaling
# =============================================================================

# Standard scaling factor for FFN weights
# Used to ensure activations are in appropriate range
FFN_SCALE_S = 16.0             # Primary scaling factor (S)

# Output scaling for normalized results
FFN_OUTPUT_SCALE = 2.0 / FFN_SCALE_S  # Standard output: 2.0/S

# =============================================================================
# Dimension Cleaning
# =============================================================================

# Large negative values used to clear dimensions
CLEAR_VALUE_LARGE = -10.0 / FFN_SCALE_S  # For dimension clearing

# =============================================================================
# Head Dimensions
# =============================================================================

# Attention head configuration
NUM_HEADS = 8
HEAD_DIM = 64
MODEL_DIM = NUM_HEADS * HEAD_DIM  # 512

# =============================================================================
# Opcode Values
# =============================================================================

# Key opcode values referenced in weight initialization
OPCODE_LEA = 0
OPCODE_IMM = 1
OPCODE_JMP = 2
OPCODE_JSR = 3
OPCODE_EXIT = 38

# =============================================================================
# Register Marker Values
# =============================================================================

# From Token class, but repeated here for convenience
# These are the token IDs that mark register outputs
MARKER_REG_PC = 257
MARKER_REG_AX = 258
MARKER_REG_SP = 259
MARKER_REG_BP = 260
MARKER_STACK0 = 261
MARKER_MEM = 262
MARKER_STEP_END = 263
MARKER_HALT = 264

# =============================================================================
# Helper Functions
# =============================================================================

def get_layer_name(layer_idx):
    """Get descriptive name for layer index."""
    layer_names = {
        0: "L0: Threshold Detection",
        1: "L1: Threshold Refinement",
        2: "L2: Threshold Decision",
        3: "L3: PC/SP/BP Init",
        4: "L4: PC Relay",
        5: "L5: Fetch & Decode",
        6: "L6: Call Routing",
        7: "L7: Operand Gather",
        8: "L8: ALU Lo",
        9: "L9: ALU Hi",
        10: "L10: Op Routing",
        11: "L11: Control Flow",
        12: "L12: Memory Ops",
        13: "L13: I/O Ops",
        14: "L14: Value Selection",
        15: "L15: Address Keys & Output",
    }
    return layer_names.get(layer_idx, f"L{layer_idx}")

def is_threshold_layer(layer_idx):
    """Check if layer is a threshold detection layer (L0-L2)."""
    return layer_idx in (LAYER_L0, LAYER_L1, LAYER_L2)

def is_alu_layer(layer_idx):
    """Check if layer is an ALU layer (L8-L9)."""
    return layer_idx in (LAYER_L8, LAYER_L9)
