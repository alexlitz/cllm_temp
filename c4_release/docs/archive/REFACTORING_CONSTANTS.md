# Magic Number Refactoring Plan

## Constants to Define

### Instruction Format (vm_step.py top)
```python
# =============================================================================
# Instruction Format Constants
# =============================================================================

# Instruction layout: 1 opcode byte + 4 immediate bytes + 3 padding bytes = 8 total
INSTR_WIDTH = 8          # Total bytes per instruction
OPCODE_SIZE = 1          # Opcode is 1 byte
IMMEDIATE_SIZE = 4       # Immediate value is 4 bytes (32-bit)
PADDING_SIZE = 3         # Padding to align to 8 bytes

# PC addressing scheme
PC_OFFSET = 0            # First instruction at address 0 (clean convention)
                         # Alternative: PC_OFFSET = 2 for quirky convention where PC points to immediate

# Token format per VM step (autoregressive output)
TOKENS_PER_STEP = 35     # Total tokens generated per VM step
TOKENS_PER_REGISTER = 5  # Marker + 4 bytes
TOKENS_FOR_MEM = 9       # Marker + 4 addr bytes + 4 value bytes
TOKENS_FOR_TERMINATOR = 1  # STEP_END or HALT

# Register field sizes
VALUE_BYTES = 4          # 32-bit values = 4 bytes
ADDR_BYTES = 4           # 32-bit addresses = 4 bytes

# Bit manipulation
BITS_PER_BYTE = 8
NIBBLE_SIZE = 4          # 4 bits per nibble
NIBBLES_PER_BYTE = 2     # 2 nibbles per byte
BYTE_MASK = 0xFF
WORD_MASK = 0xFFFFFFFF   # 32-bit mask

# Memory alignment
STACK_ALIGNMENT = 8      # Stack grows in 8-byte chunks
```

### Layer-specific Constants
```python
# =============================================================================
# Neural Layer Constants
# =============================================================================

# Layer indices (for documentation/debugging)
LAYER_L0_EMBED = 0
LAYER_L1_POS_ENCODE = 1
LAYER_L2_MARKER_DETECT = 2
LAYER_L3_PC_INIT = 3
LAYER_L4_PC_RELAY = 4
LAYER_L5_FETCH = 5
LAYER_L6_ROUTING = 6
LAYER_L7_OPERAND = 7
# ... etc

# FFN activation thresholds
OPCODE_THRESHOLD = 4.0   # Correct opcode ≈ 5.0, false positive ≈ 0
PC_INCREMENT_THRESHOLD = 0.5
CARRY_THRESHOLD = 1.5

# Attention parameters
ATTENTION_SCALE_L = 15.0  # General attention strength
FETCH_ATTENTION_L = 20.0  # L5 fetch attention strength
ANTI_LEAK_GATE = -5000.0  # Strong suppression to prevent leakage

# Nibble/dimension sizes
NIBBLE_RANGE = 16        # 0-15 for 4-bit nibbles
LOW_NIBBLE_SIZE = 16     # One-hot encoding dimension
HIGH_NIBBLE_SIZE = 16    # One-hot encoding dimension
```

## Files to Update

### 1. neural_vm/vm_step.py
**Lines to update:**
- Line 30: `PC_OFFSET = 0` (change from 2)
- Line 540: Use `PC_OFFSET` constant
- Line 1604: `ffn.W_down[BD.OUTPUT_LO + PC_OFFSET, unit]`
- All FFN functions: Replace magic 16 with `NIBBLE_RANGE`
- All FFN functions: Replace magic 5.0, 4.0 with named thresholds

### 2. neural_vm/speculative.py
**Lines to update:**
- Line 23: `return [(v >> (i * BITS_PER_BYTE)) & BYTE_MASK for i in range(VALUE_BYTES)]`
- Line 76: `self.pc = PC_OFFSET`
- Line 110: `self.pc = self.idx * INSTR_WIDTH + PC_OFFSET`
- Lines 120, 125, etc: `self.idx = (imm - PC_OFFSET) // INSTR_WIDTH`
- Line 122, 136, etc: `self.sp = (self.sp - STACK_ALIGNMENT) & WORD_MASK`
- Line 186: `assert len(tokens) == TOKENS_PER_STEP`

### 3. Layer initialization functions
Replace all:
- `16` → `NIBBLE_RANGE` (when referring to nibble one-hot dimensions)
- `8` → `INSTR_WIDTH` (when referring to instruction size)
- `4.0` → `OPCODE_THRESHOLD`
- `15.0` → `ATTENTION_SCALE_L`
- `20.0` → `FETCH_ATTENTION_L`
- `-5000.0` → `ANTI_LEAK_GATE`

## Benefits

1. **Self-documenting**: Code explains itself
2. **Single source of truth**: Change once, updates everywhere
3. **Type safety**: Can add type hints to constants
4. **Easier transitions**: Toggle between old (PC_OFFSET=2) and new (PC_OFFSET=0)
5. **Better error messages**: Reference constant names in comments
6. **Refactoring safety**: Find all usages easily

## Implementation Order

1. ✅ Define all constants at top of vm_step.py
2. ✅ Export constants that speculative.py needs
3. ✅ Update speculative.py to use constants
4. ✅ Update vm_step.py FFN functions systematically:
   - L3 FFN (PC initialization)
   - L4 FFN (fetch addressing)
   - L6 FFN (PC increment, routing)
   - L10 FFN (byte passthrough)
5. ✅ Update ADDR_KEY injection
6. ✅ Run tests to verify nothing broke
