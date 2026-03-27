# Constants Usage Examples

## How to Use the New constants.py Module

### Import Statement
```python
from .constants import (
    INSTR_WIDTH, PC_OFFSET,
    NIBBLE_RANGE, BYTE_MASK, WORD_MASK,
    STACK_ALIGNMENT, VALUE_BYTES,
    TOKENS_PER_STEP,
    OPCODE_THRESHOLD, ANTI_LEAK_GATE,
    pc_to_idx, idx_to_pc,
    opcode_address, immediate_address,
)
```

---

## speculative.py Examples

### Before: Magic Numbers
```python
def _le_bytes(val):
    """32-bit value -> 4 little-endian byte tokens."""
    v = val & 0xFFFFFFFF
    return [(v >> (i * 8)) & 0xFF for i in range(4)]

class DraftVM:
    def __init__(self, bytecode):
        self.pc = 2  # Magic number!

    def step(self):
        self.idx += 1
        self.pc = self.idx * 8 + 2  # Magic numbers!

        if op == 2:  # JMP
            self.pc = imm & 0xFFFFFFFF
            self.idx = (imm - 2) // 8  # Magic numbers!

        elif op == 13:  # PSH
            self.sp = (self.sp - 8) & 0xFFFFFFFF  # Magic numbers!

    def draft_tokens(self):
        tokens = []
        # ... build tokens ...
        assert len(tokens) == 35  # Magic number!
        return tokens
```

### After: Named Constants
```python
from .constants import (
    INSTR_WIDTH, PC_OFFSET, BITS_PER_BYTE, BYTE_MASK, WORD_MASK,
    STACK_ALIGNMENT, VALUE_BYTES, TOKENS_PER_STEP,
    pc_to_idx, idx_to_pc,
)

def _le_bytes(val):
    """32-bit value -> VALUE_BYTES little-endian byte tokens."""
    v = val & WORD_MASK
    return [(v >> (i * BITS_PER_BYTE)) & BYTE_MASK for i in range(VALUE_BYTES)]

class DraftVM:
    def __init__(self, bytecode):
        self.pc = PC_OFFSET  # Initial PC for first instruction

    def step(self):
        self.idx += 1
        self.pc = idx_to_pc(self.idx)  # Use helper function

        if op == 2:  # JMP
            self.pc = imm & WORD_MASK
            self.idx = pc_to_idx(imm)  # Use helper function

        elif op == 13:  # PSH
            self.sp = (self.sp - STACK_ALIGNMENT) & WORD_MASK

    def draft_tokens(self):
        tokens = []
        # ... build tokens ...
        assert len(tokens) == TOKENS_PER_STEP
        return tokens
```

---

## vm_step.py Examples

### Before: Magic Numbers in FFN
```python
def _set_layer3_ffn(ffn, S, BD):
    # PC FIRST-STEP DEFAULT: set PC=2
    ffn.W_down[BD.OUTPUT_LO + 2, unit] = 2.0 / S  # Magic 2!

    # PC passthrough for 16 nibbles
    for k in range(16):  # Magic 16!
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
```

### After: Named Constants
```python
from .constants import PC_OFFSET, NIBBLE_RANGE

def _set_layer3_ffn(ffn, S, BD):
    # PC FIRST-STEP DEFAULT: set PC=PC_OFFSET
    ffn.W_down[BD.OUTPUT_LO + PC_OFFSET, unit] = 2.0 / S

    # PC passthrough for all nibble positions
    for k in range(NIBBLE_RANGE):
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
```

---

### Before: Magic Numbers in L6 Routing
```python
def _set_layer6_routing_ffn(ffn, S, BD):
    unit = 0
    T = 4.0  # Magic threshold!

    # IMM: FETCH → OUTPUT
    for k in range(16):  # Magic 16!
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
```

### After: Named Constants
```python
from .constants import OPCODE_THRESHOLD, NIBBLE_RANGE

def _set_layer6_routing_ffn(ffn, S, BD):
    unit = 0

    # IMM: FETCH → OUTPUT
    for k in range(NIBBLE_RANGE):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * OPCODE_THRESHOLD
```

---

### Before: Magic Numbers in PC Increment (L6)
```python
# Low nibble increment with wrap: (k+8)%16
for k in range(16):
    new_k = (k + 8) % 16
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5  # Magic 0.5!
```

### After: Named Constants
```python
from .constants import NIBBLE_RANGE, INSTR_WIDTH, PC_INCREMENT_THRESHOLD

# Low nibble increment with wrap: (k+INSTR_WIDTH)%NIBBLE_RANGE
for k in range(NIBBLE_RANGE):
    new_k = (k + INSTR_WIDTH) % NIBBLE_RANGE
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * PC_INCREMENT_THRESHOLD
```

---

### Before: Magic Numbers in Attention
```python
def _set_layer5_fetch(attn, S, BD, HD):
    L = 20.0  # Magic number!

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0  # Magic!
    attn.W_q[base + GATE, BD.CONST] = -500.0   # Magic!
```

### After: Named Constants
```python
from .constants import FETCH_ATTENTION_L, ANTI_LEAK_GATE

def _set_layer5_fetch(attn, S, BD, HD):
    L = FETCH_ATTENTION_L

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = -ANTI_LEAK_GATE
    attn.W_q[base + GATE, BD.CONST] = ANTI_LEAK_GATE
```

---

### Before: Magic Numbers in ADDR_KEY Injection
```python
def _inject_code_addr_keys(self, token_ids, x):
    for i in range(S):
        if cs_pos is not None and tok < 256:
            addr = i - cs_pos - 1 + 2  # Magic +2!
            lo = addr & 0xF           # Magic 0xF!
            hi = (addr >> 4) & 0xF    # Magic 4 and 0xF!
```

### After: Named Constants
```python
from .constants import PC_OFFSET, NIBBLE_MASK, NIBBLE_SIZE

def _inject_code_addr_keys(self, token_ids, x):
    for i in range(S):
        if cs_pos is not None and tok < 256:
            addr = i - cs_pos - 1 + PC_OFFSET
            lo = addr & NIBBLE_MASK
            hi = (addr >> NIBBLE_SIZE) & NIBBLE_MASK
```

---

## Benefits Summary

### 1. Self-Documenting Code
```python
# Before: What does 8 mean?
self.sp = (self.sp - 8) & 0xFFFFFFFF

# After: Clear intent
self.sp = (self.sp - STACK_ALIGNMENT) & WORD_MASK
```

### 2. Single Source of Truth
```python
# Before: Change PC offset? Update 50+ locations!
self.pc = 2
self.pc = self.idx * 8 + 2
self.idx = (imm - 2) // 8

# After: Change once in constants.py
self.pc = PC_OFFSET
self.pc = idx_to_pc(self.idx)
self.idx = pc_to_idx(imm)
```

### 3. Type Safety and Validation
```python
# Constants file can include assertions
assert (TOKENS_PER_REGISTER * 5 + TOKENS_FOR_MEM + TOKENS_FOR_TERMINATOR) == TOKENS_PER_STEP

# Helper functions prevent errors
def idx_to_pc(idx):
    """Convert instruction index to PC value."""
    return idx * INSTR_WIDTH + PC_OFFSET
```

### 4. Easier Experimentation
```python
# Want to try PC_OFFSET=0? Change one line in constants.py
PC_OFFSET = 0  # Switch to clean convention

# Want different instruction width? Change one constant
INSTR_WIDTH = 16  # Try 16-byte instructions
```

---

## Migration Checklist

- [ ] Import constants at top of speculative.py
- [ ] Replace magic numbers in _le_bytes()
- [ ] Replace PC formula in DraftVM.__init__()
- [ ] Replace PC formula in DraftVM.step()
- [ ] Replace jump/branch conversions (JMP, JSR, BZ, BNZ, LEV)
- [ ] Replace stack operations (PSH, ADJ, ENT, LEV)
- [ ] Replace token count assertion
- [ ] Import constants at top of vm_step.py
- [ ] Replace L3 FFN magic numbers
- [ ] Replace L4 FFN magic numbers
- [ ] Replace L5 fetch magic numbers
- [ ] Replace L6 routing magic numbers
- [ ] Replace L10 passthrough magic numbers
- [ ] Replace ADDR_KEY injection magic numbers
- [ ] Run tests to verify nothing broke
