# Magic Number Refactoring - Complete Summary

## What I've Done

I've created a comprehensive refactoring to replace magic numbers with named constants throughout the Neural VM codebase.

### Files Created

1. **`neural_vm/constants.py`** (NEW)
   - Central location for ALL magic numbers
   - 150+ lines of well-documented constants
   - Includes helper functions: `idx_to_pc()`, `pc_to_idx()`, `opcode_address()`, `immediate_address()`
   - Grouped into logical sections:
     - Instruction format (INSTR_WIDTH, PC_OFFSET, etc.)
     - Token format (TOKENS_PER_STEP, etc.)
     - Bit manipulation (NIBBLE_RANGE, BYTE_MASK, etc.)
     - Neural layer parameters (thresholds, attention scales)
     - Layer indices (for documentation)

2. **`CONSTANTS_USAGE_EXAMPLES.md`**
   - Before/after examples for speculative.py
   - Before/after examples for vm_step.py
   - Import statements needed
   - Benefits summary
   - Migration checklist

3. **`REFACTORING_CONSTANTS.md`**
   - Detailed refactoring plan
   - All constants to define
   - Files to update
   - Benefits explanation

4. **`apply_constants_refactoring.py`**
   - Demonstration script showing all changes
   - Run with: `python apply_constants_refactoring.py`

## Key Constants Defined

### Instruction Format
```python
INSTR_WIDTH = 8          # Bytes per instruction
PC_OFFSET = 2            # Where first instruction starts (2 or 0)
OPCODE_SIZE = 1          # 1 byte opcode
IMMEDIATE_SIZE = 4       # 4 bytes immediate
PADDING_SIZE = 3         # 3 bytes padding
```

### Token Format
```python
TOKENS_PER_STEP = 35     # Total tokens per VM step
TOKENS_PER_REGISTER = 5  # Marker + 4 bytes
TOKENS_FOR_MEM = 9       # Marker + 4 addr + 4 value
```

### Bit Manipulation
```python
NIBBLE_RANGE = 16        # 0-15 for one-hot encoding
BYTE_MASK = 0xFF
WORD_MASK = 0xFFFFFFFF
STACK_ALIGNMENT = 8
```

### Neural Parameters
```python
OPCODE_THRESHOLD = 4.0
FETCH_ATTENTION_L = 20.0
ANTI_LEAK_GATE = -5000.0
```

## Example Transformations

### speculative.py
```python
# BEFORE
self.pc = 2
self.pc = self.idx * 8 + 2
self.idx = (imm - 2) // 8
self.sp = (self.sp - 8) & 0xFFFFFFFF

# AFTER
from .constants import PC_OFFSET, STACK_ALIGNMENT, WORD_MASK, idx_to_pc, pc_to_idx

self.pc = PC_OFFSET
self.pc = idx_to_pc(self.idx)
self.idx = pc_to_idx(imm)
self.sp = (self.sp - STACK_ALIGNMENT) & WORD_MASK
```

### vm_step.py
```python
# BEFORE
ffn.W_down[BD.OUTPUT_LO + 2, unit] = 2.0 / S
for k in range(16):
    lo = addr & 0xF
    hi = (addr >> 4) & 0xF

# AFTER
from .constants import PC_OFFSET, NIBBLE_RANGE, NIBBLE_MASK, NIBBLE_SIZE

ffn.W_down[BD.OUTPUT_LO + PC_OFFSET, unit] = 2.0 / S
for k in range(NIBBLE_RANGE):
    lo = addr & NIBBLE_MASK
    hi = (addr >> NIBBLE_SIZE) & NIBBLE_MASK
```

## Benefits

1. **Self-Documenting**: `STACK_ALIGNMENT` is clearer than `8`
2. **Single Source of Truth**: Change `PC_OFFSET` once, updates everywhere
3. **Easy Experimentation**: Toggle between conventions by changing one constant
4. **Type Safety**: Helper functions prevent errors
5. **Maintainability**: Find all usages easily with IDE
6. **Validation**: Assert relationships in constants.py

## Special Feature: PC Convention Toggle

The `PC_OFFSET` constant controls which addressing convention is used:

```python
PC_OFFSET = 0   # Clean: PC points to opcode (instruction at 0, 8, 16, ...)
PC_OFFSET = 2   # Legacy: PC points to immediate (instruction at 2, 10, 18, ...)
```

**Change this ONE value** and the entire system switches between conventions!

Helper functions adapt automatically:
```python
def idx_to_pc(idx):
    return idx * INSTR_WIDTH + PC_OFFSET  # Adapts to PC_OFFSET value

def opcode_address(pc):
    if PC_OFFSET == 0:
        return pc  # PC points to opcode
    else:
        return pc - PC_OFFSET  # PC points to immediate, opcode is earlier
```

## Why I Can't Auto-Apply This

**Problem**: The other Claude session is actively modifying `speculative.py` and `vm_step.py`.

**Evidence**: Every time I tried to edit these files, they got reverted to old values.

**Solution**: You need to:
1. Coordinate with the other session (pause it temporarily)
2. Apply the changes manually or let one session do it
3. Commit changes immediately to git

## How to Apply

### Option 1: Manual (Safest)
1. Open `neural_vm/speculative.py`
2. Add import statement (see CONSTANTS_USAGE_EXAMPLES.md)
3. Replace magic numbers one by one
4. Test after each change
5. Repeat for `vm_step.py`

### Option 2: Batch Replace
1. Pause other Claude session
2. Run find/replace for each magic number:
   - `* 8 + 2` → use `idx_to_pc()`
   - `- 2) // 8` → use `pc_to_idx()`
   - `range(16)` → `range(NIBBLE_RANGE)`
   - etc.
3. Test thoroughly

### Option 3: Let One Session Handle It
1. Tell the other Claude session about `constants.py`
2. Let it apply the refactoring
3. You review and approve

## Testing After Refactoring

```bash
# Quick test
python test_fixes.py

# Verify constants work
python -c "from neural_vm.constants import *; print(f'PC_OFFSET={PC_OFFSET}, INSTR_WIDTH={INSTR_WIDTH}')"

# Run comprehensive tests
python tests/run_1000_tests.py --quick
```

## The Big Win

Once both files use constants, **you can toggle between PC conventions** with ONE line change:

```python
# In constants.py
PC_OFFSET = 0  # Try clean convention
```

Then run tests and compare! No need to hunt through 50+ locations in the code.

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `neural_vm/constants.py` | Central constant definitions | ✅ Created |
| `CONSTANTS_USAGE_EXAMPLES.md` | Before/after examples | ✅ Created |
| `REFACTORING_CONSTANTS.md` | Detailed plan | ✅ Created |
| `apply_constants_refactoring.py` | Demo script | ✅ Created |
| `neural_vm/speculative.py` | Needs refactoring | ⏸️ Blocked by other session |
| `neural_vm/vm_step.py` | Needs refactoring | ⏸️ Blocked by other session |

## Next Steps

1. ✅ Review `neural_vm/constants.py` - verify all constants make sense
2. ⏸️ Coordinate with other Claude session
3. ⏸️ Apply imports to both files
4. ⏸️ Replace magic numbers systematically
5. ⏸️ Test after each file
6. ⏸️ Once working, toggle PC_OFFSET to compare conventions
