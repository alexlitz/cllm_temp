# The +2 Offset Mystery - EXPLAINED

## The Key Insight (from autoregressive_runtime.c line 543):

**"Actually PC points to the first immediate byte, so the opcode is at PC-1"**

## What This Means

### Traditional PC (Program Counter):
In a normal CPU, PC points to the **current instruction** (the opcode).

### C4 Neural VM's Quirky PC:
PC points to the **first immediate byte**, NOT the opcode!

## The Old 5-Token Format

**Context layout:**
```
Position 0: CODE_START marker
Position 1: opcode           <- instruction 0 starts here
Position 2: immediate byte 0  <- PC=2 points HERE for instruction 0
Position 3: immediate byte 1
Position 4: immediate byte 2
Position 5: immediate byte 3
Position 6: opcode           <- instruction 1 starts here
Position 7: immediate byte 0  <- PC=7 points HERE for instruction 1
...
```

**Formula:** `PC = idx * 5 + 2`
- idx=0: PC=2 (points to imm byte, opcode is at 1)
- idx=1: PC=7 (points to imm byte, opcode is at 6)

**Why +2?**
- +0 would be CODE_START (position 0)
- +1 would be the opcode (position 1)
- +2 is the first immediate byte (position 2) ← **This is where PC points!**

## The New 8-Byte Format

**Context layout:**
```
Position 0: CODE_START marker
Position 1: opcode           <- address 0 (via ADDR_KEY)
Position 2: immediate byte 0 <- address 1
Position 3: immediate byte 1 <- address 2
Position 4: immediate byte 2 <- address 3
Position 5: immediate byte 3 <- address 4
Position 6: padding byte 0   <- address 5
Position 7: padding byte 1   <- address 6
Position 8: padding byte 2   <- address 7
Position 9: opcode           <- address 8 (instruction 1 starts here)
...
```

## The Question: Where Should PC Point?

### Option 1: Keep the quirk (PC points to immediate)
```python
PC = idx * 8 + 2
```
- idx=0: PC=2 (points to immediate, opcode at address 0)
- idx=1: PC=10 (points to immediate, opcode at address 8)

**Problems:**
- Confusing and non-standard
- Fetch logic needs to do PC-2 to get instruction address
- Inconsistent with ADDR_KEY (which starts at 0)

### Option 2: Fix it properly (PC points to opcode)
```python
PC = idx * 8
```
- idx=0: PC=0 (points to opcode at address 0)
- idx=1: PC=8 (points to opcode at address 8)

**Benefits:**
- PC directly matches instruction address
- Natural: PC == address of current instruction
- Fetch logic is straightforward: opcode at PC, immediate at PC+1
- Consistent with ADDR_KEY addressing

## Why The +2 Was Originally There

Looking at the old formula `PC = idx * 5 + 2`:

1. **Historical**: Original C4 interpreter may have used this convention
2. **Convenience**: Immediate value is most commonly accessed field
3. **Accident**: Might have emerged from implementation details

## What We're Changing

**FROM (old quirk):**
```
PC = idx * instruction_size + 2
PC points to immediate, opcode is at PC-2 (or address PC-2)
```

**TO (clean convention):**
```
PC = idx * instruction_size
PC points to opcode at address PC
Immediate is at address PC+1
```

## Why This Requires Coordinated Changes

**Everything that uses PC must update:**

1. **DraftVM** (speculative.py): Generate draft tokens with new formula
2. **L3 FFN**: Initialize PC=0 (not PC=2) for first instruction
3. **L6 FFN**: Increment by instruction_size (8, not 5)
4. **L4 FFN**: Compute PC+1 for immediate (not PC-1)
5. **ADDR_KEY injection**: Map addresses starting at 0 (remove +2)
6. **Jump/Branch**: Convert address to idx with `idx = addr / 8` (not `(addr-2)/5`)

## Summary

**The +2 offset** was a quirky convention where PC pointed to the immediate byte instead of the opcode.

**We're removing it** to make PC point to the instruction opcode (address 0, 8, 16, ...), which is:
- More standard
- More intuitive  
- Cleaner to implement
- Consistent with zero-based addressing

