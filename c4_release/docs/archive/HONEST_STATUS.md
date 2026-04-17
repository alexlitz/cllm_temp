# Honest Status Assessment

## ❌ TRUTH: The Core Neural VM Is NOT Fixed Yet

Looking at the actual code in neural_vm/vm_step.py:

### Line 1600-1604 (L3 FFN):
```python
# PC FIRST-STEP DEFAULT: when MARK_PC AND NOT HAS_SE, set PC=2
ffn.W_up[unit, BD.MARK_PC] = S
ffn.b_up[unit] = -S * 0.5
ffn.b_gate[unit] = 1.0
ffn.W_down[BD.OUTPUT_LO + 2, unit] = 2.0 / S  # PC=2 (lo nibble=2)
```

**This is STILL setting PC=2, not PC=0!**

### Current Behavior:
- DraftVM (speculative.py): PC = idx*8+2 → outputs PC=10 after step 0
- Neural VM (vm_step.py): PC = 2 (L3) + 8 (L6) → outputs PC=10 after step 0
- They match each other (10 == 10) ✓
- But BOTH are wrong relative to the target (should be 8) ✗

## What Actually Got Applied

### ✅ Applied Successfully:
1. ADDR_KEY injection (line 536): Addresses start at 0
2. L4 FFN: Computes PC+1 for immediate fetch
3. L5 comments: Updated to PC/PC+1
4. L6 PC increment: Adds +8

### ❌ NOT Applied (or Reverted):
1. L3 FFN: Still sets PC=2 (should be PC=0)
2. speculative.py: Still uses idx*8+2 (should be idx*8)

## Why Fetch Still Works

The fetch works because:
- ADDR_KEY correctly maps position 1 → address 0, position 2 → address 1
- L4 computes fetch addresses relative to PC (PC for opcode, PC+1 for imm)
- Even though PC=2, the fetch logic does: fetch from address (PC mod 8)
  - Actually no, that's not right either...

Wait, let me think about this more carefully. If PC=2 and we're trying to fetch the opcode:
- L4 should put PC=2 in EMBED
- L5 should query ADDR_KEY for address 2
- But address 2 has the immediate byte (42), not the opcode!

Unless... let me check what's actually happening with the fetch.
