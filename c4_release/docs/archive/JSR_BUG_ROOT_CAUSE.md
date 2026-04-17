# JSR Neural Bug - ROOT CAUSE IDENTIFIED

## Date: 2026-04-12

## Summary

**Neural JSR is broken because L5 attention head 2 fetches the opcode from the WRONG address.**

## The Bug

### Context Structure
```
Address 0: JSR opcode (3)
Address 1: Immediate byte 0 (25)
Address 2: Immediate byte 1 (0)    <- PC_OFFSET
Address 3-7: Padding/immediate (0)
```

### What Happens
1. Initial PC = `PC_OFFSET = 2` (legacy addressing: PC points to immediate byte)
2. L5 head 2 tries to fetch opcode from address `PC_OFFSET` (address 2)
3. At address 2, there's 0 (padding/immediate byte), NOT the JSR opcode!
4. OPCODE_BYTE_LO/HI get set to 0, not JSR nibbles (3, 0)
5. L5 FFN JSR decode doesn't fire (needs OPCODE_BYTE_LO[3]=1, OPCODE_BYTE_HI[0]=1)
6. TEMP[0] never gets set to ~5.0
7. L6 FFN PC override doesn't fire
8. PC advances normally to PC+5=7 instead of jumping to 25

## The Fix

**vm_step.py line ~3058**: Change L5 head 2 to fetch from address `PC_OFFSET - 2` instead of `PC_OFFSET`

### Current Code (WRONG):
```python
# Head 2: fetch opcode for first-step (PC marker → address PC_OFFSET)
base = 2 * HD
attn.W_q[base, BD.MARK_PC] = L
attn.W_q[base, BD.HAS_SE] = -L
# Q: address PC_OFFSET (e.g., 2)
attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble = 2
attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble = 0
```

This queries for ADDR_KEY matching address 2.

### Fixed Code (CORRECT):
```python
# Head 2: fetch opcode for first-step (PC marker → address PC_OFFSET - 2)
# PC_OFFSET points to immediate byte, opcode is 2 bytes before
OPCODE_OFFSET = PC_OFFSET - 2  # e.g., 2 - 2 = 0
base = 2 * HD
attn.W_q[base, BD.MARK_PC] = L
attn.W_q[base, BD.HAS_SE] = -L
# Q: address OPCODE_OFFSET (e.g., 0)
attn.W_q[base + (OPCODE_OFFSET & 0xF), BD.CONST] = L  # lo nibble = 0
attn.W_q[base + 16 + ((OPCODE_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble = 0
```

This queries for ADDR_KEY matching address 0, where the JSR opcode actually is!

## Verification

After fix:
1. L5 head 2 fetches from address 0
2. Gets JSR opcode byte (3)
3. OPCODE_BYTE_LO[3]=1, OPCODE_BYTE_HI[0]=1
4. L5 FFN JSR decode fires
5. TEMP[0] gets set to ~5.0
6. L6 FFN PC override fires
7. PC jumps to 25 ✅

## Why This Wasn't Caught Before

The neural JSR implementation was written assuming the model fetches from PC directly. But the legacy addressing scheme has PC pointing to immediate bytes, not opcodes.

This affects ALL first-step opcode fetches, not just JSR:
- IMM first-step decode
- LEA first-step decode
- EXIT first-step decode
- JMP first-step decode
- All arithmetic/bitwise first-step decodes

All of these are broken for the same reason!

## Impact

**Breaking change**: This fix will likely break existing programs UNLESS they were also broken.

Actually, this suggests the model might not be working correctly for ANY operations on the first step. Let me check if IMM works...

Actually, our test `test_neural_no_functions.py` showed that `IMM 42; EXIT` DID work and returned 42. So either:
1. IMM doesn't use L5 head 2 (uses a different path)
2. OR the fix is more nuanced

Let me re-check the IMM path...
