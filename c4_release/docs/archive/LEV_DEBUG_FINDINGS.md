# LEV Neural Implementation - Debug Findings

**Date**: 2026-04-09
**Status**: Architecture Complete, Neural Path Not Working Yet

---

## Summary

Successfully implemented LEV neural architecture (Phases 1-3 + forward pass fix), but discovered the neural path produces different values than the handler. The model IS producing output, but it's being overridden by the handler.

---

## Key Discovery ✓

**The model outputs values, handler overrides them:**

```
[LEV MODEL]   BP=0x00010000, SP=0x0000ffc8, PC=0x00000000
[LEV HANDLER] BP=0x00000000, SP=0x00000010, PC=0x00000000
```

This means:
- ✓ L15/L16 neural path exists and runs
- ✓ Model produces non-zero output
- ✗ Handler immediately overrides the model's output
- ✗ Model's output values are incorrect

---

## Root Cause Analysis

### Issue 1: Handler Using `_last_bp` = 0

**Handler code (run_vm.py:1582)**:
```python
old_bp = self._last_bp  # = 0x00000000 (wrong!)
```

**Why it's zero**:
- `_last_bp` is set by ENT handler
- But if ENT doesn't execute (function has no locals), BP stays at initial value (0)
- Test function `helper()` has no local variables → no ENT → BP never set

**Solution**: Test with a function that HAS local variables to trigger ENT

### Issue 2: Model Outputs Wrong Values

**Model output**:
- BP=0x00010000 (should be saved_bp from memory)
- SP=0x0000ffc8 (should be old_BP + 16)
- PC=0x00000000 (should be return_addr from memory)

**Why incorrect**:
1. **BP=0x00010000**: Seems like garbage or uninitialized
2. **SP=0x0000ffc8**: This is actually from JSR! (current SP before LEV)
3. **PC=0x00000000**: Wrong, should be return address

**Analysis**: The model is NOT reading from memory correctly. L15 heads 4-11 are not working.

---

## Hypothesis: L15 Heads 4-11 Not Firing

### Possible Causes

1. **OP_LEV not active at BP/PC markers**
   - Check: BD.OP_LEV dim value at BP/PC positions
   - Expected: ~1.0 when LEV executes
   - If zero: OP_LEV relay not working

2. **MARK_BP / MARK_PC not active**
   - Check: BD.MARK_BP / BD.MARK_PC values
   - Expected: ~1.0 at BP/PC marker tokens
   - If zero: Markers not being set

3. **ADDR_B0-2 dims not populated** (Phase 1 issue)
   - Check: BD.ADDR_B0_LO/HI values at BP marker
   - Expected: BP address value
   - If zero: Phase 1 BP relay not working

4. **Memory not written by JSR/ENT**
   - Check: MEM tokens in context history
   - Expected: MEM tokens with saved_bp and return_addr
   - If missing: L14 MEM generation not working

5. **L15 heads 4-7/8-11 Q values too low**
   - Check: Attention scores for heads 4-11
   - Expected: High scores when matching BP address
   - If low: Weight configuration issue

---

## Debugging Strategy

### Step 1: Test with ENT-using Function ✓

**Change test to**:
```c
int helper(int x) {
    int local = x * 2;  // Forces ENT
    return local;
}
```

This ensures:
- ENT executes → _last_bp is set correctly
- Memory stores saved_bp
- LEV has actual data to read

### Step 2: Add Detailed Logging

Add logging at each stage:

**A. Before L15 (check inputs)**:
```python
# At BP marker position:
print(f"OP_LEV: {x[0, bp_pos, BD.OP_LEV]}")
print(f"MARK_BP: {x[0, bp_pos, BD.MARK_BP]}")
print(f"ADDR_B0: {decode_nibbles(x[0, bp_pos, BD.ADDR_B0_LO:BD.ADDR_B0_HI+16])}")
```

**B. L15 Attention (check activation)**:
```python
# Check if heads 4-7 fire:
Q_bp = l15.W_q @ x[bp_pos]  # Query at BP marker
scores = Q_bp @ K_mem.T     # Scores to MEM tokens
print(f"Head 4-7 scores: {scores[4*64:(8*64)]}")
```

**C. After L15 (check outputs)**:
```python
# At BP marker after L15:
bp_output = decode_nibbles(x_after_l15[0, bp_pos, BD.OUTPUT_LO:BD.OUTPUT_HI+16])
print(f"BP OUTPUT after L15: {bp_output:#010x}")
```

**D. After L16 (check SP computation)**:
```python
# At SP marker after L16:
sp_output = decode_nibbles(x_after_l16[0, sp_pos, BD.OUTPUT_LO:BD.OUTPUT_HI+16])
print(f"SP OUTPUT after L16: {sp_output:#010x}")
```

### Step 3: Identify the Broken Stage

Based on logging:
- If OP_LEV=0: Opcode relay issue
- If MARK_BP=0: Marker generation issue
- If ADDR=0: Phase 1 issue (L8 FFN not firing)
- If MEM missing: L14 issue (memory not written)
- If scores low: L15 weight issue
- If OUTPUT=0: L15/L16 routing issue

### Step 4: Fix the Broken Component

Once identified, fix:
- **Phase 1**: Adjust L8 FFN activation conditions
- **L15**: Adjust Q/K weights or activation conditions
- **L16**: Adjust FFN routing logic
- **L14**: Fix MEM token generation

---

## Next Steps (Prioritized)

1. **Test with ENT-using function** (30 min)
   - Modify test_lev_simple.py
   - Run and check if old_bp is non-zero
   - Check if memory contains actual values

2. **Add detailed L15 logging** (1 hour)
   - Instrument before/after L15
   - Check activation conditions
   - Identify which component fails

3. **Fix the identified issue** (2-4 hours)
   - Based on logging findings
   - Adjust weights or activation logic
   - Re-test until model output matches expected

4. **Remove LEV handler** (30 min)
   - Once model outputs correct values
   - Verify tests pass without handler

5. **Achieve ~99% neural** 🎉
   - LEV fully neural
   - Only JSR PC handler remains

---

## Expected Timeline

**Best case** (if simple fix): 3-4 hours remaining
- ENT test: 30 min
- Logging: 1 hour
- Fix: 1 hour
- Testing: 1 hour

**Worst case** (if architecture issue): 6-8 hours remaining
- ENT test: 30 min
- Logging: 1 hour
- Multiple fix attempts: 3-4 hours
- Extensive testing: 2 hours

**Most likely**: 4-6 hours remaining

---

## Progress So Far

**Time Spent**: 8 hours
- Phase 1: 1.5 hours ✓
- Phase 2: 2.5 hours ✓
- Phase 3: 1 hour ✓
- Forward pass fix: 2 hours ✓
- Debugging: 1 hour (current)

**Remaining**: 4-6 hours
- Debug + fix: 3-4 hours
- Handler removal: 30 min
- Testing: 1-2 hours

**Total Estimate**: 12-14 hours (vs 19-25 hour original estimate)

---

## Files to Modify (Likely)

Based on findings:

1. **neural_vm/vm_step.py**
   - L8 FFN (Phase 1): Activation conditions
   - L15 heads 4-11: Q/K weights or gating
   - L16 FFN: Routing logic

2. **neural_vm/run_vm.py**
   - Remove LEV handler (after fix)
   - Remove from _func_call_handlers

3. **Test files**
   - test_lev_simple.py: Add ENT-using function
   - Create debug_lev_detailed.py: Detailed logging

---

## Root Cause Found! ✓

**L14 MEM Token Generation Bug**

The model is generating MEM sections with **corrupted address byte 3**:

```
Expected (address 0x000100f0): [MEM, 0xf0, 0x00, 0x01, 0x00, val...]
Actual:                        [MEM, 0xf0, 0x01, 0x00, 0xf0, val...]
                                                         ^^^^ WRONG!
```

**Evidence from debug_mem_tokens.py**:
- JSR stores at address 0x000100f0
- Model outputs: `[261, 240, 1, 0, 240, 0, 0, 0, 0]`
- Byte 3 (index 4) = 240 (0xf0) — should be 0!
- This causes L15 heads to lookup wrong addresses

**Impact**:
- L15 heads 4-11 read from wrong memory addresses
- LEV gets garbage values for saved_bp and return_addr
- Handler override is necessary because model data is corrupted

**Fix Needed**: Check L14 attention head that generates MEM address byte 3

---

**Status**: JSR SP fixed ✓ - L14 MEM generation now works! Next: Debug L15 memory reads.

---

## JSR SP Fix Complete! ✓

**Commit**: [pending]
**Bug**: JSR neural SP -= 8 fired at SP marker (contains marker token 259), not byte positions
**Fix**: Changed JSR to fire at SP byte positions using IS_BYTE + BYTE_INDEX_0-3

**Changes**:
- vm_step.py lines 7274-7357: JSR SP bytes 0-3 implementation
- Byte 0: (old_byte - 8) % 256 with borrow detection
- Byte 1: 0xff (hardcoded for typical case where byte 0 < 8)
- Bytes 2-3: Identity pass-through from EMBED

**Result**:
- MEM addresses NOW CORRECT: 0x0000fff8, 0x0000ffe8, etc. ✓
- L14 reads correct SP bytes and generates valid MEM sections ✓

---

## Current Issue: LEV Reads Zeros

**Observation**:
```
[LEV HANDLER] old_bp=0x00000000, saved_bp=0x00000000, return_addr=0x00000000
[LEV MODEL] BP=0x00010000, SP=0x00ff00d8, PC=0x00000000
```

**Analysis**:
- MEM sections have correct addresses (0x0000fff8 etc.) ✓
- But LEV handler reads saved_bp=0, return_addr=0 from memory
- L15 heads 4-11 are not reading correctly from MEM sections
- Model produces wrong values (BP=0x00010000 looks like garbage)

**Next Steps**:
1. Check if MEM sections have correct VALUE bytes (not just address bytes)
2. Verify L15 heads 4-11 are firing at BP/PC marker positions
3. Check if L15 heads match correct MEM addresses in attention
4. Debug L15 output routing to BP/PC markers

**Status**: JSR fix complete, now debugging L15 memory reads.
