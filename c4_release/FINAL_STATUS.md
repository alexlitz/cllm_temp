# Final Status: PC=idx*8+2 Implementation

## Date
2026-03-26

## Summary

Successfully changed the PC formula from `idx*5+2` to `idx*8+2` everywhere to align with the 8-byte instruction layout (1 opcode + 4 immediate + 3 padding).

## ✅ What's Working

1. **DraftVM PC formula**: Changed to `PC = idx * 8 + 2`
   - Initial PC = 2
   - After step 0: PC = 10 (was 7)

2. **Neural VM PC initial value**: Set to 2 (matching DraftVM)

3. **Neural VM PC increment**: Changed from +5 to +8
   - L3 FFN: PC increment for subsequent steps (when HAS_SE=1)
   - L6 FFN: PC increment for step 0 (when HAS_SE=0)
   - Carry logic updated for >=8

4. **L4 FFN fetch address computation**:
   - Opcode fetch: PC-2 written to EMBED_LO/HI
   - Immediate fetch: PC-1 written to TEMP
   - For PC=2: fetches from address 0 (opcode) and 1 (immediate) ✓

5. **L5 attention fetch**: Working correctly
   - Head 0: Fetches immediate from address 1 → FETCH_LO[10]=1, FETCH_HI[2]=1 ✓
   - Head 1: Fetches opcode from address 0 → OPCODE_BYTE_LO[1]=1 ✓

6. **L5 FFN opcode decode**: OP_IMM = 5.0 ✓

7. **L6 FFN IMM routing**: Working correctly
   - At AX marker: OUTPUT_LO[10]=4.05, OUTPUT_HI[2]=4.00 ✓

8. **Final PC output**: PC_B0 = 10 (matches DraftVM) ✓

## ✗ What's Not Working

**AX byte prediction**: pred=0, expected=42

- OUTPUT is correct at AX marker (OUTPUT_LO[10], OUTPUT_HI[2])
- But AX byte 0 position (context_len+6) has OUTPUT_LO ≈ 0
- The neural VM needs to propagate OUTPUT from markers to byte positions
- This propagation likely happens in layers L7-L15, but isn't working for standalone mode

## Root Cause

The neural VM architecture expects:
1. Markers compute OUTPUT at marker positions (L0-L6)
2. Later layers propagate OUTPUT to byte positions (L7+)
3. Output head predicts byte tokens from byte positions

For speculative decoding with initial step:
- Initial step provides previous values at byte positions
- Later layers can reference these when computing new values

For standalone mode (step 0):
- No previous step → byte positions start with zeros
- Propagation from markers to bytes may depend on having previous context

## Files Modified

### 1. `neural_vm/speculative.py`
Changed DraftVM PC formula:
- Line 76: `self.pc = 2  # PC = idx * 8 + 2`
- Line 110: `self.pc = self.idx * 8 + 2`
- Lines 120, 125, 129, 133, 148: `(x - 2) // 5` → `(x - 2) // 8`

### 2. `neural_vm/vm_step.py`

**L3 FFN** (lines 1588-1657):
- Initial PC: 0 → 2
- PC increment: +5 → +8
- Carry threshold: >=11 → >=8
- Comments updated

**L4 FFN** (lines 1700-1767):
- Completely rewritten to compute PC-2 and PC-1
- PC-2 → EMBED (for opcode fetch at address 0)
- PC-1 → TEMP (for immediate fetch at address 1)
- Fixed rotation: `(k-1)` → `(k+1)` for subtraction

**L6 FFN** (lines 2095-2134):
- PC increment for step 0: +5 → +8
- Carry threshold: >=11 → >=8

## Test Results

```
Context layout (with 8-byte instructions):
  Position 1: opcode (address 0)
  Position 2: immediate byte 0 (address 1)
  Position 3-5: immediate bytes 1-3 (addresses 2-4)
  Position 6-8: padding (addresses 5-7)
  Position 9: next opcode (address 8)

After L3: PC=2 at marker ✓
After L4: EMBED=0, TEMP=1 (fetch addresses) ✓
After L5: FETCH=[10,2] (byte 42 fetched) ✓, OP_IMM=5.0 ✓
After L6: OUTPUT=[10,2] at AX marker ✓
Final: PC=10 ✓, AX=0 ✗
```

## Next Steps

To complete standalone mode, need to:

1. **Investigate L7-L15**: Find which layers propagate OUTPUT from markers to bytes
2. **Check attention patterns**: See if there's marker→byte attention
3. **Add explicit propagation**: If missing, add FFN/attention to copy OUTPUT from markers to adjacent byte positions
4. **Test with initial step**: Verify if providing initial step fixes AX prediction
5. **Consider alternative**: Accept that neural VM requires DraftVM for execution, use only for scoring

## Recommendation

The neural VM is very close to working! The core computation (fetch, decode, route) is correct. The issue is purely about OUTPUT propagation from markers to byte prediction positions.

Option A: Add explicit OUTPUT propagation in L7+ (cleanest for standalone mode)
Option B: Always provide initial step (simpler workaround)
Option C: Use neural VM only with DraftVM (current production setup)

---

**Key Achievement**: Successfully aligned DraftVM and neural VM PC formulas. Fetch, decode, and routing all work correctly!
