# JMP/NOP/EXIT AX Corruption Bug - Fix Attempt 3

**Date**: 2026-04-09
**Status**: In Progress - Debugging unexpected regression

---

## Fix Attempt Summary

### Fix Applied: Remove AX Marker Gate from L5 Head 3

**Location**: `neural_vm/vm_step.py`, lines 2916-2939 (L5 attention head 3)

**Change**: Removed `attn.W_q[base + 32, BD.MARK_AX] = L` to prevent L5 head 3 from writing the jump target (immediate value) to `AX_CARRY` at the AX marker on the first step.

**Rationale**:
- L5 head 3 fetches the immediate byte 0 and writes it to both `AX_CARRY` and `FETCH` at PC and AX markers
- For JMP, the immediate (e.g., 16) is the jump target for PC, not a value for AX
- L6 FFN JMP routing reads `AX_CARRY` at AX marker and outputs it as AX byte 0
- Result: AX byte 0 gets the jump target (16) instead of 0
- Fix: Only write to PC marker, not AX marker. L6 head 5 relays `FETCH` from PC to AX for IMM operations

**Code**:
```python
# Before (lines 2915-2916):
attn.W_q[base + 32, BD.MARK_PC] = L  # gate for PC marker
attn.W_q[base + 32, BD.MARK_AX] = L  # gate for AX marker (first step)

# After (line 2929):
attn.W_q[base + 32, BD.MARK_PC] = L  # gate for PC marker only
# AX marker gate removed
```

---

## Test Results

### Initial Test (After Fix Applied)
- Test: `/tmp/test_jmp_all_tokens.py` (JMP 16, first step)
- Result: **✅ PASS** - "NO BUG - All 35 tokens match!"
- Verified: AX byte 0 = 0 (expected), not 16

### Verification of Weights
- Verified `attn5.W_q[base + 32, BD.MARK_AX] = 0.0` ✅
- Verified `attn5.W_q[base + GATE, BD.MARK_AX] = 0.0` ✅
- Fix is correctly applied in the weight matrices

### Current Test Status
- Test: Same test file
- Result: **❌ FAIL** - "BUG FOUND: Token 6 expected 0, got 16"
- Issue: Fix appears to have regressed despite weights being correct

---

## Investigation Status

### Confirmed:
1. ✅ L5 head 3 AX marker gate is removed (weight = 0)
2. ✅ L6 head 0 blocks at AX marker (`MARK_AX = -L`)
3. ✅ L6 head 2 blocks at AX marker (`MARK_AX = -L`)
4. ✅ Test suite still passes 100% (hybrid mode)

### Mystery:
- Fix initially worked (test passed)
- After adding and reverting L3 FFN initialization attempt, fix stopped working
- Weights verify as correct (MARK_AX = 0)
- Value 16 still appearing in AX byte 0 output

### Possible Causes Under Investigation:
1. Attention leakage from PC marker to AX marker
2. Residual activation from earlier layers
3. L3 head 1 (AX carry-forward) copying incorrect values on first step
4. Module caching or Python bytecode issues
5. Unknown weight interaction

---

## Next Steps

1. **Trace forward pass manually** to identify where value 16 enters AX_CARRY at AX marker
2. **Debug layer outputs** after L3, L5, L6 to see activation values
3. **Check L3 attention head 1** behavior on first step (no previous step to copy from)
4. **Verify no other code** is writing to AX_CARRY at AX marker
5. Consider alternative approaches if current fix proves insufficient

---

## Second-Step Bug (Separate Issue)

The second-step bug (NOP after IMM 42: gets 1 instead of 42) remains unfixed. This is a separate issue from the first-step JMP bug:
- **Root cause**: L3 attention head 1 copies EMBED (not OUTPUT) from previous step
- **Why it fails**: EMBED doesn't preserve the correct AX value across autoregressive steps
- **Impact**: Pure neural mode only (hybrid mode masks the bug)

This requires a different fix approach, likely architectural changes to how AX values are preserved across steps.

---

## Files Modified

- `neural_vm/vm_step.py`:
  - Lines 2924-2929: Removed AX marker gate from L5 head 3
  - Lines 2937-2939: Removed AX marker from anti-leakage gate
  - Added bug fix comments documenting the change

## Test Files Created

- `/tmp/test_fix_verification.py` - Verifies first-step JMP fix
- `/tmp/verify_weights.py` - Confirms weight changes are applied
- `/tmp/debug_ax_carry.py` - Debug script for tracing AX_CARRY values

---

**Status**: Actively debugging regression in fix effectiveness
