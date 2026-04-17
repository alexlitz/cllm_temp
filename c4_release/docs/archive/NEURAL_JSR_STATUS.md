# Neural JSR Implementation Status

## Test Results

**Date**: 2026-04-11
**Status**: BROKEN - JSR not jumping to target

### Test Evidence

Simple JSR test program:
```
Bytecode:
  0: JSR 25     (call function at byte 25)
  1: EXIT
  5: IMM 42     (function at byte 25 = instr 5)
  6: EXIT
```

**Expected behavior**:
- Step 1: Execute JSR, jump to byte 25
- Step 2: Execute IMM 42, set AX=42
- Step 3: Execute EXIT with AX=42

**Actual behavior**:
- Step 1: PC=10 (wrong!), AX=0
- Step 2: PC=0 (looped back!), AX=208
- Exit code: 0 (failed)

**Conclusion**: JSR is NOT jumping to the target. PC is looping back to 0 instead of jumping to 25.

## Architecture Analysis

### Neural JSR Flow (from vm_step.py)

**L5 FFN First-Step Decode** (lines 3381-3391):
```python
# JSR opcode = 3 = 0x03
Condition: OPCODE_LO=3 AND OPCODE_HI=0 AND MARK_PC AND NOT HAS_SE
Threshold: 2.5
Output: TEMP[0] = 10.0 / S (≈ 5.0 after SwiGLU)
```

**L6 Attention Head 3 - JSR Relay** (lines 3682-3696):
```python
# On ALL steps (not just first), relay OP_JSR from AX to TEMP[0] at PC
Q: MARK_PC, NOT MARK_AX
K: MARK_AX
V: OP_JSR
O: TEMP[0] at PC marker
```

**L6 FFN - PC Override** (lines 7447-7481):
```python
# Cancel normal PC+5, write FETCH (jump target)
Condition: MARK_PC + TEMP[0] >= 4.0
Actions:
  1. Cancel OUTPUT_LO/HI (W_gate = -1.0)
  2. Write FETCH_LO/HI to OUTPUT
```

### Threshold Analysis

**First step (HAS_SE = 0)**:
```
L5 FFN activation = S*(opcode_lo + opcode_hi + mark_pc - has_se) - S*2.5
                  = 20*(1 + 1 + 1 - 0) - 50
                  = 60 - 50 = 10
SwiGLU output = silu(10) * gate * (10.0/S)
              ≈ 10 * 1.0 * 0.5
              = 5.0
TEMP[0] = 5.0

L6 FFN activation = S*(mark_pc + temp0) - S*T_jsr_pc
                  = 20*(1.0 + 5.0) - 20*4.0
                  = 120 - 80 = 40 ✅ SHOULD ACTIVATE
```

**Math checks out** - JSR should work!

## Possible Root Causes

### Hypothesis 1: TEMP[0] not being set
- L5 FFN decode might not be firing
- Check: opcode byte encoding in embeddings
- Check: HAS_SE on first step

### Hypothesis 2: L6 FFN not activating
- Threshold too high due to precision issues
- TEMP[0] leaking/being cleared
- MARK_PC not set correctly

### Hypothesis 3: PC override not working
- W_gate cancellation not effective
- FETCH values not correct
- OUTPUT routing broken

### Hypothesis 4: Initial PC wrong
- Program starts at wrong PC
- Byte vs instruction offset confusion
- Context prefix affecting offsets

## Next Steps

1. **Add detailed logging** to see TEMP[0] values at each step
2. **Check embeddings** - verify opcode bytes are encoded correctly
3. **Test simpler case** - JSR to instruction 1 (byte 5) instead of 5 (byte 25)
4. **Compare with handler** - run same program with handler to see expected values
5. **Check OP_JSR flag** - verify it's being set at AX marker for relay

## Quick Fix Attempts

### Option A: Lower threshold
Change `T_jsr_pc = 4.0` to `T_jsr_pc = 3.0` or `2.0`

Risk: May fire on false positives

### Option B: Increase TEMP[0] output
Change `W_down[BD.TEMP + 0, unit] = 10.0 / S` to `20.0 / S`

Risk: May cause other issues with TEMP overflow

### Option C: Re-enable handler temporarily
Uncomment JSR handler to verify basic program flow works

Status: INVESTIGATING - need more diagnostics before applying fixes
