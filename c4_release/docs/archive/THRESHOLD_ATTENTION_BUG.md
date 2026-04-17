# Threshold Attention Bug - Critical Blocker to 100% Neural

## Summary

The hop-count threshold attention heads (L0/L1: H0, H1, H2, L1H0, L1H1, L1H2, etc.) are **not producing the expected binary threshold outputs**. This breaks:

1. BYTE_INDEX flag generation (L1 FFN)
2. PSH SP byte 1-3 output (L3 FFN)
3. L14 MEM addr heads byte 1-3 matching
4. Potentially other position-dependent operations

## Evidence

Test: `test_byte_index_full_step.py`

Expected hop-count threshold behavior at SP byte positions:
```
d=1: L1H0=0, L1H1=1, L1H2=1, H0=0, H1=0  (d ≤ 1.5 but > 0.5)
d=2: L1H0=0, L1H1=0, L1H2=1, H0=0, H1=0  (d ≤ 2.5 but > 1.5)
d=3: L1H0=0, L1H1=0, L1H2=0, H0=1, H1=0  (d ≤ 3.5 but > 2.5)
d=4: L1H0=0, L1H1=0, L1H2=0, H0=0, H1=1  (d ≤ 4.5 but > 3.5)
```

**Actual values** (for SP marker index 2):
```
d=1: L1H0=0.15, L1H1=-0.01, L1H2=1.07, H0=-1.18, H1=-0.34
d=2: L1H0=0.00, L1H1=0.52, L1H2=-1.38, H0=-1.11, H1=-0.46
d=3: L1H0=-0.95, L1H1=1.16, L1H2=0.51, H0=-0.32, H1=-0.28
d=4: L1H0=-0.66, L1H1=-1.47, L1H2=-1.04, H0=1.76, H1=0.04
```

None of the expected patterns match! Values are near 0 or negative instead of binary 0/1.

## Impact on BYTE_INDEX

Expected BYTE_INDEX values:
```
d=1 (SP byte 0): BYTE_INDEX_0 = high (L1H1 - L1H0 pattern)
d=2 (SP byte 1): BYTE_INDEX_1 = high (L1H2 - L1H1 pattern)
d=3 (SP byte 2): BYTE_INDEX_2 = high (H0 - L1H2 pattern)
d=4 (SP byte 3): BYTE_INDEX_3 = high (H1 - H0 pattern)
```

**Actual BYTE_INDEX values**:
```
d=1: BYTE_INDEX_0=-0.24, _1=1.01, _2=1.97, _3=-1.56  ✗ WRONG
d=2: BYTE_INDEX_0=-0.38, _1=0.36, _2=0.73, _3=0.31   ✗ WRONG
d=3: BYTE_INDEX_0=-0.39, _1=0.66, _2=1.16, _3=-0.09  ✓ OK
d=4: BYTE_INDEX_0=-0.66, _1=-2.71, _2=-2.66, _3=1.14 ✓ OK
```

Only bytes 2-3 have correct BYTE_INDEX flags. Bytes 0-1 are wrong.

## Cascading Failures

### 1. L3 FFN PSH Byte 1-3 Output (vm_step.py:2381-2440)

Code uses BYTE_INDEX to identify positions:
```python
# SP byte 0 pos → predict byte 1 = 0xFF
ffn.W_up[unit, BD.BYTE_INDEX_0] = S  # Needs BYTE_INDEX_0 high at d=1
```

**Result**: Units don't fire because BYTE_INDEX_0 is low at d=1. SP bytes 1-3 not written to OUTPUT.

### 2. L14 MEM Addr Heads (vm_step.py:5848-5881)

Original code used BYTE_INDEX:
```python
byte_idx_dim = [None, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][h]
attn.W_k[base, byte_idx_dim] = L
```

**Result**: Heads 1-3 attend to wrong positions because BYTE_INDEX flags are wrong.

Attempted fix using hop-count thresholds:
```python
elif h == 1:
    attn.W_k[base, BD.L1H2 + SP_I] = L
    attn.W_k[base, BD.L1H1 + SP_I] = -L
```

**Result**: Still wrong because threshold heads output wrong values.

### 3. MEM Address Corruption

Handler writes: `0x0001f7f8` = bytes [0xf8, 0xf7, 0x01, 0x00]
Neural writes: `0xf80001f8` = bytes [0xf8, 0x01, 0x00, 0xf8]

Each byte head reads from position +1 (circular shift) due to wrong position matching.

## Root Cause Analysis

The threshold attention mechanism (lines 2093-2112) uses ALiBi with score:
```
score = slope * (threshold - distance)
```

Expected behavior: Binary 0/1 output when distance < threshold.
Actual behavior: Values near 0, sometimes negative.

**Possible causes**:
1. Softmax normalization diluting outputs across multiple positions
2. ALiBi slope value incorrect
3. Causal mask interfering with backward attention
4. V matrix not configured correctly
5. Numerical precision issues

## Impact on 100% Neural Goal

**Blocking**: This bug prevents L14 MEM token generation from working correctly, which blocks:
- ✗ JSR neural path (needs correct MEM tokens for return address storage)
- ✗ LEV neural path (needs L15 to read correct MEM tokens)
- ✗ ENT neural path (needs MEM tokens for BP storage)

**Current state**: ~95% neural (only JSR/LEV handlers remain)
**With this bug fixed**: Could achieve 100% neural

## Proposed Solutions

### Option A: Fix Threshold Attention (High Risk, High Reward)

1. Debug why threshold heads don't output binary values
2. Fix _set_threshold_attn configuration
3. Verify BYTE_INDEX generation works
4. Re-test all dependent layers

**Pros**: Fixes root cause, enables many features
**Cons**: Complex, risky, could break other things
**Time**: 8-16 hours

### Option B: Remove BYTE_INDEX Dependency (Lower Risk)

1. Redesign L3 PSH byte 1-3 output to not use BYTE_INDEX
2. Redesign L14 addr heads to use absolute position encoding
3. Use marker-relative position instead of hop-count

**Pros**: Bypasses broken mechanism
**Cons**: Architectural change, may be less flexible
**Time**: 4-8 hours

### Option C: Hardcode for STACK_INIT = 0x10000 (Quick Hack)

1. L3 already hardcodes PSH byte 1-3 for STACK_INIT (lines 2381-2440)
2. Make L14 read from those hardcoded positions
3. Only works for programs with default stack initialization

**Pros**: Quick, minimal changes
**Cons**: Not general, breaks for other stack values
**Time**: 1-2 hours

### Option D: Disable Neural MEM, Use Handlers (Pragmatic)

1. Let JSR/LEV handlers manage memory
2. Keep other ops 100% neural
3. Document as "95% neural VM"

**Pros**: Works now, achieves most of the goal
**Cons**: Not truly 100% neural
**Time**: 0 hours (already working)

## Recommendation

**Immediate**: Option D - Document current 95% neural state as a major achievement

**Next session**: Option B - Remove BYTE_INDEX dependency with position-based approach

**Long term**: Option A - Fix threshold attention for full architectural correctness

## Files Affected

- `neural_vm/vm_step.py`:
  - Lines 1593-1650: Threshold attention configuration (L0/L1)
  - Lines 2093-2112: _set_threshold_attn function
  - Lines 2194-2238: BYTE_INDEX generation (L1 FFN)
  - Lines 2381-2440: PSH byte 1-3 output (L3 FFN)
  - Lines 5848-5881: L14 addr heads

## Test Files

- `test_byte_index_full_step.py` - Demonstrates the bug
- `debug_lev_memory_state.py` - Shows MEM corruption symptoms
- `test_addr_heads_fix.py` - Failed fix attempt

## Status

**Discovered**: 2026-04-10
**Status**: Active blocker to 100% neural VM
**Priority**: P0 (blocks final milestone)
