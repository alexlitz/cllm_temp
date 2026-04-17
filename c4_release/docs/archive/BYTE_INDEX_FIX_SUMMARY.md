# BYTE_INDEX Workaround Implementation - Summary

**Date:** 2026-04-10
**Status:** PARTIAL SUCCESS - Addr heads fixed, val heads need more work

## Problem

BYTE_INDEX_1/2/3 flags are not being set by Layer 1 FFN, causing L14 MEM generation to fail. Symptoms:
- JSR MEM addresses: byte 3 copied byte 0 (e.g., `0xf80001f8` instead of `0x000001f8`)
- All memory operations with multi-byte addresses broken

## Root Cause Discovery

Through extensive debugging, found that:
1. ✅ **L1 attention DOES compute L1H0/1/2 correctly** for all IS_MARK markers including PC
2. ✅ **Hop-count thresholds work perfectly** - L1H2 - L1H1 fires at d∈(1.5, 2.5], etc.
3. ❌ **BYTE_INDEX FFN units don't activate** despite correct inputs (separate FFN bug)

**Key Insight:** Hop-count matching CAN replace BYTE_INDEX for byte position detection!

## Solution Implemented

### Part 1: L14 Addr Heads (Heads 0-3) - ✅ FIXED

**File:** `neural_vm/vm_step.py:5933-5963`

**Change 1: Dim 0 - Hop-count byte position matching**
```python
# Old (broken):
attn.W_k[base, BD.BYTE_INDEX_1] = L  # Doesn't work

# New (working):
if h == 1:
    attn.W_k[base, BD.L1H2 + PC_I] = L
    attn.W_k[base, BD.L1H1 + PC_I] = -L
    attn.W_k[base, BD.L1H2 + AX_I] = L
    attn.W_k[base, BD.L1H1 + AX_I] = -L
    attn.W_k[base, BD.L1H2 + SP_I] = L
    attn.W_k[base, BD.L1H1 + SP_I] = -L
# Similarly for h=2,3 using H0-L1H2 and H1-H0
```

**Change 2: Dim 3 - JSR PC source bonus**
```python
# New dimension to prefer PC bytes when JSR active
attn.W_q[base + 3, BD.CMP + 4] = L  # CMP[4] = OP_JSR relay
attn.W_k[base + 3, BD.H1 + PC_I] = L  # Match PC bytes
attn.W_k[base + 3, BD.H1 + AX_I] = -L
attn.W_k[base + 3, BD.H1 + SP_I] = -L
```

**Result:** MEM addresses now CORRECT!
```
Before: [261, 248, 1, 0, 248, ...] → 0xf80001f8 ✗
After:  [261, 248, 1, 0, 0, ...]   → 0x000001f8 ✓
```

### Part 2: L14 Val Heads (Heads 4-7) - ⚠️ PARTIAL

**File:** `neural_vm/vm_step.py:6048-6115`

**Changes made:**
1. Dim 0: Added hop-count matching (same as addr heads)
2. Dim 2: JSR PC source bonus (separated from ENT)
3. Dim 3: ENT STACK0 source bonus (separated from JSR)

**Current Status:** Addresses work, but values still corrupted
```
Expected: [..., 0, 0, 0, 5] → return address 0x00000005
Actual:   [..., 0, 0, 0, 17] → Token.STEP_END leaking
```

**Issue:** Val head 7 (byte 3) reading from wrong position. Needs further investigation of:
- Threshold attention firing positions (Q side threshold differences may be off by 1)
- Proximity to STEP_END causing score leakage
- Possible additional gating needed

## Test Results

### JSR Addr Generation
```bash
$ python test_byte_index_fix.py
✓ Addresses: FIXED (byte 3 = 0x00)
✗ Values: Still broken (byte 3 = 0x11 or 0xf1)
```

### Hop Count Verification
```bash
$ python debug_pc_hop_counts.py
✓ L1H0/1/2[PC] all correct at PC byte positions
✓ Threshold differences work: 0.984 at target positions
```

## Architecture Summary

### Hop-Count Thresholds (Layer 0/1)
- **H0** (3.5), **H1** (4.5), **H2** (5.5), **H3** (9.5), ...
- **L1H0** (0.5), **L1H1** (1.5), **L1H2** (2.5), **L1H4** (6.5)

### Byte Position Mapping
| Byte | Distance | Threshold Match |
|------|----------|-----------------|
| 0    | 1        | L1H1 - L1H0     |
| 1    | 2        | L1H2 - L1H1     |
| 2    | 3        | H0 - L1H2       |
| 3    | 4        | H1 - H0         |

### Operation Flag Relay (L13 Head 6)
- **CMP[0]** = OP_PSH
- **CMP[1]** = OP_ADJ
- **CMP[2]** = OP_ENT
- **CMP[3]** = POP group (binary ops)
- **CMP[4]** = OP_JSR ← Used in fix!

## Files Modified

- `neural_vm/vm_step.py`
  - Lines 5933-6008: L14 addr heads Dim 0 + Dim 3
  - Lines 6048-6115: L14 val heads Dim 0/2/3
- `test_byte_index_fix.py` (created)
- `debug_pc_hop_counts.py` (created)
- `BYTE_INDEX_BUG_FINDINGS.md` (reference doc)

## Next Steps

1. **Debug val head 7 position selection** (high priority)
   - Verify Q threshold differences fire at correct MEM byte positions
   - Check if STEP_END proximity causes score leakage
   - May need additional suppression gates

2. **Test addr heads with PSH/SI/SC** (medium priority)
   - Verify SP source bonus works (PSH)
   - Verify STACK0 source bonus works (SI/SC)

3. **Test JSR → LEV round trip** (blocked by val heads)
   - Need correct MEM values for JSR to work
   - Then LEV can read return address from memory

4. **Remove LEV handler** (blocked by JSR/LEV round trip)

5. **Achieve 100% neural VM!** (ultimate goal)

## Lessons Learned

1. **Trust hop counts over BYTE_INDEX** - Hop counts are computed correctly by threshold attention, while BYTE_INDEX FFN has mysterious activation bug

2. **Operation flags need relay** - OP_* flags at AX marker must be relayed to MEM marker via CMP dimensions

3. **Separate JSR and ENT bonuses** - Combining them causes both PC and STACK0 to get bonuses simultaneously

4. **Debug iteratively** - Breaking down the problem (hop counts → addr heads → val heads) makes debugging tractable

5. **Comments can be wrong** - Original val_pos comments appeared off by 1 in distance calculations

## Acknowledgments

This fix represents ~6 hours of debugging across:
- 10+ debug scripts
- 50+ test runs
- 2 failed approaches (STACK0/PC H1 flags, BYTE_INDEX FFN debugging)
- 1 successful approach (hop-count threshold matching)

The breakthrough came from verifying that L1H0/1/2 ARE correctly computed, which proved hop-count matching is viable!
