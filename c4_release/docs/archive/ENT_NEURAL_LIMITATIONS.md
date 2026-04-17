# ENT Neural Implementation: Limitations and Status

## Summary

ENT (enter stack frame) neural implementation is **partially functional** but has fundamental reliability issues due to soft ALU encoding. The handler should remain in place.

## What ENT Does

```
ENT imm:
  mem[SP-8] = BP      # Save old BP to STACK0
  BP = SP - 8         # New BP points to saved BP
  SP = SP - (8 + imm) # Allocate local variables
```

## Neural Components (Working)

1. **STACK0 = old_BP** - L6 FFN units 978-1009 ✅
2. **BP = SP - 8** - L6 FFN units 1010-1041 ✅
3. **AX passthrough** - L6 FFN units 1042-1073 ✅
4. **MEM token generation** - L14 attention ✅

## Neural Component (Unreliable)

**SP = SP - (8 + imm)** - L8/L9 FFN ENT units ⚠️

### Root Cause: Soft ALU Encoding

The ALU dimensions (ALU_LO, ALU_HI) receive values from L7 attention head 1, which gathers SP OUTPUT into ALU at the AX marker. Due to attention averaging, the values are not crisp one-hot:

```
Example for SP byte0 = 0xf8 (lo=8, hi=15):
  ALU_LO[8] = 1.522  (correct)
  ALU_LO[0] = 1.0    (spurious - from padding bytes)

  ALU_HI[15] = 1.522 (correct)
  ALU_HI[0] = 1.0    (spurious)
```

### Why This Breaks ENT

L8/L9 FFN units use threshold-based activation:
```python
activation = silu(S*60*MARK_AX + S*50*ALU_HI[sp_hi] + S*FETCH_HI[imm_hi] + ... - S*T)
```

With soft encoding, multiple units for different `sp_hi` values fire:
- Unit for sp_hi=15, imm_hi=0: activation = 57.2
- Unit for sp_hi=0, imm_hi=0: activation = 31.1

The correct unit (sp_hi=15) wins, BUT for imm=0 where the expected result is hi=15:
- hi=15 units get ~57 activation
- hi=0 units get ~31 activation

Since hi=0 units exist for sp_hi=0 (spurious) AND sp_hi=15 with different imm values, the accumulated activation can cause hi=0 to incorrectly win.

### Test Results

| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| ENT imm=24 | SP diff=32 | SP diff=32 | ✅ PASS |
| ENT imm=0 | SP diff=8, hi=15 | hi=0 wins | ❌ FAIL |
| ENT imm=8 | SP diff=16 | Not tested | ? |

## Attempted Fixes

1. **Increased ALU_HI weight to 50** - Helps differentiate correct sp_hi but doesn't eliminate spurious activations
2. **Added mutual suppression** - Prevents no-borrow/borrow overlap but doesn't fix soft encoding
3. **Increased threshold** - Filters some noise but reduces correct activations too

## Why Handler Should Remain

1. **Reliability**: Handler guarantees correct SP computation for all imm values
2. **Performance**: Handler is fast (simple subtraction)
3. **Correctness**: ENT is critical for function calls - wrong SP breaks everything

## Potential Solutions (Not Implemented)

1. **Hard one-hot encoding**: Modify L7 to produce crisp one-hot ALU values
   - Would require architectural changes to attention mechanism
   - High complexity, uncertain benefit

2. **Winner-take-all layer**: Add post-processing to convert soft to hard encoding
   - Adds latency
   - Complicates autoregressive generation

3. **Enumeration units**: Create units for all 256×256 (sp_byte, imm_byte) combinations
   - 65536 units per byte × 4 bytes = 262144 units
   - Impractical

## Conclusion

ENT neural is ~80% complete with reliable STACK0/BP/AX handling. The SP computation has fundamental limitations from soft ALU encoding that cannot be easily fixed without architectural changes.

**Recommendation**: Keep ENT handler. Focus Phase 2 efforts on LEV which has cleaner semantics (memory lookups via L15 attention).

## Session Date: 2026-04-15
