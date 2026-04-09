# ADD Operation Root Cause Analysis - SOLVED

**Date**: 2026-04-07
**Status**: Root cause identified

## Problem Statement

ADD operation returns 10 (first operand) instead of 42 (sum) when handler is disabled.

**Test Case**: `int main() { return 10 + 32; }`
- Expected: 42
- Actual (without handler): 10
- Actual (with handler): 42

## Root Cause Identified

**❌ Layer 1 threshold heads (L1H1/L1H0) are not setting byte index flags**

Specifically:
- `L1H1[AX] = 0.000` (should be 1.0 at byte 0 positions)
- `L1H0[AX] = 0.000` (should be 0.0 at byte 0 positions)

### Impact

Without L1H1/L1H0 flags, the entire AX carry mechanism fails:

```
Layer 1 FFN → L1H1/L1H0 flags (BROKEN)
              ↓
Layer 3 Attention → Identify byte 0 positions (BROKEN)
              ↓
Layer 3 Attention → Attend to previous AX (BROKEN)
              ↓
Layer 3 Attention → Populate AX_CARRY_LO/HI (BROKEN)
              ↓
Layer 8 FFN → 3-way AND needs AX_CARRY_LO (BROKEN)
              ↓
Layer 8 FFN → Cannot compute sum (BROKEN)
              ↓
Result: Returns first operand only (ALU_LO works, AX_CARRY_LO doesn't)
```

## Investigation Steps

### Step 1: Compared ADD vs IMM Execution

**Finding**: ALU_LO populated (0.676) but AX_CARRY_LO empty (0.000)

```
Step 0:
  MARK_AX: 0.000
  ALU_LO: 0.676     ← First operand gathered successfully
  AX_CARRY_LO: 0.000 ← Second operand missing!
  OUTPUT_LO: 0.000   ← No output (3-way AND incomplete)
```

**File**: `debug_compare_add_imm.py`

### Step 2: Checked Layer 3 AX Carry

**Finding**: Layer 3 attention never populates AX_CARRY_LO anywhere

```
Scanning all positions for AX_CARRY_LO activity:
  ⚠ WARNING: No AX_CARRY_LO found anywhere!
  This means Layer 3 attention is not carrying AX values forward.
```

**File**: `debug_layer3_ax_carry.py`

### Step 3: Inspected Layer 3 Attention Weights

**Finding**: Layer 3 is configured correctly in `vm_step.py:1470`:

```python
# Head 1: AX carry (prev step AX byte 0 → AX_CARRY staging)
_set_carry_forward_attn(
    attn3, 1, BD.MARK_AX, AX_I, AX_I, HD, BD.AX_CARRY_LO, BD.AX_CARRY_HI
)
```

The `_set_carry_forward_attn` function sets up the attention pattern correctly:
- Query on `MARK_AX`
- Key on `L1H1[AX_I] AND NOT L1H0[AX_I]` (byte 0 pattern)
- Value from `EMBED_LO/HI`
- Output to `AX_CARRY_LO/HI`

**Conclusion**: Layer 3 weights are correct, but depend on L1H1/L1H0 flags.

**File**: `debug_layer3_weights.py`, `vm_step.py:2119-2155`

### Step 4: Checked Layer 1 Threshold Heads

**Finding**: L1H1 and L1H0 flags are NOT being set

```
Max L1H1[AX] value: 0.000
Max L1H0[AX] value: 0.000

❌ ISSUE: L1H1/L1H0[AX] are not being set at all!
Layer 1 threshold heads are not working correctly.
```

**File**: `debug_l1_threshold.py`

## Technical Details

### L1H1/L1H0 Encoding

Layer 1 threshold heads use a 2-bit encoding for byte position:

| Byte | L1H1 | L1H0 | Binary |
|------|------|------|--------|
| 0    | 1    | 0    | 10     |
| 1    | 0    | 1    | 01     |
| 2    | 1    | 1    | 11     |
| 3    | 0    | 0    | 00     |

For each register (PC, AX, SP, BP), there are separate L1H1 and L1H0 dimensions:
- `L1H1 + 0` → PC byte index
- `L1H1 + 1` → AX byte index
- `L1H1 + 2` → SP byte index
- `L1H1 + 3` → BP byte index

### Layer 3 Carry-Forward Mechanism

From `vm_step.py:2119-2155`, `_set_carry_forward_attn`:

1. **Query**: Fire at MARK_AX positions
   ```python
   attn.W_q[base, marker_dim] = L  # L = 15.0
   ```

2. **Key**: Match positions where L1H1[AX]=1 AND L1H0[AX]=0 (byte 0)
   ```python
   attn.W_k[base, BD.L1H1 + l1h1_idx] = L
   attn.W_k[base, BD.L1H0 + l1h0_idx] = -L
   ```

3. **Value**: Read EMBED_LO/HI (the actual byte value)
   ```python
   for k in range(16):
       attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
       attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
   ```

4. **Output**: Write to AX_CARRY_LO/HI
   ```python
   for k in range(16):
       attn.W_o[out_lo + k, base + 1 + k] = 1.0
       attn.W_o[out_hi + k, base + 17 + k] = 1.0
   ```

This mechanism **requires L1H1/L1H0 to be set correctly** to identify byte 0 positions.

### Why This Breaks ADD

Layer 8 ADD implementation uses a 3-way AND gate:

```python
# From vm_step.py:3878-3915
ffn.W_up[unit, BD.MARK_AX] = S
ffn.W_up[unit, BD.ALU_LO + a] = S
ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
ffn.b_up[unit] = -S * 2.5  # 3-way AND
```

The unit only activates when **all three** inputs are present:
- MARK_AX (from embedding)
- ALU_LO (from Layer 7 attention, populated ✓)
- AX_CARRY_LO (from Layer 3 attention, NOT populated ✗)

Without AX_CARRY_LO, the 3-way AND never fires, no sum is computed.

## Why First Operand is Returned

The result is 10 (first operand) because:

1. Layer 7 successfully gathers first operand: STACK0 → ALU_LO = 10
2. Layer 8 ADD circuit doesn't fire (missing AX_CARRY_LO)
3. Some other mechanism (possibly passthrough or default) outputs ALU_LO value
4. Result: 10 instead of 42

## Comparison with Working Opcodes

**IMM works** because:
- IMM only needs simple data routing (no multi-operand computation)
- Layer 5 fetches immediate value → output
- No dependency on L1H1/L1H0 flags
- No dependency on carry-forward mechanism

**ADD fails** because:
- Requires two operands
- Second operand (AX) needs carry-forward from previous step
- Carry-forward depends on Layer 1 threshold heads
- Layer 1 threshold heads are broken

## Next Steps

### Option 1: Fix Layer 1 Threshold Heads

**Goal**: Make L1H1/L1H0 flags work correctly

**Steps**:
1. Find Layer 1 FFN weight initialization (`_set_layer1_ffn`)
2. Check if L1H1/L1H0 units are configured
3. Verify threshold logic is correct
4. Test if fixing Layer 1 resolves ADD

**Files to check**:
- `vm_step.py`: Search for `_set_layer1_ffn` or Layer 1 initialization
- Look for L1H1/L1H0 weight setting

### Option 2: Alternative AX Carry Mechanism

**Goal**: Bypass L1H1/L1H0 dependency

**Ideas**:
- Use position encoding instead of threshold flags
- Use different attention pattern (e.g., "attend to token at fixed offset")
- Simplify byte 0 identification

**Pros**: Might be simpler
**Cons**: Requires changing architecture

### Option 3: Keep Handlers Permanent

**Goal**: Accept that handlers are needed

**Rationale**:
- Layer 1 threshold heads may be fundamentally broken
- Fixing might require extensive weight changes
- Handlers work correctly
- May be architectural limitation

**Action**:
- Document that arithmetic operations require handlers
- Update documentation to clarify "neural VM" uses hybrid execution
- Remove "neural weights broken" comment, replace with "requires handler for correctness"

## Recommendation

**Try Option 1 first** (fix Layer 1 threshold heads):

1. The mechanism is well-designed and structurally correct
2. Only Layer 1 is broken (Layer 3, 7, 8 look fine)
3. If L1H1/L1H0 can be fixed, all arithmetic operations should work
4. This would significantly increase pure neural coverage (26% → potentially 50%+)

If Layer 1 cannot be fixed easily:
- Fall back to Option 3 (keep handlers)
- Document as permanent limitation
- Update architectural documentation

## Files Created During Investigation

### Debug Scripts
- `debug_compare_add_imm.py` - Compare ADD vs IMM execution
- `debug_layer3_ax_carry.py` - Check Layer 3 AX carry
- `debug_layer3_weights.py` - Inspect Layer 3 weights
- `debug_l1_threshold.py` - Check Layer 1 threshold heads
- `debug_add_simple.py` - Simple opcode flag detection
- `debug_add_neural_weights.py` - Comprehensive activation capture

### Documentation
- `docs/ADD_INVESTIGATION.md` - Investigation notes
- `docs/ADD_ROOT_CAUSE_FOUND.md` - This document
- `docs/HANDLER_DEPENDENCY_DISCOVERY.md` - Handler discovery
- `docs/SESSION_2026-04-07_NEURAL_WEIGHTS_INVESTIGATION.md` - Session summary

## Summary

**Root Cause**: Layer 1 threshold heads (L1H1/L1H0) not setting byte index flags

**Impact**: AX carry-forward broken → ADD missing second operand → returns first operand only

**Fix Strategy**: Investigate Layer 1 FFN weight initialization to fix threshold heads

**Alternative**: Accept handlers as permanent solution if Layer 1 cannot be fixed

---

**Investigation Status**: COMPLETE ✓
**Root Cause**: IDENTIFIED ✓
**Next Action**: Investigate Layer 1 FFN weights
