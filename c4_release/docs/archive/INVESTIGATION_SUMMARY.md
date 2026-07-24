# Neural Arithmetic Weights Investigation - Complete Summary

**Date**: 2026-04-07
**Status**: Root cause identified, fix strategy determined

## Executive Summary

Investigation into why arithmetic operations (ADD, SUB, MUL, etc.) fail without Python handlers has **successfully identified the root cause**:

**❌ Layer 1 threshold heads (L1H1/L1H0) are not setting byte index flags**

This breaks the AX carry-forward mechanism needed for multi-operand operations.

## Problem

```c
int main() { return 10 + 32; }
```

- **Expected**: exit_code = 42
- **Actual (without handler)**: exit_code = 10 (first operand only)
- **Actual (with handler)**: exit_code = 42

## Root Cause: Complete Failure Chain

```
❌ Layer 1 FFN
   ↓ L1H1[AX] = 0.000 (should be 1.0 at byte 0)
   ↓ L1H0[AX] = 0.000 (should be 0.0 at byte 0)
   ↓
❌ Layer 3 Attention
   ↓ Cannot identify byte 0 positions
   ↓ Cannot populate AX_CARRY_LO/HI
   ↓
❌ Layer 8 FFN (ADD circuit)
   ↓ Missing second operand (AX_CARRY_LO empty)
   ↓ 3-way AND incomplete (needs MARK_AX + ALU_LO + AX_CARRY_LO)
   ↓
✗ Returns first operand (10) instead of sum (42)
```

## Evidence

### Layer 7: Operand Gathering ✓ WORKS
```
ALU_LO: 0.676  ← First operand from STACK0 successfully gathered
```

### Layer 3: AX Carry ✗ BROKEN
```
AX_CARRY_LO: 0.000  ← Should contain second operand (32)
Cause: L1H1/L1H0 flags not set, Layer 3 can't find byte 0
```

### Layer 1: Threshold Heads ✗ BROKEN
```
Max L1H1[AX]: 0.000  ← Should be 1.0 at byte 0 positions
Max L1H0[AX]: 0.000  ← Should be 0.0 at byte 0 positions

No byte 0 positions identified → Layer 3 has nothing to attend to
```

## Technical Insight

The neural architecture is **well-designed** but Layer 1 implementation is broken:

**Designed Mechanism** (from `vm_step.py:1470`):
1. Layer 1 sets L1H1/L1H0 flags to mark byte positions
2. Layer 3 attention queries on MARK_AX
3. Layer 3 attention keys on L1H1[AX]=1 AND L1H0[AX]=0 (byte 0 pattern)
4. Layer 3 copies EMBED_LO/HI → AX_CARRY_LO/HI
5. Layer 8 computes sum using ALU_LO + AX_CARRY_LO

**Actual Behavior**:
1. Layer 1 doesn't set L1H1/L1H0 ← **BROKEN HERE**
2. Layer 3 has no targets to attend to
3. AX_CARRY_LO stays zero
4. ADD returns first operand only

## Fix Strategy

### Recommended: Fix Layer 1 Threshold Heads

**Steps**:
1. Find Layer 1 FFN initialization (`_set_layer1_ffn` in `vm_step.py`)
2. Check if L1H1/L1H0 units are configured correctly
3. Verify threshold logic:
   - Byte 0: L1H1=1, L1H0=0
   - Byte 1: L1H1=0, L1H0=1
   - Byte 2: L1H1=1, L1H0=1
   - Byte 3: L1H1=0, L1H0=0
4. Fix weight initialization
5. Test ADD without handler

**Impact if successful**:
- All arithmetic operations (ADD, SUB, MUL, DIV, MOD) should work
- All bitwise operations (OR, XOR, AND) should work
- All shift operations (SHL, SHR) should work
- Pure neural coverage: 26% → potentially 50%+

### Alternative: Keep Handlers

If Layer 1 cannot be fixed:
- Document as architectural limitation
- Keep handlers as permanent solution
- Update "neural weights broken" → "requires handler for multi-operand operations"

## Comparison with Working Opcodes

**Why IMM works but ADD doesn't**:

| Aspect | IMM | ADD |
|--------|-----|-----|
| Operands | 1 (immediate) | 2 (stack + AX) |
| Carry-forward | Not needed | Needed (AX from prev step) |
| L1H1/L1H0 dependency | No | Yes (for byte 0 identification) |
| Complexity | Simple routing | Multi-stage pipeline |
| Works neurally | ✓ Yes | ✗ No (Layer 1 broken) |

## Investigation Methodology

### 1. Activation Tracing
- Created hooks to capture layer outputs
- Compared ADD vs IMM execution
- Identified AL U_LO populated but AX_CARRY_LO empty

### 2. Layer-by-Layer Analysis
- **Layer 8**: ADD circuit correctly configured, but missing input
- **Layer 7**: Operand gathering works (ALU_LO = 0.676)
- **Layer 3**: Weights correct, but no L1H1/L1H0 targets to attend to
- **Layer 1**: Not setting threshold flags at all

### 3. Weight Inspection
- Verified Layer 3 attention configured correctly (`_set_carry_forward_attn`)
- Verified Layer 8 FFN configured correctly (3-way AND gates)
- Found Layer 1 not setting L1H1/L1H0 (root cause)

## Files Created

### Debug Scripts
```
debug_compare_add_imm.py         - Compare ADD vs IMM execution
debug_layer3_ax_carry.py         - Verify Layer 3 AX carry
debug_layer3_weights.py          - Inspect Layer 3 weights
debug_l1_threshold.py            - Check Layer 1 flags ← ROOT CAUSE IDENTIFIED HERE
debug_add_simple.py              - Opcode detection
debug_add_neural_weights.py      - Comprehensive tracing
```

### Documentation
```
docs/ADD_INVESTIGATION.md                           - Investigation notes
docs/ADD_ROOT_CAUSE_FOUND.md                        - Root cause analysis
docs/INVESTIGATION_SUMMARY.md                       - This document
docs/HANDLER_DEPENDENCY_DISCOVERY.md                - Initial findings
docs/SESSION_2026-04-07_NEURAL_WEIGHTS_INVESTIGATION.md - Session log
```

### Purity System
```
neural_vm/purity_check.py                - Purity violation detection
neural_vm/PURITY_CHECK_CHANGELOG.md      - Change tracking
docs/PURITY_PROTECTION.md                - Protection guidelines
```

## Key Metrics

### Before Investigation
- Understanding: "Neural weights are broken" (vague)
- Known cause: None
- Fix strategy: None

### After Investigation
- Understanding: "Layer 1 threshold heads not setting L1H1/L1H0 flags" (specific)
- Known cause: Identified ✓
- Fix strategy: Fix Layer 1 FFN initialization

### Opcode Coverage
- Pure Neural: 11/42 (26%)
- Handler-Dependent: 11/42 (26%) ← Could become pure if Layer 1 fixed
- Broken: 4/42 (10%)
- Untested: 16/42 (38%)

**Potential**: If Layer 1 fixed → 22/42 (52%) pure neural

## Next Actions

1. **Immediate**: Search `vm_step.py` for Layer 1 FFN initialization
2. **Verify**: Check if L1H1/L1H0 units exist in weight config
3. **Fix**: Implement correct threshold head logic
4. **Test**: Run `debug_l1_threshold.py` to verify fix
5. **Validate**: Test ADD without handler → should return 42

## Conclusion

**Investigation Status**: ✅ COMPLETE

**Key Achievement**: Identified specific root cause (Layer 1 threshold heads) rather than vague "neural weights broken"

**Next Step**: Fix Layer 1 FFN initialization to enable L1H1/L1H0 flags

**Impact**: Could unlock pure neural execution for 11 additional opcodes (all arithmetic/bitwise/shift operations)

---

**Date**: 2026-04-07
**Investigator**: Claude (with user guidance)
**Time Spent**: ~2 hours of systematic debugging
**Outcome**: Root cause identified, fix strategy determined
