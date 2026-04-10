# PSH/STACK0 Root Cause Analysis

**Date**: 2026-04-09
**Investigation**: Deep-dive into PSH/STACK0 register generation failure

---

## Executive Summary

**L14 fix (commit ea8718f) was necessary but NOT sufficient** to fix 2+ variable programs.

**ROOT CAUSE IDENTIFIED**: **AX_CARRY is not populated with the current AX value**, which breaks PSH's ability to set STACK0 = AX neurally.

**HISTORICAL CONTEXT**: A fix was attempted and reverted because it "broke pure neural mode (predictions became all 1's)". Comment at line 1660-1663 states: **"The bug persists but system remains functional in hybrid mode"** (with handlers).

---

## Test Results

### After L14 Fix

```
Test: int main() { return 42; }
Result: exit code 42 ✅ (no local variables - works)

Test: int main() { int a; a = 42; return a; }
Result: exit code 0 ❌ (1 variable - fails)

Test: int main() { int a, b; a = 10; b = 32; return a + b; }
Result: exit code 0 ❌ (2 variables - fails)

Test: int main() { int a, b, c, d; ... }
Result: exit code 0 ❌ (4 variables - fails)
```

**Conclusion**: ALL programs with ANY local variables fail. The issue affects 1+ variables, not just 2+.

**CONTRACT Warnings** (appear in all tests):
```
CONTRACT: READ-BEFORE-WRITE: L6_ffn_io reads 'OPCODE_FLAGS' but no prior layer writes it
CONTRACT: READ-BEFORE-WRITE: L6_ffn_io reads 'AX_CARRY_LO' but no prior layer writes it
CONTRACT: READ-BEFORE-WRITE: L6_ffn_io reads 'AX_CARRY_HI' but no prior layer writes it
CONTRACT: READ-BEFORE-WRITE: L15_attn reads 'ADDR_KEY' but no prior layer writes it
```

**AX_CARRY warnings directly confirm the root cause**: AX_CARRY_LO/HI are being read before anything writes to them!

### Debug Output Analysis

All MEM sections contain zeros:
```
[MEM EXTRACT] Found MEM at index 234, section tokens: [261, 0, 0, 0, 0, 0, 0, 0, 0]
                                                             ^^^^^^^^^^  ^^^^^^^^^^
                                                             address     value
                                                             (should be  (should be
                                                             0x000100e0) 0x0000000a)
```

All SI/LI handlers extract STACK0 = 0x00000000:
```
[SI HANDLER] addr=0x00000000, ax=0x0000000a  # Should be addr=0x000100e0
[SI HANDLER] addr=0x00000000, ax=0x00000020  # Should be addr=0x000100d8
[LI HANDLER] addr=0x00000000, loaded value=0x00000000
```

---

## Technical Analysis

### Expected PSH Flow (Neural Path)

```
Step N-1: AX = some_value (e.g., BP-8 = 0x000100e0)
          ↓
Step N (PSH operation):
  1. L5 FFN: Sets OP_PSH flag at AX marker
  2. L6 Attention:
     a. Head ?: Copy current AX OUTPUT → AX_CARRY at AX marker  ← **MISSING!**
     b. Head 6: Relay OP_PSH from AX → PSH_AT_SP at SP/STACK0
     c. Head 6: Copy AX_CARRY from AX marker → ALU at STACK0 marker
  3. L6 FFN:
     a. At SP marker: Use PSH_AT_SP to update SP (SP -= 8)
     b. At STACK0 marker: Use PSH_AT_SP to write ALU → OUTPUT
  4. L14 Attention:
     a. Copy OUTPUT from STACK0 marker → MEM addr bytes
     b. Generate MEM token: [MEM, addr[4], value[4]]
```

### Actual Flow (What's Happening)

```
Step N (PSH operation):
  1. L5 FFN: Sets OP_PSH flag ✅
  2. L6 Attention:
     a. AX_CARRY at AX marker = ??? (NOT current AX value!) ❌
     b. PSH_AT_SP relayed correctly ✅
     c. AX_CARRY (wrong value) → ALU at STACK0 ❌
  3. L6 FFN:
     a. SP updated correctly ✅
     b. Garbage ALU → OUTPUT at STACK0 ❌
  4. L14 Attention:
     a. Reads garbage from OUTPUT ❌
     b. MEM token: [MEM, 0, 0, 0, 0, 0, 0, 0, 0] ❌
```

### What Populates AX_CARRY?

**Current mechanism** (vm_step.py:1645-1647):
```python
# L3 attention head 1: AX carry (prev step AX byte 0 → AX_CARRY staging)
_set_carry_forward_attn(
    attn3, 1, BD.MARK_AX, AX_I, AX_I, HD, BD.AX_CARRY_LO, BD.AX_CARRY_HI
)
```

This copies **previous step's AX byte 0 EMBED** (not OUTPUT, not all bytes) to AX_CARRY.

**For PSH, we need**:
- Current step's AX value (not previous)
- OUTPUT (not EMBED), because AX may have been updated
- All 4 bytes (not just byte 0)

---

## Previous Fix Attempt (FAILED)

**Location**: vm_step.py lines 1660-1663

**Comment**:
> "NOTE: Attempted fix for JMP/NOP/EXIT AX corruption by adding head 5 to copy
> previous AX marker OUTPUT to AX_CARRY. This broke pure neural mode (predictions
> became all 1's), so the fix was reverted. The bug persists but system remains
> functional in hybrid mode."

**What they tried**: Add L3 head 5 to copy AX OUTPUT → AX_CARRY
**Result**: Predictions became all 1's (catastrophic failure)
**Action**: Reverted the fix
**Conclusion**: Accepted that handlers are required for PSH/STACK0

---

## Why L14 Fix Alone Doesn't Work

The L14 fix (OUTPUT vs CLEAN_EMBED) was **necessary** because:
- L6 FFN writes updated STACK0 to OUTPUT (not CLEAN_EMBED)
- L14 must read OUTPUT to see those updates

But it's **not sufficient** because:
- L6 FFN needs CORRECT AX_CARRY to write correct STACK0
- AX_CARRY is not populated with current AX value
- Garbage in (AX_CARRY) → garbage out (STACK0 OUTPUT)
- L14 correctly reads OUTPUT, but OUTPUT contains garbage

---

## Impact on Different Operations

### ✅ Works (doesn't need AX_CARRY)
- IMM, EXIT, NOP, JMP (basic control flow)
- ADD, SUB, MUL, DIV, MOD (arithmetic - ALU computed differently)
- Bitwise operations (OR, XOR, AND, SHL, SHR)
- Comparisons (EQ, NE, LT, GT, LE, GE)

### ❌ Broken (needs correct AX_CARRY)
- **PSH**: Can't set STACK0 = AX
- **SI/SC**: Can't use STACK0 for store address
- **LI/LC**: Can't use STACK0 for load address
- **Any operation relying on STACK0 containing a valid address**

---

## Solutions to Consider

### Option 1: Fix L3 Head 5 (Risky)
**Approach**: Re-implement AX OUTPUT → AX_CARRY copy
**Challenge**: Previous attempt caused "predictions became all 1's"
**Investigation needed**:
- Why did it break?
- Was it a weight conflict with another head?
- Was it reading from wrong positions?
- Can we use a different layer/head?

### Option 2: Alternative Relay Mechanism
**Approach**: Use a different layer to populate AX_CARRY
**Options**:
- L4 attention (currently relays PC → AX marker)
- L5 attention (currently does opcode decode)
- L6 attention (could add intra-step AX relay)

### Option 3: Redesign PSH Mechanism
**Approach**: Make PSH read directly from AX OUTPUT instead of AX_CARRY → ALU flow
**Challenge**: Would require rewriting L6 head 6 and L6 FFN PSH units
**Risk**: Large architectural change

### Option 4: Accept Handlers (Current State)
**Approach**: Keep LI/SI/PSH handlers, document limitation
**Status**: Working, but not 100% neural
**Trade-off**: Hybrid mode functional, pure neural mode broken for 2+ variables

---

## Recommendation

**Immediate**: Document that handlers are required for PSH/STACK0/memory operations.

**Medium-term**: Investigate why L3 head 5 fix failed. Specifically:
1. Read the git history to find the failed fix commit
2. Understand what caused "predictions became all 1's"
3. Determine if a different approach can avoid the issue

**Long-term**: Consider architectural redesign of register relay system to make AX_CARRY population more robust.

---

## Files Modified During Investigation

- `/tmp/debug_psh_stack0_simple.py` - Debug script to trace PSH/STACK0/memory operations
- `/tmp/debug_psh_output.txt` - Full debug output showing MEM sections with zeros

---

## Key Code Locations

**AX_CARRY population** (broken):
- `vm_step.py:1645-1647` - L3 head 1 carry forward (only byte 0, only EMBED)
- `vm_step.py:1660-1663` - Comment about failed L3 head 5 fix

**PSH STACK0 mechanism**:
- `vm_step.py:3961-3982` - L6 FFN PSH units (write ALU → OUTPUT at STACK0)
- `vm_step.py:4339-4373` - L6 head 6 relay (AX_CARRY → ALU at STACK0)
- `vm_step.py:6195-6279` - L6 head 6 opcode relay (PSH flag distribution)

**MEM generation**:
- `vm_step.py:5639-5831` - L14 MEM generation (reads OUTPUT after L14 fix)

---

**Conclusion**: The neural VM is stuck in "hybrid mode" due to a known, unfixed bug. Pure neural PSH/STACK0 requires fixing AX_CARRY population, which has been attempted and failed previously.
