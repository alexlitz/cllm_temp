# Troubleshooting Final Status

**Date**: 2026-04-10
**Session**: AX_FULL Fix + JSR/ENT/LEV Troubleshooting

---

## Summary

### ✅ Issues Fixed

1. **AX_FULL Dimension Implementation**
   - Added BD.AX_FULL (dims 467-498) to avoid attention output additivity
   - Updated L3 head 5, L6 head 6, and L6 FFN to populate/read AX_FULL
   - Prevents "predictions became all 1's" failure from previous attempt

2. **JSR/ENT/LEV Infinite Loop**
   - Re-enabled ENT handler (was incorrectly commented out)
   - Re-enabled JSR SP/STACK0 overrides (needed to push return address)
   - Fixed ENT to use `_last_sp` instead of model output
   - Added ENT PC override (neural PC update broken)

### ❌ Issues Remaining

**Programs with local variables still fail** (exit code 65536 instead of 42)

---

## Test Results

### Test 1: No Local Variables ✅

```c
int main() { return 42; }
```

**Result**: Exit code 42 ✅ **PASS**

**Bytecode**: JSR → EXIT → ENT → IMM 42 → LEV

**Execution Trace**:
```
Step 0: JSR (PC=0) → jump to main (0x12)
Step 1: ENT (PC=0x12) → setup frame
Step 2: IMM 42 (PC=0x1a) → AX=42
Step 3: LEV (PC=0x22) → return to caller
Step 4: EXIT (PC=0x0a) → exit code 42 ✅
```

### Test 2: 1 Local Variable ❌

```c
int main() { int a; a=42; return a; }
```

**Result**: Exit code 65536 ❌ **FAIL** (expected 42)

**Analysis**:
- 65536 = 0x00010000 (hex)
- This is NOT 42 (0x0000002a)
- Looks like byte extraction issue or reading wrong memory

### Test 3: 2 Local Variables ❌

```c
int main() { int a, b; a=10; b=32; return a+b; }
```

**Result**: Exit code 65536 ❌ **FAIL** (expected 42)

**Analysis**:
- Same exit code as Test 2
- Suggests systemic issue with local variable access (LEA/SI/LI)

---

## Root Cause Analysis (Exit Code 65536)

### Hypothesis 1: Byte Extraction Issue

65536 = 0x00010000 suggests we're reading:
- Byte 2 = 0x01
- Byte 1 = 0x00
- Byte 0 = 0x00

Instead of the correct value (42 = 0x0000002a).

This could indicate:
1. AX register reading wrong bytes (bytes 2-3 instead of 0-1)
2. Memory read returning wrong byte order
3. Stack offset calculation incorrect

### Hypothesis 2: Memory Operations Broken

Programs with local variables use:
- **LEA**: Load effective address (BP - offset)
- **PSH**: Push address to stack
- **SI**: Store integer at address
- **LI**: Load integer from address

If any of these operations fail, variables won't be stored/retrieved correctly.

### Hypothesis 3: BP Offset Calculation

With local variables:
- `int a` is at BP-8
- `int b` is at BP-16

If LEA computes BP-8 incorrectly, we might read from wrong memory location.

---

## What We Know

### ✅ Working Correctly
- **Basic operations**: IMM, EXIT, NOP work
- **Function calls**: JSR/ENT/LEV no longer infinite loop
- **Stack frame setup**: BP and SP are set correctly by handlers
- **PSH**: Preserves AX value correctly
- **Programs without local variables**: Work perfectly

### ❌ Still Broken
- **Programs with 1+ local variables**: Return 65536 instead of correct value
- **Memory operations** (LEA/SI/LI): Likely broken or reading wrong values
- **Exit code extraction**: May be reading wrong bytes from AX

### ⚠️ Partially Working
- **AX_FULL fix**: Works for basic operations, but may not help memory operations
- **ENT handler**: Sets up stack frame, but neural memory operations may fail

---

## Next Steps to Debug

### Step 1: Check Exit Code Extraction

Verify that `_decode_exit_code` is reading the correct bytes from AX.

**Location**: `neural_vm/run_vm.py:1611-1625`

```python
def _decode_exit_code(self, context):
    """Extract exit code from the last REG_AX before HALT."""
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            val = 0
            for j in range(4):
                val |= context[i + 1 + j] << (j * 8)
            return val & 0xFFFFFFFF
    return None
```

**Test**: Print AX bytes to see if they're 0x00, 0x01, 0x00, 0x00 (65536) or something else.

### Step 2: Trace LEA/SI/LI Operations

Add debug logging for:
- LEA: What address is computed?
- SI: What address and value are being stored?
- LI: What address is being read, what value is returned?

**Expected for `int a; a=42; return a;`**:
1. LEA: compute BP-8 → addr (e.g., 0x1f7e8)
2. PSH: push addr to stack
3. IMM 42: AX=42
4. SI: store 42 at addr
5. LEA: compute BP-8 → addr (same address)
6. LI: load from addr → should return 42
7. EXIT: exit code should be 42

**Actual**: Exit code is 65536, so something is wrong in steps 4-6.

### Step 3: Check Memory Shadow Tracking

Verify that `_memory` dict contains correct values after SI operations.

```python
print(f"Memory contents: {runner._memory}")
```

Should show `{0x1f7e8: 42}` or similar.

### Step 4: Check AX Value Before EXIT

Add debug logging right before EXIT to see AX value:

```python
if op == Opcode.EXIT:
    ax = self._extract_register(context, Token.REG_AX)
    print(f"[EXIT] AX=0x{ax:08x} ({ax})")
```

Should print `AX=0x0000002a (42)`, not `AX=0x00010000 (65536)`.

---

## Likely Root Cause

Based on exit code 65536 = 0x00010000, I suspect:

1. **LI operation returns wrong value** from memory
   - Reads 0x00010000 instead of 0x0000002a
   - Could be reading from wrong address or wrong bytes

2. **Memory not stored correctly** by SI
   - SI might not be writing to shadow memory
   - LI reads uninitialized memory (contains 0x00010000 from something else)

3. **AX bytes extracted incorrectly**
   - Exit code extraction might be reading bytes in wrong order
   - Could be endianness issue

---

## Recommendation

**Priority 1**: Add debug logging to trace LEA/SI/LI operations and identify where the wrong value (65536) is introduced.

**Priority 2**: Verify that SI is actually storing to memory and LI is reading from the correct address.

**Priority 3**: Check if the AX_FULL fix is being used correctly for these operations or if they still rely on broken AX_CARRY.

---

## Files Modified This Session

1. **`neural_vm/vm_step.py`**: AX_FULL dimension implementation
2. **`neural_vm/run_vm.py`**: JSR/ENT/LEV handler fixes, layer count fix
3. **Documentation**: AX_FULL_FIX_SUMMARY.md, JSR_ENT_LEV_FIX_SUMMARY.md, L3_HEAD5_FIX_ANALYSIS.md

---

## Conclusion

**Partial Success** ✅❌

✅ **Fixed**:
- JSR/ENT/LEV infinite loop
- Basic function calls work
- Programs without local variables execute correctly

❌ **Still Broken**:
- Programs with local variables return exit code 65536
- Memory operations (LEA/SI/LI) likely broken
- Further debugging needed to identify exact issue

**Next Action**: Debug LEA/SI/LI operations to find where 65536 is introduced.
