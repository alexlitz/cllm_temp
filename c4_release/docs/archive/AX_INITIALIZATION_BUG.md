# AX Initialization Bug - Critical Finding

**Date**: 2026-04-08
**Severity**: CRITICAL - Blocks all VM execution

## 🔍 Discovery

All programs return exit code `0x01010101` (16843009), regardless of what they do:

```python
Test 1: IMM 42, EXIT
  Exit code: 0x01010101  (should be 0x0000002A)

Test 2: IMM 255, EXIT
  Exit code: 0x01010101  (should be 0x000000FF)

Test 3: IMM 0, EXIT
  Exit code: 0x01010101  (should be 0x00000000)
```

## 💡 Analysis

### Key Observation
**The exit code is ALWAYS 0x01010101, regardless of the IMM immediate value.**

This means:
1. ❌ IMM instruction is NOT executing properly
2. ❌ AX register is NOT being written with the immediate value
3. ❌ AX is either stuck at 0x01010101 or defaulting to it

### Pattern Recognition

`0x01010101` = All bytes are `0x01`

This specific pattern suggests:
- **Not random garbage** (would vary)
- **Not zero-initialized** (would be 0x00000000)
- **Likely a default/marker value** set somewhere

Possibilities:
1. AX is initialized to 0x01010101 as a marker
2. OUTPUT dimensions default to 1's (one-hot encoding base?)
3. Some layer is writing 1's to all OUTPUT bytes

## 🔗 Connection to PC Bug

We already found that **PC doesn't advance**:
- After PSH, PC stays at 0 instead of advancing to 4
- Programs loop at instruction 0

Combined evidence:
```
Execution Flow (Broken):
1. Set opcode: IMM ✓
2. Execute IMM: ❌ (doesn't write AX)
3. PC advances: ❌ (stays at 0)
4. Set opcode: IMM again ✓
5. Loop...
6. Eventually EXIT with AX = 0x01010101
```

## 🎯 Root Cause Hypothesis

### Most Likely: L6 FFN Routing Broken

The git diff shows extensive L6 FFN changes:
```python
# Added guards to IMM routing:
ffn.W_up[unit, BD.OP_IMM] = S
ffn.W_up[unit, BD.MARK_AX] = S
ffn.W_up[unit, BD.MARK_PC] = -S    # NEW: Block at PC
ffn.W_up[unit, BD.IS_BYTE] = -S    # NEW: Block at bytes
ffn.b_up[unit] = -S * T

# Threshold changed:
T = 5.5  # Was 4.0
```

**Hypothesis**: The new guards are too restrictive and prevent IMM from firing at the AX marker position.

### Why This Breaks Everything

L6 FFN is responsible for routing operations:
- IMM: FETCH → OUTPUT (AX value)
- EXIT: AX_CARRY → OUTPUT (exit code)
- NOP: AX_CARRY → OUTPUT (passthrough)
- etc.

If L6 IMM routing doesn't fire:
1. FETCH values are NOT copied to OUTPUT
2. OUTPUT stays at default (0x01010101?)
3. AX register in next step = garbage
4. PC also broken (related routing issue)

## 🧪 How to Verify

### Test 1: Check L6 FFN Activation
```python
# Run IMM instruction
# Check if L6 FFN unit 0 (IMM LO nibble 0) fires:
# - Input: OP_IMM=5, MARK_AX=1, MARK_PC=0, IS_BYTE=0
# - Expected: 5+1-0-0 = 6 > 5.5 → fires ✓
# - If doesn't fire: guards too restrictive
```

### Test 2: Check OUTPUT After L6
```python
# After L6 FFN during IMM execution:
# - Check OUTPUT_LO[0-15] at AX marker
# - Should contain FETCH nibbles
# - If all zeros or all ones: routing failed
```

### Test 3: Revert L6 Changes
```python
git checkout <before-L6-changes>
python tests/test_trace_manual_bytecode.py
# If works: L6 changes broke it
# If fails: issue is older
```

## 🔧 Potential Fixes

### Fix 1: Adjust Threshold
```python
# If issue is threshold too high:
T = 4.5  # Lower than 5.5, higher than 4.0
# Test if IMM fires correctly
```

### Fix 2: Remove IS_BYTE Guard
```python
# If IS_BYTE guard is the problem:
# Remove this line from IMM routing:
# ffn.W_up[unit, BD.IS_BYTE] = -S
```

### Fix 3: Check Position Detection
```python
# Verify MARK_AX is correctly set:
# - Should be 1.0 at AX marker position
# - Should be 0.0 elsewhere
# If MARK_AX is wrong, guards won't help
```

## 📊 Impact

### Blocks Everything
- ✗ IMM doesn't work → can't load values
- ✗ PC doesn't advance → can't execute programs
- ✗ EXIT returns garbage → can't get correct exit codes
- ✗ Conversational I/O can't be tested end-to-end

### Affects Scope
- **All programs** (with or without conversational_io)
- **All opcodes** (not just IMM)
- **Basic execution** (even 2-instruction programs fail)

## 🎯 Priority

**HIGHEST PRIORITY** - This blocks all VM functionality.

Must be fixed before:
- ✗ Conversational I/O end-to-end testing
- ✗ Any program execution
- ✗ Further development

## 💡 Immediate Action

```bash
# 1. Bisect to find breaking commit
git bisect start
git bisect bad HEAD
git bisect good 86ca9cc  # Known good commit

# 2. For each commit, test:
python -c "
from neural_vm.run_vm import AutoregressiveVMRunner
bytecode = [1 | (42 << 8), 34 | (0 << 8)]
runner = AutoregressiveVMRunner(conversational_io=False)
_, exit_code = runner.run(bytecode, b'', [], max_steps=5)
print(f'Exit: {exit_code}')
# Expected: 42
# If 16843009: broken
"

# 3. Identify exact breaking commit
# 4. Review changes in that commit
# 5. Apply targeted fix
```

## 📋 Summary

- **Bug**: AX always 0x01010101, IMM doesn't execute
- **Cause**: Likely L6 FFN routing guards too restrictive
- **Impact**: ALL programs fail
- **Priority**: CRITICAL
- **Next**: Bisect to find breaking commit

This is the **root blocker** preventing all VM execution.
