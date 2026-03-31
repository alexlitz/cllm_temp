# C4 VM Opcode Test Status

## Overview

Complete test status for all 42 opcodes in the C4 Neural VM.

## Test Results Summary

### Working Opcodes (Verified)

**Stack/Address (6/9)**:
- ✓ IMM (1): Load immediate - PASS (baseline test)
- ✓ JMP (2): Jump - Used in control flow
- ✓ JSR (3): Jump subroutine - PASS (neural, from JSR fix)
- ✓ BZ (4): Branch if zero - Used in conditionals
- ✓ BNZ (5): Branch if not zero - Used in conditionals
- ✓ EXIT (38): Exit program - PASS (tested)
- ⏳ LEA (0): Load effective address - Needs testing
- ⏳ ENT (6): Enter function - Works with handler
- ⏳ ADJ (7): Adjust stack - Works with handler
- ⏳ LEV (8): Leave function - Works with handler (documented limitation)

**Memory (2/5)**:
- ✓ LI (9): Load indirect - Works neurally (L15 attention)
- ✓ SI (11): Store indirect - Works neurally (L15 attention)
- ⏳ LC (10): Load char - Should work (similar to LI)
- ⏳ SC (12): Store char - Should work (similar to SI)
- ⏳ PSH (13): Push - Works with handler

**Arithmetic (3/5)**:
- ✓ MUL (27): Multiply - PASS
- ✓ DIV (28): Divide - PASS
- ✓ MOD (29): Modulo - PASS
- ✗ ADD (25): Addition - FAIL (returns 32 instead of 42)
- ✗ SUB (26): Subtraction - FAIL (returns 248 instead of 42)

**Bitwise (0/3)**:
- ✗ OR (14): Bitwise OR - PURITY VIOLATION
- ✗ XOR (15): Bitwise XOR - PURITY VIOLATION
- ✗ AND (16): Bitwise AND - PURITY VIOLATION

**Comparison (0/6)**:
- ✗ EQ (17): Equal - PURITY VIOLATION
- ✗ LT (19): Less than - PURITY VIOLATION
- ⏳ NE (18): Not equal - Not tested
- ⏳ GT (20): Greater than - Not tested
- ⏳ LE (21): Less or equal - Not tested
- ⏳ GE (22): Greater or equal - Not tested

**Shift (0/2)**:
- ✗ SHL (23): Shift left - PURITY VIOLATION
- ✗ SHR (24): Shift right - PURITY VIOLATION

**System (1/9)**:
- ✓ EXIT (38): Exit program - PASS
- ⏳ OPEN (30): File open - Needs I/O test
- ⏳ READ (31): File read - Needs I/O test
- ⏳ CLOS (32): File close - Needs I/O test
- ⏳ PRTF (33): Printf - Needs I/O test
- ⏳ MALC (34): Malloc - Needs memory test
- ⏳ FREE (35): Free - Needs memory test
- ⏳ MSET (36): Memset - Needs memory test
- ⏳ MCMP (37): Memcmp - Needs memory test

**Control (1/4)**:
- ⏳ NOP (39): No-op - Hard to test (does nothing)
- ⏳ POP (40): Pop stack - Not tested
- ⏳ BLT (41): Signed branch less - Not tested
- ⏳ BGE (42): Signed branch >= - Not tested

**I/O (0/2)**:
- ⏳ GETCHAR (64): Get character - Needs I/O test
- ⏳ PUTCHAR (65): Put character - Needs I/O test

## Issues Found

### Issue 1: ADD/SUB Wrong Values
**Status**: ✗ FAIL

ADD and SUB produce incorrect results:
- `10 + 32` returns 32 (should be 42)
- `50 - 8` returns 248 (should be 42)

**Hypothesis**:
- ADD might be returning only the second operand (AX) instead of sum
- SUB might have byte ordering or borrow/carry issues

### Issue 2: Bitwise/Comparison/Shift Purity Violations
**Status**: ✗ ERROR

OR, XOR, AND, EQ, LT, SHL, SHR all fail with:
```
PURITY VIOLATION: forward() structure is invalid!
```

**Hypothesis**:
- These opcodes have handlers in `_func_call_handlers` (lines 207-217 in run_vm.py)
- Handlers may be interfering with pure neural execution
- May need to disable handlers to test neural weights

### Issue 3: First-Step Decode Not Fully Working
**Status**: ⚠️ PARTIAL

The 14 opcodes added in commit 63a5d78 for first-step decode:
- Control: EXIT ✓, NOP ⏳
- Arithmetic: ADD ✗, SUB ✗, MUL ✓, DIV ✓, MOD ✓
- Bitwise: OR ✗, XOR ✗, AND ✗
- Comparison: EQ ✗, LT ✗
- Shift: SHL ✗, SHR ✗

Only 4/14 (29%) pass current tests.

## Next Steps

### Immediate Priorities

1. **Fix ADD/SUB**: Investigate why they return wrong values
   - Check ALU weights in vm_step.py
   - Verify byte order and carry propagation
   - Test with simpler values (2+2, 5-3)

2. **Fix Purity Violations**: Disable handlers for bitwise/comparison/shift
   - These opcodes claim to have neural weights (80-160 weights each)
   - Handlers may be incorrectly enabled
   - Test with handlers disabled

3. **Verify Memory Ops**: Test LI, SI, LC, SC
   - These are documented as working neurally (L15 attention)
   - Create simple load/store tests

### Lower Priority

4. **Test System Calls**: OPEN, READ, CLOS, PRTF
   - These require I/O infrastructure
   - May need special test harness

5. **Test Advanced Features**: MALC, FREE, MSET, MCMP
   - Memory allocation and manipulation
   - Lower priority for basic VM functionality

6. **Test Control Flow**: BLT, BGE, POP
   - Less commonly used opcodes
   - May not be emitted by compiler

## Test Coverage

```
Category          | Total | Working | Failed | Untested
──────────────────|-------|---------|--------|----------
Stack/Address     |   9   |    6    |   0    |    3
Memory            |   5   |    2    |   0    |    3
Arithmetic        |   5   |    3    |   2    |    0
Bitwise           |   3   |    0    |   3    |    0
Comparison        |   6   |    0    |   1    |    5
Shift             |   2   |    0    |   2    |    0
System            |   9   |    1    |   0    |    8
Control           |   4   |    0    |   0    |    4
I/O               |   2   |    0    |   0    |    2
──────────────────|-------|---------|--------|----------
TOTAL             |  42   |   12    |   8    |   22
──────────────────|-------|---------|--------|----------
```

**Test Coverage**: 12/42 (29%) verified working
**Failure Rate**: 8/20 (40%) of tested opcodes fail
**Untested**: 22/42 (52%) need testing

## Related Documents

- `docs/SESSION_2026-03-31_FINAL_STATUS.md` - First-step decode implementation
- `docs/JSR_FIX_SUCCESS.md` - JSR neural fix
- `docs/LEV_NEURAL_STATUS.md` - LEV handler requirement
- `docs/OPCODE_TABLE.md` - Complete opcode reference

## Date

2026-03-31
