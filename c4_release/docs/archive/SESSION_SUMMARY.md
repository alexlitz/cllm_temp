# Session Summary: LEA/JMP/IMM Debugging

## Objective
Fix failing opcodes: JMP, IMM, LEA

## Major Fixes Applied

### 1. L9 ADD Hi-Nibble Units (Lines 3484-3502)
**Problem**: Units with `carry_in=1` reading CARRY[0] with weight S*2.0, causing spurious activation.

**Fix**: Reduced CARRY[0] weight from `S*2.0` to `0.01`, relaxed bias from `-S*4.5` to `-S*2.9`.

**Impact**: Eliminated OUTPUT_HI corruption where all 16 dimensions were written equal values.

### 2. L9 Carry-Out Detection Units (Lines 3622-3644)
**Problem**: 136 units activating, accumulating CARRY[1] to 18,603 instead of ~1.

**Fix**: Applied same CARRY[0] weight reduction.

**Impact**: CARRY[1] no longer accumulates abnormally.

### 3. L9 Borrow-Out Detection Units (Lines 3646-3670)
**Fix**: Applied same CARRY[0] weight reduction for SUB operations.

**Impact**: CARRY[2] handled correctly.

### 4. DivModModule Spurious Activation (Lines 363-408)
**Problem**: 414 DIV/MOD units activating for non-DIV/MOD operations, writing ~69,600 to OUTPUT_HI.

**Fix**: Changed from 6-way AND to 5-way AND + gate check - opcode gating now done via W_gate instead of W_up.

**Impact**: DivModModule now completely inactive for non-DIV/MOD operations.

## Architectural Issue Identified

### Root Cause: Imperfect One-Hot Representations

ALU_HI and AX_CARRY_HI aren't maintaining clean one-hot encoding:
- ALU_HI[0] = 0.416 (should be ~1.0)
- ALU_HI[1] = 0.012 (leakage)
- AX_CARRY_HI[0] = 2.458 (unnormalized)

This causes wrong L9 units to partially activate.

## Results

**Progress**: 
- Eliminated massive corruption (237,912 → 19.1)
- Eliminated CARRY accumulation (18,603 → 0)
- Eliminated DivMod spurious activation (414 units → 0)

**Remaining**: Operand normalization issue prevents correct final predictions.

## Files Modified
`neural_vm/vm_step.py`: Lines 363-375, 392-404, 3484-3502, 3622-3644, 3646-3670
