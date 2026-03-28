# Session Summary: IMM Investigation & PSH Revert

## What We Accomplished

### 1. Fixed Test Bug (IMM)
**Problem**: Original test showed IMM failing at token 1
**Root Cause**: Test wasn't building context incrementally
**Fix**: Added `current_context.append(draft)` in loop
**Result**: Real failure is at token 7, not token 1

### 2. Diagnosed IMM Bug
**Real Issue**: IMM 42/255 fail at token 7 (AX byte 1)
- Token 6 (AX_b0): Correct (42 or 255)
- Token 7 (AX_b1): Wrong (predicts 42/255 instead of 0)

**Root Cause**: Missing AX byte relay logic
- BYTE_INDEX increments correctly ✓
- MARK_AX not propagated (stays 0) ✗
- OUTPUT carries stale value ✗

**Key Finding**: PC bytes work, AX bytes don't

### 3. Reverted Broken PSH Fix
**Problem**: Math error in negative CMP[0] weight
**Fix**: Restored disabled state (T=100.0)
**Side Effect**: JMP 16 now fails (was passing before!)

## Current Status (Manual Tests)

**PASSING**: NOP, IMM 0, JMP 8
**FAILING**: IMM 42, IMM 255, JMP 16

## Critical Discovery

PSH revert caused JMP regression:
- Before: JMP 16 ✅, JMP 8 ❌
- After: JMP 16 ❌, JMP 8 ✅

The "broken" PSH fix was accidentally helping JMP 16!

## Priority Issues

1. **P0**: JMP 16 regression (PSH revert broke it)
2. **P1**: IMM byte relay (AX bytes fail)
3. **P2**: PSH permanent fix (architectural)

## Recommended Next Action

Fix JMP regression - understand why negative CMP[0] weight affected JMP prediction.
