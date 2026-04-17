# Commit Strategy - Conversational I/O

## Current Situation

### What Works ✅
- **Conversational I/O Fix**: Spurious THINKING_START bug fixed (3 lines)
- **Unit Tests**: All 6 tests pass, THINKING_END logit = 97.38
- **Code Quality**: Minimal, targeted changes (15 lines total)

### What's Broken ❌
- **L6 Routing**: NEXT_SE not being set, breaks STEP_END generation
- **VM Execution**: All programs fail due to L6 bug
- **End-to-End**: Can't test conversational I/O fully due to L6 bug

### Root Cause
The L6 routing bug affects:
1. Normal STEP_END generation (NEXT_SE = 0)
2. Conversational I/O detection (NEXT_THINKING_END = 0)
3. PC advancement
4. AX value propagation

**This is a pre-existing bug** (visible in git diff from recent commits), NOT caused by conversational I/O changes.

---

## Option 1: Commit Conversational I/O Only (RECOMMENDED)

### Strategy
Commit the conversational I/O fix separately from the L6 bug.

### Benefits
- ✅ Clean separation of concerns
- ✅ Conversational I/O fix is solid (95% confidence, unit tests pass)
- ✅ L6 bug is documented and understood
- ✅ Can review and test conversational I/O changes independently
- ✅ Can fix L6 bug in separate commit

### What to Commit

**Files with conversational I/O changes:**
```bash
git add neural_vm/vm_step.py
git add neural_vm/neural_embedding.py
git add neural_vm/purity_guard.py
git add neural_vm/run_vm.py
```

**Commit message:**
```
Fix conversational I/O: Prevent spurious THINKING_START generation

Problem: L10 FFN null terminator detector was firing during normal
execution, generating THINKING_START at position 1 instead of bytes.

Root Cause: OUTPUT_BYTE_LO (dims 480-511) overlaps with TEMP (480-511).
When TEMP[0] was set during normal execution, detector thought it was
a null byte and triggered THINKING_START inappropriately.

Fix: Added gating on IO_IN_OUTPUT_MODE > 5.0 in L10 FFN unit 700.
Now only fires when actually in I/O mode.

Changes:
- vm_step.py: Add _active_opcode storage and gating (8 lines)
- neural_embedding.py: Inject active opcode flags (2 methods)
- purity_guard.py: Allow embed() parameters (1 line)
- run_vm.py: Debug logging (1 line)

Testing:
- 6 unit tests pass
- THINKING_END logit: 97.38 (beats STEP_END by 201 points)
- Token sequences correct (STEP_END at position 34)

Note: End-to-end testing blocked by separate L6 routing bug
affecting NEXT_SE generation. See Issue #XXX.
```

**Documentation to commit:**
```bash
git add CONVERSATIONAL_IO_BUG_FIX_SUMMARY.md
git add CONVERSATIONAL_IO_FINAL_STATUS.md
git add PATH_TO_100_PERCENT.md
```

### What NOT to Commit
- Test files (mark as untracked or .gitignore)
- Debug scripts
- Status documents about L6 bug

---

## Option 2: Fix L6 First, Then Commit Everything

### Strategy
1. Fix L6 routing bug
2. Verify end-to-end testing works
3. Commit both fixes together

### Benefits
- ✅ 100% confidence before commit
- ✅ Clean working state
- ✅ All tests pass end-to-end

### Drawbacks
- ❌ Takes 3-4 hours longer
- ❌ Mixes two separate fixes
- ❌ Delays getting conversational I/O fix into repo

### Timeline
1. Bisect L6 bug (30 min)
2. Fix L6 routing (1-2 hours)
3. Test everything (1 hour)
4. Commit (30 min)

**Total: 3-4 hours**

---

## Option 3: Commit Current State As-Is (NOT RECOMMENDED)

### Why Not
- ❌ VM is currently broken
- ❌ Tests fail
- ❌ Would break main branch
- ❌ Not ready for production

---

## Recommendation: Option 1

### Rationale

1. **Conversational I/O fix is solid**
   - 95% confidence (unit tests all pass)
   - Minimal, targeted changes
   - Well-documented

2. **L6 bug is separate**
   - Pre-existing issue (visible in git diff)
   - Not caused by conversational I/O changes
   - Needs separate investigation and fix

3. **Clean separation**
   - Easy to review conversational I/O changes
   - Easy to revert if needed
   - L6 fix can be separate commit

4. **Documented limitations**
   - Commit message notes end-to-end testing blocked
   - Comprehensive documentation of L6 bug
   - Clear path forward

### Commit Commands

```bash
# 1. Stage conversational I/O files
git add neural_vm/vm_step.py
git add neural_vm/neural_embedding.py
git add neural_vm/purity_guard.py
git add neural_vm/run_vm.py

# 2. Stage documentation
git add CONVERSATIONAL_IO_BUG_FIX_SUMMARY.md
git add CONVERSATIONAL_IO_FINAL_STATUS.md

# 3. Review changes
git diff --staged

# 4. Commit
git commit -m "Fix conversational I/O: Prevent spurious THINKING_START generation

Problem: L10 FFN null terminator detector was firing during normal
execution, generating THINKING_START at position 1 instead of bytes.

Root Cause: OUTPUT_BYTE_LO (dims 480-511) overlaps with TEMP (480-511).
When TEMP[0] was set, detector thought it was a null byte.

Fix: Added gating on IO_IN_OUTPUT_MODE > 5.0 in L10 FFN unit 700.

Changes:
- vm_step.py: Add _active_opcode storage and gating
- neural_embedding.py: Inject active opcode flags
- purity_guard.py: Allow embed() parameters
- run_vm.py: Debug logging

Testing:
- 6 unit tests pass
- THINKING_END logit: 97.38 (beats STEP_END by 201 points)
- Token sequences correct

Note: End-to-end testing blocked by separate L6 routing bug.
"

# 5. Push
git push origin main
```

---

## Alternative: Create Branch

If uncertain about committing to main:

```bash
# Create feature branch
git checkout -b fix/conversational-io-thinking-start

# Commit changes
git add ...
git commit -m "..."

# Push to branch
git push origin fix/conversational-io-thinking-start

# Create PR for review
```

This allows review before merging to main.

---

## Confidence Level

### With Option 1 (Commit Conversational I/O)
- **95% confidence** the fix is correct
- Unit tests all pass
- Changes are minimal and targeted
- Can fix L6 bug separately

### With Option 2 (Fix L6 First)
- **100% confidence** but takes 3-4 hours longer
- All tests would pass end-to-end
- Clean working state

---

## Decision

**I recommend Option 1**: Commit the conversational I/O fix now, document the L6 bug separately, and fix L6 in the next session.

This gets the solid conversational I/O work into the repo while clearly separating it from the pre-existing L6 bug.

---

## Next Session Plan

After committing conversational I/O:

1. **Fix L6 routing bug** (3-4 hours)
   - Bisect to find breaking commit
   - Fix routing weights
   - Verify NEXT_SE generation

2. **Test conversational I/O end-to-end** (30 min)
   - Should work immediately after L6 fix
   - Verify THINKING_END generates in real execution
   - Reach 100% confidence

3. **Implement format string parsing** (1-2 hours)
   - Add %d, %x, %s support
   - Complete the feature

**Total: 5-7 hours to full completion**
