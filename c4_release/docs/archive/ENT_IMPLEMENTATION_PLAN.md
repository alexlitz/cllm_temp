# ENT Neural Implementation Plan

**Date**: 2026-04-09
**Status**: Analysis Complete - Ready to Implement
**Estimated Effort**: 8-12 hours

---

## Current ENT Handler Analysis

### What ENT Does

```c
// C semantics: ENT local_size
*--sp = bp;      // Push old BP onto stack
bp = sp;         // Set new frame pointer
sp -= local_size; // Allocate local variables
pc += 4;         // Advance to next instruction
```

### Python Handler Breakdown

```python
def _handler_ent(self, context, output):
    # 1. Get local allocation size from immediate
    imm = (instr >> 8)  # 24-bit signed immediate
    if imm >= 0x800000:
        imm -= 0x1000000

    # 2. Push old BP
    push_addr = (old_sp - 8) & 0xFFFFFFFF

    # 3. Set new BP = SP (after push)
    new_bp = push_addr

    # 4. Allocate locals
    new_sp = (new_bp - imm) & 0xFFFFFFFF

    # 5. Override registers
    context.override(REG_SP, new_sp)
    context.override(REG_BP, new_bp)
    context.override(STACK0, old_bp)
    context.override(REG_PC, next_pc)

    # 6. Write to shadow memory
    memory[push_addr] = old_bp
```

---

## Neural Implementation Strategy

### Operations Breakdown

| Step | Operation | Implementation | Status |
|------|-----------|----------------|--------|
| 1 | SP -= 8 | PSH pattern (L6 FFN) | ✅ Already works |
| 2 | STACK0 = old_bp | BP → STACK0 copy (L6 FFN) | ⏳ Need to add |
| 3 | BP = new_sp | SP → BP copy (L6 FFN) | ⏳ Need to add |
| 4 | SP -= imm | ADJ pattern (L7/L8/L9/L6) | ✅ Already works |
| 5 | PC += 4 | Standard advance | ✅ Already works |
| 6 | MEM token | L14 STACK0 source | ❌ Blocked (same as JSR) |

**Status**: 3 of 6 operations already work neurally!

---

## Implementation Approach

### Phase 1: Register Copies (Steps 2-3)

Add to **L6 FFN** routing layer (around line 3900):

```python
# === ENT: BP → STACK0 copy (for push value) ===
# At BP marker when OP_ENT active: copy BP to STACK0
for k in range(16):
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5  # 2-way AND
    ffn.W_gate[unit, BD.OUTPUT_LO + k] = 1.0  # Read BP value
    ffn.W_down[BD.STACK0_LO + k, unit] = 2.0 / S  # Write to STACK0
    unit += 1

for k in range(16):
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.OUTPUT_HI + k] = 1.0
    ffn.W_down[BD.STACK0_HI + k, unit] = 2.0 / S
    unit += 1

# === ENT: SP → BP copy (set new frame base) ===
# At BP marker when OP_ENT active: also copy SP to BP OUTPUT
# (SP has been decremented by PSH pattern)
for k in range(16):
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
    ffn.W_gate[unit, BD.SP_LO + k] = 1.0  # Read SP value
    ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S  # Write to BP OUTPUT
    unit += 1

for k in range(16):
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0
    ffn.W_gate[unit, BD.SP_HI + k] = 1.0
    ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
    unit += 1
```

**Units needed**: 64 units (32 for BP→STACK0, 32 for SP→BP)

### Phase 2: SP Adjustment (Step 4)

ENT immediate is **local allocation size**, which is **subtracted** from SP.

The ADJ pattern we implemented handles: `SP = SP + signed_immediate`

For ENT, the immediate in the instruction is already the local size, and the operation is `SP -= imm`.

**Question**: Does the compiler encode ENT's immediate as negative, or do we need to negate it?

Let me check the compiler...

Actually, looking at the handler code:
```python
imm = instr >> 8
if imm >= 0x800000:
    imm -= 0x1000000
new_sp = (new_bp - imm) & 0xFFFFFFFF  # Subtracts imm
```

So the immediate is a **positive** value representing the size to allocate, and we **subtract** it.

**Options**:
1. **Use ADJ pattern with negation**: Negate the immediate before feeding to ADJ
2. **Separate ENT pattern**: Implement subtraction separately
3. **Compiler change**: Make ENT encode negative immediate

**Recommendation**: Option 1 - Use existing ADJ pattern but negate the immediate.

But wait - ENT is more complex. It's not just `SP -= imm`. The full sequence is:
1. SP -= 8 (push slot)
2. BP = SP (copy)
3. SP -= imm (allocate)

So we need TWO SP adjustments. The neural VM can only do one per step currently.

**Solution**: Break ENT into phases:
- **Phase A**: SP -= 8, BP = SP, STACK0 = old_bp
- **Phase B**: SP -= imm (using ADJ pattern on next micro-step? No, that won't work)

Actually, I think I'm overcomplicating this. Let me re-read the handler.

```python
push_addr = (old_sp - 8) & 0xFFFFFFFF
new_bp = push_addr
new_sp = (new_bp - imm) & 0xFFFFFFFF
```

This computes:
- new_bp = old_sp - 8
- new_sp = new_bp - imm = (old_sp - 8) - imm = old_sp - 8 - imm

So ENT does: **SP = SP - 8 - imm** in total.

We could do this as a single compound operation: **SP -= (8 + imm)**

Let me add units that:
1. Read FETCH (immediate value)
2. Add 8 to it
3. Subtract from SP (or equivalently, negate and add)

This would be an ENT-specific ALU operation, similar to ADJ but with an extra +8.

---

## Revised Implementation

### L8/L9 FFN: ENT SP Adjustment

Similar to ADJ, but computes: `SP = SP - (imm + 8)`

This is equivalent to: `SP = SP + (-(imm + 8))`

**Algorithm**:
1. L8: Compute `-(FETCH + 8)` for each byte (with borrow)
2. L9: Add to SP with multi-byte arithmetic

**Complexity**: High - need to handle signed negation plus constant addition.

**Alternative**: Simpler approach...

Actually, let me reconsider. The ADJ pattern computes `SP + signed_imm`. For ENT:
- The immediate is the **local size** (positive)
- We need `SP - 8 - local_size`
- If we treat this as `SP + (-8 - local_size)`, we can use the ADJ pattern

But the ADJ pattern expects the signed immediate to come from FETCH. For ENT, we'd need:
- FETCH value: local_size (from instruction)
- Adjustment: -(8 + FETCH)

This requires negating FETCH and subtracting 8, which is complex.

**Better approach**: Separate operations
1. **PSH-like operation**: SP -= 8 (already works)
2. **ADJ-like operation**: SP -= FETCH

But we can't do both in one step...

**Actual solution**: ENT combines two operations that normally happen separately:
- Push BP (like PSH)
- Allocate locals (like ADJ)

Since we can't do both in one neural step, we have two options:
1. **Keep ENT handler** (partial: only for complex multi-op)
2. **Multi-step ENT**: Break into 2 VM steps
3. **Combined ALU**: Special ENT ALU that does both

Let me check if there's a simpler interpretation...

---

## Pragmatic Approach

Looking at the operations:
1. SP -= 8 → Works (PSH)
2. STACK0 = BP → 32 units to add
3. BP = SP → 32 units to add
4. SP -= imm → **Keep handler for this**
5. PC advance → Works
6. MEM token → **Blocked (same as JSR)**

**Recommendation**: Implement steps 2-3 neurally (64 units), keep handler for complex SP adjustment.

This gives us **partial neural ENT**:
- Register copies work neurally
- SP adjustments still use handler
- MEM token generation still uses handler

**Progress**: 2 of 6 operations neural → 4 of 6 still need handler.

Not a huge win. Let me reconsider...

---

## Alternative: Full Neural ENT with Combined Adjustment

### Key Insight

ENT's `SP -= (8 + imm)` can be implemented as a special case of the ADJ pattern.

**Implementation**:
1. L8 FFN: For each lo nibble, compute `SP_lo - FETCH_lo - 8` with borrow
2. L9 FFN: For each hi nibble, propagate borrow and compute `SP_hi - FETCH_hi - borrow`

This is essentially **multi-byte subtraction with constant offset**.

**Units required**: ~1500 units (similar to ADJ but with constant offset)

---

## Recommended Implementation

### Option A: Partial Neural (Simple)
- Add 64 units for register copies (BP→STACK0, SP→BP)
- Keep handler for SP adjustment and MEM token
- **Effort**: 2-3 hours
- **Benefit**: Minimal handler complexity reduction

### Option B: Full Neural (Complex)
- Add 64 units for register copies
- Add 1500 units for ENT-specific SP adjustment
- Still blocked on MEM token (L14 issue)
- **Effort**: 8-12 hours
- **Benefit**: ENT mostly neural (except MEM token)

### Option C: Defer Until L14 Fix
- Fix L14 MEM token generation first (JSR + ENT both need this)
- Then implement full ENT neurally
- **Effort**: 15-20 hours total (L14 + ENT)
- **Benefit**: Complete solution for both JSR and ENT

---

## Recommendation: Option C (L14 First)

**Rationale**:
1. JSR and ENT both blocked on MEM token generation
2. Fixing L14 unblocks both operations
3. More efficient to fix root cause than work around it
4. L14 fix also needed for other future operations

**Next Steps**:
1. Analyze L14 attention mechanism
2. Understand why it copies from AX instead of STACK0
3. Modify L14 to conditionally copy from STACK0 when JSR/ENT active
4. Test JSR and ENT
5. Then implement remaining ENT operations

---

## Conclusion

**Current Blocker**: MEM token generation (L14 attention)

**Path Forward**:
1. Implement L14 fix for MEM token (5-8 hours)
2. Implement ENT register copies (2-3 hours)
3. Implement ENT SP adjustment (3-5 hours)
4. Test full ENT (1-2 hours)

**Total**: 11-18 hours for complete ENT

**Alternative**: Implement partial ENT now (2-3 hours) for register copies only.

---

**Status**: Blocked on L14 MEM token generation
**Recommended**: Fix L14 first, then return to ENT
**Next**: Analyze L14 attention mechanism
