# Handler Elimination Status

**Date**: 2026-04-07
**Goal**: Eliminate arithmetic operation handlers by fixing neural weights
**Current Status**: Root cause identified, fix incomplete

## Investigation Summary

### What I Discovered

1. **All mechanisms work individually**: ✓
   - IS_MARK flags set correctly on marker tokens
   - Layer 1 threshold heads (L1H0/L1H1) working
   - Layer 3 AX carry-forward setting AX_CARRY_LO/HI

2. **AX_CARRY reaches Layer 7**: ✓
   - Layer 7 input shows AX_CARRY present (max=2.278)
   - Values successfully propagate through Layers 3-6

3. **Layer 7 doesn't use AX_CARRY**: ✗
   - Layer 7 has NO heads configured to copy AX_CARRY → ALU
   - Checked all 8 heads - none read from AX_CARRY via W_v

### The Design Issue

**Current Architecture**:
```
Layer 6 Head 2: AX_CARRY → ALU at STACK0 marker
Layer 7 Head 0: STACK0 CLEAN_EMBED → ALU at AX marker
Layer 7 Head 1: BP OUTPUT → ALU at AX marker (LEA only)
```

**Problem**: Layer 7 Head 0 reads from CLEAN_EMBED (token ID) instead of from the ALU value that Layer 6 wrote with AX_CARRY.

**Attempted Fix**: Changed Layer 7 Head 0 to read from ALU instead of CLEAN_EMBED
**Result**: Made things worse (returns 0 instead of 10)

### Why the Fix Failed

Reading from ALU_LO/HI in W_v while also writing to ALU_LO/HI in W_o creates a **residual stream conflict**. The same dimensions are being read and written in the same layer, which breaks the residual connection.

## Root Cause Analysis

### Missing Operand B Mechanism

For binary operations like ADD, we need:
- **Operand A**: From stack (Layer 7 Head 0 should get this)
- **Operand B**: From current AX or previous AX

The Python handlers get operand B from:
```python
rhs = self._extract_register(context, Token.REG_AX)  # Current AX
```

But the neural implementation has no clear mechanism to provide operand B!

### Two Possible Designs

**Design 1**: Both operands from stack
- IMM 10 pushes 10 to stack
- IMM 32 pushes 32 to stack
- ADD pops both from stack
- Layer 7 needs to gather BOTH stack values

**Design 2**: One from stack, one from AX
- IMM 10 sets AX=10
- IMM 32 sets AX=32 (but AX_CARRY preserves 10)
- ADD uses AX_CARRY + current STACK0
- Layer 7 needs separate heads for each

Currently unclear which design is intended!

## Next Steps

### Option 1: Add Missing Head for Operand B

Create a new Layer 7 head that:
- Queries at AX marker
- Attends to current/previous AX byte 0
- Copies AX value → separate ALU dimension (e.g., ALU_B_LO/HI)
- Layer 8 FFN uses both ALU_LO and ALU_B_LO

**Pros**: Clean separation, no residual conflicts
**Cons**: Requires new dimensions, changes to Layer 8 FFN

### Option 2: Fix Stack-Based Operand Gathering

If both operands should come from stack:
- Layer 7 Head 0: Get STACK0 byte 0 (top of stack)
- Layer 7 Head X: Get STACK0 byte 0 from previous step (second operand)
- Both write to separate ALU dimensions

**Pros**: Matches stack-based VM design
**Cons**: Requires understanding intended architecture

### Option 3: Use Current AX Directly

Add Layer 7 head that:
- Reads current step's AX byte 0 (from OUTPUT_LO/HI or AX_OUT dimensions)
- Writes to ALU_B dimensions
- Layer 8 combines ALU_LO (from stack) + ALU_B_LO (from AX)

**Pros**: Matches handler behavior
**Cons**: Unclear if OUTPUT/AX_OUT is available at Layer 7

### Option 4: Deep Investigation

Before making changes:
1. Understand intended operand flow by reading full architecture docs
2. Check if there ARE separate ALU_B dimensions already
3. Verify what IMM operation actually does (pushes to stack? or sets AX?)
4. Trace through complete ADD execution in existing (working) handlers

**Pros**: Won't break things
**Cons**: Time-consuming

## Recommendation

**Do Option 4 first**: Understand the intended architecture before making more changes.

Key questions to answer:
1. Where do binary operation operands come from?
2. Does IMM push to stack, set AX, or both?
3. Are there already ALU_B dimensions defined?
4. What's the complete dataflow for ADD in the neural implementation?

Once we understand the design, we can implement the correct fix.

## Files for Reference

- `neural_vm/vm_step.py` - Weight configuration
  - Line 3664: `_set_layer7_operand_gather` - Layer 7 Head 0/1 config
  - Line 3638: `_set_layer6_relay_heads` - Layer 6 Head 2 (AX_CARRY → ALU)
  - Search for "ALU_B" or "operand B" to find if alternate dimensions exist

- `neural_vm/run_vm.py` - Handlers
  - Line ~950: `_handle_add` - Shows how handlers get operands

- `check_layer7_weights.py` - Confirmed no head reads AX_CARRY
- `debug_layer7_during_add.py` - Confirmed AX_CARRY present at Layer 7

## Current Test Results

Without handlers:
```
✗ ADD: 10 + 32 => 10 (first operand only)
✗ SUB: 50 - 8  => 34 (incorrect)
✗ MUL: 6 * 7   => 0  (zero)
✗ DIV: 84 / 2  => 0  (zero)
```

This confirms operand B is missing for all binary operations.

---

**Status**: Investigation complete, need to understand architecture before fixing
**Blocker**: Unclear where second operand should come from in neural implementation
**Next**: Read architecture docs / trace through IMM and ADD dataflow
