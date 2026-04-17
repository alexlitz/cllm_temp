# Plan to Remove Remaining Handlers

## Current Status: 29/43 ops (67%) Fully Neural

After completing IMM neural implementation, here's what remains and how to neuralize it.

## Summary Table

| Category | Operations | Status | Difficulty | Path to Neural |
|----------|-----------|--------|------------|----------------|
| **Basic/ALU/Memory** | 29 ops | ✅ Neural | - | Complete |
| **Function Calls** | 3 ops | ⚠️ Handlers | Medium | Multi-byte stack ops |
| **Memory Syscalls** | 5 ops | ⚠️ Handlers | Hard | Complex, may stay |
| **I/O Boundary** | 6 ops | 🔌 External | - | Should stay external |

---

## 1. Function Call Operations (3 ops) - MEDIUM PRIORITY

### JSR (Jump Subroutine)

**Current Handler**: `_handler_jsr` (line 1451)
```python
def _handler_jsr(self, context, output):
    """JSR: push return address (PC+5), jump to target."""
    # Neural jump works, but handler ensures correct 32-bit PC push
    target = output['ax']
    return_addr = output['pc'] + 5  # PC after JSR instruction

    # Push 32-bit return address to stack
    sp = output['sp']
    for i in range(4):
        self._memory[sp + i] = (return_addr >> (i * 8)) & 0xFF

    return {'sp': sp - 8}  # Neural only handles byte 0
```

**Why Handler Needed**:
- Neural PSH works for single values
- JSR needs to push PC+5 (32-bit) as return address
- Current neural only handles byte 0 of stack operations
- Bytes 1-3 require multi-byte carry propagation

**Path to Neural**:
1. Extend Layer 6 FFN to handle multi-byte PSH
2. Add PC+5 computation in FFN (currently in handler)
3. Create 4-byte push sequence:
   - Byte 0: OUTPUT_LO → MEM
   - Bytes 1-3: Carry propagation similar to ADD/SUB
4. Use same amplification pattern as IMM (40x) for multi-byte values

**Complexity**: Medium
- Pattern exists (PSH works neurally for byte 0)
- Just need to extend to 4 bytes
- PC+5 arithmetic is simple (addition by constant)

### ENT (Enter Function)

**Current Handler**: `_handler_ent` (line 1488)
```python
def _handler_ent(self, context, output):
    """ENT: push BP, set BP=SP, allocate locals (SP -= N*8)."""
    imm = self._get_immediate(context)  # Local variable count

    # Push old BP (32-bit)
    sp = output['sp']
    old_bp = output['bp']
    for i in range(4):
        self._memory[sp + i] = (old_bp >> (i * 8)) & 0xFF
    sp -= 8

    # Set BP = old SP, then allocate locals
    bp = output['sp']
    sp -= imm * 8

    return {'sp': sp, 'bp': bp}
```

**Why Handler Needed**:
- Complex operation: push BP + set BP=SP + adjust SP
- Multi-byte BP value (32-bit)
- Variable stack allocation (imm * 8 bytes)

**Path to Neural**:
1. Break into sub-operations:
   - PSH BP (multi-byte push - same as JSR)
   - BP ← SP (register copy - can be FFN)
   - SP ← SP - (imm*8) (arithmetic - similar to ADJ)
2. Use multiple FFN layers or step-by-step execution
3. Intermediate TEMP storage for BP value

**Complexity**: Medium-Hard
- More complex than JSR (3 operations in one)
- May need multiple VM steps or complex FFN chain
- Variable allocation size adds complexity

### LEV (Leave Function)

**Current Handler**: `_handler_lev` (line 1525)
```python
def _handler_lev(self, context, output):
    """LEV: restore SP from BP, pop BP, return to caller."""
    sp = output['bp']  # SP ← BP

    # Pop old BP (32-bit)
    old_bp = 0
    for i in range(4):
        old_bp |= self._memory[sp + i] << (i * 8)
    sp += 8

    # Pop return address (32-bit) and jump
    ret_addr = 0
    for i in range(4):
        ret_addr |= self._memory[sp + i] << (i * 8)
    sp += 8

    return {'sp': sp, 'bp': old_bp, 'pc': ret_addr}
```

**Why Handler Needed**:
- Inverse of ENT: restore stack frame
- Multi-byte BP restore (32-bit load from stack)
- Multi-byte return address (32-bit load from stack)
- Multiple memory loads + register updates

**Path to Neural**:
1. SP ← BP (register copy - FFN)
2. Load BP from [SP] (4-byte load - use L15 memory attention 4 times)
3. Load return address from [SP+8] (4-byte load)
4. Update PC, SP, BP simultaneously

**Complexity**: Medium-Hard
- Similar to ENT but loads instead of stores
- L15 memory lookup already works for single bytes (LI/LC)
- Need to extend to multi-byte reads
- Multiple register updates in one step

---

## 2. Memory Operations (5 ops) - LOW PRIORITY (May Stay as Syscalls)

### ADJ (Adjust Stack)

**Current Handler**: `_syscall_adj` (syscall, not shown in handlers list)

**Why Handler Needed**:
- Adjusts SP by arbitrary amount: `SP += N`
- N is immediate value from instruction
- Multi-byte SP arithmetic

**Path to Neural**:
- **EASY** - Similar to ENT's stack allocation
- FFN can do: SP_new = SP_old + (imm * sign)
- Just needs multi-byte addition support
- **Could be neuralized with JSR/ENT/LEV work**

### MALC (Malloc), FREE, MSET, MCMP

**Current Handlers**: Various syscalls

**Why Handlers Needed**:
- **MALC**: Heap allocation requires bump allocator state
- **FREE**: Memory zeroing requires loop over allocation
- **MSET**: Fill memory region (loop operation)
- **MCMP**: Compare memory regions (loop operation)

**Path to Neural**:
- **VERY HARD** - These are complex system operations
- Require loop semantics (iterate over memory ranges)
- MALC requires persistent heap_ptr state
- May require recurrent processing or multi-step execution

**Recommendation**:
- **Keep as syscalls** even in pure mode
- These are VM "system calls" not VM instructions
- Similar to how real CPUs have syscall instructions
- Focus effort on JSR/ENT/LEV instead

---

## 3. I/O Operations (6 ops) - SHOULD STAY EXTERNAL

### PUTCHAR, GETCHAR, OPEN, READ, CLOS, PRTF

**Status**: Intentionally external

**Why External**:
- Interact with real world (files, console)
- Not part of VM execution
- Legitimate boundary between VM and environment

**Decision**: **Do NOT neuralize**
- These are working as designed
- External I/O is a feature, not a bug
- Focus on VM execution purity, not I/O purity

---

## Implementation Priority

### Phase 1: Multi-Byte Stack Operations (Enables 3 handlers)
**Target**: JSR, ENT, LEV
**Difficulty**: Medium
**Impact**: High - removes all function call handlers

**Steps**:
1. Extend Layer 6 PSH FFN to handle 4 bytes (not just byte 0)
2. Add multi-byte carry propagation (similar to ADD carry)
3. Test with JSR (simplest - just push PC+5)
4. Extend to ENT (push BP, copy register, adjust SP)
5. Extend to LEV (multi-byte loads, register updates)

**Pattern to Follow** (from IMM):
- Add flags to embeddings if needed
- Create attention relay paths for values
- Implement FFN routing with proper gating
- Amplify signals to overcome relay attenuation
- Test incrementally (JSR → ENT → LEV)

### Phase 2: Stack Adjustment (Enables 1 handler)
**Target**: ADJ
**Difficulty**: Easy
**Impact**: Low - rarely used

**Steps**:
1. Reuse multi-byte addition from Phase 1
2. Add FFN unit for SP += imm
3. Test with various stack adjustments

### Phase 3: Keep as Syscalls
**Target**: MALC, FREE, MSET, MCMP
**Decision**: Keep external
**Reason**: Complex operations, better as system calls

---

## Technical Approach for Multi-Byte Operations

### Strategy: Extend Existing Patterns

**Current**: PSH works neurally for byte 0
```
Layer 6 FFN:
  IF OP_PSH AND MARK_SP:
    OUTPUT_LO ← AX_LO (byte 0 only)
    SP_LO ← SP_LO - 8 (byte 0 only)
```

**Needed**: PSH for all 4 bytes
```
Layer 6 FFN:
  IF OP_PSH AND MARK_SP:
    For byte_idx in [0, 1, 2, 3]:
      MEM[SP + byte_idx] ← AX_BYTE[byte_idx]

Layer 7+ FFN:
  IF OP_PSH AND MARK_SP:
    Carry propagation for SP -= 8 across all bytes
    (Similar to SUB instruction carry)
```

### Challenges

1. **Memory Write Pattern**:
   - Current: Single byte OUTPUT → prediction
   - Needed: 4-byte write to MEM section
   - Solution: Generate 4 MEM tokens in sequence?

2. **Multi-Byte Arithmetic**:
   - Current: L8-L10 ALU handles multi-byte ADD/SUB
   - Needed: Same for SP operations
   - Solution: Reuse ALU patterns for stack pointer

3. **Register Updates**:
   - Current: One register per step (AX byte 0)
   - Needed: Multiple registers (SP, BP, PC)
   - Solution: May need multiple steps or complex FFN

### Concrete Implementation for JSR

**Goal**: `JSR target` pushes PC+5 to stack, jumps to target

**Step 1**: Compute PC+5
- Layer 5/6 FFN: PC_plus_5 = PC + 5
- Use TEMP dims for intermediate value
- Similar to PC increment (already works)

**Step 2**: Push PC+5 (4 bytes)
- Layer 6 FFN: Route PC_plus_5 → MEM writes
- Generate 4 MEM tokens with correct bytes
- Use same pattern as PSH but with PC_plus_5 instead of AX

**Step 3**: Update SP
- Layer 6/7 FFN: SP = SP - 8
- Multi-byte subtraction (reuse SUB carry logic)

**Step 4**: Jump to target
- Already works (JMP is neural)
- PC ← AX (target address)

**Estimate**: 2-3 days of work
- Day 1: Multi-byte PSH extension
- Day 2: PC+5 computation + integration
- Day 3: Testing and debugging

---

## Expected Outcome

After Phase 1 completion:
- **32/43 ops (74%) fully neural**
- All function call handlers removed
- Clean function call semantics
- Path clear for remaining operations

After Phase 2 completion:
- **33/43 ops (77%) fully neural**
- Only syscalls and I/O remain as handlers
- ~80% neural execution achieved

**Final State** (intended):
- **33/43 ops neural** (77%)
- 5 memory syscalls (MALC/FREE/MSET/MCMP/ADJ if not done)
- 6 I/O boundaries (intentional)
- **All VM execution logic is neural**

---

## Key Insight from IMM Implementation

The IMM neural implementation taught us:

1. **Signal Amplification**: When relaying through attention heads, weaker signals need amplification (40x for FETCH)

2. **Multi-Value Relay**: Relaying many values through one head causes attenuation (37x for FETCH vs 17x for OP_IMM)

3. **Incremental Testing**: Test each component (embedding → relay → routing) separately

4. **Position Matters**: Wrong marker position (REG_BP vs REG_AX) can cause mysterious failures

**Apply to JSR/ENT/LEV**:
- Expect signal attenuation for multi-byte values
- Use amplification proactively (40x for byte values)
- Test PC+5 computation separately from stack push
- Verify marker positions for SP, BP registers

---

## Files to Update

### For JSR/ENT/LEV Neural Implementation:

1. **`neural_vm/vm_step.py`**:
   - Layer 6 FFN: Extend PSH routing to 4 bytes
   - Layer 6/7 FFN: Add PC+5 computation
   - Layer 7+ FFN: Multi-byte SP carry propagation
   - May need new attention heads for multi-byte relay

2. **`neural_vm/run_vm.py`**:
   - Remove JSR/ENT/LEV from `_func_call_handlers`
   - Update comments to mark them as neural
   - Keep in mind multi-byte correctness

3. **Test files**:
   - Create `test_neural_jsr.py`
   - Create `test_neural_ent_lev.py`
   - Verify function calls work without handlers

### For Documentation:

1. **`HANDLER_STATUS.md`** - Update as handlers removed
2. **`REMAINING_HANDLERS_PLAN.md`** - This file, track progress
3. **New**: `MULTI_BYTE_NEURAL_IMPLEMENTATION.md` - Document patterns

---

## Conclusion

**Immediate Next Steps**:
1. ✅ IMM is complete - celebrate!
2. 🎯 **Next target: JSR** (simplest multi-byte operation)
3. 📋 Document multi-byte patterns as you implement
4. 🧪 Test incrementally (one operation at a time)

**Long-term Vision**:
- All VM execution logic neural (~80%)
- Only syscalls and I/O use Python
- Clean separation: VM execution (neural) vs OS services (Python)

The Neural VM is already a remarkable achievement. Each handler removed brings us closer to a truly neural computer!
