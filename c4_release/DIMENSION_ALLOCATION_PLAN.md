# Dimension Allocation Plan for Pure Autoregressive Mode

**Date**: 2026-04-09
**Status**: Dimensions Available - Implementation Ready

---

## Executive Summary

**Previous Assessment**: Only 6 dimensions available [298-303]
**Corrected Assessment**: **79 dimensions available** across 3 reserved blocks!

This enables full implementation of remaining handlers (ADJ, ENT, LEV, memory syscalls) without model retraining.

---

## Available Dimension Blocks

### Block 1: RESERVED_297_327 (Primary)
**Location**: dims 297-327
**Size**: 31 dimensions
**Status**: Fully available (marked "Reserved (future STACK0/IO)")
**Allocated To**:
- ADJ implementation: dims 297-314 (18 dims)
- ENT/LEV staging: dims 315-327 (13 dims)

### Block 2: RESERVED_400_415
**Location**: dims 400-415
**Size**: 16 dimensions
**Status**: Fully available (marked "Reserved (future PC binary encoding/IO)")
**Allocated To**:
- Memory syscall state: dims 400-415 (16 dims)

### Block 3: TEMP
**Location**: dims 480-511
**Size**: 32 dimensions
**Status**: Available when conversational_io=False
**Allocated To**:
- MCMP buffers: dims 480-511 (32 dims)

---

## Allocation Strategy

### Priority 1: ADJ (Stack Adjustment)

**Dimensions Needed**: 18 dims
**Allocation**: dims 297-314

**Breakdown**:
- SP_OLD_LO [297-304]: 8 dims (SP old value, low nibbles, 4 bytes)
- SP_OLD_HI [305-312]: 8 dims (SP old value, high nibbles, 4 bytes)
- CARRY_BITS [313-314]: 2 dims (multi-byte carry propagation)

**Implementation Pattern**: Copy LEA (BP + signed_imm) but use SP
- L7 attention: Gather SP → ALU dims
- L8-L9 FFN: 4-byte signed addition with carry
- L6 FFN: Write result to OUTPUT at SP marker

**Estimated Effort**: 10-15 hours

---

### Priority 2: ENT/LEV State

**Dimensions Needed**: 13 dims
**Allocation**: dims 315-327

**ENT Requirements** (8 dims):
- OLD_BP_LO [315-318]: 4 dims (saved BP low nibbles, 2 bytes)
- OLD_BP_HI [319-322]: 4 dims (saved BP high nibbles, 2 bytes)

**LEV Requirements** (5 dims):
- RETURN_ADDR_TEMP [323-327]: 5 dims (loaded return address staging)

**Note**: ENT depends on ADJ (step 4: SP -= imm local allocation)

**Estimated Effort**: 15-25 hours (after ADJ)

---

### Priority 3: Memory Syscalls

**Dimensions Needed**: 47 dims total
**Allocation**:
- Block 2 (16 dims): HEAP_PTR + MALC state
- Block 3 (32 dims): MCMP buffers

**MALC (Memory Allocation)**:
- HEAP_PTR [400-403]: 4 dims (persistent heap pointer, 4 bytes)
- MALC_SIZE_ALIGNED [404-407]: 4 dims (aligned allocation size)

**FREE (Memory Deallocation)**:
- FREE_MARKER [408]: 1 dim (mark freed regions)

**MSET (Memory Fill)**:
- MSET_ADDR [409-412]: 4 dims (fill start address)
- MSET_VAL [413]: 1 dim (fill value byte)
- MSET_SIZE [414-415]: 2 dims (fill size, up to 64KB)

**MCMP (Memory Compare)**:
- MCMP_BUF_A [480-495]: 16 dims (buffer A, 16 bytes)
- MCMP_BUF_B [496-511]: 16 dims (buffer B, 16 bytes)

**Estimated Effort**: 50-80 hours (complex operations)

---

## Implementation Sequence

### Phase 1: ADJ Implementation (Immediate)

**Goal**: Remove ADJ handler, enable neural SP += imm

**Steps**:
1. Allocate SP_OLD_LO/HI, CARRY_BITS in dim_registry.py
2. Copy LEA L7 attention pattern (BP gather → ALU)
3. Modify to gather SP instead of BP
4. Copy LEA L8-L9 FFN addition circuit
5. Implement 4-byte signed addition with carry propagation
6. L6 FFN: Route result to SP OUTPUT

**Test Program**:
```c
int main() {
    int a, b, c;  // ADJ -12 (allocate 3 ints)
    a = 1;
    b = 2;
    c = 3;
    return a + b + c;  // ADJ +12 (deallocate), should return 6
}
```

**Success Criteria**: All 1096 tests pass with ADJ handler removed

---

### Phase 2: ENT Partial Implementation

**Goal**: Remove ENT handler partially (depends on ADJ)

**ENT Operations**:
1. SP -= 8 (push slot) → Already works (PSH pattern)
2. Memory write (old BP) → Needs MEM token (L14 modification)
3. BP = SP (frame base) → Add L6 FFN relay
4. SP -= imm (local allocation) → **Uses ADJ** (now available!)

**With ADJ working**: Steps 1, 3, 4 can be neural. Step 2 needs L14 fix (same as JSR).

**Success Criteria**: ENT works with handler only for MEM token generation

---

### Phase 3: LEV Multi-Step Decomposition

**Goal**: Remove LEV handler (most complex)

**Challenge**: LEV needs 3 memory lookups + 4 register updates in one VM step

**Options**:

**Option A: Multi-step execution**
- Break LEV into 3-4 VM steps
- Step 1: Load saved_bp from memory[BP]
- Step 2: Load return_addr from memory[BP+8]
- Step 3: Compute new_sp = BP + 16
- Step 4: Update registers (SP, BP, PC, STACK0)

**Option B: Parallel L15 lookups**
- Use multiple L15 attention heads for parallel loads
- Head 0: Load memory[BP] → saved_bp
- Head 1: Load memory[BP+8] → return_addr
- Combine results in L16 FFN

**Recommendation**: Start with Option B (parallel lookups)

**Success Criteria**: Recursive functions work neurally

---

### Phase 4: Memory Syscalls

**Goal**: Remove all memory syscall handlers

**MALC**:
- L11 FFN: Detect OP_MALC, compute aligned size
- Read HEAP_PTR from dims[400-403]
- Write result to AX (allocated address)
- Increment HEAP_PTR by aligned_size

**FREE**:
- L11 FFN: Detect OP_FREE
- Inject FREE_MARKER token into context
- L15 attention: Suppress matches in freed ranges

**MSET**:
- L12 FFN: Detect OP_MSET
- Extract addr, val, size from stack
- Generate MEM_FILL metadata tokens
- L15: Range check on lookups, return fill value if in range

**MCMP**:
- L15: Parallel lookups for up to 16 bytes from each buffer
- L12 FFN: Byte-by-byte comparison
- Return difference on first mismatch

**Success Criteria**: malloc/free/memset/memcmp work without handlers (up to size limits)

---

## Risk Mitigation

### Risk 1: Dimension Interference

**Concern**: New dimensions might interfere with existing logic

**Mitigation**:
- Test after each allocation
- Use different blocks for independent operations
- Verify no unexpected cross-talk in attention

### Risk 2: Performance Regression

**Concern**: Additional FFN units slow forward pass

**Mitigation**:
- Use MoE gating where possible (opcode-specific)
- Measure tests/sec after each phase
- Target: >7500 tests/sec (85% of current 8581)

### Risk 3: Complexity Creep

**Concern**: Implementations become overly complex

**Mitigation**:
- Follow existing patterns (LEA for ADJ, PSH for ENT)
- Incremental testing at each step
- Rollback option if tests fail

---

## Success Metrics

### Phase 1 Complete (ADJ)
- ✅ ADJ handler removed
- ✅ All 1096 tests passing
- ✅ ADJ test program returns 6
- ✅ Tests/sec >= 8000

### Phase 2 Complete (ENT Partial)
- ✅ ENT handler simplified (only MEM tokens)
- ✅ All function frame tests passing
- ✅ Recursive Fibonacci works

### Phase 3 Complete (LEV)
- ✅ LEV handler removed
- ✅ Deep recursion works (10+ levels)
- ✅ All 1096 tests passing

### Phase 4 Complete (Memory Syscalls)
- ✅ MALC/FREE/MSET/MCMP handlers removed
- ✅ Dynamic allocation tests pass
- ✅ Memory utility tests pass
- ✅ 100% pure autoregressive execution achieved

---

## Timeline Estimate

| Phase | Task | Effort | Running Total |
|-------|------|--------|---------------|
| 1 | ADJ implementation | 10-15 hrs | 15 hrs |
| 1 | ADJ testing & debug | 3-5 hrs | 20 hrs |
| 2 | ENT partial implementation | 8-12 hrs | 32 hrs |
| 2 | ENT testing | 3-5 hrs | 37 hrs |
| 3 | LEV multi-step design | 10-15 hrs | 52 hrs |
| 3 | LEV implementation | 15-20 hrs | 72 hrs |
| 3 | LEV testing | 5-8 hrs | 80 hrs |
| 4 | MALC/FREE | 15-20 hrs | 100 hrs |
| 4 | MSET/MCMP | 20-30 hrs | 130 hrs |
| 4 | Memory syscalls testing | 10-15 hrs | 145 hrs |
| - | Documentation | 5-10 hrs | 155 hrs |
| **Total** | | **~155 hours** | **~4 weeks full-time** |

---

## Recommendation

### Immediate Action: Implement ADJ

**Rationale**:
1. Dimensions are available (31 free, need 18)
2. Pattern exists (LEA) - copy and adapt
3. Unblocks ENT (step 4: SP -= imm)
4. High impact (removes 1 handler, enables 1 more)
5. Low risk (well-understood operation)

**Start Now**: Allocate dims 297-314, begin ADJ implementation following LEA pattern

### Medium Term: Complete Function Call Support

**After ADJ**: Implement ENT/LEV to achieve neural function calls

**Impact**: Removes 3/5 remaining handlers (ADJ, ENT, LEV)

### Long Term: Full Pure Autoregressive

**After ENT/LEV**: Implement memory syscalls

**Impact**: Removes all 5 remaining handlers, achieves 100% pure autoregressive execution

---

## Conclusion

**Previous blocker**: "Only 6 dimensions available, cannot implement remaining operations"

**Corrected reality**: **79 dimensions available** - full pure autoregressive mode is achievable!

**Next step**: Allocate ADJ dimensions and begin implementation. The path to 100% autoregressive execution is clear.

---

**Report by**: Claude Sonnet 4.5
**Date**: 2026-04-09
**Status**: READY TO IMPLEMENT
