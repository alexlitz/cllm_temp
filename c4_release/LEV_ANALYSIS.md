# LEV (Leave Function) Analysis

**Date**: 2026-04-09
**Status**: Handler Analysis - Planning Neural Implementation
**Current**: 10% Neural (AX passthrough only)

---

## Current Handler Implementation

**File**: `neural_vm/run_vm.py` (lines 1540-1560)

```python
def _handler_lev(self, context, output):
    """LEV -- restore frame, return.

    sp = bp;  bp = *sp++;  pc = *sp++;
    So: saved_bp = mem[old_bp], return_addr = mem[old_bp + 8]
    new_sp = old_bp + 16, new_bp = saved_bp, pc = return_addr
    """
    old_bp = self._last_bp
    saved_bp = self._mem_load_word(old_bp)          # Memory read 1
    return_addr = self._mem_load_word(old_bp + 8)   # Memory read 2
    new_sp = (old_bp + 16) & 0xFFFFFFFF

    # Override all registers
    self._override_register_in_last_step(context, Token.REG_SP, new_sp)
    self._override_register_in_last_step(context, Token.REG_BP, saved_bp)
    self._override_register_in_last_step(context, Token.REG_PC, return_addr)
    # STACK0 = *new_sp (top of stack after restoring frame)
    stack0_val = self._mem_load_word(new_sp)        # Memory read 3
    self._override_register_in_last_step(context, Token.STACK0, stack0_val)
```

---

## Operation Breakdown

| # | Operation | Complexity | Neural Feasibility |
|---|-----------|------------|-------------------|
| 1 | saved_bp = mem[BP] | Memory read | ⏳ Requires L15 extension |
| 2 | return_addr = mem[BP+8] | Memory read | ⏳ Requires L15 extension |
| 3 | new_sp = BP + 16 | Arithmetic | ✅ Simple (L6 FFN) |
| 4 | BP = saved_bp | Assignment | ⏳ Depends on #1 |
| 5 | PC = return_addr | Assignment | ⏳ Depends on #2 |
| 6 | STACK0 = mem[new_sp] | Memory read | ⏳ Depends on #3 |

**Dependency Chain**:
- #3 independent
- #1, #2 can be parallel
- #4 depends on #1
- #5 depends on #2
- #6 depends on #3

---

## Current Neural Implementation

### AX Passthrough (L6 FFN: lines 6850-6864)

**Units**: 32 (16 lo + 16 hi)

```python
# LEV AX passthrough (preserve return value)
for k in range(16):
    ffn.W_up[unit, BD.OP_LEV] = S
    ffn.W_up[unit, BD.MARK_AX] = S
    ffn.b_up[unit] = -S * T
    ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
    ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
```

**Status**: ✅ Working (AX preserved during function return)

---

## Challenges for Full Neural Implementation

### Challenge 1: Multiple Memory Reads

LEV requires **3 memory reads**:
1. `saved_bp` from `memory[BP]`
2. `return_addr` from `memory[BP+8]`
3. `stack0_val` from `memory[new_sp]`

**Current L15** (lines 5705-5848):
- 4 heads (one per output byte)
- Supports 2 address sources: AX (for LI/LC) and SP (for STACK0)
- Loads **one 4-byte value** per operation

**Problem**: LEV needs to load **2-3 values** simultaneously.

**Solutions**:

**A. Multi-Step LEV** (2-3 VM steps):
- Step 1: Load saved_bp via extended L15 at BP
- Step 2: Load return_addr via extended L15 at BP+8
- Step 3: Load stack0_val via L15 at new_sp
- Each step partial register updates

**Pros**:
- Reuses existing L15 infrastructure
- Clear separation of concerns

**Cons**:
- LEV becomes 2-3 instructions instead of 1
- Breaks C4 instruction semantics
- Requires intermediate state storage

**B. Parallel L15 Lookups** (8 heads instead of 4):
- Heads 0-3: Load saved_bp from memory[BP]
- Heads 4-7: Load return_addr from memory[BP+8]
- Both happen in same transformer step

**Pros**:
- Single-step LEV (maintains semantics)
- True parallel execution

**Cons**:
- Requires doubling L15 heads (4 → 8)
- Head dimension increase (64 → 128)
- Architectural change

**C. Sequential L15 with State** (use TEMP dims):
- L15 loads saved_bp → TEMP[0-31]
- L16 loads return_addr → TEMP[32-63]
- L17 assembles final output

**Pros**:
- Doesn't change L15 head count
- Uses existing TEMP storage

**Cons**:
- Requires 2 new layers (L16, L17)
- Complex coordination
- Increases model depth

**D. Hybrid: Neural arithmetic + Handler memory** (RECOMMENDED):
- Neural: SP = BP + 16 (simple L6 FFN)
- Handler: Memory reads + register assignment
- Similar to ENT approach

**Pros**:
- Simple, incremental progress
- Doesn't require L15 changes
- ~17% neural (1 of 6 operations)

**Cons**:
- Still mostly handler-based
- Limited neural execution

---

### Challenge 2: Multiple Simultaneous Register Updates

LEV updates **4 registers** in one step:
- SP = new_sp
- BP = saved_bp
- PC = return_addr
- STACK0 = stack0_val

**Current Model**: Register updates happen at specific marker positions:
- L6 FFN writes to OUTPUT at PC marker
- L6 FFN writes to OUTPUT at SP marker
- L6 FFN writes to OUTPUT at BP marker
- L6 FFN writes to OUTPUT at STACK0 marker

**Problem**: All 4 updates need values from memory lookups, which aren't available until L15.

**Solution**: Layer ordering
- L15: Memory lookups → results in TEMP or AX_CARRY
- L16: Final routing to OUTPUT at all 4 marker positions

But this requires coordination that doesn't currently exist.

---

## Feasibility Analysis

### Option 1: Minimal Neural LEV (~17% neural)

**Implementation**: Add SP arithmetic only

**L6 FFN** (32 new units):
```python
# LEV: SP = BP + 16
T_lev = 1.5
for k in range(16):
    new_k = (k + 16) % 16  # Add 16 to lo nibble
    ffn.W_up[unit, BD.OP_LEV] = S
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.b_up[unit] = -S * T_lev
    ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
    # Read BP value (need to relay BP → SP somehow)
    # Actually this is complex because we need BP's value at SP marker
    # Would need another relay head...
```

**Problem**: Even SP = BP + 16 requires BP's value at SP marker, which needs a relay.

**Conclusion**: Not worth the complexity for just one operation.

---

### Option 2: Full Neural LEV (~90% neural)

**Implementation**: Multi-step with extended L15

**Step 1 - Load saved_bp**:
1. Extend L15 to activate at BP marker (like STACK0)
2. Load memory[BP] → 4 bytes to TEMP[0-31]
3. SP unchanged, PC unchanged

**Step 2 - Load return_addr and finalize**:
1. Extend L15 to load from BP+8
2. Load memory[BP+8] → 4 bytes to TEMP[32-63]
3. L6 FFN: SP = BP + 16 (from TEMP or original BP)
4. L6 FFN: BP = TEMP[0-31] (saved_bp)
5. L6 FFN: PC = TEMP[32-63] (return_addr)
6. L15: Load memory[new_sp] → STACK0

**Complexity**:
- Extend L15 BP support: 4-6 hours
- Multi-step coordination: 6-10 hours
- Register routing: 4-6 hours
- **Total**: 14-22 hours

**Benefit**: 90% neural LEV

---

### Option 3: Keep Handler (0% change)

**Implementation**: None

**Rationale**:
- LEV is rare (only at function returns)
- Complexity vs benefit doesn't justify effort
- JSR (100% neural) and ENT (80% neural) cover most function call overhead
- LEV handler works correctly

**Effort**: 0 hours

**Benefit**: Focus on other improvements

---

## Recommendation

### Phase 1: Keep Handler (Current Session)

**Action**: Document current state, no implementation changes.

**Rationale**:
- JSR and ENT already ~90% neural
- LEV is rare operation
- Handler is small and correct
- Better to focus on:
  - Testing JSR/ENT neural implementations
  - Improving ENT to 100% (ENT-specific ALU for SP adjustment)
  - Other high-value improvements

### Phase 2: Full Neural LEV (Future Work)

**Prerequisites**:
1. JSR and ENT tested and stable
2. L15 extension designed and proven
3. Multi-step execution framework in place

**Estimated Effort**: 14-22 hours

**Implementation Plan**:
1. Design multi-step LEV semantics
2. Extend L15 for BP-based memory lookups
3. Implement state passing via TEMP dims
4. Add register routing in L6/L16
5. Test with complex recursive programs
6. Remove handler

---

## Current Status Summary

| Component | Status | Neural % | Handler Lines |
|-----------|--------|----------|---------------|
| AX passthrough | ✅ Neural | 100% | 0 |
| SP arithmetic | ❌ Handler | 0% | 1 |
| Memory reads (3×) | ❌ Handler | 0% | 3 |
| Register updates (4×) | ❌ Handler | 0% | 4 |
| **Total** | **Hybrid** | **~10%** | **8** |

---

## Comparison with JSR and ENT

| Operation | Neural % | Handler Lines | Complexity |
|-----------|----------|---------------|------------|
| JSR | 100% | 0 | Medium (1 memory write) |
| ENT | 80% | 13 | Medium (1 memory write, SP arithmetic) |
| LEV | 10% | 8 | **High (3 memory reads, 4 register updates)** |

**Key Difference**: LEV has multiple **reads** (not writes) and complex **dependencies**.

---

## Future Neural LEV Design (Outline)

### Architecture Changes Required

**1. L15 Extension - BP Memory Lookup**

Add BP marker support similar to STACK0:

```python
# L15 Query modification
attn.W_q[base, BD.MARK_BP] = 2000.0  # Activate at BP marker

# Address encoding: use BP value (from previous step)
# Load from address = BP (saved_bp) or BP+8 (return_addr)
```

**2. Multi-Head Lookup**

Use all 8 heads in L15:
- Heads 0-3: Load saved_bp from memory[BP]
- Heads 4-7: Load return_addr from memory[BP+8]

**3. State Passing**

Store loaded values in TEMP:
- TEMP[0-31]: saved_bp
- TEMP[32-63]: return_addr

**4. L16/L17 Register Assembly**

New layers to route TEMP → OUTPUT at multiple markers:
- SP marker: BP + 16
- BP marker: TEMP[0-31] (saved_bp)
- PC marker: TEMP[32-63] (return_addr)
- STACK0 marker: (requires another L15 lookup)

**Estimated New Code**:
- L15 extension: 100-150 lines
- L16/L17 layers: 200-300 lines
- Testing: 100+ lines
- **Total**: ~500 lines

---

## Testing Requirements

For full neural LEV, need to test:

**1. Simple Return**
```c
int helper() { return 42; }
int main() { return helper(); }
```

**2. Nested Calls**
```c
int c() { return 42; }
int b() { return c(); }
int a() { return b(); }
int main() { return a(); }
```

**3. With Locals**
```c
int func(int x) {
    int local = x * 2;
    return local;
}
int main() { return func(21); }
```

**4. Recursive**
```c
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}
int main() { return fib(7); }
```

---

## Conclusion

**Current Approach**: Keep LEV handler as-is.

**Rationale**:
- LEV complexity (3 memory reads, 4 register updates) requires significant architectural changes
- JSR (100% neural) + ENT (80% neural) already provide ~90% neural function calls
- Better to focus on testing and stability
- LEV handler is small (8 lines) and correct

**Future Work**: Full neural LEV is achievable but requires:
- L15 extension for BP lookups
- Multi-head parallel memory reads
- State passing infrastructure
- New layers for register assembly
- Estimated 14-22 hours effort

**Next Priority**: Test JSR/ENT, then consider ENT 100% neural (3-5 hours) before LEV.

---

**Date**: 2026-04-09
**Author**: Claude Sonnet 4.5
**Status**: Analysis Complete - Handler Retained
