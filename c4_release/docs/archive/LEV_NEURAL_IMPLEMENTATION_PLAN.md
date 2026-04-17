# LEV Neural Implementation Plan - April 9, 2026

## Executive Summary

**Goal**: Complete LEV neural implementation to achieve ~99% neural VM

**Current Status**: LEV is ~10% neural (only AX passthrough implemented)

**Required Work**:
1. Extend L15 from 4 → 12 heads (3 parallel memory reads)
2. Add L16 routing layer (~600 FFN units)
3. Add BP address relay support
4. Test and validate implementation

**Estimated Time**: 18-24 hours
**Complexity**: High (architectural changes)
**Risk**: Medium (requires careful attention to attention score budgets)

---

## Current LEV Handler Behavior

The LEV (leave function) operation performs the following:

```python
# From run_vm.py:_handler_lev (lines ~1540-1560)
def _handler_lev(self, state):
    # 1. Read saved_bp from memory[BP]
    saved_bp = self.shadow_mem.get(state.bp, 0)

    # 2. Read return_addr from memory[BP+8]
    return_addr = self.shadow_mem.get(state.bp + 8, 0)

    # 3. Compute new_sp = BP + 16
    new_sp = state.bp + 16

    # 4. Read stack0_val from memory[new_sp]
    stack0_val = self.shadow_mem.get(new_sp, 0)

    # 5. Update registers
    state.sp = new_sp
    state.bp = saved_bp
    state.pc = return_addr
    state.stack0 = stack0_val

    # 6. AX passthrough (return value preserved)
    # state.ax = state.ax  # No change
```

**Neural Status**:
- ✅ AX passthrough: L6 FFN (identity carry)
- ❌ Memory reads (3): Need L15 extension
- ❌ Register routing (4): Need L16 layer
- ❌ BP address encoding: Need relay support

---

## Architecture Design

### Challenge: 3 Parallel Memory Reads

LEV requires **3 memory reads in one step**:
1. `saved_bp = mem[BP]` - 4 bytes
2. `return_addr = mem[BP+8]` - 4 bytes
3. `stack0_val = mem[new_sp]` where `new_sp = BP + 16` - 4 bytes

**Current L15**: 4 heads (1 read × 4 bytes)
**Required L15**: 12 heads (3 reads × 4 bytes)

### Solution: Extend L15 to 12 Heads

```
L15 Extension: 12 heads total
  Heads 0-3: Existing functionality (LI/LC at AX, *SP at STACK0)
  Heads 4-7: LEV read #1 - saved_bp from mem[BP]
  Heads 8-11: LEV read #2 - return_addr from mem[BP+8]
```

**Note**: The third read (stack0_val) can be deferred or handled differently:
- Option A: Add heads 12-15 (extend to 16 heads)
- Option B: Reuse existing STACK0 lookup (heads 0-3) after SP update
- Option C: Defer to next step (acceptable for function returns)

**Recommendation**: Option B (reuse existing heads after SP update in L16)

### L16 New Layer: Register Routing

```
L16 FFN Layer (~600 units):
  1. Route saved_bp (from L15 heads 4-7) → BP marker OUTPUT
  2. Route return_addr (from L15 heads 8-11) → PC marker OUTPUT
  3. Compute SP = BP + 16 → SP marker OUTPUT (arithmetic)
  4. Preserve AX (identity carry)
  5. Handle STACK0 routing (optional)
```

---

## Implementation Steps

### Phase 1: Add BP Address Relay (2-3 hours)

**Goal**: Enable L15 activation at BP marker (like existing STACK0 support)

**Current State**: L15 can activate at:
- AX marker (via OP_LI_RELAY, OP_LC_RELAY)
- STACK0 marker (via MARK_STACK0)

**Needed**: Activation at BP marker when OP_LEV active

**Implementation** (vm_step.py):

1. **Option A - L5 Extension**: Add BP address relay to L5
   ```python
   # L5: Add BP → address encoding dims (like STACK0)
   # When OP_LEV active at BP marker, copy BP OUTPUT → ADDR_B0-2 dims
   ```

2. **Option B - Dedicated Layer**: Use L7 or another layer
   ```python
   # L7: Already has BP handling for LEA
   # Extend to relay BP → address dims when OP_LEV active
   ```

3. **Option C - L15 Direct**: Read BP directly in L15
   ```python
   # L15: Use BP OUTPUT dims directly (may need encoding)
   ```

**Recommendation**: Option C (simplest, no additional layers)

**Implementation Details**:
```python
# In L15 heads 4-7 and 8-11:
# Q: Fire when OP_LEV active at BP marker
attn.W_q[base, BD.OP_LEV] = 2000.0
attn.W_q[base, BD.MARK_BP] = 2000.0

# K: Match memory addresses using binary encoding
# Read address from BP OUTPUT dims (BD.OUTPUT_LO/HI)
# Use same binary encoding as existing heads (dims 4-27)
```

### Phase 2: Extend L15 to 12 Heads (6-8 hours)

**Goal**: Add heads 4-11 for LEV memory reads

**File**: neural_vm/vm_step.py, function `_set_layer15_memory_lookup`

**Current Structure** (line 5911):
```python
for h in range(4):
    base = h * HD
    # ... existing code
```

**New Structure**:
```python
for h in range(12):
    base = h * HD

    if h < 4:
        # Heads 0-3: Existing (LI/LC/STACK0)
        # ... existing code unchanged

    elif h < 8:
        # Heads 4-7: LEV saved_bp read from mem[BP]
        # ... new code for BP-based lookup

    else:  # h < 12
        # Heads 8-11: LEV return_addr read from mem[BP+8]
        # ... new code for BP+8-based lookup
```

**Implementation for Heads 4-7 (saved_bp at BP)**:

```python
elif h < 8:
    byte_idx = h - 4  # 0, 1, 2, 3

    # === Dim 0: Bias — fire at BP marker only when OP_LEV ===
    attn.W_q[base, BD.CONST] = -2000.0
    attn.W_q[base, BD.OP_LEV] = 2000.0
    attn.W_q[base, BD.MARK_BP] = 2000.0
    attn.W_k[base, BD.CONST] = 10.0

    # === Dim 1: Store anchor ===
    attn.W_q[base + 1, BD.OP_LEV] = 50.0
    attn.W_q[base + 1, BD.MARK_BP] = 50.0
    attn.W_k[base + 1, BD.MEM_STORE] = 100.0
    attn.W_k[base + 1, BD.CONST] = -50.0

    # === Dim 2: ZFOD offset ===
    attn.W_q[base + 2, BD.CONST] = -96.0
    attn.W_k[base + 2, BD.MEM_STORE] = 50.0

    # === Dim 3: Byte selection ===
    BS = 60.0
    byte_q_flags = [BD.MARK_BP, BD.BP_BYTE1, BD.BP_BYTE2, BD.BP_BYTE3]
    attn.W_q[base + 3, byte_q_flags[byte_idx]] = BS
    # K side: same as existing heads (MEM_VAL_B0-3 flags)
    if byte_idx == 0:
        attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
        attn.W_k[base + 3, BD.H1 + MEM_I] = -BS
    else:
        attn.W_k[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

    # === Dims 4-27: Binary address encoding ===
    # Use BP OUTPUT dims as address source
    addr_dim = 4
    scale = 10.0
    for ab_lo, ab_hi in addr_bases:
        for nibble_base in [ab_lo, ab_hi]:
            for bit in range(4):
                # Q: Read from BP OUTPUT (not ADDR dims)
                # Need to map BP OUTPUT → address encoding
                # This is the tricky part - may need intermediate dims

                # CHALLENGE: BP is in OUTPUT dims, but we need
                # address encoding in ADDR_B0-2 dims for K matching
                #
                # SOLUTION: Use TEMP dims or add relay in earlier layer
                for k in range(16):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    # Q: Read BP value from OUTPUT (needs encoding)
                    # attn.W_q[base + addr_dim, ???] = scale * bit_val
                    # K: Match against memory address bits
                    attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                addr_dim += 1

    # === Dim 28: Position gate ===
    attn.W_q[base + 28, BD.CONST] = -500.0
    attn.W_q[base + 28, byte_q_flags[byte_idx]] = 500.0
    attn.W_k[base + 28, BD.CONST] = 5.0

    # === V/O: Copy byte value to TEMP dims (not OUTPUT) ===
    # Store in TEMP[0-31] for saved_bp (4 bytes × 2 nibbles × 4 heads)
    # TEMP layout: TEMP[0-7]=byte0, TEMP[8-15]=byte1, etc.
    for k in range(16):
        # V: Copy from memory value byte (CLEAN_EMBED)
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # V[0]: Cancel L3 default
    attn.W_v[base + 0, BD.CONST] = 1.0

    # O: Write to TEMP dims for saved_bp
    temp_offset = byte_idx * 8  # 0, 8, 16, 24
    for k in range(16):
        attn.W_o[BD.TEMP + temp_offset + (k % 16), base + 1 + k] = 1.0
    # O: Cancel L3 default
    attn.W_o[BD.TEMP + temp_offset, base + 0] = -1.0
```

**Implementation for Heads 8-11 (return_addr at BP+8)**:

```python
else:  # h < 12
    byte_idx = h - 8  # 0, 1, 2, 3

    # Same structure as heads 4-7, but:
    # 1. Address matching is BP+8 instead of BP
    # 2. Output goes to TEMP[32-63] instead of TEMP[0-31]

    # ... similar code with adjustments for +8 offset
    # O: Write to TEMP dims for return_addr
    temp_offset = 32 + byte_idx * 8  # 32, 40, 48, 56
```

**Key Challenge**: Address encoding for BP value

The main technical challenge is that BP is stored in OUTPUT dims, but we need it encoded in ADDR_B0-2 dims for K-side matching. Solutions:

1. **Add BP → ADDR relay in L14 or earlier**
   - When OP_LEV active, copy BP OUTPUT → ADDR dims
   - Allows L15 to read address directly

2. **Use OUTPUT dims directly in Q**
   - Modify Q weights to read BP from OUTPUT
   - Requires different encoding strategy

3. **Hybrid approach**
   - Use TEMP dims as intermediate storage
   - L14: BP OUTPUT → TEMP (encoded)
   - L15: TEMP → Q (address matching)

**Recommendation**: Option 1 (add relay in earlier layer for cleanliness)

### Phase 3: Add L16 Routing Layer (6-8 hours)

**Goal**: Route LEV memory reads to output registers

**File**: neural_vm/vm_step.py, new function `_set_layer16_lev_routing`

**Architecture**:
```
Input (from L15):
  - TEMP[0-31]: saved_bp (4 bytes, 32 nibbles)
  - TEMP[32-63]: return_addr (4 bytes, 32 nibbles)
  - OUTPUT: All other state (AX, etc.)

Output (to markers):
  - BP marker OUTPUT: saved_bp
  - PC marker OUTPUT: return_addr
  - SP marker OUTPUT: BP + 16 (computed)
  - AX marker OUTPUT: identity (preserve)
```

**Implementation**:

```python
def _set_layer16_lev_routing(attn, ffn, S, BD, HD):
    """L16: Route LEV memory reads to output registers.

    LEV routing:
    1. saved_bp (TEMP[0-31]) → BP marker OUTPUT
    2. return_addr (TEMP[32-63]) → PC marker OUTPUT
    3. SP = BP + 16 → SP marker OUTPUT (arithmetic)
    4. AX → AX marker OUTPUT (identity)
    """
    unit = 0

    # ===================================================================
    # Part 1: Route saved_bp → BP marker OUTPUT (32 nibbles)
    # ===================================================================
    for k in range(32):
        # Fire when OP_LEV at BP marker
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_BP] = S
        ffn.b_up[unit] = -S * 1.5

        # Gate on saved_bp value from TEMP
        ffn.W_gate[unit, BD.TEMP + k] = 1.0

        # Output to BP OUTPUT dims
        ffn.W_down[BD.OUTPUT_LO + (k % 16), unit] = 2.0 / S
        unit += 1

    # ===================================================================
    # Part 2: Route return_addr → PC marker OUTPUT (32 nibbles)
    # ===================================================================
    for k in range(32):
        # Fire when OP_LEV at PC marker
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 1.5

        # Gate on return_addr value from TEMP
        ffn.W_gate[unit, BD.TEMP + 32 + k] = 1.0

        # Output to PC OUTPUT dims
        ffn.W_down[BD.OUTPUT_LO + (k % 16), unit] = 2.0 / S
        unit += 1

    # ===================================================================
    # Part 3: Compute SP = BP + 16 → SP marker OUTPUT
    # ===================================================================
    # Reuse ADD pattern: BP + 16 (constant immediate)
    # This is similar to ADJ/ENT but simpler (constant offset)

    # Lo nibble: SP_lo = (BP_lo + 16) % 16 = BP_lo (since 16 % 16 = 0)
    for bp_lo in range(16):
        result = (bp_lo + 16) % 16  # = bp_lo

        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.OUTPUT_LO + bp_lo] = S  # BP lo nibble
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.CONST] = 1.0  # Always active

        ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
        unit += 1

    # Hi nibble byte 1: Add carry from 16
    # 16 in bytes: [0x10, 0x00, 0x00, 0x00]
    # Byte 0: 0x10 → lo=0x0, hi=0x1
    # So byte 1 needs +1 carry
    for bp_hi in range(16):
        result = (bp_hi + 1) % 16

        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.OUTPUT_HI + bp_hi] = S  # BP hi nibble
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.CONST] = 1.0

        ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
        unit += 1

    # Bytes 2-3: No carry (16 is only 2 bytes)
    # Identity carry for remaining bytes

    # ===================================================================
    # Part 4: AX identity (preserve return value)
    # ===================================================================
    # Handled by existing L6 FFN (identity carry for AX)
    # No additional units needed

    print(f"L16 LEV routing: {unit} FFN units allocated")
    return unit
```

**Total Units**: ~64 (saved_bp) + 64 (return_addr) + 32 (SP arithmetic) = ~160 FFN units

**(Revised estimate: 160 units, not 600 - much simpler than initially thought!)**

### Phase 4: Update Model Architecture (2 hours)

**Goal**: Add L16 layer to the model

**File**: neural_vm/vm_step.py, class `AutoregressiveVM`

**Changes**:

1. **Update n_layers** (line ~2500):
   ```python
   def __init__(self, n_layers=17, ...):  # Changed from 16 to 17
   ```

2. **Add L16 initialization** (line ~2700):
   ```python
   # After L15 setup
   elif layer_idx == 16:
       # L16: LEV routing
       _set_layer16_lev_routing(attn, ffn, self.swiglu_scale, BD, HD)
   ```

3. **Update layer comments and documentation**

### Phase 5: Testing and Validation (2-4 hours)

**Goal**: Verify LEV neural implementation works correctly

**Test Suite**:

1. **Simple function return**:
   ```c
   int helper() { return 42; }
   int main() { return helper(); }
   ```

2. **Function with arguments**:
   ```c
   int add(int a, int b) { return a + b; }
   int main() { return add(20, 22); }
   ```

3. **Function with locals**:
   ```c
   int func() {
       int x = 21;
       return x * 2;
   }
   int main() { return func(); }
   ```

4. **Nested function calls**:
   ```c
   int c() { return 42; }
   int b() { return c(); }
   int a() { return b(); }
   int main() { return a(); }
   ```

5. **Recursive fibonacci**:
   ```c
   int fib(int n) {
       if (n <= 1) return n;
       return fib(n-1) + fib(n-2);
   }
   int main() { return fib(7); }  // = 13
   ```

**Test Commands**:
```bash
# Run existing function tests
pytest tests/test_vm.py -k "function or lev" -v

# Run full test suite
python tests/run_1000_tests.py

# Expected: All 1096 tests still pass
```

### Phase 6: Handler Removal (1 hour)

**Goal**: Remove LEV handler, validate tests

**Changes** (neural_vm/run_vm.py):

1. **Comment out handler** (line ~227):
   ```python
   self._func_call_handlers = {
       Opcode.JSR: self._handler_jsr,
       # REMOVED 2026-04-09: LEV now works fully neurally (L15/L16 implementation complete)
       # Neural path: L15 reads mem[BP] and mem[BP+8], L16 routes to registers
       # Opcode.LEV: self._handler_lev,
   }
   ```

2. **Delete handler function** (optional - can keep for reference):
   - `_handler_lev` at lines ~1540-1560

3. **Test and commit**:
   ```bash
   python tests/run_1000_tests.py  # Should still pass 1096/1096
   git add neural_vm/vm_step.py neural_vm/run_vm.py
   git commit -m "Complete LEV neural - now 100% neural (L15/L16)"
   ```

---

## Technical Challenges and Solutions

### Challenge 1: BP Address Encoding

**Problem**: BP value is in OUTPUT dims, but L15 needs it encoded in ADDR_B0-2 dims for K-side matching.

**Solution Options**:
1. Add BP → ADDR relay in L14 when OP_LEV active
2. Read BP directly from OUTPUT in L15 Q (non-standard)
3. Use intermediate TEMP dims for encoded address

**Recommendation**: Option 1 (cleanest architecture)

### Challenge 2: TEMP Dimension Limits

**Problem**: Need 64 dims for saved_bp + return_addr storage, but TEMP has limited space.

**Solution**:
- Use TEMP[0-31] for saved_bp (4 bytes × 8 nibbles = 32 dims) ✓
- Use TEMP[32-63] for return_addr (4 bytes × 8 nibbles = 32 dims) ✓
- Total: 64 dims needed, may need to expand TEMP

**Verification**: Check current TEMP allocation in dim_registry.py

### Challenge 3: Attention Score Budget

**Problem**: L15 heads 4-11 need similar score budgets to heads 0-3 for correct softmax1 behavior.

**Solution**: Reuse exact same score budget strategy:
- Dim 0: Bias (-2500 non-target, 0 target)
- Dim 1: Store anchor (+312.5 store, -312.5 non-store)
- Dim 2: ZFOD offset (-600 at stores)
- Dim 3: Byte selection (+450 correct byte)
- Dims 4-27: Address matching (+300 match, 0 random)

**Total**: Same as existing heads (proven to work)

### Challenge 4: Third Memory Read (stack0_val)

**Problem**: LEV needs to read `stack0_val = mem[new_sp]` after computing `new_sp = BP + 16`.

**Solution Options**:
1. Add heads 12-15 for third read (extend to 16 heads)
2. Defer to next step (STACK0 will be re-read when needed)
3. Reuse existing heads 0-3 after SP update in L16

**Recommendation**: Option 2 (simplest - stack0_val rarely used immediately)

---

## Timeline and Milestones

### Detailed Timeline

| Phase | Task | Hours | Cumulative |
|-------|------|-------|------------|
| **Phase 1** | BP address relay | 2-3 | 2-3 |
| **Phase 2** | L15 extension (12 heads) | 6-8 | 8-11 |
| **Phase 3** | L16 routing layer | 6-8 | 14-19 |
| **Phase 4** | Model architecture update | 2 | 16-21 |
| **Phase 5** | Testing and validation | 2-4 | 18-25 |
| **Phase 6** | Handler removal | 1 | 19-26 |
| **Total** | **All phases** | **19-26 hours** | - |

### Milestones

1. **M1**: BP address relay working → Can read BP value in L15
2. **M2**: L15 heads 4-7 working → Can read saved_bp from mem[BP]
3. **M3**: L15 heads 8-11 working → Can read return_addr from mem[BP+8]
4. **M4**: L16 routing working → Registers updated correctly
5. **M5**: Simple function returns pass → Basic LEV neural works
6. **M6**: Recursive functions pass → Full LEV neural complete
7. **M7**: Handler removed → ~99% neural VM achieved! 🎉

---

## Success Criteria

### Phase Success Criteria

**Phase 1 (BP Address Relay)**:
- ✅ BP value visible in address encoding dims when OP_LEV active
- ✅ L15 can read address from BP marker

**Phase 2 (L15 Extension)**:
- ✅ Heads 4-7 correctly read saved_bp from mem[BP]
- ✅ Heads 8-11 correctly read return_addr from mem[BP+8]
- ✅ Values stored in TEMP dims correctly

**Phase 3 (L16 Routing)**:
- ✅ saved_bp routed to BP marker OUTPUT
- ✅ return_addr routed to PC marker OUTPUT
- ✅ SP = BP + 16 computed and routed to SP marker OUTPUT
- ✅ AX preserved (identity)

**Phase 4 (Model Architecture)**:
- ✅ 17-layer model initializes correctly
- ✅ No errors during weight initialization
- ✅ Model forward pass works

**Phase 5 (Testing)**:
- ✅ Simple function returns work
- ✅ Nested function calls work
- ✅ Recursive functions work (fibonacci, factorial)
- ✅ Full test suite passes (1096/1096)

**Phase 6 (Handler Removal)**:
- ✅ LEV handler removed from run_vm.py
- ✅ Tests still pass 1096/1096
- ✅ No LEV-related failures in logs

### Final Success Criteria

- ✅ All 1096+ tests pass without LEV handler
- ✅ Function returns work correctly (simple, nested, recursive)
- ✅ No performance regression (< 15% slowdown acceptable)
- ✅ LEV removed from `_func_call_handlers`
- 🎯 **Milestone: ~99% NEURAL VM** (only JSR handler remains)

---

## Risks and Mitigation

### Risk 1: BP Address Encoding Complexity
**Probability**: High
**Impact**: High (blocks L15 extension)
**Mitigation**:
- Start with simplest solution (add relay in earlier layer)
- Test address encoding independently before full LEV
- Have fallback: use handler for BP, neural for rest

### Risk 2: TEMP Dimension Overflow
**Probability**: Medium
**Impact**: High (no space for memory read results)
**Mitigation**:
- Audit current TEMP usage before starting
- Expand TEMP if needed (dim_registry.py)
- Alternative: use OUTPUT overlay (risky but possible)

### Risk 3: Attention Score Budget Errors
**Probability**: Medium
**Impact**: High (incorrect memory reads)
**Mitigation**:
- Reuse exact same score budget as existing heads
- Test each head independently before combining
- Extensive validation with debug logging

### Risk 4: L16 Routing Bugs
**Probability**: Medium
**Impact**: Medium (incorrect register updates)
**Mitigation**:
- Test each routing path independently
- Validate with simple programs before complex ones
- Keep handler as fallback during development

### Risk 5: Performance Regression
**Probability**: Low
**Impact**: Medium (~20-30% slowdown)
**Mitigation**:
- Extra layer adds one forward pass (~6% overhead)
- Speculative execution compensates
- Target: < 15% slowdown acceptable

---

## Next Steps After This Plan

### Immediate: Phase 1 (BP Address Relay)

Start with BP address relay implementation:

1. **Audit current address encoding**:
   - Understand how ADDR_B0-2 dims are populated
   - Identify where to add BP → ADDR relay

2. **Implement relay** (choose cleanest location):
   - Option A: L14 (with MEM token generation)
   - Option B: L7 (with BP handling)
   - Option C: L15 Q directly (non-standard)

3. **Test relay independently**:
   - Verify BP value appears in ADDR dims when OP_LEV active
   - Validate with simple LEV instruction

4. **Document and commit**:
   ```bash
   git commit -m "Add BP address relay for LEV (Phase 1)"
   ```

### Then: Phase 2 (L15 Extension)

Proceed to extending L15 once BP address relay works.

---

## Conclusion

LEV neural implementation is complex but achievable. The main challenges are:
1. BP address encoding (architectural)
2. L15 extension (attention score budgets)
3. L16 routing (register updates)

With careful implementation and testing, LEV can be completed in 19-26 hours of focused work.

**Result**: ~99% neural VM with only 1 handler remaining (JSR)

**Impact**: Massive step toward 100% neural VM - function returns fully neural!

---

**Plan Date**: 2026-04-09 22:15 UTC-4
**Status**: Ready for implementation
**Next Phase**: Phase 1 (BP address relay)
**Estimated Completion**: 19-26 hours from start

🚀 **Ready to achieve ~99% neural VM!**
