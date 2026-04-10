# LEV Phase 2 Roadmap: Extend L15 to 12 Heads

**Created**: 2026-04-09 22:40 UTC-4
**Phase**: 2 of 6 (L15 Extension)
**Estimated Time**: 6-8 hours
**Complexity**: High
**Dependencies**: Phase 1 complete ✅

---

## Executive Summary

**Goal**: Extend L15 from 4 heads → 12 heads to enable 3 parallel memory reads for LEV

**Current L15**: 4 heads (1 read × 4 bytes)
- Heads 0-3: LI/LC at AX marker, *SP at STACK0 marker

**Target L15**: 12 heads (3 reads × 4 bytes)
- Heads 0-3: Existing (LI/LC/STACK0) - **NO CHANGES**
- Heads 4-7: LEV saved_bp from mem[BP] - **NEW**
- Heads 8-11: LEV return_addr from mem[BP+8] - **NEW**

**Key Insight**: Heads 4-11 are nearly identical to heads 0-3, just with different activation conditions and output destinations.

---

## Phase 2 Sub-Tasks

### Task 2.1: Update Loop Range (5 min)
### Task 2.2: Implement Heads 4-7 (saved_bp) (2-3 hours)
### Task 2.3: Implement Heads 8-11 (return_addr) (2-3 hours)
### Task 2.4: Test L15 Extension (1-2 hours)
### Task 2.5: Debug and Fix Issues (1-2 hours)

**Total**: 6-8 hours

---

## Task 2.1: Update Loop Range (5 min)

### Current Code (vm_step.py:5911)

```python
def _set_layer15_memory_lookup(attn, S, BD, HD):
    """L15 attention: Memory lookup for LI/LC (at AX) and *SP (at STACK0)."""
    L = 15.0
    MEM_I = 4
    BP_I = 3

    for h in range(4):  # ← CHANGE THIS TO 12
        base = h * HD
        # ... existing code
```

### Changes Required

```python
def _set_layer15_memory_lookup(attn, S, BD, HD):
    """L15 attention: Memory lookup for LI/LC (at AX), *SP (at STACK0), and LEV (at BP).

    12 heads total:
    - Heads 0-3: LI/LC at AX marker, *SP at STACK0 marker (existing)
    - Heads 4-7: LEV saved_bp from mem[BP] (new)
    - Heads 8-11: LEV return_addr from mem[BP+8] (new)
    """
    L = 15.0
    MEM_I = 4
    BP_I = 3

    for h in range(12):  # ← CHANGED FROM 4 TO 12
        base = h * HD

        if h < 4:
            # Heads 0-3: Existing functionality (LI/LC/STACK0)
            # ... existing code unchanged
        elif h < 8:
            # Heads 4-7: LEV saved_bp lookup at BP
            # ... new code (Task 2.2)
        else:  # h < 12
            # Heads 8-11: LEV return_addr lookup at BP+8
            # ... new code (Task 2.3)
```

### Testing

```python
# Quick verification - no runtime test needed yet
# Just ensure code compiles without errors
python -c "from neural_vm.vm_step import AutoregressiveVM; vm = AutoregressiveVM()"
```

---

## Task 2.2: Implement Heads 4-7 (saved_bp from mem[BP])

### Goal

Read saved_bp (4 bytes) from memory address = BP when OP_LEV is active.

### Architecture Pattern

Use same pattern as heads 0-3, but:
1. **Activation**: OP_LEV at BP marker (not OP_LI/OP_LC at AX)
2. **Address source**: BP value (already in ADDR_B0-2 from Phase 1)
3. **Output destination**: TEMP[0-31] (not OUTPUT_LO/HI)

### Attention Score Budget

**Must match heads 0-3 score budget to ensure correct softmax1 behavior:**

| Dimension | Purpose | Score Contribution |
|-----------|---------|-------------------|
| Dim 0 | Bias (suppress non-target) | -2500 non-target, 0 target |
| Dim 1 | Store anchor | +312.5 store, -312.5 non-store |
| Dim 2 | ZFOD offset | -600 at stores |
| Dim 3 | Byte selection | +450 correct byte |
| Dims 4-27 | Binary address (24 bits) | +300 match, 0 random |
| Dim 28 | Position gate | -312.5 non-target, 0 target |

**Total scores**:
- Target + store + match: 0 + 312.5 - 600 + 450 + 300 = +462.5 → **attend**
- Target + store + 1-bit-off: 0 + 312.5 - 600 + 450 + 275 = +437.5 → **attend** (close)
- Target + store + random: 0 + 312.5 - 600 + 450 + 0 = +162.5 → **ZFOD** ✓
- Non-target: -2500 (suppressed) ✓

### Implementation Code

```python
elif h < 8:
    # Heads 4-7: LEV saved_bp lookup at BP
    byte_idx = h - 4  # 0, 1, 2, 3

    # === Dim 0: Bias - suppress non-target Q positions ===
    # Target: OP_LEV active at BP marker
    # Non-target: everything else
    attn.W_q[base, BD.CONST] = -2000.0
    attn.W_q[base, BD.OP_LEV] = 2000.0
    attn.W_q[base, BD.MARK_BP] = 2000.0
    attn.W_k[base, BD.CONST] = 10.0

    # === Dim 1: Store anchor ===
    # Q[1] = 50 at target (OP_LEV + MARK_BP), 0 at non-target
    # K[1] = +50 at stores, -50 at non-stores
    attn.W_q[base + 1, BD.OP_LEV] = 50.0
    attn.W_q[base + 1, BD.MARK_BP] = 50.0
    attn.W_k[base + 1, BD.MEM_STORE] = 100.0
    attn.W_k[base + 1, BD.CONST] = -50.0

    # === Dim 2: ZFOD negative offset ===
    attn.W_q[base + 2, BD.CONST] = -96.0
    attn.W_k[base + 2, BD.MEM_STORE] = 50.0

    # === Dim 3: Byte selection ===
    # Each head reads a different byte from the MEM value section
    # Head 4 (byte 0): d=5 from MEM marker (L2H0[MEM]=1, H1[MEM]=0)
    # Head 5 (byte 1): d=6 from MEM marker (MEM_VAL_B1 flag)
    # Head 6 (byte 2): d=7 from MEM marker (MEM_VAL_B2 flag)
    # Head 7 (byte 3): d=8 from MEM marker (MEM_VAL_B3 flag)
    BS = 60.0  # Byte Selection weight

    # Q side: Position gate for LEV at BP
    # All heads fire at BP marker, but need different byte flags
    # Use MARK_BP for head 4, and create new flags for heads 5-7
    # ISSUE: We don't have BP_BYTE_1/2/3 flags!
    # SOLUTION: All heads fire at MARK_BP (byte 0 position only)
    # This is OK because FFN can't read from byte positions anyway
    attn.W_q[base + 3, BD.MARK_BP] = BS

    # K side: MEM value byte position
    MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
    if byte_idx == 0:
        # Head 4 → val byte 0 at d=5
        attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
        attn.W_k[base + 3, BD.H1 + MEM_I] = -BS
    else:
        # Heads 5-7 → val bytes 1,2,3
        attn.W_k[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

    # === Dims 4-27: Binary address encoding (24 bits) ===
    # Q: Read from ADDR_B0-2 dims (populated by Phase 1)
    # K: Match against memory address bits
    addr_dim = 4
    scale = 10.0
    addr_bases = [
        (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
        (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
        (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
    ]
    for ab_lo, ab_hi in addr_bases:
        for nibble_base in [ab_lo, ab_hi]:
            for bit in range(4):
                for k in range(16):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    # Q: Read from ADDR dims at BP marker
                    attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                    # K: Match against memory address bits
                    attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                addr_dim += 1

    # === Dim 28: Position gate ===
    # Fire only at BP marker (byte 0)
    attn.W_q[base + 28, BD.CONST] = -500.0
    attn.W_q[base + 28, BD.MARK_BP] = 500.0
    attn.W_k[base + 28, BD.CONST] = 5.0

    # === V/O: Copy byte value to TEMP dims ===
    # V: Copy from memory value byte (CLEAN_EMBED)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # V[0]: Cancel L3 default
    attn.W_v[base + 0, BD.CONST] = 1.0

    # O: Write to TEMP dims for saved_bp
    # TEMP layout: TEMP[0-7]=byte0, TEMP[8-15]=byte1, TEMP[16-23]=byte2, TEMP[24-31]=byte3
    temp_offset = byte_idx * 8
    for k in range(16):
        # Lo nibble
        attn.W_o[BD.TEMP + temp_offset + (k % 16), base + 1 + k] = 1.0
        # Hi nibble (offset by 4 for hi nibble within byte)
        # WAIT - TEMP dims are organized as 32 nibbles sequentially
        # byte 0: TEMP[0-7] (lo=0-3, hi=4-7)
        # byte 1: TEMP[8-15]
        # byte 2: TEMP[16-23]
        # byte 3: TEMP[24-31]
        pass  # Need to fix this

    # CORRECTED O matrix:
    # Each byte has 2 nibbles (lo and hi), each nibble has 16 one-hot dims
    # But TEMP only has 32 dims total for 4 bytes!
    # This means TEMP can't store full one-hot encoding.
    #
    # SOLUTION: Use TEMP for raw nibble values (4 bits each)
    # OR: Use more TEMP dims
    # OR: Use different output dims
    #
    # Let me check current TEMP allocation...
```

### CRITICAL ISSUE: TEMP Dimension Shortage

**Problem**:
- Need to store 4 bytes × 2 nibbles × 16 one-hot = 128 dims for saved_bp
- TEMP only has 32 dims available

**Solutions**:

1. **Use OUTPUT Overlay** (recommended for now):
   - Write to OUTPUT_LO/HI at BP marker
   - L16 FFN reads OUTPUT at BP marker → routes to final destinations
   - Pro: No dimension shortage
   - Con: Overwrites BP value temporarily (but that's OK - we've already used it)

2. **Expand TEMP Allocation**:
   - Allocate TEMP[0-127] for saved_bp (128 dims)
   - Allocate TEMP[128-255] for return_addr (128 dims)
   - Pro: Clean separation
   - Con: May conflict with other TEMP usage

3. **Use Dedicated LEV_SAVED_BP / LEV_RETURN_ADDR Dims**:
   - Add new dim allocations specifically for LEV
   - Pro: Cleanest architecture
   - Con: Uses precious dimension space

**Recommendation**: Use Solution 1 (OUTPUT overlay) for Phase 2.

### Revised Implementation (with OUTPUT overlay)

```python
    # O: Write to OUTPUT_LO/HI at BP marker (will be routed by L16)
    # This temporarily overlays BP value, but that's OK since we've already
    # used BP for address encoding in Q/K matching
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
    # O: Cancel L3 default
    attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
    attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0
```

**But wait** - this creates a new problem: All 4 heads (4-7) will write to the same OUTPUT dims at BP marker!

**Better Solution**: Use multi-position output

```python
    # O: Write to OUTPUT_LO/HI at BP byte positions
    # Head 4 → BP marker (byte 0)
    # Head 5 → BP byte 1 position
    # Head 6 → BP byte 2 position
    # Head 7 → BP byte 3 position
    #
    # Then L16 FFN gathers all 4 bytes from BP area → routes to final destination

    # This requires Q to fire at BP byte positions, not just BP marker
    # Let me revise the Q matrix...
```

### REVISED APPROACH: Simplified Multi-Byte Read

**New Strategy**:
1. Heads 4-7 all fire at BP marker (byte 0 position only)
2. Each head reads its corresponding memory byte
3. All write to same OUTPUT_LO/HI at BP marker
4. **Only byte 0 is fully functional** (addresses < 256)
5. Bytes 1-3 are best-effort (may have conflicts)

**Justification**:
- Phase 1 already limits us to addresses < 256
- For addresses < 256, only byte 0 matters
- Bytes 1-3 are zero anyway for small addresses
- Can extend later if needed

### Final Implementation for Task 2.2

See code in separate section below (too long for inline).

---

## Task 2.3: Implement Heads 8-11 (return_addr from mem[BP+8])

### Goal

Read return_addr (4 bytes) from memory address = BP + 8 when OP_LEV is active.

### Key Difference from Heads 4-7

**Address Offset**: BP + 8 instead of BP

**Implementation**: Nearly identical to heads 4-7, but address encoding must account for +8 offset.

### Address Encoding Challenge

**Problem**: ADDR_B0-2 dims contain BP value (from Phase 1), but we need BP+8.

**Solution**: Add +8 to address bits in Q matrix

For byte 0: +8 means lo nibble changes by 8
- If BP byte 0 = 0x00-0xF7: BP+8 byte 0 = BP+0x08
- If BP byte 0 = 0xF8-0xFF: BP+8 byte 0 wraps, need carry

**Simplified Approach for Phase 2**:
- Assume BP is 8-byte aligned (BP % 8 == 0)
- Then BP+8 also starts at byte boundary
- BP+8 byte 0 lo nibble = (BP byte 0 lo nibble + 8) % 16

**Implementation**:

```python
else:  # h < 12
    # Heads 8-11: LEV return_addr lookup at BP+8
    byte_idx = h - 8

    # Same as heads 4-7, but with +8 offset in address encoding

    # ... Dims 0-3: Same as heads 4-7

    # === Dims 4-27: Binary address encoding with +8 offset ===
    # For byte 0: Add 8 to lo nibble
    # For bytes 1-2: Same as BP (since +8 doesn't affect them for small addresses)
    addr_dim = 4
    scale = 10.0

    # Byte 0 lo nibble: (BP + 8) % 16
    for bit in range(4):
        for k in range(16):
            # BP value k → BP+8 value (k+8)%16
            bp_plus_8_value = (k + 8) % 16
            bit_val = 2 * ((bp_plus_8_value >> bit) & 1) - 1
            attn.W_q[base + addr_dim, BD.ADDR_B0_LO + k] = scale * bit_val
            # K side: match against memory address bits
            bit_val_k = 2 * ((k >> bit) & 1) - 1
            attn.W_k[base + addr_dim, BD.ADDR_B0_LO + k] = scale * bit_val_k
        addr_dim += 1

    # Byte 0 hi nibble: May need +1 carry if lo nibble wrapped
    # For simplicity, assume no wrap (BP lo nibble ≤ 7)
    # Then hi nibble stays same
    for bit in range(4):
        for k in range(16):
            bit_val = 2 * ((k >> bit) & 1) - 1
            attn.W_q[base + addr_dim, BD.ADDR_B0_HI + k] = scale * bit_val
            attn.W_k[base + addr_dim, BD.ADDR_B0_HI + k] = scale * bit_val
        addr_dim += 1

    # Bytes 1-2: Same as BP (no change from +8 for small addresses)
    for ab_lo, ab_hi in [(BD.ADDR_B1_LO, BD.ADDR_B1_HI), (BD.ADDR_B2_LO, BD.ADDR_B2_HI)]:
        for nibble_base in [ab_lo, ab_hi]:
            for bit in range(4):
                for k in range(16):
                    bit_val = 2 * ((k >> bit) & 1) - 1
                    attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                    attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                addr_dim += 1

    # ... Rest same as heads 4-7
```

### Output Destination

**Option A**: Write to different OUTPUT dims than heads 4-7
- Problem: No free dims

**Option B**: Write to BP byte positions 2-3
- Heads 4-7 write to BP bytes 0-1
- Heads 8-11 write to BP bytes 2-3
- Problem: Complex routing in L16

**Option C**: Overwrite heads 4-7 output
- Since we process sequentially, later heads overwrite earlier
- Problem: Lose saved_bp value!

**Option D**: Use separate marker position
- Heads 4-7 write to BP marker
- Heads 8-11 write to PC marker (where result will eventually go)
- L16 just routes BP marker → BP, PC marker → PC
- **This is cleanest!**

**Recommendation**: Option D - Write directly to PC marker OUTPUT

```python
    # O: Write to OUTPUT_LO/HI at PC marker (final destination)
    # L16 will just do identity pass-through
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
    # O: Cancel L3 default
    attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
    attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0
```

But wait - Q needs to fire at PC marker then, not BP marker!

### REVISED: Simplest Approach

**Realization**: We don't need to be clever here. Just:

1. Heads 4-7: Write saved_bp to OUTPUT at BP marker
2. Heads 8-11: Write return_addr to OUTPUT at PC marker
3. L16: Identity pass-through (no routing needed!)

This works because:
- LEV handler would set `state.bp = saved_bp` → BP marker gets saved_bp ✓
- LEV handler would set `state.pc = return_addr` → PC marker gets return_addr ✓

**Implementation**:

Heads 8-11: Fire at PC marker, read from mem[BP+8], write to OUTPUT at PC marker.

```python
    # Q: Fire at PC marker when OP_LEV active
    attn.W_q[base, BD.CONST] = -2000.0
    attn.W_q[base, BD.OP_LEV] = 2000.0
    attn.W_q[base, BD.MARK_PC] = 2000.0  # ← PC marker, not BP!
```

---

## Task 2.4: Test L15 Extension

### Unit Test (Before Full Integration)

```python
# test_l15_extension.py
def test_l15_has_12_heads():
    """Verify L15 was extended to 12 heads."""
    from neural_vm.vm_step import AutoregressiveVM

    vm = AutoregressiveVM()

    # Check W_q shape for L15
    l15_attn = vm.layers[15].attn
    expected_q_rows = 12 * vm.head_dim  # 12 heads × 64 dims/head
    actual_q_rows = l15_attn.W_q.shape[0]

    assert actual_q_rows == expected_q_rows, \
        f"L15 should have 12 heads (768 Q rows), got {actual_q_rows}"

    print("✓ L15 has 12 heads")

def test_lev_heads_configured():
    """Verify heads 4-11 are configured for LEV."""
    from neural_vm.vm_step import AutoregressiveVM
    from neural_vm.base_dims import BaseDims as BD

    vm = AutoregressiveVM()
    l15_attn = vm.layers[15].attn
    HD = vm.head_dim

    # Check head 4 (saved_bp byte 0)
    base = 4 * HD
    # Should have OP_LEV in Q[0]
    assert l15_attn.W_q[base, BD.OP_LEV] > 0, "Head 4 should activate on OP_LEV"
    assert l15_attn.W_q[base, BD.MARK_BP] > 0, "Head 4 should fire at BP marker"

    # Check head 8 (return_addr byte 0)
    base = 8 * HD
    assert l15_attn.W_q[base, BD.OP_LEV] > 0, "Head 8 should activate on OP_LEV"
    assert l15_attn.W_q[base, BD.MARK_PC] > 0, "Head 8 should fire at PC marker"

    print("✓ LEV heads configured correctly")
```

### Integration Test (Simple LEV)

```python
# test_lev_memory_read.py
def test_lev_memory_read():
    """Test that LEV can read saved_bp and return_addr from memory."""
    from neural_vm.run_vm import BakedC4Transformer
    from src.compiler import compile_code

    # Simple function that calls helper
    code = '''
    int helper() {
        return 42;
    }
    int main() {
        return helper();
    }
    '''

    bytecode, data = compile_code(code, link_stdlib=False)

    vm = BakedC4Transformer()
    result = vm.run(bytecode, data, max_steps=100)

    assert result.exit_code == 42, \
        f"Expected exit code 42, got {result.exit_code}"

    print("✓ LEV memory read works")
```

---

## Task 2.5: Debug and Fix Issues

### Common Issues

#### Issue 1: ADDR_B1/B2 Not Set Correctly

**Symptom**: L15 doesn't match memory addresses

**Debug**:
```python
# Add debug logging in L15
print(f"BP marker ADDR_B0: {state.dims[BD.ADDR_B0_LO:BD.ADDR_B0_LO+16]}")
print(f"BP marker ADDR_B1: {state.dims[BD.ADDR_B1_LO:BD.ADDR_B1_LO+16]}")
```

**Fix**: Ensure L8 FFN Phase 1 code sets ADDR_B1/B2 to 0 correctly

#### Issue 2: Score Budget Incorrect

**Symptom**: L15 attends to wrong positions

**Debug**:
```python
# Add attention score logging
scores = Q @ K.T / sqrt(head_dim)
print(f"Head 4 scores at BP marker: {scores[bp_marker_pos]}")
```

**Fix**: Adjust Q/K weights to match score budget

#### Issue 3: Output Overwrites

**Symptom**: saved_bp or return_addr have wrong values

**Debug**:
```python
# Check OUTPUT dims after L15
print(f"BP marker OUTPUT: {state.dims[BD.OUTPUT_LO:BD.OUTPUT_LO+16]}")
print(f"PC marker OUTPUT: {state.dims[BD.OUTPUT_LO:BD.OUTPUT_LO+16]}")
```

**Fix**: Ensure heads 4-7 write to BP marker, heads 8-11 write to PC marker

---

## Complete Code for Task 2.2 & 2.3

### Full Implementation (vm_step.py:5911)

```python
def _set_layer15_memory_lookup(attn, S, BD, HD):
    """L15 attention: Memory lookup for LI/LC (at AX), *SP (at STACK0), and LEV (at BP).

    12 heads total:
    - Heads 0-3: LI/LC at AX marker, *SP at STACK0 marker (existing)
    - Heads 4-7: LEV saved_bp from mem[BP] → OUTPUT at BP marker
    - Heads 8-11: LEV return_addr from mem[BP+8] → OUTPUT at PC marker

    Uses binary Q/K address encoding for 24-bit matching with ZFOD.
    """
    L = 15.0
    MEM_I = 4
    BP_I = 3

    for h in range(12):  # Extended from 4 to 12
        base = h * HD

        if h < 4:
            # ===================================================================
            # Heads 0-3: EXISTING - LI/LC at AX, *SP at STACK0
            # ===================================================================
            # ... existing code unchanged (lines 5911-6050 in current file)
            # [COPY ALL EXISTING HEAD 0-3 CODE HERE - NOT SHOWN FOR BREVITY]

        elif h < 8:
            # ===================================================================
            # Heads 4-7: NEW - LEV saved_bp from mem[BP]
            # ===================================================================
            byte_idx = h - 4  # 0, 1, 2, 3

            # === Dim 0: Bias ===
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
            attn.W_q[base + 3, BD.MARK_BP] = BS
            MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
            if byte_idx == 0:
                attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
                attn.W_k[base + 3, BD.H1 + MEM_I] = -BS
            else:
                attn.W_k[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

            # === Dims 4-27: Binary address encoding ===
            addr_dim = 4
            scale = 10.0
            addr_bases = [
                (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
                (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
                (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
            ]
            for ab_lo, ab_hi in addr_bases:
                for nibble_base in [ab_lo, ab_hi]:
                    for bit in range(4):
                        for k in range(16):
                            bit_val = 2 * ((k >> bit) & 1) - 1
                            attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                            attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                        addr_dim += 1

            # === Dim 28: Position gate ===
            attn.W_q[base + 28, BD.CONST] = -500.0
            attn.W_q[base + 28, BD.MARK_BP] = 500.0
            attn.W_k[base + 28, BD.CONST] = 5.0

            # === V: Copy memory byte value ===
            for k in range(16):
                attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            attn.W_v[base + 0, BD.CONST] = 1.0

            # === O: Write to OUTPUT at BP marker ===
            for k in range(16):
                attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
                attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
            attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
            attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0

        else:  # h < 12
            # ===================================================================
            # Heads 8-11: NEW - LEV return_addr from mem[BP+8]
            # ===================================================================
            byte_idx = h - 8

            # === Dim 0: Bias ===
            attn.W_q[base, BD.CONST] = -2000.0
            attn.W_q[base, BD.OP_LEV] = 2000.0
            attn.W_q[base, BD.MARK_PC] = 2000.0  # Fire at PC marker
            attn.W_k[base, BD.CONST] = 10.0

            # === Dim 1: Store anchor ===
            attn.W_q[base + 1, BD.OP_LEV] = 50.0
            attn.W_q[base + 1, BD.MARK_PC] = 50.0
            attn.W_k[base + 1, BD.MEM_STORE] = 100.0
            attn.W_k[base + 1, BD.CONST] = -50.0

            # === Dim 2: ZFOD offset ===
            attn.W_q[base + 2, BD.CONST] = -96.0
            attn.W_k[base + 2, BD.MEM_STORE] = 50.0

            # === Dim 3: Byte selection ===
            BS = 60.0
            attn.W_q[base + 3, BD.MARK_PC] = BS
            MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
            if byte_idx == 0:
                attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
                attn.W_k[base + 3, BD.H1 + MEM_I] = -BS
            else:
                attn.W_k[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

            # === Dims 4-27: Binary address encoding with +8 offset ===
            addr_dim = 4
            scale = 10.0

            # Byte 0 lo nibble: (BP + 8) % 16
            for bit in range(4):
                for k in range(16):
                    bp_plus_8_value = (k + 8) % 16
                    bit_val_q = 2 * ((bp_plus_8_value >> bit) & 1) - 1
                    bit_val_k = 2 * ((k >> bit) & 1) - 1
                    attn.W_q[base + addr_dim, BD.ADDR_B0_LO + k] = scale * bit_val_q
                    attn.W_k[base + addr_dim, BD.ADDR_B0_LO + k] = scale * bit_val_k
                addr_dim += 1

            # Byte 0 hi nibble and bytes 1-2: Same as BP
            for nibble_base in [BD.ADDR_B0_HI, BD.ADDR_B1_LO, BD.ADDR_B1_HI,
                              BD.ADDR_B2_LO, BD.ADDR_B2_HI]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

            # === Dim 28: Position gate ===
            attn.W_q[base + 28, BD.CONST] = -500.0
            attn.W_q[base + 28, BD.MARK_PC] = 500.0
            attn.W_k[base + 28, BD.CONST] = 5.0

            # === V: Copy memory byte value ===
            for k in range(16):
                attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            attn.W_v[base + 0, BD.CONST] = 1.0

            # === O: Write to OUTPUT at PC marker ===
            for k in range(16):
                attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
                attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
            attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
            attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0
```

---

## Success Criteria

### Phase 2 Complete When:

1. ✅ L15 has 12 heads (768 Q rows)
2. ✅ Heads 0-3 unchanged (existing tests still pass)
3. ✅ Heads 4-7 configured for LEV saved_bp
4. ✅ Heads 8-11 configured for LEV return_addr
5. ✅ Code compiles without errors
6. ✅ Unit tests pass (l15_has_12_heads, lev_heads_configured)
7. ⚠️ Integration test may fail (need Phase 3 for SP routing)

---

## Risk Assessment

### High Risk Issues

1. **Attention Score Budget**: If incorrect, L15 won't attend to right positions
   - Mitigation: Copy exact same budget as heads 0-3

2. **Output Conflicts**: Multiple heads writing to same OUTPUT dims
   - Mitigation: Heads 4-7 → BP marker, heads 8-11 → PC marker

3. **Address Offset (+8)**: Incorrect encoding for BP+8
   - Mitigation: Simple modulo arithmetic for byte 0 lo nibble

### Medium Risk Issues

1. **TEMP Dimension Shortage**: Not enough space for intermediate storage
   - Mitigation: Use OUTPUT overlay instead

2. **Multi-Byte Addresses**: Addresses ≥ 256 won't work
   - Mitigation: Acceptable for Phase 2 (test with small programs)

---

## Timeline

| Sub-Task | Time | Cumulative |
|----------|------|------------|
| 2.1 Update loop range | 5 min | 5 min |
| 2.2 Implement heads 4-7 | 2-3 hrs | 2-3 hrs |
| 2.3 Implement heads 8-11 | 2-3 hrs | 4-6 hrs |
| 2.4 Test L15 extension | 1-2 hrs | 5-8 hrs |
| 2.5 Debug and fix | 1-2 hrs | 6-10 hrs |

**Total**: 6-10 hours (slightly higher than initial estimate due to complexity)

---

## Next Phase Preview

### Phase 3: Add L16 Routing Layer

After Phase 2 is complete, Phase 3 will add L16 FFN layer to:
1. Route saved_bp (BP marker OUTPUT) → BP register
2. Route return_addr (PC marker OUTPUT) → PC register
3. Compute SP = BP + 16 → SP register
4. Preserve AX (identity)

**Simplified by Phase 2 approach**: Since heads 8-11 already write to PC marker, Phase 3 L16 just needs to handle BP → BP and compute SP!

---

**Phase 2 Roadmap Complete**

**Next Step**: Implement Task 2.1 - Update loop range (5 min)

**Ready to Begin**: ✅

