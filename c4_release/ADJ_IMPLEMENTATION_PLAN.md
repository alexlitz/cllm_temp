# ADJ Neural Implementation Plan

**Date**: 2026-04-09
**Status**: Dimensions Allocated - Ready to Implement
**Estimated Effort**: 10-15 hours

---

## Dimensions Allocated ✅

Added to `neural_vm/dim_registry.py`:
- SP_OLD_LO [297-304]: 8 dims (old SP low nibbles)
- SP_OLD_HI [305-312]: 8 dims (old SP high nibbles)
- ADJ_CARRY [313-314]: 2 dims (carry propagation)

---

## Implementation Steps

### Step 1: L7 Head 1 - Add ADJ Gathering

**Location**: `vm_step.py` line ~4275 in `_set_layer7_operand_gather`

**Current**: LEA gathers BP OUTPUT → ALU at AX marker
**Add**: ADJ gathers SP OUTPUT → ALU at AX marker

```python
# Head 1: LEA and ADJ operand gather
base = 1 * HD

# LEA path (existing)
attn.W_q[base, BD.OP_LEA] = L
attn.W_k[base, BD.MARK_BP] = L

# ADD: ADJ path (new)
attn.W_q[base, BD.OP_ADJ] = L  # Also fires when ADJ active
attn.W_k[base, BD.MARK_SP] = L  # Also attends to SP marker

# V/O unchanged: copy OUTPUT → ALU
```

### Step 2: L8 FFN - Add ADJ Lo Nibble Units

**Location**: `vm_step.py` line ~4462 in `_set_layer8_alu`

**Add after LEA units** (~line 4475):
```python
# === ADJ: lo nibble (256 units) ===
# Like LEA but gates on OP_ADJ
for a in range(16):
    for b in range(16):
        result = (a + b) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.ALU_LO + a] = S
        ffn.W_up[unit, BD.FETCH_LO + b] = S
        ffn.b_up[unit] = -S * 15.5  # High threshold like LEA
        ffn.W_gate[unit, BD.OP_ADJ] = 1.0
        ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
        unit += 1

# === ADJ carry detection (120 units) ===
for a in range(16):
    for b in range(16):
        if a + b >= 16:
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.FETCH_LO + b] = S
            ffn.b_up[unit] = -S * 15.5
            ffn.W_gate[unit, BD.OP_ADJ] = 1.0
            ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
            unit += 1
```

### Step 3: L9 FFN - Add ADJ Hi Nibble Units

**Location**: `vm_step.py` line ~4595 in `_set_layer9_alu`

**Add after LEA hi nibble units**:
```python
# === ADJ: hi nibble (768 units: 256 × 3 carry states) ===
for a in range(16):
    for b in range(16):
        for carry_in in [0, 1, 2]:  # 0=no carry, 1=carry, 2=don't care
            if carry_in == 2:
                continue  # Skip don't-care state
            result = (a + b + carry_in) % 16
            has_carry = (a + b + carry_in >= 16)

            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_HI + a] = S
            ffn.W_up[unit, BD.FETCH_HI + b] = S
            if carry_in == 1:
                ffn.W_up[unit, BD.CARRY + 0] = S
                threshold = 18.5  # 4-way AND
            else:
                ffn.W_up[unit, BD.CARRY + 0] = -S
                threshold = 15.5  # 3-way AND + 1 suppress
            ffn.b_up[unit] = -S * threshold
            ffn.W_gate[unit, BD.OP_ADJ] = 1.0
            ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
            if has_carry:
                ffn.W_down[BD.CARRY + 1, unit] = 2.0 / (S * 5.0)
            unit += 1
```

### Step 4: L13 FFN - Route AX Result to SP

**Location**: `vm_step.py` line ~5100 in layer 13 FFN

**Add SP writeback for ADJ**:
```python
# === ADJ: Route AX result → SP OUTPUT ===
# At SP marker when OP_ADJ active: cancel identity, write AX_CARRY
for k in range(16):
    ffn.W_up[unit, BD.OP_ADJ] = S
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
    ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Write result
    ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
    unit += 1

for k in range(16):
    ffn.W_up[unit, BD.OP_ADJ] = S
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0
    ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
    ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
    unit += 1
```

### Step 5: Remove ADJ Handler

**Location**: `neural_vm/run_vm.py` line ~54-60

**Comment out** ADJ from `_RUNNER_VM_MEMORY_OPS`:
```python
_RUNNER_VM_MEMORY_OPS = {
    # Opcode.ADJ,  # Now works fully neurally!
    Opcode.MALC,
    Opcode.FREE,
    Opcode.MSET,
    Opcode.MCMP,
}
```

---

## Testing Plan

### Test 1: Simple ADJ
```c
int main() {
    int a, b, c;  // ADJ -12 (allocate 3 ints)
    a = 1;
    b = 2;
    c = 3;
    return a + b + c;  // ADJ +12 (deallocate)
}
// Expected: 6
```

### Test 2: Large Adjustment
```c
int main() {
    char buf[1000];  // ADJ -1000
    return 42;  // ADJ +1000
}
// Expected: 42
```

### Test 3: Negative Adjustment
```c
int f(int x) {
    int local;  // ADJ -4
    local = x * 2;
    return local;  // ADJ +4
}
int main() {
    return f(21);
}
// Expected: 42
```

### Test 4: Full Test Suite
```bash
python tests/test_suite_1000.py
# Expected: 1096/1096 passing
```

---

## Success Criteria

1. ✅ All 1096 tests pass
2. ✅ ADJ test programs return correct values
3. ✅ Tests/sec >= 8000 (within 5% of current)
4. ✅ ADJ handler can be removed
5. ✅ ENT can now use ADJ for local allocation

---

## Next Steps After ADJ

With ADJ working, we can proceed to:

1. **ENT Partial Implementation** (8 hours)
   - 3 out of 4 operations go neural
   - Only MEM token generation remains

2. **LEV Implementation** (25-35 hours)
   - Parallel L15 lookups
   - Multi-register coordination

3. **Memory Syscalls** (45-65 hours)
   - MALC/FREE/MSET/MCMP full neural

**Total to 100% pure autoregressive**: ~88-118 hours from ADJ completion

---

**Status**: Ready to implement - dimensions allocated, plan complete
**Next Action**: Modify vm_step.py following steps 1-4 above
