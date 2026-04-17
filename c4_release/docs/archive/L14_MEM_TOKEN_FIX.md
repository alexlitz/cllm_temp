# L14 MEM Token Generation Fix

**Date**: 2026-04-09
**Status**: Solution Identified - Ready to Implement
**Estimated Effort**: 4-6 hours

---

## Problem Statement

**Current Behavior**: L14 value heads (heads 4-7) copy MEM value bytes from **AX only**

**Needed**: JSR and ENT need to copy from **STACK0** instead of AX

**Impact**: Blocks neural implementation of JSR and ENT

---

## Current L14 Implementation

### Value Heads (4-7) - Current Code

```python
# === Heads 4-7: MEM val byte generation ===
for h in range(4):
    head = 4 + h
    base = head * HD

    # K targets AX byte positions ONLY
    attn.W_k[base, byte_idx_dim] = L
    attn.W_k[base, BD.H1 + AX_I] = L  # AX area bonus

    # V: copy CLEAN_EMBED from AX
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
```

**Current Support**:
- ✅ PSH: Copies from AX (pushes accumulator value)
- ✅ SI/SC: Copies from AX (stores result)
- ❌ JSR: Needs STACK0 (return address)
- ❌ ENT: Needs STACK0 (old BP)

---

## Solution Design

### Add Conditional STACK0 Support

Similar to how address heads use `MEM_ADDR_SRC`, create `MEM_VAL_SRC`:
- MEM_VAL_SRC = 0 → Copy from **AX** (PSH, SI, SC)
- MEM_VAL_SRC = 1 → Copy from **STACK0** (JSR, ENT)

### Implementation Steps

#### Step 1: Add MEM_VAL_SRC Dimension

**Location**: Dimension registry (if not exists) or use existing OP flags

**Option A**: Use existing opcode flags
```python
# In query: add conditional based on OP_JSR or OP_ENT
attn.W_q[base + X, BD.OP_JSR] = bonus
attn.W_q[base + X, BD.OP_ENT] = bonus
```

**Option B**: Create new dimension
```python
# Add to dim_registry.py
MEM_VAL_SRC = X  # 1 dim, 0=AX, 1=STACK0
```

**Recommendation**: Option A (use existing OP flags) - simpler, no new dimensions needed

#### Step 2: Modify Value Head K (Attention Keys)

**Current**:
```python
# K targets AX only
attn.W_k[base, byte_idx_dim] = L
attn.W_k[base, BD.H1 + AX_I] = L  # AX area bonus
```

**Modified**:
```python
# K targets AX or STACK0 based on opcode
attn.W_k[base, byte_idx_dim] = L

# Dim 1: AX source bonus (PSH, SI, SC - default)
attn.W_q[base + 1, BD.CONST] = L  # Default to AX
attn.W_q[base + 1, BD.OP_JSR] = -L  # Not when JSR
attn.W_q[base + 1, BD.OP_ENT] = -L  # Not when ENT
attn.W_k[base + 1, BD.H1 + AX_I] = L  # AX area
attn.W_k[base + 1, BD.H1 + BP_I] = -L  # Exclude BP (overlaps)

# Dim 2: STACK0 source bonus (JSR, ENT only)
attn.W_q[base + 2, BD.OP_JSR] = L  # Active for JSR
attn.W_q[base + 2, BD.OP_ENT] = L  # Active for ENT

# For each byte, attend to STACK0 positions
if h == 0:
    # Byte 0: STACK0_BYTE0 flag (d=6 from BP)
    attn.W_k[base + 2, BD.STACK0_BYTE0] = L
elif h == 1:
    # Byte 1: H2[BP] - L1H4[BP] (d=7 from BP)
    attn.W_k[base + 2, BD.H2 + BP_I] = L
    attn.W_k[base + 2, BD.L1H4 + BP_I] = -L
elif h == 2:
    # Byte 2: H3[BP] - H2[BP] (d=8 from BP)
    attn.W_k[base + 2, BD.H3 + BP_I] = L
    attn.W_k[base + 2, BD.H2 + BP_I] = -L
elif h == 3:
    # Byte 3: H4[BP] - H3[BP] (d=9 from BP)
    attn.W_k[base + 2, BD.H4 + BP_I] = L
    attn.W_k[base + 2, BD.H3 + BP_I] = -L

# Suppress non-STACK0 areas
attn.W_k[base + 2, BD.H1 + AX_I] = -L
attn.W_k[base + 2, BD.H1 + SP_I] = -L
attn.W_k[base + 2, BD.MARK_STACK0] = -L
```

**Note**: This mirrors the pattern used in address heads (heads 0-3).

#### Step 3: Test with JSR and ENT

**Test JSR**:
```c
int helper() { return 42; }
int main() { return helper(); }
```

Expected MEM section after JSR:
- Address: Stack position (SP - 8)
- Value: Return address (from STACK0, set by JSR)

**Test ENT**:
```c
int func(int x) {
    int local;
    local = x * 2;
    return local;
}
int main() { return func(21); }
```

Expected MEM section after ENT:
- Address: Stack position (SP - 8)
- Value: Old BP (from STACK0, set by ENT)

---

## Detailed Implementation

### Code Changes to vm_step.py

**Location**: Lines 5604-5647 (_set_layer14_mem_generation, value heads)

**Before**:
```python
# === Heads 4-7: MEM val byte generation ===
for h in range(4):
    head = 4 + h
    base = head * HD
    pos_up, pos_down = val_pos[h]
    byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][h]

    # Dim 0: Q position selection + K targets AX byte positions.
    attn.W_q[base, pos_up] = L
    attn.W_q[base, pos_down] = -L
    attn.W_k[base, byte_idx_dim] = L
    attn.W_k[base, BD.H1 + AX_I] = L  # AX area bonus (+28 points)

    # Dim 33: Position gate ...
    # Dim 34: MEM_STORE gate ...

    # V: copy CLEAN_EMBED nibbles (AX bytes have correct value here)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
```

**After**:
```python
# === Heads 4-7: MEM val byte generation ===
for h in range(4):
    head = 4 + h
    base = head * HD
    pos_up, pos_down = val_pos[h]
    byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][h]

    # Dim 0: Q position selection + K targets byte positions
    attn.W_q[base, pos_up] = L
    attn.W_q[base, pos_down] = -L
    attn.W_k[base, byte_idx_dim] = L

    # Dim 1: AX source bonus (default: PSH, SI, SC)
    # Q = L by default, -L when JSR/ENT
    attn.W_q[base + 1, BD.CONST] = L
    attn.W_q[base + 1, BD.OP_JSR] = -2 * L  # Disable for JSR
    attn.W_q[base + 1, BD.OP_ENT] = -2 * L  # Disable for ENT
    attn.W_k[base + 1, BD.H1 + AX_I] = L  # AX area bonus
    attn.W_k[base + 1, BD.H1 + BP_I] = -L  # Exclude BP area

    # Dim 2: STACK0 source bonus (JSR, ENT only)
    # Q = L when JSR or ENT, 0 otherwise
    attn.W_q[base + 2, BD.OP_JSR] = L
    attn.W_q[base + 2, BD.OP_ENT] = L

    # K: STACK0 byte positions (same pattern as address heads)
    if h == 0:
        attn.W_k[base + 2, BD.STACK0_BYTE0] = L
    elif h == 1:
        attn.W_k[base + 2, BD.H2 + BP_I] = L
        attn.W_k[base + 2, BD.L1H4 + BP_I] = -L
    elif h == 2:
        attn.W_k[base + 2, BD.H3 + BP_I] = L
        attn.W_k[base + 2, BD.H2 + BP_I] = -L
    elif h == 3:
        attn.W_k[base + 2, BD.H4 + BP_I] = L
        attn.W_k[base + 2, BD.H3 + BP_I] = -L

    # Suppress non-STACK0 areas
    attn.W_k[base + 2, BD.H1 + AX_I] = -L
    attn.W_k[base + 2, BD.H1 + SP_I] = -L
    attn.W_k[base + 2, BD.MARK_STACK0] = -L

    # Dim 33: Position gate (unchanged)
    attn.W_q[base + 33, BD.CONST] = -500.0
    attn.W_q[base + 33, pos_up] = 500.0
    attn.W_q[base + 33, pos_down] = -500.0
    attn.W_k[base + 33, BD.CONST] = 5.0

    # Dim 34: MEM_STORE gate (unchanged)
    attn.W_q[base + 34, BD.CONST] = -250.0
    attn.W_q[base + 34, BD.MEM_STORE] = 250.0
    attn.W_k[base + 34, BD.CONST] = 5.0

    # V: copy CLEAN_EMBED nibbles (from AX or STACK0)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # V[0]: cancel L3 MEM default
    attn.W_v[base + 0, BD.CONST] = 1.0

    # O: write to OUTPUT (unchanged)
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
    attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
    attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0
```

---

## Testing Plan

### Test 1: JSR MEM Token

```c
int helper() { return 42; }
int main() { return helper(); }
```

**Expected**:
1. JSR executes
2. STACK0 set to return address
3. L14 generates MEM token with:
   - Address: SP - 8
   - Value: Return address (from STACK0)
4. LEV can load return address from memory

### Test 2: ENT MEM Token

```c
int func(int x) {
    int local;
    local = x;
    return local;
}
int main() { return func(42); }
```

**Expected**:
1. ENT executes
2. STACK0 set to old BP
3. L14 generates MEM token with:
   - Address: SP - 8
   - Value: Old BP (from STACK0)
4. LEV can restore BP from memory

### Test 3: Ensure PSH Still Works

```c
int main() {
    int x = 5;
    int y = 3;
    return x + y;  // Binary ADD pops and adds
}
```

**Expected**:
- PSH still uses AX source (not broken by STACK0 addition)
- Returns 8

---

## Impact Analysis

### Operations Unblocked

1. **JSR** - Can generate MEM token for return address
2. **ENT** - Can generate MEM token for old BP

### Operations Still Needed

After L14 fix:
- JSR: Still needs PC override, STACK0 = return_addr (handler or neural)
- ENT: Still needs SP adjustment, BP copy (handler or partial neural)

But the critical blocker (MEM token) is removed.

---

## Effort Estimate

**Implementation**: 2-3 hours
- Modify L14 value heads (1 hour)
- Add dimension checks (30 min)
- Code review and cleanup (30 min)

**Testing**: 2-3 hours
- Create test programs (30 min)
- Run and debug (1-2 hours)
- Verify no regressions (30 min)

**Total**: 4-6 hours

---

## Next Steps After L14 Fix

1. **JSR Neural Implementation** (3-5 hours)
   - PC override already works (L6 FFN)
   - STACK0 = return_addr (add to L6 FFN)
   - Remove handler

2. **ENT Neural Implementation** (5-8 hours)
   - Register copies (BP→STACK0, SP→BP)
   - SP adjustment (complex, may keep handler)
   - Remove or simplify handler

**Total Path**: 12-19 hours to complete JSR + ENT

---

**Status**: Solution designed, ready to implement
**Blocker**: None
**Next Action**: Modify L14 value heads in vm_step.py
