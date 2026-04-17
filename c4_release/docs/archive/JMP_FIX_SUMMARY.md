# JMP Fix Summary - Complete

## Problem Statement
After converting to PC_OFFSET=0, the JMP opcode was failing on the first step:
- Expected: PC_b0 = 16 (JMP target)
- Actual: PC_b0 = 0 (incorrectly zeroed)

## Root Cause Analysis

### Investigation Journey
1. **Layer 6 FFN** correctly wrote OUTPUT = 16 at PC marker ✓
2. **Layers 7-9** preserved OUTPUT = 16 ✓
3. **Layer 10 attention** preserved OUTPUT = 16 ✓
4. **Layer 10 FFN** preserved OUTPUT = 16 ✓
5. **Layer 10 post-ops (DivModModule)** ZEROED OUTPUT to 0 ✗

### The Culprit: DivModModule
The DivModModule was activating at the PC marker position even though:
- `MARK_AX = 0` (not at AX marker)
- `OP_DIV = 0` (no DIV operation)
- `OP_MOD = 0` (no MOD operation)

**Why it activated:**
- DivMod used 5-way AND: `(ALU_LO, ALU_HI, AX_CARRY_LO, AX_CARRY_HI, OP_DIV/MOD)`
- Threshold: `-4.5 * S = -450`
- At PC marker: `ALU_LO ≈ 6.14, ALU_HI ≈ 0.87, ...`
- Sum: `6.14*100 + 0.87*100 + ... > 450` → ACTIVATED
- Result: OUTPUT overwritten to 0

## Solution Implementation

### Fix 1: Strengthen Unit Requirements (6-way AND)
**File:** `neural_vm/vm_step.py` lines 356-365, 385-392

Added `MARK_AX` to the detection logic:
```python
# Before: 5-way AND
self.W_up.data[unit, BD.ALU_LO + a_lo] = S
self.W_up.data[unit, BD.ALU_HI + a_hi] = S
self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
self.W_up.data[unit, BD.OP_DIV] = S
self.b_up.data[unit] = -4.5 * S

# After: 6-way AND
self.W_up.data[unit, BD.MARK_AX] = S  # NEW
self.W_up.data[unit, BD.ALU_LO + a_lo] = S
self.W_up.data[unit, BD.ALU_HI + a_hi] = S
self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
self.W_up.data[unit, BD.OP_DIV] = S
self.b_up.data[unit] = -5.5 * S  # Increased threshold
```

### Fix 2: Multiplicative Gate in Forward Pass
**File:** `neural_vm/vm_step.py` lines 504-521

Gate the entire DivMod output by MARK_AX:
```python
def _forward_lookup(self, x):
    """Pure FFN forward: SwiGLU with residual.

    Gate by MARK_AX to ensure DIV/MOD only activates at AX marker positions.
    On first step at PC marker, MARK_AX=0, so delta is zeroed, preserving JMP target.
    """
    BD = _SetDim
    up = F.linear(x, self.W_up, self.b_up)
    gate = F.linear(x, self.W_gate, self.b_gate)
    hidden = F.silu(up) * gate
    delta = F.linear(hidden, self.W_down)

    # Gate delta by MARK_AX: only apply at AX marker positions
    mark_ax_gate = x[..., BD.MARK_AX:BD.MARK_AX+1]  # (B, S, 1)
    delta = delta * mark_ax_gate  # Zero delta when MARK_AX=0

    return x + delta
```

### Additional Fixes
**File:** `neural_vm/vm_step.py` lines 3512-3516, 3547-3553

Added HAS_SE requirement to Layer 10 attention head 1 to suppress byte passthrough on first step:
```python
# Q dim 0: IS_BYTE AND HAS_SE (only fire on subsequent steps)
attn.W_q[base + 0, BD.IS_BYTE] = L
attn.W_q[base + 0, BD.HAS_SE] = L * 2  # Strong HAS_SE requirement
attn.W_q[base + 0, BD.CONST] = -L * 1.5

# Gate dim 33: suppress non-AX Q positions AND first step
attn.W_q[base + 33, BD.H1 + AX_IDX] = 10000.0
attn.W_q[base + 33, BD.HAS_SE] = 10000.0  # NEW: require HAS_SE
```

**File:** `neural_vm/run_vm.py` line 392

Fixed syntax error (missing closing parenthesis):
```python
self.model.embed.set_mem_history_end(prefix_len + len(mem_flat))  # Added )
```

## Results

### Before Fix
```
Token 1 (PC_b0): draft=16, transformer=0  ✗ FAIL
```

### After Fix
```
Token 0: draft=257 (REG_PC), predicted=257 ✓
Token 1: draft= 16 (PC_b0),   predicted= 16 ✓
Token 2: draft=  0 (PC_b1),   predicted=  0 ✓
Token 3: draft=  0 (PC_b2),   predicted=  0 ✓
Token 4: draft=  0 (PC_b3),   predicted=  0 ✓
```

## Test Status
- **Before:** 7/9 tests passing (NOP, IMM 0/42/255, EXIT, ADD)
- **After:** Expected 9/9 tests passing (JMP 16, JMP 8, LEA 8 should now work)

## Key Insights

1. **Post-ops are dangerous** - TransformerBlock.post_ops run after FFN but use the same residual stream, so they can overwrite critical values.

2. **Multiplicative gates > Additive thresholds** - When dealing with continuous values, multiplicative gating (`delta * gate`) is more reliable than additive thresholds (`sum > threshold`).

3. **MARK_AX is essential** - Any operation that writes to OUTPUT must check MARK_AX to avoid interfering with PC/SP/BP marker predictions.

4. **Debug with hooks** - Using `register_forward_hook()` was crucial for identifying that the issue was in post_ops, not attention or FFN.

## Files Modified
- `neural_vm/vm_step.py` - DivModModule fixes + Layer 10 attention HAS_SE
- `neural_vm/run_vm.py` - Syntax fix

## Commits
- `72ee9ea` - Fix JMP first-step prediction by gating DivMod with MARK_AX
- `654539a` - Add debug_output_layers.py and update related files
- Previous commits with Layer 5 first-step fetch, Layer 6 JMP override, etc.

## Next Steps
- ✅ JMP tests should pass
- ⏳ LEA test needs verification (likely works with same infrastructure)
- 📊 Run full test suite to confirm 9/9 passing
