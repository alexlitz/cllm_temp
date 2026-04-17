# BYTE_INDEX FFN Activation Bug - Findings and Solutions

**Date:** 2026-04-10
**Status:** Critical bug blocking JSR/LEV neural implementation

## Summary

L14 MEM generation relies on BYTE_INDEX_0/1/2/3 flags to distinguish between byte positions within registers. However, **BYTE_INDEX_1/2/3 are not being set by Layer 1 FFN**, causing L14 to read from wrong byte positions and generating corrupt MEM addresses.

## Bug Evidence

### Test Output
MEM address bytes show byte 3 copying byte 0:
```
Raw MEM section tokens: [261, 248, 255, 0, 248, 0, 0, 0, 0]
Address bytes: ['f8', 'ff', '00', 'f8'] → 0xf800fff8
                                    ^^^^
Expected:      ['f8', 'ff', '00', '00'] → 0x0000fff8
```

### BYTE_INDEX Values

**SP bytes (positions 11-14):**
```
SP byte 0 (pos 11): BYTE_INDEX_0=0.970, others=0.000
SP byte 1 (pos 12): ALL = 0.000  ← BUG!
SP byte 2 (pos 13): ALL = 0.000  ← BUG!
SP byte 3 (pos 14): ALL = 0.000  ← BUG!
```

**Expected:** Only one BYTE_INDEX should be 1.0 at each byte position:
- Byte 0: BYTE_INDEX_0 = 1.0
- Byte 1: BYTE_INDEX_1 = 1.0
- Byte 2: BYTE_INDEX_2 = 1.0
- Byte 3: BYTE_INDEX_3 = 1.0

### FFN Input Verification

At SP byte 1 (position 12), Layer 1 FFN inputs are CORRECT:
```
IS_BYTE:  1.000 ✓
L1H2:     0.987 ✓ (any marker within dist 2.5)
L1H1:     0.003 ✓ (no marker within dist 1.5)
```

**Expected activation:**
```python
up = S * (IS_BYTE + L1H2 - 1.5) = 100 * (1.0 + 0.987 - 1.5) = 48.7 > 0 ✓
gate = 1.0 - L1H1 = 1.0 - 0.003 = 0.997 ≈ 1.0 ✓
output = swiglu(48.7, 0.997) * (2.0 / 100) ≈ 0.02
```

**Actual output:** 0.000 (unit not activating)

## Root Cause Analysis

### BYTE_INDEX Computation (vm_step.py lines 2193-2233)

```python
# BYTE_INDEX_1: IS_BYTE AND L1H2 AND NOT L1H1
ffn.W_up[unit, BD.IS_BYTE] = S
for i in range(NM):
    ffn.W_up[unit, BD.L1H2 + i] = S
ffn.b_up[unit] = -S * 1.5
for i in range(NM):
    ffn.W_gate[unit, BD.L1H1 + i] = -1.0
ffn.b_gate[unit] = 1.0
ffn.W_down[BD.BYTE_INDEX_1, unit] = 2.0 / S
```

**Issue:** The FFN unit has correct inputs but produces zero output.

### Possible Causes

1. **SwiGLU Implementation Bug**: The FFN forward pass may have a bug in activation or gating logic
2. **Residual Connection**: BYTE_INDEX output may be getting zeroed by residual stream
3. **Weight Initialization**: FFN weights may not be initialized correctly
4. **Dimension Conflict**: BYTE_INDEX dimensions may be overwritten by another layer

## Impact

### Affected Operations

**Directly Blocked:**
- JSR MEM generation (corrupt addresses)
- Any MEM operation with multi-byte addresses

**Indirectly Blocked:**
- LEV (requires JSR to work)
- Full neural VM (needs LEV)

### Current Workaround

JSR handler override still active in run_vm.py:1520-1530, providing correct MEM tokens but blocking neural execution.

## Proposed Solutions

### Option 1: Fix Layer 1 FFN (Preferred)

**Pros:** Fixes root cause, enables all BYTE_INDEX users
**Cons:** Requires deep FFN debugging, may uncover more bugs

**Steps:**
1. Add FFN forward pass logging to trace activation
2. Check if SwiGLU gate computation is correct
3. Verify residual connection doesn't zero BYTE_INDEX
4. Test with isolated FFN unit

### Option 2: Replace BYTE_INDEX with Hop-Count Matching

**Pros:** Avoids FFN bug, proven to work for H dimensions
**Cons:** More complex attention logic, higher compute cost

**Steps:**
1. Modify L14 addr heads 1-3 to use hop-count matching like L14 addr head 0
2. For byte 1: Use L1H1 - L1H0 (fires at d∈(0.5, 1.5])
3. For byte 2: Use L1H2 - L1H1 (fires at d∈(1.5, 2.5])
4. For byte 3: Use H0 - L1H2 (fires at d∈(2.5, 3.5])

### Option 3: L14 Reads PC Directly for JSR (Implemented)

**Status:** Implemented in vm_step.py:6029-6036
**Pros:** Simple, bypasses STACK0/BYTE_INDEX issues
**Cons:** Only fixes JSR, doesn't fix ENT or general BYTE_INDEX bug

**Current implementation:**
```python
# L14 heads 4-7: Read from PC bytes for JSR
attn.W_k[base + 2, BD.H1 + PC_I] = L  # Match PC area
```

**Issue:** Still relies on L14 addr heads working correctly, which need BYTE_INDEX_1/2/3!

## Next Steps

### Immediate (Fix JSR)

1. **Verify Option 3 works** - Test if L14 reading PC fixes JSR MEM values
2. **If not**: Implement Option 2 for L14 addr heads to fix address corruption
3. **Then**: Re-test JSR MEM generation

### Short-term (Fix BYTE_INDEX)

1. **Debug FFN activation** - Add logging to Layer 1 FFN forward pass
2. **Identify bug** - Pinpoint why units don't activate despite correct inputs
3. **Fix and test** - Repair bug, verify all BYTE_INDEX_0/1/2/3 work

### Long-term (100% Neural)

1. Fix JSR (MEM addresses + values)
2. Fix ENT (already mostly working)
3. Fix LEV (blocked by JSR)
4. Remove all handlers
5. Achieve 100% neural VM

## Related Files

- `neural_vm/vm_step.py:2193-2233` - BYTE_INDEX computation
- `neural_vm/vm_step.py:5907-5995` - L14 addr heads (need BYTE_INDEX)
- `neural_vm/vm_step.py:6000-6053` - L14 val heads (need BYTE_INDEX)
- `neural_vm/vm_step.py:7329-7412` - JSR SP -= 8 (needs BYTE_INDEX)
- `neural_vm/run_vm.py:1520-1530` - JSR handler (workaround)

## Debug Commands

### Check BYTE_INDEX at all positions:
```bash
python -c "
import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim as BD, Token, set_vm_weights
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)
context = [Token.REG_PC, 0, 0, 0, 0, Token.REG_AX, 0, 0, 0, 0,
           Token.REG_SP, 0, 0, 1, 0, Token.REG_BP, 0, 0, 1, 0,
           Token.STACK0, 0, 0, 0, 0, Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0, Token.STEP_END]
x = model.embed(torch.tensor([context], dtype=torch.long))
x = model.blocks[0].attn(x); x = model.blocks[0].ffn(x)
x = model.blocks[1].attn(x); x = model.blocks[1].ffn(x)
for pos in [11, 12, 13, 14]:
    print(f'SP byte {pos-11} (pos {pos}):')
    for i in range(4):
        val = x[0, pos, BD.BYTE_INDEX_0 + i * 9].item()
        print(f'  BYTE_INDEX_{i}: {val:.3f}')
"
```

### Check MEM generation:
```bash
python debug_mem_tokens.py 2>&1 | grep "Address bytes"
```

## Commit History

- `fafb2fc` - Refactor PureFFN and PureAttention to use nn.Linear
- `acab975` - Fix L10 byte passthrough, L15 memory lookup, DivMod binarize
- `2d14419` - Merge: GPU support + batch runners
- `34aa9b3` - Fix JSR neural SP -= 8 and STACK0 byte writes (INCOMPLETE - BYTE_INDEX bug blocks)
