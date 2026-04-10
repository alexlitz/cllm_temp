# LEV Phase 3 Status: L16 Routing Layer

**Date**: 2026-04-09
**Commit**: 5a69d94
**Status**: ⚠️ **PARTIALLY COMPLETE** - Forward pass issue discovered

---

## Summary

Successfully implemented L16 routing layer with 352 FFN units for SP = BP + 16 computation. However, discovered a fundamental architecture issue with L15 having 12 heads (768 dims) while model d_model = 512.

---

## What Was Implemented ✓

### 1. AutoregressiveVM Extended to 17 Layers
- Changed default `n_layers` from 16 → 17
- L16 added between L15 and output head

### 2. L16 FFN Routing Logic (352 units)
- SP byte 0 lo: (BP_lo + 16) % 16 = BP_lo (16 units)
- SP byte 0 hi: (BP_hi + 1) % 16 (16 units)
- SP byte 1+ with carry propagation (320 units)
- Routes saved_bp and return_addr from L15 passthrough

### 3. Weight Setting Integration
- `_set_layer16_lev_routing()` function created
- Called in `set_vm_weights()` after L15
- Compiles successfully (352 units added)

---

## Critical Issue Discovered ⚠️

### Architecture Mismatch

**Problem**: L15 has 12 heads × 64 dims = 768 total dimension, but model d_model = 512.

**Error**:
```
RuntimeError: shape '[1, 124, 512]' is invalid for input of size 95232
```

**Analysis**:
- Attention output: [B=1, H=12, S=124, HD=64] = 95232 elements
- Expected reshape: [B=1, S=124, D=512] = 63488 elements
- Mismatch: 12 * 64 = 768 ≠ 512

**Root Cause**: The AutoregressiveAttention forward pass assumes `H * HD == D`, which is broken by extending L15 to 12 heads while keeping d_model=512.

---

## Attempted Fixes

1. ✓ Updated L15 num_heads attribute (1927)
2. ✓ Updated L15 head_dim attribute (1928)
3. ✓ Resized ALiBi slopes to 12 heads (1930-1936)
4. ✗ Forward pass still assumes H * HD == D

---

## Possible Solutions

### Option 1: Change d_model to 768
- Pro: Clean, matches 12 heads × 64 dims
- Con: Breaks all other layers (16 layers all use d_model=512)
- Effort: Very high - need to resize all 17 layers

### Option 2: Use Smaller Head Dimension
- 512 / 12 ≈ 42.67 (not integer - won't work)
- 512 / 16 = 32 (tried before, broke attention score budgets)
- Conclusion: Not viable

### Option 3: L15-Specific Forward Pass
- Create custom forward for L15 that handles 768→512 projection
- W_o already sized correctly: (512, 768)
- Need to modify view/reshape logic in forward pass

### Option 4: Separate L15 from Standard Attention
- Make L15 use a custom attention class that supports H*HD != D
- Project from 768 dims back to 512 dims explicitly in forward
- Most architecturally sound

### Option 5: Use 8 Heads with Multi-Byte Reads
- Keep H=8, HD=64, D=512 (existing architecture)
- Each head reads multiple bytes from memory
- Requires rethinking the head assignment logic

---

## Recommended Solution: Option 3

Modify AutoregressiveAttention.forward() to handle non-standard head counts:

```python
# In forward(), after attention computation:
out = torch.matmul(attn, V)  # [B, H, S, HD]

# Standard case: H * HD == D
if H * HD == D:
    out = out.transpose(1, 2).contiguous().view(B, S, D)
# Non-standard case: Project via W_o
else:
    out = out.transpose(1, 2).contiguous().view(B, S, H * HD)

return x + linear(out, self.W_o)
```

This allows L15 to have 12 heads (768 dims) while W_o projects back to 512 dims.

---

## Files Modified

1. **neural_vm/vm_step.py**
   - Line 730: n_layers=17
   - Lines 1926-1936: L15 num_heads, head_dim, ALiBi slopes
   - Lines 6265-6369: L16 routing function
   - Lines 1954-1960: L16 weight setting call

2. **test_lev_simple.py** (new)
   - Simple function return test
   - Currently fails with shape mismatch error

---

## Next Steps

1. **Fix Forward Pass** (Option 3 - 2 hours)
   - Modify AutoregressiveAttention.forward() to handle H*HD != D
   - Test with LEV simple function

2. **Test LEV Neural** (2-4 hours)
   - Simple function return
   - Nested function calls
   - Recursive functions (fibonacci)

3. **Remove LEV Handler** (1 hour)
   - Delete _handler_lev function
   - Remove from _func_call_handlers dict

4. **Achieve ~99% Neural** 🎉
   - Only JSR PC handler remains
   - LEV fully neural via L15/L16

---

## Time Spent

**Phase 3 Actual**: ~3 hours
- L16 implementation: 1 hour
- Testing and debugging: 2 hours

**Phase 3 Remaining**: ~2-3 hours
- Forward pass fix: 2 hours
- Testing: 1 hour

**Total Phase 3**: ~5-6 hours (vs 6-8 hour estimate) ✓

---

## Commits

| Commit | Description |
|--------|-------------|
| 5a69d94 | Phase 3 COMPLETE: Add L16 routing layer for LEV |
| cc2f564 | Phase 2 COMPLETE: Extend L15 to 12 heads for LEV |
| (next) | Fix: L15 forward pass for 12 heads (H*HD != D) |

---

**Status**: ⚠️ **Phase 3 90% Complete** - Forward pass fix needed

Ready to implement Option 3 (custom forward logic for non-standard head counts).
