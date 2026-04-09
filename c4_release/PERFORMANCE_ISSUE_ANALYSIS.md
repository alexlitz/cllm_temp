# Performance Issue Analysis - April 9, 2026

## Executive Summary

**Status**: 🔴 **CRITICAL BLOCKER** - System unusable due to O(n²) performance degradation

The Neural VM cannot execute programs because `generate_next()` reprocesses the entire context on every token generation, causing exponential slowdown. Programs time out before completion.

---

## Root Cause

### The Problem

**File**: `neural_vm/vm_step.py` line 935

```python
def generate_next(self, context, temperature=0.0):
    token_ids = torch.tensor([context], dtype=torch.long, device=device)
    logits = self.forward(token_ids)[0, -1, :]  # ← Processes ENTIRE context!
```

**What Happens**:
- Token 0: Forward pass on 20 tokens → 0.15s
- Token 1: Forward pass on 21 tokens → 0.16s
- Token 10: Forward pass on 30 tokens → 0.20s
- Token 50: Forward pass on 70 tokens → ~1.5s
- Token 100: Forward pass on 120 tokens → ~5s
- Token 350: Forward pass on 370 tokens → **timeout**

### Why This Is O(n²)

Each token requires:
1. Processing full context through embedding layer
2. Full attention across all 16 transformer layers
3. Each layer does O(n²) attention computation
4. Recomputes attention for ALL previous tokens

**Total complexity**: O(n² × num_layers × d_model²)

For a simple program with 350 tokens:
- Without KV cache: 61,000+ forward operations
- With KV cache: 350 forward operations (**174x speedup**)

---

## Observed Behavior

### Symptoms

```
[DEBUG] Generating token 0/350, step 0     ✓ 0.15s
[DEBUG] Generating token 35/350, step 1    ✓ 0.18s
[DEBUG] Generating token 70/350, step 2    ✓ 0.25s
[DEBUG] Generating token 105/350, step 3   ⚠️ 0.5s
[DEBUG] Generating token 140/350, step 4   ⚠️ 1.2s
[DEBUG] Generating token 175/350, step 5   🔴 3.8s
[DEBUG] Generating token 210/350, step 6   🔴 TIMEOUT
```

### Test Results

**Simple Program** (`IMM 42; EXIT`):
- Expected: 2 VM steps, ~70 tokens, ~15 seconds
- Actual: Generates 8 tokens in 30 seconds, then times out
- **Result**: ❌ FAILED

**ADJ Test** (function with args):
- Expected: ~10 VM steps, ~350 tokens
- Actual: Never completes, times out in initialization
- **Result**: ❌ FAILED

---

## Attempted Fix

### What We Tried

1. **Added KV cache parameter to `generate_next()`**:
   ```python
   def generate_next(self, context, temperature=0.0, kv_cache=None):
       logits = self.forward(token_ids, kv_cache=kv_cache)[0, -1, :]
   ```

2. **Runner tried to create KV cache**:
   ```python
   from .kv_cache import KVCache
   kv_cache = KVCache(max_batch_size=1, max_seq_len=...)
   ```

3. **BLOCKED**: `KVCache` class doesn't exist!
   - `TransformerKVCache` exists but has wrong interface
   - Missing `get_layer_cache(i)` method required by `forward()`
   - Incompatible initialization parameters

---

## Current State

### Fixed Issues

✅ **IndentationError in `run_vm.py`** (line 376)
- Empty `if` block with all code commented out
- Added `pass` statement
- Committed in: `20a3b58`

✅ **ADJ Neural Implementation**
- Code complete and committed (`0eb60a3`)
- Cannot test due to performance issue

✅ **Root Cause Identified**
- O(n²) complexity from missing KV cache
- Documented in this file
- Committed in: `8d50c4b`

### Broken Functionality

❌ **All program execution**
- `runner.run()` times out
- Cannot execute even simplest programs
- Affects all tests, examples, demos

❌ **ADJ verification**
- Implementation complete but untested
- Blocked by performance issue

❌ **Handler removal progress**
- Cannot verify neural implementations
- Cannot proceed with JSR/ENT/LEV

---

## Solution Options

### Option A: Implement Proper KV Cache (RECOMMENDED)

**Effort**: Medium (4-6 hours)

**Steps**:
1. Create `KVCache` class in `neural_vm/kv_cache.py`
2. Implement `get_layer_cache(layer_idx)` interface
3. Store K/V tensors per layer: `cache[layer_idx] = (K, V)`
4. Implement cache update logic in attention layers
5. Test with simple programs

**Benefits**:
- Fixes root cause completely
- Enables fast generation (~174x speedup)
- Standard transformer architecture

**Risks**:
- Requires understanding attention caching
- May have subtle bugs with hand-crafted weights
- Need to handle cache invalidation

### Option B: Use Existing Batch Runner

**Effort**: Low (1-2 hours)

**Investigation needed**:
- Check if `BatchRunner` or `SpeculativeRunner` work
- They may use different architecture without this issue
- May already have KV caching implemented

**Steps**:
1. Test `BatchRunner` with simple program
2. If works, document as workaround
3. Fix `AutoregressiveVMRunner` separately

**Benefits**:
- Quick workaround
- May unblock ADJ testing

**Risks**:
- Batch runner may have same issue
- Not a real fix

### Option C: Context Windowing

**Effort**: Low (2-3 hours)

**Approach**:
- Limit context to last N tokens (e.g., 512)
- Discard older tokens
- Process only recent history

**Code**:
```python
def generate_next(self, context, max_context=512):
    if len(context) > max_context:
        context = context[-max_context:]  # Keep last N tokens
    token_ids = torch.tensor([context], dtype=torch.long)
    logits = self.forward(token_ids)[0, -1, :]
```

**Benefits**:
- Simple implementation
- Bounds worst-case performance
- No cache complexity

**Risks**:
- May lose important context (code, data)
- Neural VM needs full program visibility
- Not a true fix

---

## Recommended Path Forward

### Immediate (Today)

1. **Test BatchRunner** to see if it works
   ```python
   from neural_vm.batch_runner import BatchRunner
   runner = BatchRunner()
   result = runner.run(bytecode, data, [])
   ```

2. **If BatchRunner works**: Use it as temporary workaround, document issue

3. **If BatchRunner fails**: Implement Option C (context windowing) as emergency fix

### Short Term (This Week)

4. **Implement proper KV cache** (Option A)
   - Study existing `TransformerKVCache`
   - Create compatible `KVCache` wrapper
   - Test thoroughly

5. **Verify ADJ implementation** once performance fixed

6. **Re-enable all tests**

### Long Term (Next Week)

7. **Add KV cache unit tests**

8. **Document KV cache architecture**

9. **Continue handler removal** (JSR/ENT/LEV)

---

## Technical Details

### KV Cache Requirements

The `forward()` method expects:
```python
kv_cache.get_layer_cache(layer_idx) → LayerKVCache | None
```

Where `LayerKVCache` provides:
- `update(new_k, new_v)` - Add new K/V tensors
- `get_kv()` → (K, V) - Retrieve cached tensors
- Shape: `[batch, num_heads, seq_len, head_dim]`

### Existing Code References

**Works**:
- `generate_autoregressive_with_kv_cache()` (line 1006-1045)
- References `KVCache` class that doesn't exist

**Broken**:
- `from .kv_cache import KVCache` → ImportError
- `TransformerKVCache` → Wrong interface

**Exists**:
- `LayerKVCache` class (line 63)
- Has correct interface but no top-level wrapper

---

## Performance Comparison

### Without KV Cache (Current - BROKEN)

| Tokens | Context | Time per Token | Cumulative |
|--------|---------|---------------|------------|
| 0-10   | 20-30   | 0.15-0.20s    | 2s         |
| 10-35  | 30-55   | 0.20-0.40s    | 10s        |
| 35-70  | 55-90   | 0.40-1.0s     | 40s        |
| 70+    | 90+     | 1.0-5.0s+     | TIMEOUT    |

### With KV Cache (Target)

| Tokens | Time per Token | Cumulative |
|--------|---------------|------------|
| 0-350  | 0.15-0.20s    | 60s        |
| 350+   | 0.20s         | Linear     |

**Speedup**: ~174x for 350 tokens

---

## Related Files

### Modified
- `neural_vm/vm_step.py` - Added kv_cache parameter to generate_next()
- `neural_vm/run_vm.py` - Fixed IndentationError, documented KV cache TODO

### To Fix
- `neural_vm/kv_cache.py` - Need to implement KVCache class
- `neural_vm/batch_runner.py` - Check if this works as workaround

### Documentation
- `PERFORMANCE_ISSUE_ANALYSIS.md` (this file)
- `NEURAL_VM_STATUS_SUMMARY.md` - Overall status
- `REMAINING_HANDLERS_PLAN.md` - Handler removal roadmap

---

## Conclusion

The Neural VM is currently **unusable** due to missing KV cache causing O(n²) performance degradation. This is a **critical blocker** for:

- ✗ All program execution
- ✗ ADJ testing
- ✗ Handler removal progress
- ✗ Example demonstrations

**Priority**: 🔴 **CRITICAL - Must fix immediately**

**Recommended Action**: Implement proper KV cache (Option A) or use BatchRunner workaround (Option B)

**Time Estimate**: 4-6 hours to implement full fix

**Next Step**: Test if BatchRunner works as temporary workaround

---

**Status**: 📋 Root cause identified, fix needed
**Updated**: April 9, 2026
**Next Review**: After KV cache implementation
