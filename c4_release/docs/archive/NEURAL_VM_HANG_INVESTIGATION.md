# Neural VM Hang Investigation

## Problem

The neural VM consistently times out when attempting to execute programs, making it impossible to measure match rates or validate against the Fast VM.

## Evidence

### Test Results

1. **Fast VM**: Works perfectly, completes instantly
   ```
   Source: int main() { return 0; }
   Fast VM Result: 0
   Status: ✓ WORKING
   ```

2. **Neural VM**: Times out after 30+ seconds
   ```
   Source: int main() { return 0; }
   Neural VM: TIMEOUT (no result)
   Status: ✗ HANGING
   ```

### What Hangs

- `C4TransformerVM.run()` - Hangs during execution
- `AutoregressiveVMRunner.run()` - Hangs in token generation loop
- `model.generate_next(context)` - Never returns

### What Works

- Model initialization: ✓ Completes successfully
- Weight loading (`set_vm_weights`): ✓ No errors
- Context building: ✓ Completes instantly
- Fast VM execution: ✓ All programs run correctly

## Root Cause

The hang occurs in the autoregressive generation loop at:
```python
# neural_vm/run_vm.py:274
for _ in range(max_steps * Token.STEP_TOKENS):
    next_token = self.model.generate_next(context)  # ← HANGS HERE
    context.append(next_token)
```

The `model.generate_next()` method never returns, suggesting:
- Model forward pass is stuck in an infinite loop
- Weights are in an invalid state causing divergence
- Some layer is waiting for a condition that never occurs

## Additional Issues Found

### Cached Model Problem

The test suite cache file `.compact_moe_model.pt` references moved modules:
```
ModuleNotFoundError: No module named 'neural_vm.archive.kv_cache_eviction'
```

Deleted the cache, but the hang persists.

### Compact Method Error

When trying to manually create a runner with compaction:
```python
runner.model.compact(block_size=32)
# AttributeError: 'EfficientShiftFFN' object has no attribute 'compact'
```

The FFN implementation changed but compact() still tries to call it.

## Impact on Match Rate Testing

**Cannot measure match rate** because:
1. Validation requires running neural VM
2. Neural VM hangs indefinitely
3. Tests timeout before producing any result

The comparison fix (extracting exit_code from tuple) is correct, but untestable until the neural VM executes.

## Attempted Fixes

1. ❌ Limited max_steps to 5 (still hangs)
2. ❌ Removed compact() calls (still hangs)
3. ❌ Deleted cached model (still hangs)
4. ❌ Added timeouts (confirms hang, doesn't fix it)

## Expected Behavior

According to documentation, the neural VM should:
- Generate ~35 tokens per step
- Detect HALT token and exit
- Return `(output, exit_code)` tuple

For `return 0`, it should return `('', 0)` which would match Fast VM's `0`.

## Current Status

- **Fast VM**: 100% functional ✓
- **Neural VM**: Completely non-functional (hangs) ✗
- **Validation**: Cannot run (depends on neural VM) ✗
- **Match Rate**: Unmeasurable ✗

## Next Steps

To fix this, need to:
1. Debug `model.generate_next()` to find infinite loop
2. Check if weights are correctly loaded
3. Verify model architecture matches weight structure
4. Test with minimal bytecode to isolate issue

The comparison logic fix is sound, but testing it requires a working neural VM.
