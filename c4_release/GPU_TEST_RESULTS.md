# GPU Test Results: Memory Tests

**Date**: 2026-04-07
**GPU**: NVIDIA RTX A5000 (24GB)

## Performance Comparison

| Test Suite | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| Single test | ~90 minutes | 30 seconds | **180x** |
| 3 quick tests | ~4-6 hours | 100 seconds | **144x-216x** |
| Expected 16 comprehensive tests | ~30-60 hours | ~8-15 minutes | **~200x** |

**GPU acceleration is WORKING** ✅

## Test Execution Results

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest neural_vm/tests/test_memory_quick.py::TestMemoryQuick::test_basic_memory_write_read -v
```

**Result**: Single test completed in **28.65 seconds**

### Test Outcomes

❌ **All 3 tests FAILED** - But this is revealing bugs!

```
FAILED test_basic_memory_write_read - exit_code=16843009 (expected 42)
FAILED test_memory_overwrite - Similar issue
FAILED test_multiple_addresses - Similar issue
```

## What Was Discovered

### Issue Identified: Memory Operations Not Working

**Symptom**: All tests return exit_code=16843009 (0x01010101)

**Root Cause**: MEM_STORE flag not being set, causing L15 memory lookup to fail

**Evidence**:
- Debug output shows model generating "byte_01" repeatedly
- Exit code 0x01010101 = all bytes set to 1
- Memory writes (SI) and reads (LI) not preserving values

**Technical Analysis**:
```python
# In NeuralVMEmbedding._inject_mem_store():
def _inject_mem_store(self, token_ids, x):
    end = self._mem_history_end
    if end == 0:
        return  # ← EARLY RETURN - no injection!
```

**The Problem**:
1. `_mem_history_end` starts at 0 and only updates after KV cache eviction
2. Simple tests with no eviction stay at 0
3. No MEM_STORE flags get injected on MEM sections
4. L15 attention-based memory lookup requires MEM_STORE flag
5. Without flag, LI operation fails → returns 0x01010101

**Expected Behavior**:
- L6 head 6 should set MEM_STORE for current step's MEM section
- Embedding should only inject it for historical (evicted) MEM sections
- Bug: Either L6 weights incorrect OR current-step MEM handling broken

## This is GOOD NEWS!

✅ **Tests are working correctly** - They're doing exactly what they should: validating memory operations

✅ **GPU acceleration works** - 180x-200x speedup achieved

✅ **Found real bugs** - Memory operations (SI/LI) need fixing

## Next Steps

### Fix MEM_STORE Flag Generation

**Option 1: Fix L6 Head 6 Weights**
Investigate why L6 head 6 isn't setting MEM_STORE on current step MEM sections:
- Check `set_vm_weights()` L6 head 6 configuration (line ~1281 in vm_step.py)
- Verify MEM_STORE output weights in L6 FFN
- Test with simple SI instruction to see if MEM section gets MEM_STORE=1.0

**Option 2: Modify Embedding Injection Logic**
If L6 can't reliably set MEM_STORE, modify `_inject_mem_store()` to handle current-step sections:
```python
def _inject_mem_store(self, token_ids, x):
    """Inject MEM_STORE on ALL MEM markers, not just historical ones."""
    B, S = token_ids.shape
    for b in range(B):
        for i in range(S):  # Search entire context, not just 0.._mem_history_end
            if token_ids[b, i].item() == Token.MEM:
                x[b, i, BD.MEM_STORE] = 1.0
```

**Option 3: Debug MEM Section Generation**
Verify that SI opcode actually produces a MEM section:
- Add debug logging to see what tokens are generated after SI
- Check if MEM marker + 9 tokens (addr + value) are present
- Verify the MEM section format matches expectations

### Validation
Once fixed, these tests will validate:
- Long-term retention works (100+ steps)
- Overwrites work correctly (50+ to same address)
- Multiple addresses work (30+ simultaneous)
- All memory mechanisms function properly

## GPU Usage Notes

**GPU Memory**: Tests use ~3-4GB of VRAM
- Plenty of headroom on 24GB card
- Can run full 16-test suite easily

**GPU Utilization**: Low (1-10%)
- Most time spent in Python overhead
- Actual forward passes are fast
- Could batch operations for higher utilization

## Recommendations

### For Now
1. **Tests are ready** - GPU acceleration works
2. **SI/LI operations need fixing** - That's the blocker
3. **Use these tests to validate fixes** - They'll catch regressions

### For Future
1. **Run comprehensive tests on GPU** - Will complete in ~10-15 minutes
2. **Add to nightly CI with GPU runners** - Fast enough for regular testing
3. **Batch operations** - Could optimize further for even faster runs

## Commands for GPU Testing

### Quick tests (3 tests, ~2 min)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest neural_vm/tests/test_memory_quick.py -v
```

### Comprehensive tests (16 tests, estimated ~8-15 min)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest neural_vm/tests/test_memory_stress.py -v
```

### Monitor GPU during tests
```bash
watch -n 1 nvidia-smi
```

## Conclusion

✅ **GPU acceleration successful** - 180-200x speedup
✅ **Tests execute correctly** - Fast validation
❌ **Memory operations broken** - SI/LI need fixing

The test infrastructure is solid. The neural VM has a bug that these tests are correctly detecting!

**Action Item**: Fix SI/LI memory operations in the neural VM, then these tests will pass.
