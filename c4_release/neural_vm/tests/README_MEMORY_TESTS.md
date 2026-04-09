# Memory Stress Tests

**File**: `test_memory_stress.py`
**Purpose**: Comprehensive testing of Neural VM memory retention, overwrites, and capacity limits

## Overview

These tests validate the attention-based memory mechanism (Layer 15 softmax) under realistic workloads. They cover scenarios not tested in the basic test suite:

- Long-term retention (500+ steps)
- Multiple overwrites (100+ to same address)
- Capacity limits (50+ simultaneous addresses)
- Time-distributed writes (across execution timeline)
- Mixed byte/word operations

## Test Classes

### 1. TestMemoryRetention

**Tests memory values persist over long execution sequences.**

- `test_memory_retention_across_100_steps`: Write value, execute 100 NOPs, verify value persists
- `test_memory_retention_with_arithmetic`: Write value, execute 50 arithmetic ops, verify retention
- `test_memory_retention_multiple_values`: Write 10 values at different addresses, verify all persist

**Why**: Validates attention mechanism doesn't degrade over many instructions.

### 2. TestMemoryOverwrites

**Tests multiple overwrites to same address work correctly.**

- `test_10_overwrites_same_address`: Overwrite 10 times, verify latest wins
- `test_50_overwrites_same_address`: Overwrite 50 times, verify latest wins
- `test_overwrite_pattern_alternating`: Alternating values 20 times
- `test_byte_overwrites_sc`: SC (byte store) overwrites

**Why**: Validates "latest write wins" semantic with many overwrites.

### 3. TestMemoryCapacity

**Tests memory capacity limits and attention span.**

- `test_20_different_addresses`: Write to 20 addresses, verify random one persists
- `test_30_different_addresses`: Write to 30 addresses, verify first and last persist
- `test_sparse_address_pattern`: Non-contiguous addresses (0x1000, 0x2000, 0x5000, etc.)

**Why**: Identifies attention span limits - how many unique addresses can Layer 15 handle?

### 4. TestTimeDistributedMemory

**Tests memory writes distributed across execution timeline.**

- `test_early_and_late_writes`: Write at step 1 and step 50, read both
- `test_interleaved_writes_and_reads`: Write-read-write-read pattern
- `test_write_read_write_same_address`: Overwrite same address across time

**Why**: Real programs don't write all memory at once - validates temporal distribution.

### 5. TestMixedMemoryOperations

**Tests mixing byte (SC/LC) and word (SI/LI) operations.**

- `test_si_then_lc`: Store word (SI), load byte (LC)
- `test_multiple_sc_then_li`: Store bytes (SC), read word (LI)

**Why**: Validates byte vs word semantics and ZFOD (zero-fill-on-demand) for unwritten bytes.

## Running the Tests

```bash
# Run all memory stress tests
python3 -m pytest neural_vm/tests/test_memory_stress.py -v

# Run specific test class
python3 -m pytest neural_vm/tests/test_memory_stress.py::TestMemoryRetention -v

# Run single test
python3 -m pytest neural_vm/tests/test_memory_stress.py::TestMemoryOverwrites::test_50_overwrites_same_address -v

# Run with output
python3 -m pytest neural_vm/tests/test_memory_stress.py -v -s
```

## Expected Results

All tests should **PASS** if:
- Layer 15 attention correctly handles 30+ unique memory addresses
- Memory values persist across 100+ instructions
- "Latest write wins" semantic works for 50+ overwrites
- ZFOD (zero-fill-on-demand) works correctly

## Performance Notes

**Model Initialization**: Each test class initializes model once (via `setUpClass`), then reuses for all tests.

**Execution Time**:
- Retention tests: ~5-10 seconds each
- Overwrite tests: ~3-5 seconds each
- Capacity tests: ~10-15 seconds each
- Full suite: ~2-5 minutes

**Memory Usage**: ~4-6GB GPU memory (model + activations)

## Implementation Details

### Memory Mechanism (Layer 15)

The Neural VM uses **attention-based memory** implemented in Layer 15:

```python
# L15: Memory lookup (softmax1 for ZFOD)
# Q: ADDR_KEY (current memory address)
# K: MEM_STORE markers (historical write addresses)
# V: Byte values at those addresses
# Softmax1: Returns 0 for unwritten addresses (ZFOD semantic)
```

### Potential Issues Tested

1. **Attention span**: Can softmax attend to 50+ memory locations?
2. **Recency bias**: Does recent history dominate old history?
3. **Key collision**: Do similar addresses interfere?
4. **KV cache eviction**: Does cache eviction drop memory values?
5. **Temporal degradation**: Does memory accuracy decrease over time?

## Test Coverage Gaps Filled

These tests fill gaps identified in the basic test suite (`test_opcodes.py`):

| Gap | Basic Suite | Memory Stress Suite |
|-----|-------------|---------------------|
| Long-term retention | ❌ Not tested | ✅ 100+ instructions |
| Many overwrites | ⚠️ Only 2 writes | ✅ 50+ overwrites |
| Multiple addresses | ⚠️ 1-2 addresses | ✅ 30+ addresses |
| Time distribution | ❌ Not tested | ✅ Distributed writes |
| Byte/word mixing | ⚠️ Basic only | ✅ Comprehensive |

## Debugging Failed Tests

If tests fail, check:

1. **Memory retention failure**: Layer 15 attention may be degrading over time
   - Check attention weights in Layer 15
   - Verify MEM_STORE markers are set correctly
   - Check if KV cache is evicting memory entries

2. **Overwrite failure**: "Latest write wins" not working
   - Verify attention bias (should favor recent writes)
   - Check if historical writes interfere

3. **Capacity failure**: Too many addresses
   - Layer 15 softmax may not scale to N addresses
   - Check if attention is spreading too thin
   - May need architectural change for large N

4. **ZFOD failure**: Unwritten addresses not returning 0
   - Verify softmax1 bias configuration
   - Check if Layer 15 output routing works correctly

## Integration with CI/CD

Add to test suite:

```yaml
# .github/workflows/test.yml
- name: Run memory stress tests
  run: python3 -m pytest neural_vm/tests/test_memory_stress.py -v
```

Or add to existing test runs:

```bash
# Run all neural_vm tests including memory stress
python3 -m pytest neural_vm/tests/ -v
```

## Future Enhancements

Possible additions:

1. **Massive capacity test**: Write to 100+ unique addresses
2. **Extreme retention test**: Persist values for 1000+ steps
3. **Concurrent read/write test**: Interleaved reads and writes
4. **Memory aliasing test**: Test overlapping byte/word writes
5. **Performance benchmarks**: Measure memory operation latency
6. **Fuzzing tests**: Random memory access patterns

## References

- Main test suite: `neural_vm/tests/test_opcodes.py`
- Memory mechanism: `neural_vm/vm_step.py` (Layer 15, lines ~1700+)
- Test coverage analysis: `docs/MEMORY_TEST_COVERAGE.md`
- Architecture overview: `docs/README.md`
