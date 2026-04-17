# RoPE/ALiBi Integration - Bugs and Test Coverage Gaps

**Date**: 2026-04-08
**Status**: 🐛 Critical bugs found, test coverage incomplete

---

## Critical Bugs 🔴

### 1. Export/Import Tools Missing Guards
**Severity**: Critical - Will crash in RoPE mode

#### `src/prompt_baking.py` Line 52
```python
# BUG: Crashes when loading RoPE models
attn.alibi_slopes.copy_(torch.from_numpy(layer_w['alibi_slopes']))
```
**Fix**: Guard with `if attn.alibi_slopes is not None`

#### `tools/export_autoregressive.py` Line 133
```python
# BUG: Crashes when exporting RoPE models
write_tensor(f, attn.alibi_slopes.detach().cpu().numpy(), sparse=False)
```
**Fix**: Check `if attn.alibi_slopes is not None` before export

#### `tools/export_autoregressive.py` Line 255
```python
# BUG: Always loads alibi_slopes even for RoPE models
layer['alibi_slopes'] = _expand_sparse(data, n_heads)
```
**Fix**: Make alibi_slopes loading conditional

#### `tools/bundle_autoregressive_quine.py` Lines 183, 2180
```python
# BUG: Bundles alibi_slopes without checking existence
array_parts.append(emit_dense(f'{pfx}_alibi', layer['alibi_slopes']))
```
**Fix**: Guard bundle operations

### 2. RoPE Cache Bounds Not Validated
**Severity**: High - Silent errors or crashes with long sequences

#### `neural_vm/vm_step.py` Lines 291-294
```python
# BUG: No bounds checking for sequences beyond max_seq_len
cos_q = self._rope_cos[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)
sin_q = self._rope_sin[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)
cos_k = self._rope_cos[0:S_kv].unsqueeze(0).unsqueeze(0)
sin_k = self._rope_sin[0:S_kv].unsqueeze(0).unsqueeze(0)
```

**Issue**: If `S_kv > max_seq_len` or `q_offset + S_q > max_seq_len`:
- Slicing returns empty/partial tensors
- Causes shape mismatches
- No error message

**Fix**: Add bounds validation:
```python
if S_kv > self.max_seq_len:
    raise ValueError(f"Sequence length {S_kv} exceeds max_seq_len {self.max_seq_len}")
if q_offset + S_q > self.max_seq_len:
    raise ValueError(f"Query offset {q_offset + S_q} exceeds max_seq_len {self.max_seq_len}")
```

### 3. Config System Lacks Validation
**Severity**: Medium - Silent failures and incorrect behavior

#### `neural_vm/config.py` Line 62
```python
# BUG: Invalid env var silently defaults to alibi
pos_encoding = os.environ.get("NEURAL_VM_POS_ENCODING", "alibi")
if pos_encoding == "rope":
    _global_config = VMConfig.rope_mode()
elif pos_encoding == "hybrid":
    _global_config = VMConfig.hybrid_mode()
else:
    _global_config = VMConfig.alibi_mode()  # <- Silent fallback!
```

**Issue**: `NEURAL_VM_POS_ENCODING=typo` silently uses alibi
**Fix**: Add validation and warning

#### VMConfig Dataclass No Validation
```python
@dataclass
class VMConfig:
    positional_encoding: Literal["alibi", "rope", "hybrid"] = "alibi"
    # BUG: No runtime validation of literal values
```

**Issue**: `VMConfig(positional_encoding="invalid")` accepted at runtime
**Fix**: Add `__post_init__` validation

### 4. Thread Safety Not Guaranteed
**Severity**: Medium - Race conditions in multi-threaded environments

#### `neural_vm/config.py` Lines 60-69
```python
# BUG: Global config not thread-safe
_global_config: VMConfig = None

def get_config() -> VMConfig:
    global _global_config
    if _global_config is None:  # <- Race condition!
        # Multiple threads can enter here
        _global_config = ...
```

**Fix**: Use threading.Lock or make config immutable once set

---

## Test Coverage Gaps 📊

### High Priority Gaps

#### 1. KV Cache with RoPE ⚠️
**Status**: NOT TESTED
**Risk**: High - Critical feature combination

**Missing tests**:
- RoPE with incremental generation
- Cache hit/miss scenarios
- Position offset calculation correctness
- Cache size edge cases

**Test needed**:
```python
def test_rope_with_kv_cache():
    """Test RoPE works correctly with KV caching."""
    set_config(VMConfig.rope_mode())
    model = AutoregressiveVM()

    # Test incremental generation
    tokens_1 = torch.randint(0, 256, (1, 10))
    tokens_2 = torch.randint(0, 256, (1, 15))  # Extends sequence

    # First forward pass
    out1 = model(tokens_1)

    # Second pass with cache
    # Should compute RoPE positions [10, 15) for new tokens
    out2 = model(tokens_2)

    # Verify correctness
    ...
```

#### 2. Sequence Length Edge Cases ⚠️
**Status**: NOT TESTED
**Risk**: High - Will crash in production

**Missing tests**:
- Sequences exactly at max_seq_len
- Sequences beyond max_seq_len
- Empty sequences (length 0)
- Single token sequences (length 1)
- Boundary conditions

**Test needed**:
```python
def test_rope_sequence_length_bounds():
    """Test RoPE handles sequence length edge cases."""
    set_config(VMConfig.rope_mode())
    max_len = 100
    attn = AutoregressiveAttention(256, 4, max_seq_len=max_len)

    # Test exactly at max
    x = torch.randn(1, max_len, 256)
    out = attn(x)  # Should work

    # Test beyond max
    x = torch.randn(1, max_len + 1, 256)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        out = attn(x)  # Should raise clear error
```

#### 3. Config Validation ⚠️
**Status**: NOT TESTED
**Risk**: Medium - User confusion

**Missing tests**:
- Invalid NEURAL_VM_POS_ENCODING values
- Invalid VMConfig.positional_encoding values
- Config changes after model creation
- Config state pollution between tests

**Test needed**:
```python
def test_config_validation():
    """Test config validates inputs."""
    # Invalid env var should warn
    os.environ["NEURAL_VM_POS_ENCODING"] = "invalid"
    with pytest.warns(UserWarning, match="Unknown positional encoding"):
        config = get_config()
    assert config.positional_encoding == "alibi"  # Fallback

    # Invalid direct construction should raise
    with pytest.raises(ValueError):
        VMConfig(positional_encoding="invalid")
```

#### 4. Model Serialization ⚠️
**Status**: NOT TESTED
**Risk**: High - Export/import will fail

**Missing tests**:
- Save/load RoPE models
- Save/load hybrid models
- Cross-mode loading (save ALiBi, load as RoPE)
- ONNX export with RoPE
- Weight export/import tools

**Test needed**:
```python
def test_rope_model_serialization():
    """Test RoPE models can be saved and loaded."""
    set_config(VMConfig.rope_mode())
    model1 = AutoregressiveVM()

    # Save
    torch.save(model1.state_dict(), "rope_model.pt")

    # Load
    model2 = AutoregressiveVM()
    model2.load_state_dict(torch.load("rope_model.pt"))

    # Verify identical
    x = torch.randint(0, 256, (1, 10))
    assert torch.allclose(model1(x), model2(x))
```

### Medium Priority Gaps

#### 5. Speculator Compatibility
**Status**: Partially tested (runs but not verified)
**Risk**: Medium - Used in production

**Missing tests**:
- Speculator with RoPE mode
- Draft model with different config than main model
- Hybrid mode speculator

#### 6. Concurrent Model Creation
**Status**: NOT TESTED
**Risk**: Medium - Multi-threaded inference

**Missing tests**:
- Thread safety of get_config()
- Multiple models with different configs
- Race conditions in config initialization

#### 7. Hybrid Mode Boundaries
**Status**: Basic test only
**Risk**: Low - Simple logic but should verify

**Missing tests**:
- Exactly layer 2 (should be ALiBi)
- Exactly layer 3 (should be RoPE)
- Verify all 16 layers have correct encoding

### Low Priority Gaps

#### 8. Numerical Stability
**Status**: NOT TESTED
**Risk**: Low - Unlikely but possible

**Missing tests**:
- Very large RoPE base values
- Very small head dimensions
- Float16 vs Float32
- Gradient flow through RoPE

#### 9. Edge Input Shapes
**Status**: NOT TESTED
**Risk**: Low - Unlikely in practice

**Missing tests**:
- Batch size > 1 with RoPE
- Different batch sizes between forward passes
- Non-contiguous tensors

---

## Summary Statistics

### Bugs by Severity
- 🔴 Critical: 4 bugs (export/import tools, bounds checking)
- 🟡 High: 2 bugs (config validation, thread safety)
- **Total**: 6 bugs

### Test Coverage by Priority
- ⚠️ High Priority: 4 gaps (KV cache, bounds, config, serialization)
- 🟨 Medium Priority: 3 gaps (speculator, concurrency, hybrid boundaries)
- 🟦 Low Priority: 2 gaps (numerical, edge shapes)
- **Total**: 9 test coverage gaps

### Files Needing Fixes
1. `src/prompt_baking.py`
2. `tools/export_autoregressive.py`
3. `tools/bundle_autoregressive_quine.py`
4. `neural_vm/vm_step.py`
5. `neural_vm/config.py`
6. `tests/test_positional_encoding.py` (add missing tests)

---

## Impact Assessment

### Current State
✅ **Basic functionality works** - All 1096 tests pass in all three modes
❌ **Production readiness incomplete** - Export/import broken, edge cases not handled

### Risk Level by Use Case

| Use Case | ALiBi Mode | RoPE Mode | Hybrid Mode |
|----------|------------|-----------|-------------|
| Basic inference | ✅ Safe | ✅ Safe | ✅ Safe |
| Long sequences (>4096) | ✅ Works | 🔴 Crashes | 🔴 Crashes |
| KV cache | ✅ Safe | ⚠️ Untested | ⚠️ Untested |
| Model export | ✅ Works | 🔴 Broken | 🔴 Broken |
| Model import | ✅ Works | 🔴 Broken | 🔴 Broken |
| Multi-threading | ⚠️ Untested | ⚠️ Untested | ⚠️ Untested |
| Speculator | ✅ Tested | ⚠️ Untested | ⚠️ Untested |

### Recommended Actions

#### Immediate (Block Production)
1. Fix export/import tool bugs
2. Add bounds checking for RoPE sequences
3. Add config validation
4. Test KV cache with RoPE

#### Short-term (Before Release)
1. Add comprehensive edge case tests
2. Test model serialization
3. Fix thread safety issues
4. Document limitations

#### Long-term (Nice to Have)
1. Support dynamic sequence extension
2. Optimize RoPE cache memory usage
3. Add gradient checkpointing support
4. Profile performance differences

---

**Next Steps**: Create comprehensive test suite and apply all fixes
