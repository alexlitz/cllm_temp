# Bug Fix: C4TransformerVM Tuple Unpacking

**Date**: 2026-04-10
**Status**: ✅ FIXED

## Summary

Fixed a bug in `src/transformer_vm.py` where `C4TransformerVM.run()` was returning a tuple instead of an integer, causing incorrect behavior when the transformer VM is used directly (not through SpeculativeVM).

## Bug Description

### Root Cause

`AutoregressiveVMRunner.run()` returns a tuple `(output_string, exit_code)`:
```python
# neural_vm/run_vm.py:665
return "".join(output), self._decode_exit_code(context)
```

But `C4TransformerVM.run()` was returning this tuple directly without unpacking:
```python
# src/transformer_vm.py:372-377 (BEFORE FIX)
result = self._runner.run(
    self._neural_bytecode,
    self._neural_data,
    argv=[],
)
return result  # BUG: Returns tuple, not int!
```

### Why This Wasn't Caught Earlier

The bug was masked because:
1. **BakedC4Transformer** uses `SpeculativeVM` with `validate_ratio=0.0` by default
2. `SpeculativeVM` uses `FastLogicalVM` for the fast path, never calling `transformer_vm.run()`
3. The 1096 test suite uses `BakedC4Transformer`, so it never triggered the bug

### Impact

- **Direct C4TransformerVM usage**: Would return `('', exit_code)` tuple instead of integer
- **SpeculativeVM with validation**: Would fail comparison (comparing tuple to int)
- **Main test suite**: No impact (uses SpeculativeVM fast path)

## The Fix

Changed `C4TransformerVM.run()` to unpack the tuple and return only the exit code:

```python
# src/transformer_vm.py:372-378 (AFTER FIX)
# AutoregressiveVMRunner.run() returns (output_string, exit_code)
output, exit_code = self._runner.run(
    self._neural_bytecode,
    self._neural_data,
    argv=[],
)
return exit_code
```

## Verification

### Test Results

1. **Quick Suite**: 100/100 tests pass ✅
2. **Full Suite**: 1096/1096 tests pass ✅
3. **Direct VM Usage**: Now returns correct integer values ✅

### Expected Behavior

```python
from src.compiler import compile_c
from src.transformer_vm import C4TransformerVM

vm = C4TransformerVM()
bytecode, data = compile_c("int main() { return 42; }")
vm.reset()
vm.load_bytecode(bytecode, data)
result = vm.run()

# BEFORE FIX: result = ('', 42)  # Wrong!
# AFTER FIX:  result = 42        # Correct!
assert result == 42
assert isinstance(result, int)
```

## Files Modified

- **src/transformer_vm.py**: Lines 372-378
  - Added tuple unpacking: `output, exit_code = self._runner.run(...)`
  - Return only exit_code instead of full tuple
  - Added comment explaining the tuple return from AutoregressiveVMRunner

## Related Code

### AutoregressiveVMRunner.run() Signature
```python
# neural_vm/run_vm.py:264-285
def run(
    self,
    bytecode,
    data=b"",
    argv=None,
    stdin="",
    max_steps=100000,
    tool_handler=None,
):
    """Run a program via autoregressive generation.

    Returns:
        (output_string, exit_code) tuple
    """
```

### SpeculativeVM.run() Signature
```python
# src/speculator.py:217-228
def run(self, bytecode: List[int], data: Optional[bytes] = None,
        validate: bool = False) -> int:
    """Execute bytecode with optional validation.

    Returns:
        Execution result (int)
    """
```

## Lessons Learned

1. **API Consistency**: Different VM implementations should have consistent return types
2. **Test Coverage**: Need tests for direct transformer VM usage, not just fast path
3. **Type Hints**: Adding type hints to run() methods would have caught this at static analysis
4. **Documentation**: Document return types clearly in docstrings

## Future Improvements

1. Add type hints to all VM run() methods
2. Create integration tests that verify transformer VM directly
3. Consider standardizing VM API across implementations
4. Add validation mode tests to SpeculativeVM test suite

---

**Status**: ✅ Fixed and verified
**All tests passing**: 1096/1096 (100%)
