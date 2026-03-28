# Critical Discovery: Test Suite Uses Fallback Interpreter

## Finding

The test suite in `tests/test_vm.py` **does NOT test the neural VM**. It uses a simple Python fallback interpreter instead.

## Evidence

### Test Code Pattern
```python
class TestVM(unittest.TestCase):
    def setUp(self):
        self.vm = C4TransformerVM()

    def test_immediate_and_exit(self):
        self.vm.reset()
        self.vm.load([(1, 42), (38, 0)])  # ← Uses load(), not load_bytecode()
        result = self.vm.run()
        self.assertEqual(result, 42)
```

### How It Works

**File**: `src/transformer_vm.py`

1. **`load()` method** (line 318):
   - Sets `self._bytecode` directly
   - Does NOT set `self._neural_bytecode`

2. **`run()` method** (line 350):
   ```python
   def run(self, max_steps=100000):
       if self._use_neural_vm and hasattr(self, '_neural_bytecode'):
           # Use neural VM
           return self._runner.run(...)
       
       # Fallback: simple interpreter
       return self._run_fallback(max_steps)
   ```

3. **Result**: When using `load()`, `_neural_bytecode` is not set, so `run()` uses `_run_fallback()` - a simple Python interpreter!

### Comparison

| Method | Sets _neural_bytecode? | VM Used |
|--------|----------------------|----------|
| `load()`          | ❌ NO | Fallback Python interpreter |
| `load_bytecode()` | ✅ YES | Neural VM (AutoregressiveVMRunner) |

## Implications

1. **Test suite passes** because it uses working fallback interpreter
2. **Neural VM untested** by the main test suite
3. **Bugs I found** affect only the neural VM path (which tests don't exercise)
4. **JSR purity work** would apply to neural VM, which isn't being used by tests

## What This Means

- The 4 bugs I fixed are real and important for neural VM
- But tests pass because they don't use neural VM
- Need to either:
  1. Fix neural VM to match fallback behavior
  2. Update tests to use `load_bytecode()` and test neural VM
  3. Accept that neural VM is experimental/incomplete

## Next Steps

1. Check if `load_bytecode()` path works with my fixes
2. Investigate compiler CALL target alignment issue
3. Determine if neural VM is production-ready or experimental
