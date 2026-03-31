# I/O Implementation - Quick Start Guide

**Goal:** Add printf/read support to DraftVM in 4-6 hours

**Current Status:** 99% complete - just need I/O in speculation mode

---

## TL;DR - What to Do

### Files to Modify
1. `neural_vm/speculative.py` - Add I/O handlers (~150 lines)
2. `neural_vm/batch_runner.py` - Wire up I/O (~10 lines)
3. `tests/test_io_speculation.py` - Tests (~200 lines, NEW FILE)

### Key Changes
1. **DraftVM gets I/O support:**
   - Load data section into memory
   - Add printf handler with format parsing
   - Add read handler for stdin
   - Add output buffer

2. **BatchedRunner passes data:**
   - Pass data to DraftVM on creation
   - Collect output from DraftVM

---

## Implementation Steps (Ordered)

### Step 1: Add Data Loading (15 min)
```python
# In neural_vm/speculative.py, add to __init__:
self.output = []
self.stdin_buf = ""
self.stdin_pos = 0

# Add method:
def load_data(self, data):
    if data:
        for i, b in enumerate(data):
            self.memory[0x10000 + i] = b
```

### Step 2: Add String Reader (15 min)
```python
# In neural_vm/speculative.py:
def _read_string(self, addr):
    chars = []
    while True:
        word = self._mem_read(addr)
        byte_val = word & 0xFF
        if byte_val == 0:
            break
        chars.append(chr(byte_val))
        addr += 1
        if len(chars) > 10000:
            break
    return ''.join(chars)
```

### Step 3: Add Printf Handler (60 min)
Copy the complete implementation from `docs/IO_IMPLEMENTATION_PLAN_ACTIONABLE.md` Task 1.3

Key parts:
- Peek ahead for argc (ADJ instruction)
- Read format string from stack
- Parse format specifiers (%d, %x, %c, %s)
- Handle escape sequences (\n, \t, \\)
- Accumulate output

### Step 4: Wire Up in step() (5 min)
```python
# In neural_vm/speculative.py step() method:
elif op == 33:  # PRTF
    self._handle_printf()
```

### Step 5: Add Output Accessor (5 min)
```python
def get_output(self):
    return ''.join(self.output)
```

### Step 6: Update BatchRunner (30 min)
```python
# In neural_vm/batch_runner.py:

# Update _BlockedDraftVM:
def __init__(self, bytecode, data=None):
    self._vm = DraftVM(bytecode)
    if data:
        self._vm.load_data(data)

def get_output(self):
    return self._vm.get_output()

# Update run_batch():
self.draft_vms = [_BlockedDraftVM(bc, data_list[i]) for i, bc in enumerate(bytecodes)]

# At end of run_batch():
output = self.draft_vms[i].get_output()
results.append((output, exit_code))
```

### Step 7: Create Tests (30 min)
Create `tests/test_io_speculation.py` with 8 tests (see actionable plan)

### Step 8: Run Tests (5 min)
```bash
python -m pytest tests/test_io_speculation.py -v
python tests/run_1000_tests.py  # Verify no regression
```

---

## Expected Results

**Before:**
```
Arithmetic tests: 1096/1096 ✅
I/O tests:        0/8 ❌
```

**After:**
```
Arithmetic tests: 1096/1096 ✅
I/O tests:        8/8 ✅
Total:            1104/1104 (100%)
```

---

## Common Issues

### Issue 1: Format strings are zeros
**Cause:** Data section not loaded
**Fix:** Verify `load_data()` is called in `_BlockedDraftVM.__init__()`

### Issue 2: Output is empty
**Cause:** `get_output()` not called in `run_batch()`
**Fix:** Add `output = self.draft_vms[i].get_output()` at line ~286

### Issue 3: Tests hang
**Cause:** max_steps too low
**Fix:** Increase max_steps to 200-500 in tests

---

## Testing Checklist

- [ ] Simple printf works: `printf("Hello\n")`
- [ ] Integer format works: `printf("%d", 42)`
- [ ] Multiple args work: `printf("%d + %d", 10, 32)`
- [ ] Hex format works: `printf("%x", 255)` → "ff"
- [ ] Char format works: `printf("%c", 65)` → "A"
- [ ] Negative numbers work: `printf("%d", -42)` → "-42"
- [ ] Multiple printfs work
- [ ] Printf in loops works
- [ ] Arithmetic tests still pass (1096/1096)

---

## Full Documentation

See `docs/IO_IMPLEMENTATION_PLAN_ACTIONABLE.md` for:
- Complete code for all methods
- Detailed task breakdown
- Risk mitigation strategies
- Performance testing
- Edge case testing
- Alternative approaches

---

## Quick Commands

```bash
# Run I/O tests
python -m pytest tests/test_io_speculation.py -v

# Run all tests
python tests/run_1000_tests.py

# Benchmark performance
python tests/benchmark_io.py

# Debug single test
python -m pytest tests/test_io_speculation.py::test_simple_printf -v -s
```

---

## Time Budget

| Phase | Time | Tasks |
|-------|------|-------|
| Core Implementation | 2h | Steps 1-5 |
| Integration | 1h | Steps 6-8 |
| Additional Syscalls | 1h | memset/memcmp |
| Polish & Testing | 1h | Edge cases, docs |
| **Total** | **4-6h** | **Full I/O support** |

---

## Success Criteria

✅ `printf("Hello\n")` works
✅ Format specifiers work (%d, %x, %c)
✅ All 8 basic tests pass
✅ No regression in arithmetic tests
✅ Performance < 30% slower

When all boxes checked: **I/O implementation complete!**
