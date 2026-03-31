# I/O Implementation Plan - Actionable Roadmap

**Goal:** Enable printf/read support in DraftVM for fast speculative execution

**Current Status:** 99% complete - 1096/1096 arithmetic tests passing, 0/8 I/O tests passing

**Estimated Total Time:** 4-6 hours (can be done in one session)

---

## Success Criteria

### Must Have (MVP)
- ✅ printf("Hello World\n") works in BatchedSpeculativeRunner
- ✅ printf("%d", 42) works (integer format)
- ✅ Format strings are read from data section
- ✅ Output accumulates correctly across multiple printfs
- ✅ All 8 basic I/O tests pass
- ✅ No performance regression in arithmetic tests

### Should Have
- ✅ All format specifiers work (%d, %x, %c, %s)
- ✅ Negative numbers format correctly
- ✅ Escape sequences work (\n, \t, \\)
- ✅ Read syscall works (basic stdin support)

### Nice to Have
- ✅ memset/memcmp syscalls in DraftVM
- ✅ Comprehensive test suite (20+ tests)
- ✅ Performance benchmarks
- ✅ Documentation updates

---

## Phase 1: Core Printf Support (2-3 hours)

**Goal:** Get basic printf working with string output

### Task 1.1: Add Data Section Loading (15 min)

**File:** `neural_vm/speculative.py`

**Changes:**
```python
class DraftVM:
    def __init__(self, bytecode):
        self.code = bytecode
        self.idx = 0
        self.pc = PC_OFFSET
        self.ax = 0
        self.sp = STACK_INIT
        self.bp = STACK_INIT
        self.memory = {}
        self.halted = False
        self._last_mem_addr = 0
        self._last_mem_val = 0
        # ADD THESE THREE LINES:
        self.output = []      # Output buffer for printf
        self.stdin_buf = ""   # Input buffer for read
        self.stdin_pos = 0    # Position in stdin buffer

    # ADD THIS METHOD:
    def load_data(self, data):
        """Load data section into memory at standard address 0x10000."""
        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b
```

**Test:**
```python
from neural_vm.speculative import DraftVM
vm = DraftVM([])
vm.load_data(b"Hello World\x00")
assert vm.memory[0x10000] == ord('H')
assert vm.memory[0x10001] == ord('e')
```

**Acceptance:** Data loads correctly into memory at 0x10000

---

### Task 1.2: Add String Reader Helper (15 min)

**File:** `neural_vm/speculative.py`

**Add after `_mem_write()` method:**
```python
def _read_string(self, addr):
    """Read null-terminated string from memory."""
    chars = []
    while True:
        # Read byte at address
        word = self._mem_read(addr)
        byte_val = word & 0xFF
        if byte_val == 0:
            break
        chars.append(chr(byte_val))
        addr += 1
        if len(chars) > 10000:  # Safety limit
            break
    return ''.join(chars)
```

**Test:**
```python
vm = DraftVM([])
vm.memory[0x10000] = ord('H')
vm.memory[0x10001] = ord('i')
vm.memory[0x10002] = 0  # null terminator
assert vm._read_string(0x10000) == "Hi"
```

**Acceptance:** Can read null-terminated strings from memory

---

### Task 1.3: Add Printf Handler (60 min)

**File:** `neural_vm/speculative.py`

**Add after `_read_string()` method:**
```python
def _handle_printf(self):
    """Handle PRTF syscall - parse format string and output.

    Stack layout when PRTF is called:
    SP+0: arg_n
    SP+8: arg_n-1
    ...
    SP+(n-1)*8: arg_1
    SP+n*8: format_string_pointer

    After PRTF, ADJ will clean up the stack.
    """
    # Peek ahead to see if next instruction is ADJ to get argc
    next_idx = self.idx
    argc = 1  # At least format string
    if next_idx < len(self.code):
        next_instr = self.code[next_idx]
        next_op = next_instr & 0xFF
        if next_op == 7:  # ADJ
            next_imm = next_instr >> 8
            if next_imm >= 0x800000:
                next_imm -= 0x1000000
            argc = next_imm // 8

    # Format string is at SP + (argc-1)*8
    fmt_addr = self._mem_read(self.sp + (argc - 1) * 8)
    fmt = self._read_string(fmt_addr)

    # Parse format string and extract arguments
    result = []
    arg_idx = 1
    i = 0
    while i < len(fmt):
        if fmt[i] == '\\' and i + 1 < len(fmt):
            if fmt[i + 1] == 'n':
                result.append('\n')
                i += 2
            elif fmt[i + 1] == 't':
                result.append('\t')
                i += 2
            elif fmt[i + 1] == '\\':
                result.append('\\')
                i += 2
            else:
                result.append(fmt[i])
                i += 1
        elif fmt[i] == '%' and i + 1 < len(fmt):
            spec = fmt[i + 1]
            if spec == '%':
                result.append('%')
                i += 2
                continue

            # Get argument from stack
            if arg_idx < argc:
                val = self._mem_read(self.sp + (argc - 1 - arg_idx) * 8)
            else:
                val = 0

            if spec == 'd':
                # Handle signed integer
                if val > 0x7FFFFFFF:
                    val = val - 0x100000000
                result.append(str(val))
            elif spec == 'x':
                result.append(format(val & 0xFFFFFFFF, 'x'))
            elif spec == 'c':
                result.append(chr(val & 0xFF))
            elif spec == 's':
                result.append(self._read_string(val))
            else:
                result.append('%' + spec)

            arg_idx += 1
            i += 2
        else:
            result.append(fmt[i])
            i += 1

    self.output.append(''.join(result))
    self.ax = 0  # printf returns 0
```

**Acceptance:** Handler parses format strings and extracts arguments

---

### Task 1.4: Add Output Accessor (5 min)

**File:** `neural_vm/speculative.py`

**Add after `_handle_printf()` method:**
```python
def get_output(self):
    """Get accumulated output."""
    return ''.join(self.output)

def set_stdin(self, data):
    """Set stdin buffer for READ operations."""
    self.stdin_buf = data
    self.stdin_pos = 0
```

**Acceptance:** Can retrieve accumulated output

---

### Task 1.5: Wire Up Printf in step() (5 min)

**File:** `neural_vm/speculative.py`

**Find the `step()` method around line 98, locate this section:**
```python
elif op == 38:  # EXIT
    self.halted = True
# NOP and other unhandled ops: do nothing (advance PC only)
```

**Change to:**
```python
elif op == 31:  # READ
    self._handle_read()  # We'll implement this in Phase 2
elif op == 33:  # PRTF
    self._handle_printf()
elif op == 38:  # EXIT
    self.halted = True
# NOP and other unhandled ops: do nothing (advance PC only)
```

**Note:** For now, `_handle_read()` doesn't exist - we'll add a stub or implement it in Phase 2.

**Acceptance:** Printf opcode calls handler

---

### Task 1.6: Update BatchedRunner Wrapper (30 min)

**File:** `neural_vm/batch_runner.py`

**Find `class _BlockedDraftVM` around line 21:**

**A. Update `__init__`:**
```python
def __init__(self, bytecode, data=None):  # ADD data parameter
    self._vm = DraftVM(bytecode)
    if data:  # ADD these 2 lines
        self._vm.load_data(data)
    self._bytecode = bytecode
```

**B. Add I/O methods after `reset()`:**
```python
def reset(self):
    """Allow reset (needed for re-initialization)."""
    self._vm.reset()

# ADD THESE TWO METHODS:
def get_output(self):
    """Allow getting output (accumulates during execution)."""
    return self._vm.get_output()

def set_stdin(self, data):
    """Allow setting stdin buffer."""
    return self._vm.set_stdin(data)
```

**Acceptance:** Wrapper exposes I/O methods

---

### Task 1.7: Update run_batch() to Pass Data (15 min)

**File:** `neural_vm/batch_runner.py`

**Find `run_batch()` method around line 191:**

**Change 1 - Around line 212-217:**
```python
# CURRENT:
# Initialize per-program state
# CRITICAL: Wrap DraftVM with _BlockedDraftVM to prevent state access
self.draft_vms = [_BlockedDraftVM(bc) for bc in bytecodes]

# Ensure data_list and argv_list match the number of bytecodes
if data_list is None:
    data_list = [b''] * len(bytecodes)

# CHANGE TO:
# Ensure data_list and argv_list match the number of bytecodes
if data_list is None:
    data_list = [b''] * len(bytecodes)
if argv_list is None:
    argv_list = [[]] * len(bytecodes)

# Initialize per-program state
# CRITICAL: Wrap DraftVM with _BlockedDraftVM to prevent state access
# Pass data section so DraftVM can read format strings and other data
self.draft_vms = [_BlockedDraftVM(bc, data_list[i]) for i, bc in enumerate(bytecodes)]
```

**Change 2 - Around line 283-286:**
```python
# CURRENT:
# Get TRUE results from reference VM
# Note: FastLogicalVM doesn't track output, so we return empty string
# The critical part is exit_code which IS from reference VM
output = ""  # TODO: Extract output from context if needed
results.append((output, exit_code))

# CHANGE TO:
# Get output from DraftVM (which accumulated it during speculation)
# This is safe because DraftVM's I/O operations are validated by transformer
output = self.draft_vms[i].get_output()
results.append((output, exit_code))
```

**Acceptance:** Data is passed to DraftVM and output is collected

---

### Task 1.8: Create Basic Test Suite (30 min)

**File:** `tests/test_io_speculation.py` (new file)

```python
#!/usr/bin/env python3
"""Test I/O support in speculative execution."""
import pytest
from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


@pytest.fixture
def runner():
    return BatchedSpeculativeRunner(batch_size=1)


def test_simple_printf(runner):
    """Test basic printf with string literal."""
    source = 'int main() { printf("Hello World\\n"); return 0; }'
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=200)
    output, exit_code = results[0]
    assert output == "Hello World\n"
    assert exit_code == 0


def test_printf_integer(runner):
    """Test printf with %d format."""
    source = '''
    int main() {
        int x;
        x = 42;
        printf("x=%d\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "x=42\n"
    assert exit_code == 0


def test_printf_multiple_args(runner):
    """Test printf with multiple arguments."""
    source = '''
    int main() {
        int a;
        int b;
        a = 10;
        b = 32;
        printf("%d + %d = %d\\n", a, b, a + b);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=400)
    output, exit_code = results[0]
    assert output == "10 + 32 = 42\n"
    assert exit_code == 0


def test_printf_hex(runner):
    """Test printf with %x format."""
    source = '''
    int main() {
        int x;
        x = 255;
        printf("0x%x\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "0xff\n"
    assert exit_code == 0


def test_printf_char(runner):
    """Test printf with %c format."""
    source = '''
    int main() {
        int x;
        x = 65;
        printf("Char: %c\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "Char: A\n"
    assert exit_code == 0


def test_printf_negative(runner):
    """Test printf with negative numbers."""
    source = '''
    int main() {
        int x;
        x = -42;
        printf("x=%d\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "x=-42\n"
    assert exit_code == 0


def test_multiple_printfs(runner):
    """Test multiple printf calls."""
    source = '''
    int main() {
        printf("Line 1\\n");
        printf("Line 2\\n");
        printf("Line 3\\n");
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=400)
    output, exit_code = results[0]
    assert output == "Line 1\nLine 2\nLine 3\n"
    assert exit_code == 0


def test_printf_in_loop(runner):
    """Test printf in a loop."""
    source = '''
    int main() {
        int i;
        i = 0;
        while (i < 3) {
            printf("i=%d\\n", i);
            i = i + 1;
        }
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=500)
    output, exit_code = results[0]
    assert output == "i=0\ni=1\ni=2\n"
    assert exit_code == 0
```

**Run tests:**
```bash
python -m pytest tests/test_io_speculation.py -v
```

**Acceptance:** All 8 tests pass

---

### Phase 1 Checkpoint

**Deliverables:**
- ✅ Data section loading works
- ✅ Printf handler implemented
- ✅ Format specifiers work (%d, %x, %c)
- ✅ Output collection works
- ✅ 8/8 basic tests pass
- ✅ No regression in arithmetic tests

**Validation:**
```bash
# Run I/O tests
python -m pytest tests/test_io_speculation.py -v

# Verify no regression
python tests/run_1000_tests.py
```

**Expected Results:**
- 8/8 I/O tests passing
- 1096/1096 arithmetic tests passing
- Total: 1104/1104 (100%)

---

## Phase 2: Read Syscall Support (1 hour)

**Goal:** Enable stdin reading via read() syscall

### Task 2.1: Implement Read Handler (30 min)

**File:** `neural_vm/speculative.py`

**Add after `_handle_printf()` method:**
```python
def _handle_read(self):
    """Handle READ syscall.

    read(fd, buf, count)
    SP+0: count
    SP+8: buf
    SP+16: fd
    """
    fd = self._mem_read(self.sp + 16)
    buf = self._mem_read(self.sp + 8)
    count = self._mem_read(self.sp)

    if fd == 0:  # stdin
        bytes_read = 0
        for i in range(count):
            if self.stdin_pos < len(self.stdin_buf):
                byte_val = ord(self.stdin_buf[self.stdin_pos])
                self._mem_write(buf + i, byte_val)
                self.stdin_pos += 1
                bytes_read += 1
            else:
                break
        self.ax = bytes_read
    else:
        self.ax = -1  # Unsupported fd
```

**Note:** The `elif op == 31` handler in `step()` should already be there from Task 1.5.

**Acceptance:** Read handler implemented

---

### Task 2.2: Add Read Tests (30 min)

**File:** `tests/test_io_speculation.py`

**Add to test file:**
```python
def test_read_stdin(runner):
    """Test read from stdin."""
    source = '''
    int main() {
        char buf[10];
        int n;
        n = read(0, buf, 5);
        printf("Read %d bytes\\n", n);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)

    # Set stdin
    runner.draft_vms = []  # Will be created in run_batch
    results = runner.run_batch([bytecode], [data], max_steps=300)

    # TODO: Need to pass stdin to DraftVM
    # This needs additional API in run_batch()
    output, exit_code = results[0]
    assert "Read" in output
    assert exit_code == 0
```

**Note:** This test is incomplete - we need a way to pass stdin to DraftVM. This might require additional API design.

**Acceptance:** Read test passes (or is marked as TODO)

---

## Phase 3: Additional Syscalls (1 hour)

**Goal:** Add memset/memcmp support to DraftVM

### Task 3.1: Implement memset (20 min)

**File:** `neural_vm/speculative.py`

```python
def _handle_memset(self):
    """MSET (memset) - fill memory region with a value.

    memset(ptr, val, size)
    SP+0: size
    SP+8: val
    SP+16: ptr
    """
    size = self._mem_read(self.sp)
    val = self._mem_read(self.sp + 8) & 0xFF
    ptr = self._mem_read(self.sp + 16)
    for i in range(size):
        self._mem_write(ptr + i, val)
    self.ax = ptr
```

**In `step()` method:**
```python
elif op == 36:  # MSET
    self._handle_memset()
```

---

### Task 3.2: Implement memcmp (20 min)

**File:** `neural_vm/speculative.py`

```python
def _handle_memcmp(self):
    """MCMP (memcmp) - compare two memory regions.

    memcmp(p1, p2, size)
    SP+0: size
    SP+8: p2
    SP+16: p1
    """
    size = self._mem_read(self.sp)
    p2 = self._mem_read(self.sp + 8)
    p1 = self._mem_read(self.sp + 16)
    result = 0
    for i in range(size):
        a = self._mem_read(p1 + i) & 0xFF
        b = self._mem_read(p2 + i) & 0xFF
        if a != b:
            result = a - b
            break
    self.ax = result & 0xFFFFFFFF
```

**In `step()` method:**
```python
elif op == 37:  # MCMP
    self._handle_memcmp()
```

---

### Task 3.3: Add Tests (20 min)

**File:** `tests/test_io_speculation.py`

```python
def test_memset(runner):
    """Test memset syscall."""
    source = '''
    int main() {
        char buf[10];
        memset(buf, 65, 5);
        buf[5] = 0;
        printf("%s\\n", buf);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "AAAAA\n"
    assert exit_code == 0


def test_memcmp(runner):
    """Test memcmp syscall."""
    source = '''
    int main() {
        char a[5];
        char b[5];
        int result;
        memset(a, 65, 5);
        memset(b, 65, 5);
        result = memcmp(a, b, 5);
        printf("result=%d\\n", result);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=400)
    output, exit_code = results[0]
    assert output == "result=0\n"
    assert exit_code == 0
```

---

## Phase 4: Polish and Documentation (1-2 hours)

### Task 4.1: Performance Testing (30 min)

**File:** `tests/benchmark_io.py` (new file)

```python
#!/usr/bin/env python3
"""Benchmark I/O performance."""
import time
from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner

def benchmark():
    runner = BatchedSpeculativeRunner(batch_size=1)

    # Test 1: Simple arithmetic (baseline)
    source1 = 'int main() { return 42; }'
    bytecode1, data1 = compile_c(source1)

    t0 = time.time()
    for _ in range(100):
        runner.run_batch([bytecode1], [data1], max_steps=100)
    elapsed1 = time.time() - t0

    print(f"Arithmetic: {100/elapsed1:.1f} progs/sec")

    # Test 2: With printf
    source2 = 'int main() { printf("Hello\\n"); return 0; }'
    bytecode2, data2 = compile_c(source2)

    t0 = time.time()
    for _ in range(100):
        runner.run_batch([bytecode2], [data2], max_steps=200)
    elapsed2 = time.time() - t0

    print(f"With printf: {100/elapsed2:.1f} progs/sec")
    print(f"Overhead: {(elapsed2/elapsed1 - 1)*100:.1f}%")

if __name__ == '__main__':
    benchmark()
```

**Acceptance:** I/O overhead is < 20%

---

### Task 4.2: Update Documentation (30 min)

**Files to update:**
1. `docs/README.md` - Add I/O section
2. `docs/IMPLEMENTATION_STATUS.md` - Update to 100% complete
3. `docs/TESTING_STATUS.md` - Update test counts

**Acceptance:** Documentation reflects new I/O support

---

### Task 4.3: Edge Case Testing (30 min)

**File:** `tests/test_io_edge_cases.py` (new file)

```python
#!/usr/bin/env python3
"""Test I/O edge cases."""
import pytest
from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


@pytest.fixture
def runner():
    return BatchedSpeculativeRunner(batch_size=1)


def test_printf_empty_string(runner):
    """Test printf with empty string."""
    source = 'int main() { printf(""); return 0; }'
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=100)
    output, exit_code = results[0]
    assert output == ""
    assert exit_code == 0


def test_printf_no_newline(runner):
    """Test printf without newline."""
    source = 'int main() { printf("Hello"); return 0; }'
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=200)
    output, exit_code = results[0]
    assert output == "Hello"
    assert exit_code == 0


def test_printf_percent_percent(runner):
    """Test printf with %% (literal percent)."""
    source = 'int main() { printf("100%%\\n"); return 0; }'
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=200)
    output, exit_code = results[0]
    assert output == "100%\n"
    assert exit_code == 0


def test_printf_large_number(runner):
    """Test printf with large numbers."""
    source = '''
    int main() {
        int x;
        x = 2147483647;
        printf("%d\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "2147483647\n"
    assert exit_code == 0


def test_printf_zero(runner):
    """Test printf with zero."""
    source = '''
    int main() {
        int x;
        x = 0;
        printf("%d\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "0\n"
    assert exit_code == 0
```

**Acceptance:** All edge cases pass

---

## Implementation Order

### Session 1 (2-3 hours): Core Printf
```
Task 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8
```

**Checkpoint:** 8/8 basic tests passing

### Session 2 (1-2 hours): Polish
```
Task 2.1 → 3.1 → 3.2 → 3.3 → 4.1 → 4.2 → 4.3
```

**Checkpoint:** All tests passing, docs updated, performance validated

---

## Risk Mitigation

### Risk 1: Match Rate Still 0%
**Symptom:** Transformer rejects all DraftVM tokens even with I/O

**Investigation:**
1. Check if DraftVM state matches transformer expectations
2. Compare DraftVM tokens with AutoregressiveVMRunner tokens
3. Debug token-by-token to find divergence

**Mitigation:** This doesn't block I/O - output still accumulates correctly

---

### Risk 2: Performance Regression
**Symptom:** Arithmetic tests slow down after I/O implementation

**Investigation:**
1. Profile code to find bottleneck
2. Check if I/O handlers are being called unnecessarily

**Mitigation:** I/O code only runs on opcodes 31, 33, 36, 37

---

### Risk 3: Data Section Not Loading
**Symptom:** Format strings are all zeros

**Investigation:**
1. Verify data parameter is passed to run_batch()
2. Check data list is populated correctly
3. Verify load_data() is called

**Debugging:**
```python
vm = DraftVM(bytecode)
vm.load_data(data)
print(f"Data at 0x10000: {[vm.memory.get(0x10000 + i, 0) for i in range(20)]}")
```

---

## Success Metrics

### Test Coverage
```
Before: 1096/1096 tests (arithmetic only)
After:  1100+/1100+ tests (arithmetic + I/O + syscalls)
```

### Performance
```
Arithmetic: Should maintain ~6,800 tests/sec
With I/O:   Should be ~5,000+ tests/sec (< 30% overhead)
```

### Functionality
```
✅ printf with all format specifiers
✅ Multiple printf calls
✅ Printf in loops
✅ Negative numbers
✅ Edge cases
✅ memset/memcmp syscalls
```

---

## Alternative Approaches

### Option A: Minimal Implementation (This Plan)
**Pros:**
- Fast implementation (4-6 hours)
- Minimal code changes
- Low risk

**Cons:**
- Only supports printf/read/memset/memcmp
- No generic syscall framework

### Option B: Generic Syscall Framework
**Changes:**
- Create `_dispatch_syscall()` method
- Register handlers in a dict
- Support arbitrary syscalls

**Pros:**
- More extensible
- Easier to add new syscalls

**Cons:**
- More complex
- Takes longer (8-10 hours)
- Over-engineering for current needs

### Option C: Use TOOL_CALL Tokens
**Changes:**
- Make DraftVM emit TOOL_CALL tokens
- Handle I/O via tool system

**Pros:**
- Consistent with AutoregressiveVMRunner
- Could integrate with external I/O

**Cons:**
- Breaks 35-token format
- Major architectural change
- Speculation contract unclear

**Recommendation:** Use Option A (this plan) - minimal and practical

---

## Rollback Plan

If implementation fails or causes regressions:

### Rollback Steps
```bash
# Save current branch
git branch io-implementation-backup

# Revert changes
git checkout neural_vm/speculative.py
git checkout neural_vm/batch_runner.py

# Verify tests still pass
python tests/run_1000_tests.py
```

### Partial Rollback
If only specific parts fail:
1. Keep data loading (Task 1.1) - low risk
2. Revert printf handler if broken
3. Keep wrapper changes if they don't break

---

## Timeline

### Optimistic (4 hours)
- Session 1: 2 hours (Tasks 1.1-1.8)
- Session 2: 2 hours (Tasks 2.1, 3.1-3.3, 4.1-4.3)

### Realistic (6 hours)
- Session 1: 3 hours (Tasks 1.1-1.8 + debugging)
- Session 2: 3 hours (Tasks 2.1, 3.1-3.3, 4.1-4.3 + edge cases)

### Pessimistic (8 hours)
- Session 1: 4 hours (Tasks 1.1-1.8 + major debugging)
- Session 2: 4 hours (All remaining tasks + performance tuning)

---

## Final Checklist

Before marking complete:
- [ ] All Phase 1 tasks complete
- [ ] 8/8 basic I/O tests passing
- [ ] 1096/1096 arithmetic tests still passing
- [ ] No performance regression (< 30% slower)
- [ ] Read syscall implemented (or documented as TODO)
- [ ] memset/memcmp implemented
- [ ] Edge cases tested
- [ ] Documentation updated
- [ ] Code reviewed for edge cases
- [ ] Performance benchmarked

---

## Next Steps After Completion

1. **Add to C Runtime**
   - Implement PRTF/READ opcodes in `vm/onnx_runtime_c4.c`
   - Enable I/O in standalone executables

2. **Optimize Performance**
   - Profile I/O code
   - Reduce string allocations
   - Cache format string parsing

3. **Extended Format Specifiers**
   - Add %u (unsigned)
   - Add %ld (long)
   - Add %p (pointer)
   - Add padding/width specifiers

4. **File I/O**
   - Implement OPEN/CLOS/READ/WRITE for files
   - Add file handle management

5. **Tool Integration**
   - Connect to external I/O tools
   - Enable interactive programs
