# I/O Implementation Plan for DraftVM

## Current Situation

**Problem:** DraftVM treats I/O opcodes (PRTF=33, READ=31) as NOPs (line 170 in `speculative.py`):
```python
# NOP and other unhandled ops: do nothing (advance PC only)
```

**Impact:** Programs with printf/read work in AutoregressiveVMRunner but fail in fast speculation mode.

---

## What's Needed for I/O Support

### 1. Data Section Loading
**Status:** ❌ Missing

**Issue:** Format strings like "Hello World\n" are stored in the data section at address 0x10000, but DraftVM doesn't load them.

**Implementation:**
```python
class DraftVM:
    def __init__(self, bytecode):
        # ... existing code ...
        self.memory = {}  # Already exists

    def load_data(self, data):
        """Load data section into memory at standard address 0x10000."""
        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b
```

**Where to call:** In `BatchedSpeculativeRunner.run_batch()`, when creating DraftVM instances.

---

### 2. Printf Handler (PRTF opcode 33)
**Status:** ❌ Missing

**Reference:** `neural_vm/run_vm.py:1068` has the AutoregressiveVMRunner implementation.

**Required Components:**

#### A. Output Buffer
```python
def __init__(self, bytecode):
    # ... existing code ...
    self.output = []  # Accumulate printf output
```

#### B. String Reader
```python
def _read_string(self, addr):
    """Read null-terminated string from memory."""
    chars = []
    while True:
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

#### C. Printf Handler
```python
def _handle_printf(self):
    """Handle PRTF syscall - parse format string and output.

    Stack layout when PRTF is called:
    SP+0: arg_n
    SP+8: arg_n-1
    ...
    SP+(n-1)*8: arg_1
    SP+n*8: format_string_pointer
    """
    # Peek ahead to get argc from next ADJ instruction
    next_idx = self.idx
    argc = 1
    if next_idx < len(self.code):
        next_instr = self.code[next_idx]
        next_op = next_instr & 0xFF
        if next_op == 7:  # ADJ
            next_imm = next_instr >> 8
            if next_imm >= 0x800000:
                next_imm -= 0x1000000
            argc = next_imm // 8

    # Get format string from stack
    fmt_addr = self._mem_read(self.sp + (argc - 1) * 8)
    fmt = self._read_string(fmt_addr)

    # Parse format string and extract arguments
    result = []
    arg_idx = 1
    i = 0
    while i < len(fmt):
        if fmt[i] == '\\' and i + 1 < len(fmt):
            # Handle escape sequences: \n, \t, \\
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

#### D. Output Accessor
```python
def get_output(self):
    """Get accumulated output."""
    return ''.join(self.output)
```

#### E. Update step() Method
```python
def step(self):
    # ... existing code ...
    elif op == 33:  # PRTF
        self._handle_printf()
    elif op == 38:  # EXIT
        self.halted = True
    # ... rest of code ...
```

---

### 3. Read Handler (READ opcode 31)
**Status:** ❌ Missing (but lower priority than printf)

**Implementation:**
```python
def __init__(self, bytecode):
    # ... existing code ...
    self.stdin_buf = ""  # Input buffer
    self.stdin_pos = 0   # Current read position

def set_stdin(self, data):
    """Set stdin buffer for READ operations."""
    self.stdin_buf = data
    self.stdin_pos = 0

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

---

### 4. BatchedSpeculativeRunner Integration
**Status:** ❌ Missing

**Required Changes:**

#### A. Update `_BlockedDraftVM` Wrapper
```python
class _BlockedDraftVM:
    def __init__(self, bytecode, data=None):
        self._vm = DraftVM(bytecode)
        if data:
            self._vm.load_data(data)
        self._bytecode = bytecode

    def get_output(self):
        """Allow getting output (accumulates during execution)."""
        return self._vm.get_output()

    def set_stdin(self, data):
        """Allow setting stdin buffer."""
        return self._vm.set_stdin(data)
```

#### B. Update `run_batch()` Method
```python
def run_batch(self, bytecodes, data_list=None, argv_list=None, max_steps=10000):
    # Ensure data_list is populated
    if data_list is None:
        data_list = [b''] * len(bytecodes)

    # Pass data when creating DraftVMs
    self.draft_vms = [_BlockedDraftVM(bc, data_list[i]) for i, bc in enumerate(bytecodes)]

    # ... execution loop ...

    # Collect results INCLUDING output
    results = []
    for i, (bytecode, data, ctx) in enumerate(zip(bytecodes, data_list, self.contexts)):
        # Run reference VM for exit code
        ref_vm = FastLogicalVM()
        ref_vm.reset()
        ref_vm.load(bytecode, data)
        exit_code = ref_vm.run(max_steps=100000)

        # Get output from DraftVM
        output = self.draft_vms[i].get_output()
        results.append((output, exit_code))

    return results
```

---

## Format Specifier Support

| Specifier | Description | Priority | Status |
|-----------|-------------|----------|--------|
| `%d` | Signed integer | HIGH | ❌ Missing |
| `%x` | Hexadecimal | MEDIUM | ❌ Missing |
| `%c` | Character | MEDIUM | ❌ Missing |
| `%s` | String | LOW | ❌ Missing |
| `%%` | Literal % | LOW | ❌ Missing |

**Escape Sequences:**
- `\n` - Newline (HIGH)
- `\t` - Tab (MEDIUM)
- `\\` - Backslash (LOW)

---

## Files to Modify

1. **`neural_vm/speculative.py`** (Core implementation)
   - Add `load_data()` method
   - Add output/stdin buffers to `__init__`
   - Add `_read_string()` helper
   - Add `_handle_printf()` handler
   - Add `_handle_read()` handler
   - Add `get_output()` and `set_stdin()` accessors
   - Update `step()` to handle opcodes 31 and 33

2. **`neural_vm/batch_runner.py`** (Integration)
   - Update `_BlockedDraftVM.__init__()` to accept data
   - Add `get_output()` and `set_stdin()` to wrapper
   - Update `run_batch()` to pass data and collect output

---

## Testing Plan

### Phase 1: Basic Printf
```python
# Test 1: Simple string
source = 'int main() { printf("Hello\\n"); return 0; }'
# Expected: output="Hello\n", exit=0

# Test 2: Integer format
source = 'int main() { int x; x = 42; printf("x=%d\\n", x); return 0; }'
# Expected: output="x=42\n", exit=0
```

### Phase 2: Format Specifiers
```python
# Test 3: Hex
source = 'int main() { printf("0x%x\\n", 255); return 0; }'
# Expected: output="0xff\n", exit=0

# Test 4: Character
source = 'int main() { printf("%c\\n", 65); return 0; }'
# Expected: output="A\n", exit=0
```

### Phase 3: Complex Cases
```python
# Test 5: Multiple arguments
source = 'int main() { printf("%d + %d = %d\\n", 10, 32, 42); return 0; }'
# Expected: output="10 + 32 = 42\n", exit=0

# Test 6: Loop
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
# Expected: output="i=0\ni=1\ni=2\n", exit=0
```

---

## Estimated Effort

| Task | Time | Priority |
|------|------|----------|
| Data loading | 15 min | HIGH |
| Printf handler | 1 hour | HIGH |
| Output integration | 30 min | HIGH |
| Basic testing | 30 min | HIGH |
| Read handler | 30 min | MEDIUM |
| Comprehensive testing | 1 hour | MEDIUM |
| **Total** | **3-4 hours** | - |

---

## Library Functions (memset, memcmp, memcpy)

### ✅ Available Syscalls

**From `src/compiler.py:350`:**
```python
syscalls = ['open', 'read', 'close', 'printf', 'malloc', 'free', 'memset', 'memcmp', 'exit']
```

**Opcodes:**
- `MSET = 36` - memset(ptr, val, size)
- `MCMP = 37` - memcmp(p1, p2, size)

**Handler Status:**

| Function | Opcode | AutoregressiveVMRunner | DraftVM | FastLogicalVM |
|----------|--------|------------------------|---------|---------------|
| memset | 36 | ✅ Implemented | ❌ Missing | ❌ Missing |
| memcmp | 37 | ✅ Implemented | ❌ Missing | ❌ Missing |

**AutoregressiveVMRunner Handlers:**
- `_syscall_mset()` at line 1121 - fills memory region with byte value
- `_syscall_mcmp()` at line 1135 - compares two memory regions

### ❌ Not Available

**memcpy:** No syscall exists. Users must implement manually:
```c
void* memcpy(void* dest, void* src, int n) {
    char* d = (char*)dest;
    char* s = (char*)src;
    int i;
    i = 0;
    while (i < n) {
        d[i] = s[i];
        i = i + 1;
    }
    return dest;
}
```

**Other common functions also missing:**
- strlen - must implement manually
- strcpy - must implement manually
- strcmp - must implement manually (but memcmp exists)

### Implementation Note

**If you want memset/memcmp in DraftVM:**

Add to `speculative.py`:
```python
def _handle_memset(self):
    """MSET (memset) - fill memory with byte value."""
    size = self._mem_read(self.sp)
    val = self._mem_read(self.sp + 8) & 0xFF
    ptr = self._mem_read(self.sp + 16)
    for i in range(size):
        self._mem_write(ptr + i, val)
    self.ax = ptr

def _handle_memcmp(self):
    """MCMP (memcmp) - compare memory regions."""
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

# In step() method:
elif op == 36:  # MSET
    self._handle_memset()
elif op == 37:  # MCMP
    self._handle_memcmp()
```

---

## Alternative: Baked Library Functions

**Question:** Could we compile C implementations to bytecode and bake them into the prompt?

**Answer:** Yes, but not currently implemented. This would require:

1. Write C implementations of library functions
2. Compile them to C4 bytecode
3. Include bytecode in the initial context
4. Set up symbol table with function addresses
5. Functions would call via JSR instead of syscalls

**Trade-offs:**
- ✅ More flexible (any C code)
- ✅ No special VM support needed
- ❌ Slower (function call overhead)
- ❌ Uses more context tokens
- ❌ Complexity in setup

**Recommendation:** Keep syscalls for performance-critical operations (printf, memset, malloc). Use C implementations for less common functions.

---

## Summary

**What's Missing for I/O:**
1. Data section loading in DraftVM
2. Printf handler with format parsing
3. Read handler (lower priority)
4. Output buffer accumulation
5. Integration in BatchedSpeculativeRunner

**What's Available:**
- ✅ memset syscall (opcode 36) - in AutoregressiveVMRunner only
- ✅ memcmp syscall (opcode 37) - in AutoregressiveVMRunner only
- ❌ memcpy - must implement manually in C
- ❌ Most libc functions - must implement manually

**Effort:** ~3-4 hours of focused implementation to get I/O working in speculation mode.
