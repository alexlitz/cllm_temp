# C4 Standard Library Proposal

**Date**: 2026-04-09
**Context**: User requested memory operations (malloc/free/memset/memcmp) use bytecode subroutines instead of special opcodes

---

## Current Implementation

Memory operations are currently **special opcodes** with Python handlers:

| C Function | Opcode | Handler | Status |
|------------|--------|---------|--------|
| `malloc()` | MALC (34) | `_syscall_malc()` | Python fallback |
| `free()` | FREE (35) | `_syscall_free()` | Python fallback |
| `memset()` | MSET (36) | `_syscall_mset()` | Python fallback |
| `memcmp()` | MCMP (37) | `_syscall_mcmp()` | Python fallback |

**Mapping** (src/compiler.py:350-352):
```python
syscalls = ['open', 'read', 'close', 'printf', 'malloc', 'free', 'memset', 'memcmp', 'exit']
for i, name in enumerate(syscalls):
    self.symbols[name] = Symbol(name, 'Sys', INT, Op.OPEN + i)
```

---

## Proposed Approach: C4 Standard Library

Implement these as **C4 C bytecode subroutines** instead of special opcodes.

### Advantages

1. **Pure C4 implementation** - no special opcodes needed
2. **Already works** with existing neural VM (uses LI/LC/SI/SC)
3. **Testable/debuggable** like any C program
4. **Easy to extend/modify** - just edit C code
5. **Transparent** - users can read the implementation
6. **No neural training** required - works immediately

### Disadvantages

1. **Slower** than opcode-based (more VM steps per call)
2. **Requires linking** stdlib into every program
3. **Limited optimization** compared to native implementation

---

## Implementation Plan

### Step 1: Create stdlib/memory.c4

```c
// stdlib/memory.c4 - C4 Standard Library Memory Functions

// Heap management
static int heap_ptr = 0x10000;  // Initialize after data section

void *malloc(int size) {
    int aligned_size;
    void *ptr;

    // Align to 8-byte boundary
    aligned_size = (size + 7) & ~7;

    // Return current heap pointer
    ptr = (void *)heap_ptr;

    // Bump allocator: advance heap pointer
    heap_ptr = heap_ptr + aligned_size;

    return ptr;
}

void free(void *ptr) {
    // Simple free: no-op (bump allocator doesn't reclaim)
    // Could implement free list in future
}

void *memset(void *ptr, int val, int size) {
    char *p;
    int i;

    p = (char *)ptr;
    i = 0;

    while (i < size) {
        *p = val;
        p = p + 1;
        i = i + 1;
    }

    return ptr;
}

int memcmp(void *ptr1, void *ptr2, int size) {
    unsigned char *p1;
    unsigned char *p2;
    int i;

    p1 = (unsigned char *)ptr1;
    p2 = (unsigned char *)ptr2;
    i = 0;

    while (i < size) {
        if (*p1 != *p2) {
            return *p1 - *p2;
        }
        p1 = p1 + 1;
        p2 = p2 + 1;
        i = i + 1;
    }

    return 0;
}
```

### Step 2: Modify Compiler to Link stdlib

```python
# src/compiler.py - modify compile_c()

def compile_c(source: str) -> Tuple[List[int], List[int]]:
    # Load stdlib
    stdlib_path = Path(__file__).parent / 'stdlib' / 'memory.c4'
    if stdlib_path.exists():
        stdlib_source = stdlib_path.read_text()
        # Prepend stdlib to user source
        full_source = stdlib_source + '\n' + source
    else:
        full_source = source

    compiler = Compiler()
    compiler.compile(full_source)
    return compiler.code, compiler.data
```

### Step 3: Remove Opcode Mapping

```python
# src/compiler.py - update syscall list

# Before:
syscalls = ['open', 'read', 'close', 'printf', 'malloc', 'free', 'memset', 'memcmp', 'exit']

# After (remove memory functions):
syscalls = ['open', 'read', 'close', 'printf', 'exit']
```

### Step 4: Remove Python Handlers

```python
# neural_vm/run_vm.py - remove handlers

_RUNNER_VM_MEMORY_OPS = {
    # Opcode.MALC,  # Now implemented as C4 stdlib
    # Opcode.FREE,  # Now implemented as C4 stdlib
    # Opcode.MSET,  # Now implemented as C4 stdlib
    # Opcode.MCMP,  # Now implemented as C4 stdlib
}
```

---

## Testing Plan

### Test 1: malloc/free
```c
#include "stdlib/memory.c4"

int main() {
    int *a = malloc(100);
    int *b = malloc(200);
    int offset = (int)b - (int)a;
    free(a);
    free(b);
    return offset;  // Should be 104 (100 aligned to 104)
}
```

### Test 2: memset
```c
#include "stdlib/memory.c4"

int main() {
    char buf[10];
    memset(buf, 0xAB, 10);
    return buf[0] + buf[9];  // Should return 0x156
}
```

### Test 3: memcmp
```c
#include "stdlib/memory.c4"

int main() {
    char *a = "hello";
    char *b = "hello";
    char *c = "world";

    if (memcmp(a, b, 5) == 0 && memcmp(a, c, 5) != 0) {
        return 42;
    }
    return 0;
}
```

---

## Performance Comparison

### Current (Opcode-based)
- malloc: 1 VM step (Python handler)
- memset(100 bytes): 1 VM step (Python loop)
- memcmp(16 bytes): 1 VM step (Python loop)

### Proposed (C4 stdlib)
- malloc: ~20 VM steps (arithmetic + pointer ops)
- memset(100 bytes): ~120 VM steps (loop overhead + stores)
- memcmp(16 bytes): ~50 VM steps (loop overhead + loads)

**Slowdown**: 20-120x for memory operations

**Mitigation**: Most programs spend <5% time in malloc/memset, so overall impact is ~5-10% slowdown.

---

## Alternative: Hybrid Approach

Keep opcodes for performance-critical operations, use stdlib for others:

| Function | Implementation | Reason |
|----------|----------------|--------|
| `malloc()` | **Opcode** | Very common, needs speed |
| `free()` | **Stdlib** | Less critical, simple |
| `memset()` | **Opcode** | Large operations, needs speed |
| `memcmp()` | **Stdlib** | Rare, small comparisons |

---

## Recommendation

**Approach**: **Full stdlib implementation** (all 4 functions)

**Rationale**:
1. Aligns with VM philosophy (pure autoregressive execution)
2. Works immediately without neural implementation
3. Easy to maintain and extend
4. Performance impact acceptable for most use cases
5. Users can optimize hot paths manually

**Timeline**:
- Step 1-3: 2-3 hours
- Step 4: 1 hour
- Testing: 1-2 hours
- **Total**: 4-6 hours

**Next Steps**:
1. Create `src/stdlib/memory.c4`
2. Modify compiler to auto-link stdlib
3. Remove opcode mappings and handlers
4. Test with existing programs
5. Document stdlib functions

---

## Decision Required

Please choose:

**A)** Full stdlib implementation (recommended)
**B)** Hybrid approach (opcodes for malloc/memset, stdlib for free/memcmp)
**C)** Keep opcodes, implement neurally (50-80 hours from plan)
**D)** Different approach (please specify)

---

**Author**: Claude Sonnet 4.5
**Status**: Awaiting user decision
