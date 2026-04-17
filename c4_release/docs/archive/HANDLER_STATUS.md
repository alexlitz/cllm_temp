# Runner Handler Status - What Remains

This document lists all operations that still use Python runner handlers (fallbacks) instead of pure neural execution.

## ✅ Fully Neural (NO Handlers)

These instructions execute 100% through transformer weights:

### Basic Instructions
- **IMM** - Load immediate ✓ (just completed)
- **LEA** - Load effective address ✓
- **PSH** - Push to stack ✓

### Arithmetic Operations (L8-L10 ALU)
- **ADD** - Addition ✓
- **SUB** - Subtraction ✓
- **MUL** - Multiplication ✓
- **DIV** - Division ✓
- **MOD** - Modulo ✓

### Bitwise Operations (L8-L10 ALU)
- **OR** - Bitwise OR ✓
- **XOR** - Bitwise XOR ✓
- **AND** - Bitwise AND ✓
- **SHL** - Shift left ✓
- **SHR** - Shift right ✓

### Comparison Operations (L8-L10 ALU)
- **EQ** - Equal ✓
- **NE** - Not equal ✓
- **LT** - Less than ✓
- **GT** - Greater than ✓
- **LE** - Less or equal ✓
- **GE** - Greater or equal ✓

### Memory Operations (L15 Attention)
- **LI** - Load int (32-bit) ✓ (L15 softmax1 memory lookup)
- **LC** - Load char (8-bit) ✓ (L15 softmax1 memory lookup)
- **SI** - Store int ✓ (MEM section tracking)
- **SC** - Store char ✓ (MEM section tracking)

### Control Flow
- **JMP** - Jump ✓
- **JZ** - Jump if zero ✓
- **JNZ** - Jump if not zero ✓
- **EXIT** - Exit program ✓
- **NOP** - No operation ✓

## ⚠️ Partial Handlers (Function Calls)

These use handlers for correctness but the core operation is neural:

### Function Call Operations
**Location**: `neural_vm/run_vm.py` lines 215-237

- **JSR** (Jump Subroutine) - `_handler_jsr`
  - **Purpose**: Pushes return address (PC+5) to stack
  - **Status**: Neural jump works, handler ensures correct multi-byte PC push
  - **Why needed**: 32-bit return address requires precise stack manipulation

- **ENT** (Enter Function) - `_handler_ent`
  - **Purpose**: Sets up stack frame (push BP, set BP=SP, allocate locals)
  - **Status**: Neural parts work, handler ensures multi-byte BP correctness
  - **Why needed**: Complex multi-step operation with 32-bit values

- **LEV** (Leave Function) - `_handler_lev`
  - **Purpose**: Tears down stack frame (SP=BP, pop BP, return)
  - **Status**: Neural parts work, handler ensures multi-byte correctness
  - **Why needed**: Complex multi-step operation with 32-bit values

**Comment from code** (line 212-214):
> "Function-call handlers dispatched by exec_pc (not output PC).
> These are runner-side compatibility shims for full 32-bit correctness
> while the corresponding neural memory paths are being completed."

## ⚠️ Runner Memory Operations (Can be Blocked)

These operations use runner shadow memory. **Blocked in `pure_attention_memory=True` mode**.

**Location**: Lines 54-60, syscall handlers 168-176

- **ADJ** (Adjust Stack) - `_syscall_adj`
  - **Purpose**: Adjust stack pointer (SP += N)
  - **Status**: Uses runner shadow memory tracking
  - **Blockable**: Yes (in pure mode)

- **MALC** (Malloc) - `_syscall_malc`
  - **Purpose**: Allocate heap memory
  - **Status**: Bump allocator in runner (heap_ptr tracking)
  - **Blockable**: Yes (in pure mode)

- **FREE** (Free) - `_syscall_free`
  - **Purpose**: Free heap memory
  - **Status**: Zero-fill via runner shadow memory
  - **Blockable**: Yes (in pure mode)

- **MSET** (Memset) - `_syscall_mset`
  - **Purpose**: Fill memory region with byte value
  - **Status**: Uses runner shadow memory
  - **Blockable**: Yes (in pure mode)

- **MCMP** (Memcmp) - `_syscall_mcmp`
  - **Purpose**: Compare memory regions
  - **Status**: Uses runner shadow memory
  - **Blockable**: Yes (in pure mode)

## 🔌 I/O Operations (Intentional External Boundary)

These are **intentionally external** - they interact with the outside world and are not meant to be neural.

**Location**: Lines 161-174

### Character I/O
- **PUTCHAR** - `_syscall_putchar`
  - **Purpose**: Output a character
  - **Status**: Neural routing (AX→OUTPUT→byte), runner reads output
  - **Type**: External I/O boundary (not a fallback)

- **GETCHAR** - `_syscall_getchar`
  - **Purpose**: Read a character from stdin
  - **Status**: Runner injects stdin byte into AX
  - **Type**: External I/O boundary (not a fallback)

### File I/O
- **OPEN** - `_syscall_open`
  - **Purpose**: Open file
  - **Status**: Python file operations
  - **Type**: External I/O boundary

- **READ** - `_syscall_read`
  - **Purpose**: Read from file
  - **Status**: Python file operations
  - **Type**: External I/O boundary

- **CLOS** - `_syscall_clos`
  - **Purpose**: Close file
  - **Status**: Python file operations
  - **Type**: External I/O boundary

- **PRTF** (Printf) - `_syscall_prtf`
  - **Purpose**: Formatted output
  - **Status**: Python string formatting
  - **Type**: External I/O boundary

**Comment from code** (lines 11-13):
> "Tool/I/O boundary handlers (`OPEN/READ/CLOS/PRTF/GETCHAR/PUTCHAR`) are
> intentionally external."

## Summary Statistics

| Category | Count | Status |
|----------|-------|--------|
| **Fully Neural** | 29 ops | ✅ 100% neural execution |
| **Function Calls** | 3 ops | ⚠️ Handlers for multi-byte correctness |
| **Memory Ops** | 5 ops | ⚠️ Runner fallback (blockable in pure mode) |
| **I/O Ops** | 6 ops | 🔌 Intentional external boundary |

## Pure Attention Memory Mode

When `pure_attention_memory=True`:
- ✅ All 29 basic/ALU/comparison/memory ops work neurally
- ✅ Function calls (JSR/ENT/LEV) still work with handlers
- ❌ Memory ops (ADJ/MALC/FREE/MSET/MCMP) are **blocked** and reported
- ✅ I/O ops still work (they're meant to be external)

## Removed Handlers (Previously Used, Now Neural)

From code comments (lines 216-236):

```python
# REMOVED: IMM and LEA now work fully neurally (L6 FFN relay)
# Opcode.IMM: self._handler_imm,  ← JUST COMPLETED!
# Opcode.LEA: self._handler_lea,

# REMOVED: PSH now works fully neurally (L6 FFN SP -= 8)
# Opcode.PSH: self._handler_psh,

# REMOVED: Arithmetic operations now work fully neurally (L8-L10 ALU)
# Opcode.ADD: self._handler_add,
# Opcode.SUB: self._handler_sub,
# Opcode.MUL: self._handler_mul,
# Opcode.DIV: self._handler_div,
# Opcode.MOD: self._handler_mod,

# REMOVED: Bitwise operations now work fully neurally (L8-L10 ALU)
# Opcode.OR: self._handler_or,
# Opcode.XOR: self._handler_xor,
# Opcode.AND: self._handler_and,

# REMOVED: Shift operations now work fully neurally (L8-L10 ALU)
# Opcode.SHL: self._handler_shl,
# Opcode.SHR: self._handler_shr,
```

Also removed (from syscall comments, lines 163-167):
```python
# DIV/MOD removed — neural (DivModModule after L10)
# LI/LC handlers removed: L15 softmax1 memory lookup produces
# correct AX values with MEM section retention + MEM_STORE injection.
# SI/SC handlers removed: _track_memory_write provides equivalent
# shadow memory updates via L14-generated MEM section extraction.
```

## Next Steps for Full Neural Execution

To eliminate remaining handlers:

### 1. Function Calls (JSR/ENT/LEV)
**Challenge**: Multi-byte (32-bit) stack manipulation
**Approach**:
- Extend existing neural stack operations (PSH works neurally)
- Add multi-byte push/pop support in transformer
- May require additional FFN layers for complex operations

### 2. Memory Operations (ADJ/MALC/FREE/MSET/MCMP)
**Challenge**: Complex memory management
**Approach**:
- **ADJ**: Can be neural (similar to PSH SP adjustment)
- **MALC/FREE**: Require heap allocator state (complex)
- **MSET/MCMP**: Require loop semantics (very complex)
- These may remain as "system calls" even in pure mode

### 3. Keep I/O External
**Decision**: I/O operations should remain external
**Reason**: They interact with the real world and aren't VM execution

## Conclusion

**Current Status**: **29/43 operations (67%) are fully neural**, including all arithmetic, bitwise, comparison, and basic memory operations.

The remaining handlers fall into three categories:
1. **Function calls** (3): Need multi-byte correctness (fixable)
2. **Memory ops** (5): Complex system operations (may stay external)
3. **I/O ops** (6): Intentionally external (should stay external)

**The IMM completion demonstrates the path forward**: Each handler can be eliminated by:
1. Adding proper flags to embeddings
2. Creating attention relay paths
3. Implementing FFN routing logic
4. Amplifying signals to overcome attenuation
