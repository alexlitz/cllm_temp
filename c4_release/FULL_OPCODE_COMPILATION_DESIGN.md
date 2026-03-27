# Full C4 Opcode Compilation - 100% Coverage Design

## Goal

Compile **ALL 42 C4 opcodes** to PureFFN + Attention primitives. No opcodes left unsupported.

## Current Coverage

- ✅ **18/42 opcodes** (42.9%) - Already compilable to PureFFN
- ⚠️ **24/42 opcodes** (57.1%) - Need new primitives

## Strategy: FFN → Attention Interface

The key insight: **FFN computes, Attention accesses memory**

```
FFN Layer:
  - Compute addresses
  - Compute values
  - Set request flags (MEM_READ, MEM_WRITE, PC_JUMP, etc.)
  - Prepare data in mailbox slots

Attention Layer:
  - Read flags from FFN output
  - Perform memory operations based on flags
  - Write results back to mailbox slots

Next FFN Layer:
  - Read results from mailbox slots
  - Continue computation
```

## New Primitive Categories

### 1. Memory Request Primitives (FFN Operations)

These are **pure FFN operations** that prepare memory requests:

```python
# New OpType additions
class OpType(Enum):
    # ... existing ops ...

    # Memory request primitives (FFN layer)
    MEM_READ_REQUEST = "mem_read_request"      # Set MEM_READ flag + address
    MEM_WRITE_REQUEST = "mem_write_request"    # Set MEM_WRITE flag + address + data
    STACK_PUSH_REQUEST = "stack_push_request"  # Push to stack (FFN computes address)
    STACK_POP_REQUEST = "stack_pop_request"    # Pop from stack

    # Control flow primitives (FFN layer)
    PC_SET = "pc_set"                          # Set PC to computed value
    PC_CONDITIONAL = "pc_conditional"          # PC = cond ? target : PC+1

    # I/O primitives (FFN layer)
    IO_PUTCHAR_REQUEST = "io_putchar_request"  # Set IO_OUTPUT_READY flag
    IO_GETCHAR_REQUEST = "io_getchar_request"  # Set IO_NEED_INPUT flag

    # System call primitives (FFN layer)
    SYSCALL_REQUEST = "syscall_request"        # Set TOOL_CALL_TYPE flag
```

### 2. Mailbox Slots (Embedding Dimensions)

Already defined in `E` class, but we'll use them explicitly:

```python
# From neural_vm/embedding.py
E.MEM_ADDR_BASE = 136      # [136:144] Memory address (8 nibbles)
E.MEM_DATA_BASE = 144      # [144:152] Memory data (8 nibbles)
E.MEM_WRITE = 152          # 1.0 = write request pending
E.MEM_READ = 153           # 1.0 = read request pending
E.MEM_READY = 154          # 1.0 = operation complete

E.IO_CHAR = 80             # [80:87] I/O character (8 nibbles)
E.IO_OUTPUT_READY = 88     # 1.0 = PUTCHAR ready
E.IO_NEED_INPUT = 90       # 1.0 = GETCHAR needs input

E.IO_TOOL_CALL_TYPE = 93   # Tool call type (OPEN, READ, etc.)
E.IO_TOOL_CALL_ID = 94     # Unique call ID
```

## Implementation Plan by Opcode Category

### Category A: Memory Operations (4 opcodes)

**LI (Load Int)**: `AX = *AX`

```
Layer N (FFN): MEM_READ_REQUEST
  - Copy AX → MEM_ADDR_BASE
  - Set MEM_READ = 1.0

Layer N+1 (Attention): Memory Lookup
  - Query: MEM_ADDR_BASE
  - Key: Historical MEM sections (address matching)
  - Value: Data from matching MEM section
  - Write result → MEM_DATA_BASE
  - Set MEM_READY = 1.0

Layer N+2 (FFN): Result Copy
  - If MEM_READY: Copy MEM_DATA_BASE → AX
  - Clear MEM_READ, MEM_READY
```

**LC (Load Char)**: `AX = *(char*)AX`
- Same as LI but only load 1 byte

**SI (Store Int)**: `*pop = AX`

```
Layer N (FFN): MEM_WRITE_REQUEST
  - Pop address from stack → MEM_ADDR_BASE
  - Copy AX → MEM_DATA_BASE
  - Set MEM_WRITE = 1.0

Layer N+1 (Attention): Memory Write
  - Generate MEM token with address + data
  - Append to context (for future lookups)

Layer N+2 (FFN): Clear Flags
  - Clear MEM_WRITE
```

**SC (Store Char)**: `*(char*)pop = AX`
- Same as SI but only store 1 byte

### Category B: Stack Operations (4 opcodes)

**PSH (Push)**: `SP -= 8; *SP = AX`

```
Layer N (FFN): Compute Stack Address
  - new_SP = SP - 8
  - Copy new_SP → MEM_ADDR_BASE
  - Copy AX → MEM_DATA_BASE
  - Set MEM_WRITE = 1.0
  - Update SP register

Layer N+1 (Attention): Write to Stack
  - Generate MEM token

Layer N+2 (FFN): Clear Flags
  - Clear MEM_WRITE
```

**JSR (Jump Subroutine)**: `push PC; PC = imm`

```
Layer N (FFN): Multi-step
  1. Push current PC to stack (like PSH)
  2. Set PC = immediate

This is a composite: STACK_PUSH_REQUEST + PC_SET
```

**ENT (Enter)**: `push BP; BP = SP; SP -= imm`

```
Layer N (FFN): Multi-step
  1. Push BP to stack
  2. Copy SP → BP
  3. Subtract immediate → SP

Composite: STACK_PUSH_REQUEST + MOVE + SUB
```

**LEV (Leave)**: `SP = BP; BP = pop; PC = pop`

```
Layer N (FFN): Compute addresses for pops
Layer N+1 (Attention): Read BP value from stack
Layer N+2 (FFN): Update BP, compute next pop
Layer N+3 (Attention): Read PC value from stack
Layer N+4 (FFN): Update PC
```

### Category C: Control Flow (5 opcodes)

**JMP (Jump)**: `PC = imm`

```
Layer N (FFN): PC_SET
  - Copy immediate → PC register
  - Simple MOVE operation
```

**BZ (Branch if Zero)**: `if (AX == 0) PC = imm`

```
Layer N (FFN): PC_CONDITIONAL
  - cond = (AX == 0)  [CMP_EQ with 0]
  - target = immediate [CONST]
  - fallthrough = PC + INSTR_WIDTH [ADD]
  - result = SELECT(cond, target, fallthrough)
  - Copy result → PC

Composite: CMP_EQ + CONST + ADD + SELECT + MOVE
```

**BNZ (Branch if Not Zero)**: `if (AX != 0) PC = imm`
- Same as BZ but with CMP_NE

**BLT (Branch if Less Than)**: `if (result < 0) PC = imm`
- Neural VM extension, similar pattern

**BGE (Branch if Greater or Equal)**: `if (result >= 0) PC = imm`
- Neural VM extension, similar pattern

### Category D: System Calls (13 opcodes)

**Principle**: FFN sets up call request, tool handler executes externally

**OPEN (File Open)**: `AX = open(filename, mode)`

```
Layer N (FFN): SYSCALL_REQUEST
  - Set IO_TOOL_CALL_TYPE = OPEN
  - Set IO_TOOL_CALL_ID = unique_id
  - Copy filename address to mailbox
  - Copy mode to mailbox

External: Tool handler reads request, executes open()

Layer N+1 (FFN): Read Response
  - Copy IO_TOOL_RESPONSE → AX (file descriptor)
  - Clear IO_TOOL_CALL_TYPE
```

**READ, CLOS, PRTF**: Similar pattern with different call types

**MALC (Malloc)**: `AX = malloc(size)`

```
Layer N (FFN): Bump Allocator Arithmetic
  - Load HEAP_PTR
  - result = HEAP_PTR [current allocation]
  - new_ptr = HEAP_PTR + size [ADD]
  - Check: new_ptr < HEAP_END [CMP_LT]
  - If valid: Update HEAP_PTR, return result
  - If invalid: Return 0 (NULL)

Pure FFN: ADD + CMP + SELECT + MOVE
No attention needed!
```

**FREE**: No-op for bump allocator (FFN just returns)

**MSET (memset)**: `memset(ptr, val, size)`

```
Layer N (FFN): Loop Setup
  - Compute addresses [ptr, ptr+1, ..., ptr+size-1]
  - For each: MEM_WRITE_REQUEST with val

Layer N+1 (Attention): Batch Memory Writes
  - Generate multiple MEM tokens

Needs loop unrolling or sequential execution
```

**MCMP (memcmp)**: `AX = memcmp(p1, p2, size)`

```
Layer N (FFN): Compute addresses for both pointers
Layer N+1 (Attention): Read both memory regions
Layer N+2 (FFN): Compare values, return result
```

**EXIT**: `exit(code)`

```
Layer N (FFN): Set Exit Flag
  - Set IO_PROGRAM_END = 1.0
  - Copy code → IO_EXIT_CODE

Pure FFN flag setting
```

**GETCHAR**: `AX = getchar()`

```
Layer N (FFN): IO_GETCHAR_REQUEST
  - Set IO_NEED_INPUT = 1.0

External: Runner provides input character

Layer N+1 (FFN): Read Input
  - Copy IO_CHAR → AX
  - Clear IO_NEED_INPUT
```

**PUTCHAR**: `putchar(AX)`

```
Layer N (FFN): IO_PUTCHAR_REQUEST
  - Copy AX → IO_CHAR
  - Set IO_OUTPUT_READY = 1.0

External: Runner reads and outputs character

Layer N+1 (FFN): Clear Flags
  - Clear IO_OUTPUT_READY
```

## Primitive Operation Summary

### Pure FFN Primitives (Already Have)

| Primitive | Operations |
|-----------|------------|
| Arithmetic | ADD, SUB, MUL, DIV, MOD |
| Comparison | EQ, NE, LT, GT, LE, GE |
| Logical | AND, OR, NOT, XOR |
| Bitwise | BIT_AND, BIT_OR, BIT_XOR, SHL, SHR |
| Conditional | SELECT, IF_THEN |
| Register | MOVE, CLEAR, CONST, SWAP |

### New FFN Primitives (Need to Add)

| Primitive | Description | Used By |
|-----------|-------------|---------|
| **MEM_READ_REQUEST** | Set MEM_READ flag + address | LI, LC |
| **MEM_WRITE_REQUEST** | Set MEM_WRITE flag + address + data | SI, SC, PSH |
| **PC_SET** | Copy value to PC register | JMP, JSR, LEV |
| **PC_CONDITIONAL** | Conditional PC update | BZ, BNZ, BLT, BGE |
| **IO_PUTCHAR_REQUEST** | Set IO_OUTPUT_READY flag | PUTCHAR |
| **IO_GETCHAR_REQUEST** | Set IO_NEED_INPUT flag | GETCHAR |
| **SYSCALL_REQUEST** | Set IO_TOOL_CALL_TYPE flag | OPEN, READ, CLOS, PRTF |
| **FLAG_SET** | Set arbitrary flag to 1.0 | EXIT, etc. |
| **FLAG_CLEAR** | Clear flag to 0.0 | Cleanup operations |

### Attention Operations (Existing)

| Operation | Description |
|-----------|-------------|
| **Memory Lookup** | Q·K^T matching on addresses |
| **Memory Write** | Append MEM token to context |
| **Stack Access** | Special case of memory lookup |

## Layer Allocation Strategy

Since we can be "as wide or deep as needed", let's allocate layers by function:

```
Layer 0-1:   Token Embedding + Position Encoding
Layer 2-4:   Instruction Fetch (Attention)
Layer 5-6:   Opcode Decode (FFN)
Layer 7-8:   Operand Fetch (Attention)

Layer 9-11:  ALU Operations (FFN) ← COMPILED WEIGHTS
             - Arithmetic, comparison, bitwise
             - Pure computation, no memory

Layer 12:    Memory Request Setup (FFN) ← NEW
             - Prepare MEM_ADDR, MEM_DATA
             - Set request flags

Layer 13:    Memory Operations (Attention)
             - Read/write based on flags
             - Stack access

Layer 14:    Control Flow (FFN) ← NEW
             - PC update logic
             - Branch conditions

Layer 15:    I/O and System Calls (FFN) ← NEW
             - I/O flag setting
             - System call requests

Layer 16:    Output & Next Token Prediction (FFN + Head)
```

## Implementation Phases

### Phase 1: Extend OpType Enum ✅ (Next Step)

Add new primitive types to `graph_weight_compiler.py`:
- Memory request primitives
- Control flow primitives
- I/O primitives

### Phase 2: Implement FFN Emitters

For each new primitive, implement weight emission in `nibble_weight_compiler.py`:

```python
def emit_mem_read_request(self, node: IRNode, graph: ComputationGraph):
    """Emit: Copy address to MEM_ADDR_BASE, set MEM_READ flag."""
    # Unit 0-1: Copy source → MEM_ADDR_BASE (8 nibbles)
    # Unit 2-3: Set MEM_READ = 1.0 (flag)
    pass

def emit_pc_conditional(self, node: IRNode, graph: ComputationGraph):
    """Emit: PC = cond ? target : fallthrough"""
    # This is just SELECT on PC register
    # Already have SELECT primitive!
    pass
```

### Phase 3: Extend OpcodeMapper

Map all 42 opcodes to computation graphs:

```python
def _compile_memory_op(self, opcode: C4Opcode, **params):
    """Compile LI/LC/SI/SC to memory request primitives."""
    graph = ComputationGraph()

    if opcode == C4Opcode.LI:
        # LI: AX = *AX
        ax = graph.add_input("AX")
        # Copy AX → MEM_ADDR_BASE
        addr_copy = graph.add_op(OpType.MOVE, [ax], "MEM_ADDR")
        # Set MEM_READ flag
        read_req = graph.add_op(OpType.MEM_READ_REQUEST, [addr_copy], "MEM_READ")

    return graph
```

### Phase 4: Test All 42 Opcodes

Create comprehensive test suite:
- Graph generation for all opcodes
- Weight emission for all primitives
- Integration with attention layers

### Phase 5: Multi-Layer Compilation

For opcodes that need multiple layers (e.g., LI needs FFN → Attention → FFN):

```python
def compile_opcode_multilayer(self, opcode: C4Opcode):
    """Compile opcode to multiple layer weights."""
    layers = []

    if opcode == C4Opcode.LI:
        # Layer 1: FFN - Setup memory read request
        layer1 = self.compile_operation(OpType.MEM_READ_REQUEST, opcode)
        layers.append(('ffn', layer1))

        # Layer 2: Attention - Memory lookup (manual weights)
        layer2 = self.get_memory_lookup_attention_weights()
        layers.append(('attention', layer2))

        # Layer 3: FFN - Copy result to AX
        layer3 = self.compile_operation(OpType.MOVE, opcode)
        layers.append(('ffn', layer3))

    return layers
```

## Benefits of This Approach

### 100% Coverage
- All 42 C4 opcodes compilable
- No special cases or manual implementations
- Uniform compilation pipeline

### Composability
- Complex opcodes = compositions of primitives
- JSR = STACK_PUSH_REQUEST + PC_SET
- BZ = CMP_EQ + SELECT + PC_SET

### Extensibility
- Easy to add new opcodes
- New primitives can be added incrementally
- Graph optimization applies to all opcodes

### Debugging
- Each primitive testable independently
- Clear layer boundaries
- Explicit data flow through mailbox slots

## Next Steps

1. ✅ Document design (this file)
2. 🔄 Extend OpType enum with new primitives
3. 🔄 Implement new FFN emitters
4. 🔄 Map all 42 opcodes to graphs
5. 🔄 Test 100% compilation coverage
6. 🔄 Benchmark performance

## Success Criteria

✅ All 42 C4 compiler opcodes have compilation paths
✅ Each opcode maps to PureFFN + optional Attention
✅ All primitives have weight emission implementations
✅ Test suite validates all opcode compilations
✅ Sparsity remains >99% across all operations
✅ No manual weight tuning required
