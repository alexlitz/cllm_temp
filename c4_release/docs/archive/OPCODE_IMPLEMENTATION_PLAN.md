# C4 VM Opcode Implementation Plan

## Overview

The C4 compiler emits **42 opcodes**: 39 core opcodes (0-38) plus 3 I/O opcodes (64-66). This document outlines the plan to support all opcodes through the graph weight compiler.

**Note:** Opcodes 39-42 (NOP, POP, BLT, BGE) are neural VM extensions defined but not emitted by the C4 compiler.

## Current Status

**Implemented in Graph Compiler: 24 primitives covering 12 opcodes**

| Category | Opcodes | Status | Primitives |
|----------|---------|--------|------------|
| Arithmetic | ADD, SUB, MUL, DIV, MOD | ✅ Complete | ADD, SUB, MUL, DIV, MOD |
| Comparison | EQ, NE, LT, GT, LE, GE | ✅ Complete | CMP_EQ, CMP_NE, CMP_LT, CMP_GT, CMP_LE, CMP_GE |
| Bitwise | AND, OR, XOR | ✅ Complete | BIT_AND, BIT_OR, BIT_XOR |
| Shifts | SHL, SHR | ✅ Complete | SHL, SHR |

## Implementation Strategy

### Phase 1: Stack Manipulation Operations ✓ PRIORITY
These opcodes manipulate registers (SP, BP, PC, AX) and can be implemented with FFN primitives.

| Opcode # | Name | Semantics | Implementation | Units | Priority |
|----------|------|-----------|----------------|-------|----------|
| 0 | LEA | AX = BP + imm | Compose: ADD(BP, imm) | 4 | HIGH |
| 1 | IMM | AX = imm | Use CONST primitive | 2 | HIGH |
| 7 | ADJ | SP += imm | Compose: ADD(SP, imm) | 4 | HIGH |
| 40 | POP | SP += 8 | Compose: ADD(SP, CONST(8)) | 4 | MEDIUM |

**New Primitives Needed:**
1. `LOAD_IMM` - Load immediate value to register
2. `ADD_IMM` - Add immediate to register (for LEA, ADJ)
3. `REG_INC` - Increment register by constant

### Phase 2: Control Flow Operations ✓ PRIORITY
Branch and jump operations that modify PC based on conditions.

| Opcode # | Name | Semantics | Implementation | Units | Priority |
|----------|------|-----------|----------------|-------|----------|
| 2 | JMP | PC = imm | New: JUMP primitive | 2 | HIGH |
| 4 | BZ | if AX==0: PC = imm | Compose: CMP_EQ(AX,0) + SELECT(cond, imm, PC) | 6 | HIGH |
| 5 | BNZ | if AX!=0: PC = imm | Compose: CMP_NE(AX,0) + SELECT(cond, imm, PC) | 6 | HIGH |
| 41 | BLT | if AX<0: PC = imm | Compose: CMP_LT(AX,0) + SELECT(cond, imm, PC) | 6 | MEDIUM |
| 42 | BGE | if AX>=0: PC = imm | Compose: CMP_GE(AX,0) + SELECT(cond, imm, PC) | 6 | MEDIUM |

**New Primitives Needed:**
1. `BRANCH` - Conditional jump: out_PC = cond ? target_PC : current_PC + 1
2. `JUMP_IMM` - Unconditional jump to immediate value

### Phase 3: Function Call/Return Operations 🔄 COMPLEX
Stack frame management for function calls.

| Opcode # | Name | Semantics | Implementation | Units | Priority |
|----------|------|-----------|----------------|-------|----------|
| 3 | JSR | push PC, PC = imm | Needs STACK_PUSH(PC) + JUMP | 6-8 | MEDIUM |
| 6 | ENT | push BP, BP=SP, SP-=imm | Needs STACK_PUSH(BP) + REG_MOVE + ADD | 8 | MEDIUM |
| 8 | LEV | SP=BP, pop BP, pop PC | Needs REG_MOVE + STACK_POP × 2 | 8 | MEDIUM |
| 13 | PSH | push AX | Needs STACK_PUSH(AX) | 4 | MEDIUM |

**New Primitives Needed:**
1. `STACK_PUSH` - Decrement SP, write value to memory[SP]
2. `STACK_POP` - Read value from memory[SP], increment SP
3. `MULTI_REG_OP` - Operations that modify multiple registers at once

**Challenge:** These require **memory writes** which use attention, not pure FFN. Options:
- **Option A**: Implement as composite operation that outputs (new_SP, new_BP, value_to_write)
- **Option B**: Create "memory write request" output that attention layer processes
- **Option C**: Mark as "requires attention integration"

### Phase 4: Memory Operations ⚠️ ATTENTION-BASED
These **cannot** be implemented with pure FFN - they require attention for content-addressable lookup.

| Opcode # | Name | Semantics | Why Attention Needed |
|----------|------|-----------|---------------------|
| 9 | LI | AX = *AX | Memory read via KV cache attention |
| 10 | LC | AX = *(char*)AX | Memory read via KV cache attention |
| 11 | SI | *pop = AX | Memory write via KV cache attention |
| 12 | SC | *(char*)pop = AX | Memory write via KV cache attention |

**Implementation:** Already handled in `neural_vm/` via `MemoryReadAttention` and `MemoryWriteAttention`.

**Graph Compiler Role:** Can generate "memory request" outputs:
- `MEM_READ_REQUEST(address) → (address, read_flag)`
- `MEM_WRITE_REQUEST(address, value) → (address, value, write_flag)`

### Phase 5: I/O Operations ⚠️ ATTENTION-BASED
Character I/O requires cascaded ALiBi extraction for 32-bit positions.

| Opcode # | Name | Semantics | Why Attention Needed |
|----------|------|-----------|---------------------|
| 64 | GETCHAR | AX = getchar() | 8-layer cascaded nibble extraction + attention |
| 65 | PUTCHAR | putchar(AX) | Same as GETCHAR |
| 33 | PRTF | printf(fmt,...) | Character loop + formatting |

**Implementation:** Already handled in `neural_vm/io/` via cascaded ALiBi layers.

**Graph Compiler Role:** Can generate I/O flags:
- `IO_INPUT_REQUEST → set input_needed_flag`
- `IO_OUTPUT_REQUEST(char) → (char_nibbles, output_ready_flag)`

### Phase 6: System Operations 🔧 SPECIAL HANDLING

| Opcode # | Name | Semantics | Implementation |
|----------|------|-----------|----------------|
| 34 | MALC | AX = malloc(n) | Bump allocator: HEAP_PTR += n, return old HEAP_PTR |
| 35 | FREE | free(ptr) | Zero overwrite via attention |
| 36 | MSET | memset(p,v,n) | Loop: write v to [p..p+n) |
| 37 | MCMP | memcmp(a,b,n) | Loop: compare [a..a+n) with [b..b+n) |
| 38 | EXIT | exit(code) | Set IO_PROGRAM_END flag |
| 30 | OPEN | open(file) | Tool call or return -1 |
| 31 | READ | read(fd,buf,n) | Tool call or stdin via attention |
| 32 | CLOS | close(fd) | Tool call or no-op |
| 39 | NOP | no operation | Identity (zero delta) |

**Implementation Strategy:**
- **MALC**: Pure FFN - just ADD(HEAP_PTR, n) and return old value
- **EXIT**: Pure FFN - set flag in embedding
- **NOP**: Trivial - identity operation
- **MSET/MCMP**: Hybrid - FFN generates loop parameters, attention executes
- **FREE/OPEN/READ/CLOS**: External to compiler (tool calls or attention-based)

### Phase 7: Utility Operations

| Opcode # | Name | Semantics | Implementation |
|----------|------|-----------|----------------|
| 39 | NOP | no operation | Identity (already works) |
| 40 | POP | SP += 8 | Compose from ADJ(8) |
| 66 | PRINTF2 | printf variant | Unused (skip) |

## Parameter Budget

### Current Primitives (24 ops): 12,392 non-zero parameters

| Category | Operations | Non-Zero Params |
|----------|-----------|-----------------|
| Scalar | 16 | 296 |
| One-Hot | 3 | 4,416 |
| Bitwise | 5 | 7,680 |

### Estimated New Primitives (15 ops): ~500 additional parameters

| Category | Operations | Est. Non-Zero |
|----------|-----------|---------------|
| Stack (LEA, IMM, ADJ, POP) | 4 | ~80 |
| Control (JMP, BZ, BNZ, BLT, BGE) | 5 | ~200 |
| Function (JSR, ENT, LEV, PSH) | 4 | ~160 |
| System (MALC, EXIT, NOP) | 3 | ~60 |

**Total after expansion: ~12,892 non-zero parameters** (still >98% sparse)

## Implementation Phases

### ✅ Phase 1: Core Primitives (COMPLETE)
- [x] Arithmetic: ADD, SUB, MUL, DIV, MOD
- [x] Comparisons: CMP_EQ, CMP_NE, CMP_LT, CMP_GT, CMP_LE, CMP_GE
- [x] Bitwise: BIT_AND, BIT_OR, BIT_XOR, SHL, SHR
- [x] Conditional: SELECT, IF_THEN
- [x] Register: MOVE, CLEAR, CONST

### 🔄 Phase 2: Stack & Control Flow (IN PROGRESS)
- [ ] LOAD_IMM - Load immediate to register
- [ ] ADD_IMM - Add immediate to register
- [ ] JUMP - Unconditional jump to target
- [ ] BRANCH - Conditional branch (cond ? target : PC+1)
- [ ] Test with stack manipulation programs

### 🔜 Phase 3: Function Calls (NEXT)
- [ ] STACK_PUSH - Push value to stack
- [ ] STACK_POP - Pop value from stack
- [ ] Design memory request interface
- [ ] Integrate with attention layers

### ⏳ Phase 4: Integration & Testing
- [ ] Create opcode → primitive mapping
- [ ] Write tests for all 46 opcodes
- [ ] Integration with full neural VM
- [ ] Performance benchmarking

## Technical Challenges

### 1. Memory Access
**Problem:** Stack operations (PSH, POP, ENT, LEV) need memory writes.
**Solution:** Generate "memory request" outputs that attention processes:
```python
# FFN outputs memory write request
mem_write_addr = SP  # Stack pointer
mem_write_value = AX  # Value to push
mem_write_flag = 1.0  # Signal to attention layer
```

### 2. Multiple Register Updates
**Problem:** Operations like ENT modify 3 registers (BP, SP, both values).
**Solution:** Use separate output dimensions for each register:
```python
# Output has dimensions for: AX_new, SP_new, BP_new, PC_new
output[REG_AX] = ...
output[REG_SP] = SP - imm
output[REG_BP] = SP
```

### 3. Conditional PC Update
**Problem:** Branches need to conditionally update PC.
**Solution:** Use SELECT primitive:
```python
# BZ implementation
is_zero = CMP_EQ(AX, 0)
new_PC = SELECT(is_zero, target, PC + 1)
```

### 4. Scope Boundaries
**Problem:** Some operations fundamentally need attention (LI, LC, SI, SC).
**Solution:**
- Graph compiler generates **request signals**
- Attention layers in neural_vm/ **execute the requests**
- Clean separation of concerns

## Opcode → Primitive Mapping

### Direct Mapping (12 opcodes)
```
ADD (25) → ADD
SUB (26) → SUB
MUL (27) → MUL
DIV (28) → DIV
MOD (29) → MOD
EQ  (17) → CMP_EQ
NE  (18) → CMP_NE
LT  (19) → CMP_LT
GT  (20) → CMP_GT
LE  (21) → CMP_LE
GE  (22) → CMP_GE
AND (16) → BIT_AND
OR  (14) → BIT_OR
XOR (15) → BIT_XOR
SHL (23) → SHL
SHR (24) → SHR
```

### Composite Mapping (9 opcodes)
```
LEA (0)  → ADD(BP, CONST(imm))
IMM (1)  → MOVE(CONST(imm), AX)
ADJ (7)  → ADD(SP, CONST(imm))
POP (40) → ADD(SP, CONST(8))
BZ  (4)  → SELECT(CMP_EQ(AX, 0), imm, PC+1)
BNZ (5)  → SELECT(CMP_NE(AX, 0), imm, PC+1)
BLT (41) → SELECT(CMP_LT(AX, 0), imm, PC+1)
BGE (42) → SELECT(CMP_GE(AX, 0), imm, PC+1)
NOP (39) → MOVE(input, output)  # Identity
```

### New Primitives (6 opcodes)
```
JMP (2)  → JUMP(imm)           # New primitive
JSR (3)  → JSR(imm)            # New primitive
ENT (6)  → ENTER(imm)          # New primitive
LEV (8)  → LEAVE()             # New primitive
PSH (13) → PUSH(AX)            # New primitive
MALC(34) → MALLOC(n)           # New primitive (bump allocator)
```

### Attention-Based (11 opcodes - outside compiler)
```
LI    (9)  → MemoryReadAttention(AX)
LC    (10) → MemoryReadAttention(AX, byte=True)
SI    (11) → MemoryWriteAttention(pop, AX)
SC    (12) → MemoryWriteAttention(pop, AX, byte=True)
GETCHAR(64)→ CascadedALiBiInput()
PUTCHAR(65)→ CascadedALiBiOutput(AX)
READ  (31) → InputStreamAttention(fd, buf, n)
OPEN  (30) → ToolCallRequest("open", filename)
CLOS  (32) → ToolCallRequest("close", fd)
PRTF  (33) → CharacterLoopOutput(fmt, args)
EXIT  (38) → SetFlag(IO_PROGRAM_END, code)
```

## Testing Strategy

### Unit Tests (per primitive)
```python
def test_load_imm():
    result = LOAD_IMM(42)
    assert result == 42

def test_branch_taken():
    result = BRANCH(cond=1, target=100, fallthrough=10)
    assert result == 100

def test_branch_not_taken():
    result = BRANCH(cond=0, target=100, fallthrough=10)
    assert result == 10
```

### Integration Tests (per opcode)
```python
def test_opcode_lea():
    # LEA with BP=0x1000, imm=8
    result = compile_opcode(OpType.LEA, bp=0x1000, imm=8)
    assert result.ax == 0x1008

def test_opcode_bz_taken():
    # BZ with AX=0, target=0x2000
    result = compile_opcode(OpType.BZ, ax=0, target=0x2000, pc=0x1000)
    assert result.pc == 0x2000

def test_opcode_bz_not_taken():
    # BZ with AX=42, target=0x2000
    result = compile_opcode(OpType.BZ, ax=42, target=0x2000, pc=0x1000)
    assert result.pc == 0x1001  # PC + 1
```

### Full Program Tests (C programs)
```python
def test_full_program_factorial():
    # Compile: int factorial(int n) { ... }
    program = compile_c_to_graph("factorial.c")
    result = execute_graph(program, input=5)
    assert result == 120
```

## Success Criteria

1. ✅ All 12 computational opcodes implemented
2. 🔄 All 9 stack/control opcodes implemented
3. 🔜 All 6 function call opcodes implemented
4. ⏳ Memory/IO opcodes integrated with attention
5. ⏳ 100% opcode coverage in tests
6. ⏳ 1000+ C program test suite passing

## Next Steps

1. **Immediate:** Implement Phase 2 operations (LOAD_IMM, JUMP, BRANCH)
2. **Short-term:** Implement Phase 3 operations (PUSH, POP, JSR, ENT, LEV)
3. **Medium-term:** Design memory request interface for stack operations
4. **Long-term:** Integrate with full neural VM and run 1000+ test suite
