# Neural VM Opcode Table

Complete reference for the Neural VM instruction set with C4 compiler compatibility.

---

## C4 Registers

| Name          |           Size | Description                                           |                      Initial Value |
| ------------- | -------------: | ----------------------------------------------------- | ---------------------------------: |
| `PC`          |         32-bit | **Program Counter** — address of the next instruction |                                `0` |
| `AX`          |         32-bit | **Accumulator** — main working register for results   |                                `0` |
| `SP`          |         32-bit | **Stack Pointer** — top of stack                      |            `0x10000` (or `0x8000`) |
| `BP`          |         32-bit | **Base Pointer** — base of current stack frame        |            `0x10000` (or `0x8000`) |
| **Immediate** | 32-bit operand | Literal constant embedded in the instruction stream   | Read from code at `PC` when needed |

---

## C4 Opcodes

**L** = Layers, **W** = Non-Zero Weights

|                 # | Name    |   L |   W | Description                      |
| ----------------: | ------- | --: | --: | -------------------------------- |
| **Stack/Address** |         |     |     |                                  |
|                 0 | LEA     |   1 |  20 | AX = BP + imm                    |
|                 1 | IMM     |   0 |   0 | AX = imm                         |
|                 2 | JMP     |   1 |  12 | PC = imm                         |
|                 3 | JSR     |   2 |  30 | push PC, PC = imm                |
|                 4 | BZ      |   1 |  10 | if AX==0: PC = imm               |
|                 5 | BNZ     |   1 |  10 | if AX!=0: PC = imm               |
|                 6 | ENT     |   2 |  40 | push BP, BP=SP, SP-=imm          |
|                 7 | ADJ     |   1 |  16 | SP += imm                        |
|                 8 | LEV     |   2 |  40 | SP=BP, pop BP, pop PC            |
|        **Memory** |         |     |     |                                  |
|                 9 | LI      |   1 | 500 | AX = *AX (KV attention)          |
|                10 | LC      |   1 | 500 | AX = *(char*)AX                  |
|                11 | SI      |   1 | 500 | *pop = AX (KV write)             |
|                12 | SC      |   1 | 500 | *(char*)pop = AX                 |
|                13 | PSH     |   2 |  80 | push AX                          |
|       **Bitwise** |         |     |     |                                  |
|                14 | OR      |   1 |  80 | AX = pop \| AX                   |
|                15 | XOR     |   1 |  80 | AX = pop ^ AX                    |
|                16 | AND     |   1 |  80 | AX = pop & AX                    |
|    **Comparison** |         |     |     |                                  |
|                17 | EQ      |   1 |  80 | AX = (pop == AX)                 |
|                18 | NE      |   1 |  80 | AX = (pop != AX)                 |
|                19 | LT      |   2 | 100 | AX = (pop < AX)                  |
|                20 | GT      |   2 | 100 | AX = (pop > AX)                  |
|                21 | LE      |   2 | 110 | AX = (pop <= AX)                 |
|                22 | GE      |   2 | 110 | AX = (pop >= AX)                 |
|         **Shift** |         |     |     |                                  |
|                23 | SHL     |   2 | 160 | AX = pop << AX                   |
|                24 | SHR     |   2 | 160 | AX = pop >> AX                   |
|    **Arithmetic** |         |     |     |                                  |
|                25 | ADD     |   2 | 120 | AX = pop + AX                    |
|                26 | SUB     |   2 | 120 | AX = pop - AX                    |
|                27 | MUL     |   4 | 320 | AX = pop * AX                    |
|                28 | DIV     |   6 | 250 | AX = pop / AX                    |
|                29 | MOD     |   7 | 370 | AX = pop % AX                    |
|        **System** |         |     |     |                                  |
|                30 | OPEN    |   1 |  20 | AX = open(file) via tool call    |
|                31 | READ    |   9 | 250 | AX = read(fd,buf,n) via input KV |
|                32 | CLOS    |   1 |  10 | close(fd) → 0                    |
|                33 | PRTF    |   1 |  20 | printf(fmt,...)                  |
|                34 | MALC    |   1 |  51 | AX = malloc(n) bump alloc        |
|                35 | FREE    |   1 |  27 | free(ptr) zero overwrite         |
|                36 | MSET    |   1 |  16 | memset(p,v,n) loop               |
|                37 | MCMP    |   1 |  16 | memcmp(a,b,n) loop               |
|                38 | EXIT    |   1 |   9 | exit(code)                       |
|       **Control** |         |     |     |                                  |
|                39 | NOP     |   0 |   0 | no-op                            |
|                40 | POP     |   1 |  16 | SP += 8                          |
|                41 | BLT     |   1 |  17 | branch if signed <               |
|                42 | BGE     |   1 |  17 | branch if signed >=              |
|           **I/O** |         |     |     |                                  |
|                64 | GETCHAR |   9 | 220 | AX = getchar()                   |
|                65 | PUTCHAR |   9 | 220 | putchar(AX)                      |
|                66 | PRINTF2 |   1 |  20 | printf variant (unused)          |

**Note:** Opcodes 39-42 are neural VM extensions defined in `embedding.py` but not emitted by the compiler:
- **BLT (41)**: Direct signed branch if less than (combines LT + BZ into one instruction)
- **BGE (42)**: Direct signed branch if greater or equal (combines GE + BNZ into one instruction)
- **NOP (39)**: No operation (identity)
- **POP (40)**: Pop from stack (SP += 8)

---

## Summary by Category

| Category | Count | Max L | Weights |
|----------|------:|------:|--------:|
| Stack/Address | 9 | 2 | 178 |
| Memory | 5 | 2 | 2,080 |
| Bitwise | 3 | 1 | 240 |
| Comparison | 6 | 2 | 580 |
| Shift | 2 | 2 | 320 |
| Arithmetic | 5 | 7 | 1,180 |
| System | 9 | 9 | 419 |
| Control | 4 | 1 | 50 |
| I/O | 3 | 9 | 460 |
| **Total** | **46** | **9** | **5,507** |

**Note**: 42 opcodes emitted by compiler (0-38, 64-66), 4 neural VM extensions (39-42) defined but unused.

---

## Weight Sharing & Redundancy

| Sharing Type       | Reported | Unique | Factor | Examples                   |
|--------------------|----------|--------|--------|----------------------------|
| 8 nibbles parallel | 400      | 50     | 8x     | OR, XOR, AND, EQ, NE       |
| 8 cascade layers   | 930      | 150    | 6x     | GETCHAR, PUTCHAR, ADD, SUB |
| 6 iterations       | 250      | 50     | 5x     | DIV (Newton-Raphson)       |
| KV projections     | 2,000    | 400    | 5x     | LI, LC, SI, SC             |
| No sharing         | 397      | 397    | 1x     | Stack ops, EXIT, etc.      |

**Total: 5,487 reported → 1,397 unique weights**

| Counting Method            | Tensor Size | Non-Zero |
|----------------------------|-------------|----------|
| Naive (all ops separate)   | 153,232     | 5,487    |
| With nibble/layer sharing  | 153,232     | 1,397    |
| With full op-class sharing | 88,548      | ~800     |

**The entire Neural VM needs only ~800 unique non-zero weights.**

### Shared Modules

| Module | Weights | Description |
|--------|--------:|-------------|
| Memory KV (LI/LC/SI/SC) | 100 | Same Q/K/V projection |
| I/O Cascade | 35 | Same nibble extraction × 8 layers |
| Nibble Arith | 20 | Same sum+carry × 8 nibbles |
| Bitwise | 10 | One formula (a+b-ab) for all |
| Comparison | 40 | diff + cascade priority |
| Shift | 40 | Nibble routing |

The tensors are **96.4% sparse** — most entries are zero. The actual computation is controlled by ~800 carefully placed non-zero weights.

---

## Attention Architecture

### Purity Guarantee

**All computation uses ONLY:**
- `PureFFN.forward()`: SwiGLU → `x + W_down @ (silu(W_up @ x) * (W_gate @ x))`
- `PureAttention.forward()`: Standard attention → `x + W_o @ softmax(Q @ K^T) @ V`

No Python control flow, no `.sum()`, no `.clone()`. The `pure_alu.py` is an `nn.Sequential` with NO custom forward() method.

### Two Attention Contexts

| Context | Sequence Length | What are the positions? | Purpose |
|---------|----------------:|-------------------------|---------|
| **ALU (8 nibbles)** | 8 | Nibble 0-7 of a 32-bit value | Cross-nibble routing via flattened FFN |
| **KV Cache** | Variable | Memory writes, bytecode, I/O | Memory read, instruction fetch |

**ALU operations** (carry propagation, nibble gather) use **flattened FFN** over a fixed length-8 sequence. By flattening [batch, 8, dim] → [batch, 8*dim], linear weights can route data across nibble positions without attention.

**KV Cache attention** operates over the growing context (CODE tokens, DATA tokens, MEM_WRITE tokens). This is standard transformer KV cache attention for memory and bytecode lookup.

### Context Layout (Token Sequence)

```
[CODE_START] <inst₀> <inst₁> ... <instₙ> [CODE_END]
[DATA_START] <byte₀> <byte₁> ... [DATA_END]
[ARGV_START] <char₀> <char₁> ... [ARGV_END]
[REG: AX, SP, BP, PC]
[MEM_WRITE tokens...]
[OUTPUT tokens...]
[INPUT tokens...]
```

Each token is 256 floats:
- `[0]`: Token type (CODE=262, DATA=263, ARGV=264, OUTPUT=265, INPUT=266)
- `[1:33]`: 32-bit binary address encoding (+1/-1 per bit)
- `[33]`: Opcode or byte value
- `[34]`: Immediate value
- `[35]`: Sign flag

### Attention Head Types (KV Cache Operations)

| Head | Q Source | K Source | V Source | Output | Q/K Dim | Weights |
|------|----------|----------|----------|--------|--------:|--------:|
| **Instruction Fetch** | PC binary enc | CODE token addrs | (opcode, imm) | Current instr | 32 | 64 |
| **Memory Read** | MEM_ADDR nibbles | DATA/MEM_WRITE addrs | byte values | MEM_DATA | 16 | 64 |
| **Memory Write** | MEM_ADDR nibbles | position addresses | MEM_DATA | KV cache append | 16 | 64 |
| **Register Load** | Register ID | REG token types | register values | embedding slot | 8 | 32 |
| **User Input** | input index binary | INPUT token positions | char values | IO_CHAR slot | 32 | 64 |
| **Argv Fetch** | argv index binary | ARGV_CHAR positions | char values | IO_CHAR slot | 32 | 64 |
| **Output Position** | output count | OUTPUT token count | - | relative pos | 16 | 32 |

### ALU Cross-Nibble Operations (All via Flattened FFN)

**Key insight:** Cross-nibble routing within the 8-nibble ALU is done via **flattened FFN**, NOT attention. By flattening `[batch, 8, dim]` → `[batch, 8*dim]`, linear weights route data across positions in a single layer.

| Operation | FFN Class | Function |
|-----------|-----------|----------|
| Carry propagation | `CarryPropagateFFN` | Routes CARRY_OUT[i-1] → CARRY_IN[i] |
| Compare reduce (EQ) | `CompareReduceEqFFNNew` | Sum TEMP from 8 positions → RAW_SUM[0] |
| Compare reduce (NE) | `CompareReduceNeFFNNew` | Sum TEMP from 8 positions → RAW_SUM[0] |
| Branch broadcast | `BranchConditionFFN` | Copy TEMP[0] → RAW_SUM at all positions |
| Cmp broadcast | `CmpBroadcastResultFFN` | Copy TEMP[7] → RESULT[0] |
| BZ reduce | `BzReduceFFN` | Sum zero flags across nibbles |
| BNZ reduce | `BnzReduceFFN` | Sum non-zero flags across nibbles |
| MCMP reduce | `McmpReduceFFN` | Sum memcmp diffs across nibbles |

### Sequential Dependencies

```
Layer 0: Position Encoding (ALiBi)
         └─ Writes nibble position to E.POS

Layer 1: Instruction Fetch (Attention)
         ├─ Q = binary(PC), K = CODE addrs, V = (op, imm)
         └─ Writes opcode to OP_START + opcode

Layer 1: Operand Setup (parallel with fetch)
         └─ NibbleGatherFFN: flattened FFN assembles 32-bit values
            └─ Flatten [batch, 8, dim] → [batch, 8*dim]
            └─ Weights read NIB_A/B from each position, scale by 16^k
            └─ Sum to A_FULL/B_FULL slots (all 8 nibbles in ONE layer)

Layer 2+: Arithmetic/Logic (depends on operands)
         ├─ ADD/SUB: RawSum FFN → CarryPropagateFFN (flattened)
         ├─ MUL: Partial products → accumulation → carry propagation
         ├─ DIV: Newton-Raphson iterations (FFN)
         └─ Bitwise/Compare: Single FFN layer
```

**ADD/SUB (2 layers per iteration, 10 iterations):**
```
Per iteration:
  FFN: CarryPropagateFFN (flattened)
       └─ Flatten [batch, 8, dim] → [batch, 8*dim]
       └─ Linear weights route CARRY_OUT[i-1] → CARRY_IN[i]
       └─ All 7 carries propagate in ONE layer
  FFN stages: ZeroFirst → ClearOut → Iterate → ClearIn
             └─ Detect new carries, update RESULT

Total: 50 layers (10 iterations × 5 layers each)
Effective: Most iterations are no-ops after carries stabilize
```

**MUL (4 layers + carry propagation):**
```
Layer 1-8: Schoolbook partial products
           └─ 64 products: a[i] × b[j] for all i,j where i+j < 8
Layer 9+: Carry propagation (same as ADD/SUB)
```

**DIV/MOD (64+ layers via div32_ops.py):**
```
Phase 1: Clear + Gather (3 layers)
         └─ ClearDivSlots: clear TEMP, TEMP+1, RESULT at position 0
         └─ GatherA: sum(NIB_A[i] × 16^i) → TEMP[0] (dividend)
         └─ GatherB: sum(NIB_B[i] × 16^i) → TEMP+1[0] (divisor)
Phase 2: Iterative Division (64 layers)
         └─ DivIterFlatFFN: if TEMP >= TEMP+1 then TEMP -= TEMP+1, RESULT += 1
         └─ Uses step functions: silu(S*(dividend - divisor + 1)) - silu(S*(dividend - divisor))
Phase 3: Copy + Clear + Scatter (3 layers)
         └─ Copy RESULT (quotient) or TEMP (remainder) to CARRY_OUT[0]
         └─ Clear RESULT at all positions
         └─ Scatter: extract nibbles via mod 16 step functions
```

**I/O (9 layers):**
```
Layer 1: Set IO_NEED_INPUT or IO_OUTPUT_READY flag
Layer 2-9: Cascaded ALiBi bit extraction
         └─ Layer k extracts bit (8-k) via slope = 1/2^(8-k)
         └─ 8 bits sufficient for 256-char buffer index
Layer 9: Final attention fetches char at computed position
```

### Flattened FFN for Cross-Position Routing

**Key Insight:** Both carry propagation and nibble gather use **flattened FFN** (not attention). By flattening [batch, 8, dim] → [batch, 8*dim], linear weights can route data across nibble positions in a single layer.

**CarryPropagateFFN:**
```python
# Flatten: [batch, 8, dim] → [batch, 8*dim]
# Weights route CARRY_OUT[i-1] → CARRY_IN[i]
# Position 0: no carry in (LSB)
# Positions 1-7: copy from previous position's CARRY_OUT
x_flat = x.reshape(B, 8 * dim)
delta = W_down @ (W_up @ x_flat)  # Linear copy
return (x_flat + delta).reshape(B, 8, dim)
```

**NibbleGatherFFN:**
```python
# Flatten: [batch, 8, dim] → [batch, 8*dim]
# Weights read NIB_A from position k, scale by 16^k, sum to A_FULL
# All 8 nibbles gathered in ONE layer (no 8-head attention needed)
x_flat = x.reshape(B, 8 * dim)
# W_up[k] reads NIB_A at position k with scale 16^k
# W_down sums all to A_FULL at all positions
delta = W_down @ (W_up @ x_flat)
return (x_flat + delta).reshape(B, 8, dim)
```

Carry propagation is iterated 10 times to handle cascading carries (conservative; 8 would suffice).

### Binary Position Encoding

All address-based lookups use **binary encoding** for exact matching:

```python
# Encode address as ±1 vector
def encode(addr: int, bits: int = 32) -> Tensor:
    return [(2 * ((addr >> k) & 1) - 1) for k in range(bits)]
    # addr=42 → [+1,-1,+1,-1,+1,-1,0,0,...] (binary 101010)

# Dot product gives exact match score
score = Q · K  # = bits when all match, < bits otherwise
```

**Why it works:**
- Matching bit: `(+1)*(+1) = +1` or `(-1)*(-1) = +1`
- Mismatched bit: `(+1)*(-1) = -1`
- Perfect match: score = 32 (all bits agree)
- Any mismatch: score < 32

### Attention for Each Operation

**Instruction Fetch:**
```
Q = binary_encode(PC)           # 32-dim query
K = [binary_encode(inst.addr) for inst in CODE]  # 32-dim keys
V = [(inst.opcode, inst.immediate)]              # values
score = softmax(Q·K / 0.1)      # sharp temperature
output = weighted_sum(V, score) # (opcode, imm)
```

**Memory Read (LI/LC):**
```
Q = binary_encode(AX)           # address from accumulator
K = [binary_encode(write.addr) for write in MEM_WRITES] + DATA_addrs
V = [write.value, ...] + DATA_values
# Later writes shadow earlier (position bias)
score = softmax(Q·K + position * 0.01)
output = weighted_sum(V, score)
```

**User Input (GETCHAR):**
```
# Uses cascaded ALiBi to extract binary input position
# 32 layers, each extracts one bit via slope = 1/2^k
Layer k: slope = 1/2^(31-k)
         score = -slope * distance_to_input_start
         bit[31-k] = threshold(attention_pattern)

After 32 layers: have binary offset
Final: attend to INPUT token at that offset
```

### Memory Attention (softmax1)

Memory operations use **softmax1** for sparse reads:

```
softmax1(x) = exp(x) / (1 + sum(exp(x)))
```

- Address not found → all scores low → softmax1 → ~0 output
- This enables **ZFOD** (Zero Fill On Demand): unwritten memory reads as 0
- Zero writes are not stored; reading them returns 0 via softmax1

**Eviction Policy (Address-Based):**
| Range | Type | Priority |
|-------|------|----------|
| 0xFFFF0000+ | I/O registers | Never evict |
| 0xFFFE0000-0xFFFF0000 | VM registers | Never evict |
| 0x00000000-0x0000FFFF | Code segment | Low priority |
| Other | Data/heap | Normal eviction |

### Attention Parameter Counts

| Head Type | Heads | Q/K Params | V/O Params | **Total** |
|-----------|------:|-----------:|-----------:|----------:|
| Instruction Fetch | 1 | 32×2=64 | 2+64=66 | **130** |
| Memory Read | 1 | 16×2=32 | 8+8=16 | **48** |
| Memory Write | 1 | 16×2=32 | 8+8=16 | **48** |
| Register Load | 4 | 8×2×4=64 | 8×4=32 | **96** |
| User Input | 1 | 32×2=64 | 8+8=16 | **80** |
| Argv Fetch | 1 | 32×2=64 | 8+8=16 | **80** |
| Output Position | 1 | 16×2=32 | 8+8=16 | **48** |
| Nibble Gather | 8 | 4×2×8=64 | 4×8=32 | **96** |
| ALiBi Position | 1 | 0 (bias) | 8 | **8** |
| **Total** | **19** | **~448** | **~218** | **~634** |

**Note:** Carry propagation uses flattened FFN (~56 weights), not attention.

### Files

| File | Contents |
|------|----------|
| `token_sequence_vm.py` | Instruction fetch, memory read/write via standard attention |
| `neural_state.py` | VMState slots, MemoryReadAttention, MemoryWriteAttention |
| `position_encoding.py` | ALiBiPositionAttention for nibble position derivation |
| `cascaded_binary_io.py` | AlibiBitAttention for input position extraction |
| `optimized_ops.py` | MultiHeadGatherAttention for nibble assembly |
| `carry_ffn.py` | Carry propagation between nibbles |

---

## Layer Structure & Sequential Dependencies

### How PureALU is Built

The ALU is constructed as `nn.Sequential` with no custom `forward()` method:

```python
def build_pure_alu(num_carry_iters=10, num_div_iters=16) -> nn.Sequential:
    layers = []
    # Build layers...
    return nn.Sequential(*layers)  # No forward() - just stacked layers
```

Only `PureFFN.forward()` and `PureAttention.forward()` execute.

### Layer Baking

Each operation-specific FFN **bakes** its weights at init time:

```python
class AddRawSumFFN(PureFFN):
    def _bake_weights(self):  # Called once at init
        S = E.SCALE
        # Set specific weights for ADD raw sum computation
        self.W_up[0, E.NIB_A] = S
        self.W_gate[0, E.NIB_B] = S
        self.W_down[E.RAW_SUM, 0] = 1.0 / (S * S)
```

Weights are **frozen after baking** - no training, no gradient updates.

### SoftMoE Routing

Operations share layers via **Soft Mixture of Experts**:

```python
class SoftMoEFFN(nn.Module):
    """Routes inputs to operation-specific experts based on opcode one-hot."""

    def forward(self, x):
        # Each expert runs on the input
        outputs = [expert(x) for expert in self.experts]

        # Opcode one-hot determines which expert's output is used
        # x[:, :, E.OP_START + opcode] = 1.0 for active opcode
        weighted_outputs = sum(
            x[:, :, E.OP_START + op:E.OP_START + op + 1] * out
            for op, out in zip(self.opcodes, outputs)
        )
        return weighted_outputs
```

**Key insight:** ALL experts run, but only the active opcode's output is kept.

### Complete Layer Sequence

```
STAGE 1a (FFN): Raw computation
├─ ADD: AddRawSumFFN
├─ SUB: SubRawDiffFFN
├─ DIV: DivInitFFN
├─ MOD: ModInitFFN
├─ AND/OR/XOR: ClearBitSlotsFFN
├─ EQ/NE: CompareDiffFFN
├─ LT/GT/LE/GE: CmpRawDiffFFN
├─ JMP/BEQ/BNE/BLT/BGE: BranchFFN variants
├─ LOAD/STORE/PUSH/POP: MemoryFFN variants
├─ GETCHAR/PUTCHAR/EXIT: I/O FFNs
├─ LEA/IMM/BZ/BNZ/ENT/ADJ: Stack frame ops
└─ OPEN/READ/CLOS/PRTF: File I/O (tool calls)

STAGE 1b (FFN): Init/second step
├─ ADD: InitResultFFN
├─ SUB: SubInitResultFFN
├─ AND/OR/XOR: ExtractBit3FFN
├─ EQ/NE: CompareEqNibbleFFN
└─ LT/GT/LE/GE: CmpBorrowDetectFFN

STAGE 1c (FFN): Third step
├─ ADD: CarryDetectFFN
├─ SUB: BorrowDetectFFN
├─ AND/OR/XOR: ExtractBit2FFN
└─ EQ/NE: ClearRawSumFFN

STAGE 1d-1f (FFN): Bitwise extraction
├─ ExtractBit1FFN
├─ ExtractBit0FFN
└─ BitwiseAndCombineFFN / OrCombineFFN / XorCombineFFN

SCHOOLBOOK MUL (multiple layers)
└─ Partial products + accumulation

CARRY PROPAGATION (×10 iterations)
├─ SoftMoEFFN: CarryPropagateFFN (flattened, routes across 8 positions)
├─ SoftMoEFFN: ZeroFirstCarryFFN
├─ SoftMoEFFN: ClearCarryOutFFN
├─ SoftMoEFFN: CarryIterFFN
└─ SoftMoEFFN: ClearCarryInFFN

DIV ITERATIONS (×16)
└─ DivIterFFN

MOD ITERATIONS (×16)
└─ ModIterFFN

VARIABLE SHIFTS
├─ ShiftLeftLayers
└─ ShiftRightLayers

ATTENTION STAGES
├─ CompareReduceEqAttention
├─ CompareReduceNeAttention
├─ BranchConditionAttention
├─ CmpBroadcastResultAttention
└─ BzReduceAttention / BnzReduceAttention / McmpReduceAttention

FINALIZATION (FFN)
├─ ClearBitsFFN (bitwise cleanup)
├─ MulClearTempFFN
├─ ModResultFFN
├─ CompareReduceEqFFN / NeFFN
├─ BranchCopyTargetFFN / ClearTempFFN
├─ BzBranchFFN / BnzBranchFFN
└─ CmpInvertResultFFN (LE/GE)
```

### Why This Structure?

**Sequential dependencies dictate layer order:**

1. **Raw sum/diff must come before carry detect**
   - Can't detect overflow until we compute the nibble sum

2. **Carry propagate must come after carry detect**
   - Need to know which nibbles overflow before propagating

3. **Carry iterations must be sequential**
   - Nibble 1 needs nibble 0's carry, nibble 2 needs nibble 1's carry, etc.
   - BUT: Using flattened attention, ALL carries propagate in ONE layer

4. **Division iterations are independent**
   - Each Newton-Raphson step only depends on previous step
   - 16 sequential iterations (could be reduced with better initial guess)

5. **Comparison broadcast must come after borrow propagate**
   - MSB borrow determines LT/GT result
   - Must propagate borrows first

### Iteration Counts

| Operation | Iterations | Reason |
|-----------|------------|--------|
| Carry/Borrow | 10 | Conservative; 8 would suffice for 8 nibbles |
| DIV | 16 | Newton-Raphson for 32-bit precision |
| MOD | 16 | Uses DIV result, same iterations |
| Shift | 5 | 5 bits can shift 0-31 positions |

### Layer Count Summary

| Stage | Layers | Type |
|-------|-------:|------|
| Stage 1a-1f | 6 | FFN |
| Schoolbook MUL | ~8 | FFN |
| Carry propagation | 50 | 10 iter × (1 attn + 4 FFN) |
| DIV iterations | 16 | FFN |
| MOD iterations | 16 | FFN |
| Variable shifts | ~10 | FFN |
| Attention stages | ~8 | Attention |
| Finalization | ~4 | FFN |
| **Total** | **~120** | Mixed |

**Note:** Most layers are no-ops for most opcodes (SoftMoE routing). Effective depth depends on operation.

---

## Per-Instruction Execution (Token Generation)

### Execution Model

Each VM instruction is executed via **one forward pass** through the ~120 layers. The embedding tensor flows through ALL layers, with SoftMoE routing ensuring only the relevant expert's output is kept.

```
Input Embedding: [batch, 8, 160]  # 8 nibble positions × 160 dim
    ↓
┌─────────────────────────────────────────┐
│ Layer 1 (SoftMoEFFN): Stage 1a          │
│   - Identity experts run for inactive   │
│   - Only active opcode's expert computes│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 2-6 (SoftMoEFFN): Stages 1b-1f    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 7-14 (FFN): Schoolbook MUL        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 15-64 (Attn+FFN): Carry ×10       │
│   - 5 layers per iteration              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 65-80 (FFN): DIV iterations ×16   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 81-96 (FFN): MOD iterations ×16   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 97-106 (FFN): Variable shifts     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 107-114 (Attn): Reductions        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Layer 115-120 (FFN): Finalization       │
└─────────────────────────────────────────┘
    ↓
Output Embedding: [batch, 8, 160]  # Results in RESULT slots
```

### What Happens in Each Layer

**Layer enters with:**
```python
x: [batch, 8, dim]  # 8 nibble positions
# x[:, :, E.NIB_A]  = operand A nibbles
# x[:, :, E.NIB_B]  = operand B nibbles
# x[:, :, E.OP_START + opcode] = 1.0  # Active opcode
# x[:, :, E.RESULT] = result nibbles (accumulating)
# x[:, :, VMState.PC_BASE:PC_BASE+8] = PC nibbles
# ...
```

**Layer exits with:**
```python
# Same shape, but specific slots modified
# E.g., for ADD at carry layer:
#   x[:, :, E.CARRY_IN] updated from prev nibble's CARRY_OUT
#   x[:, :, E.RESULT] updated with carried sum
```

### Embedding Slot Usage Per Operation

| Slot | ADD/SUB | MUL | DIV | CMP | Bitwise | Memory |
|------|:-------:|:---:|:---:|:---:|:-------:|:------:|
| NIB_A | operand A | a | dividend | a | a | address |
| NIB_B | operand B | b | divisor | b | b | value |
| RAW_SUM | a+b | partial prod | quotient | a-b | - | - |
| CARRY_IN | from prev | from prev | - | borrow | - | - |
| CARRY_OUT | to next | to next | - | borrow | - | - |
| RESULT | sum | product | quotient | 0/1 | result | loaded |
| TEMP | - | accumulate | remainder | msb_borrow | bits | - |

### Control Flow Between Layers

**Opcode gating:** Each SoftMoE layer outputs:
```python
output = sum(
    x[:, :, E.OP_START + op] * expert_output
    for op, expert_output in zip(opcodes, expert_outputs)
)
```

Only the active opcode (where `x[:, :, E.OP_START + op] ≈ 1.0`) contributes.

**Identity behavior:** Operations not relevant to current layer see identity:
- FFN: `x + 0 = x` (weights output zero for inactive opcodes)
- Attention: Attends to self only (identity mask)

### Output Token Generation

For I/O operations (PUTCHAR, GETCHAR, EXIT), the VM sets flags in embedding:

```
PUTCHAR:
  1. Layer writes character nibbles to IO_CHAR slots
  2. Layer sets IO_OUTPUT_READY = 1.0
  3. External handler reads IO_CHAR, generates output token

GETCHAR:
  1. Layer sets IO_NEED_INPUT = 1.0
  2. External handler waits for input token
  3. Handler writes char to IO_CHAR, sets IO_INPUT_READY = 1.0
  4. Next forward pass reads input

EXIT:
  1. Layer sets IO_PROGRAM_END = 1.0
  2. Layer writes exit code to IO_EXIT_CODE
  3. External handler terminates execution
```

### Timing: One Forward Pass = One Instruction

| Phase | Layers | What Happens |
|-------|-------:|--------------|
| Fetch | 0 | PC → instruction (via inter-token attention to CODE) |
| Decode | 1 | Opcode one-hot written to embedding |
| Execute | 2-119 | ALU computation via SoftMoE layers |
| Writeback | 120 | Results in RESULT slot, PC incremented |

**Total: ~120 layers per instruction, executed as one forward pass.**

This differs from traditional VMs where execution is iterative. Here, all operations execute simultaneously, with the opcode one-hot selecting the correct result.

---

## Optimization Opportunities

### Multi-Head Gather (2-Layer Comparison)

The current implementation uses sequential nibble-by-nibble operations (40+ layers for comparison).
An optimized approach uses **multi-head attention** to gather all nibbles in parallel:

| Current | Optimized | Improvement |
|---------|-----------|-------------|
| 40 layers | 2 layers | 20x fewer |

**How it works:**
1. **Layer 1 (8-head attention)**: Each head reads one nibble with scale 16^k, all sum to same slot
   - Head 0: nibble[0] × 1
   - Head 1: nibble[1] × 16
   - Head 2: nibble[2] × 256
   - ... Head 7: nibble[7] × 16^7
   - Output: Full 32-bit value in single slot

2. **Layer 2 (FFN)**: Compute (a - b) and step(result) → comparison result

This applies to all comparison ops (EQ, NE, LT, GT, LE, GE) and can also optimize ADD/SUB.

**Implementation:** `optimized_ops.py` - `MultiHeadGatherAttention` + `OptimizedCompareFFN`.

---

## V7 Key Features

### Dynamic Memory (Bump Allocator)
- `HEAP_BASE`, `HEAP_PTR`, `HEAP_END` embedding slots
- Malloc: Bump HEAP_PTR, return old value
- Free: Write zeros (softmax1 ignores zeros = ZFOD)
- All I/O buffers dynamically allocated

### I/O Modes
| Mode | Protocol | Use Case |
|------|----------|----------|
| Native | stdio | Direct terminal |
| Streaming | `<NEED_INPUT/>`, `<PROGRAM_END/>` | LLM generation |
| Tool-Use | `TOOL_CALL:type:id:{params}` | Agentic execution |

### Non-Tool-Use Behavior
- `getchar()`/`putchar()`: Use streaming I/O
- `open()`/`read()`/`close()`: Return -1
- Only stdin/stdout supported without tool-use

---

## ONNX Export & Runtime

### ONNX Export (PyTorch → ONNX)

Export the PureALU to ONNX format:
```bash
python -m neural_vm.onnx_export --verify --compare
```

**Export Configuration:**
- Opset version: 18 (required by PyTorch's new exporter)
- File size: ~9 MB (with external data file)
- Input: `[batch, 8, 160]` embedding
- Output: `[batch, 8, 160]` with results in `RESULT` slot

**ONNX Test Results (March 2026):**

| Category | Tests | Passed | Notes |
|----------|------:|-------:|-------|
| ADD | 5 | 5 | 0+0, 255+1, 65535+1, 12345+67890 |
| SUB | 3 | 3 | 10-3, 100-100, 1000-1 |
| MUL | 3 | 3 | 6×7, 255×255, 100×100 |
| DIV | 4 | 4 | 6/2, 42/6, 100/10, 1000/33 ✓ |
| MOD | 4 | 4 | 10%3, 42%6, 1000%33, 127%4 ✓ |
| AND/OR/XOR | 3 | 3 | All bitwise ops work |
| EQ/NE | 4 | 4 | Equal and not-equal |
| LT/GT/LE/GE | 4 | 4 | All comparisons work |
| SHL/SHR | 2 | 2 | Shift operations |
| **Total** | **32** | **32** | **100% pass rate** |

**DIV/MOD Implementation (March 2026 update):**
- Flattened 32-bit gather/divide/scatter pipeline (`div32_ops.py`)
- Gathers 8 nibbles into scalar value at position 0
- Iterative subtraction-based division (64 iterations)
- Scatters result back to nibble positions
- File size: ~11 MB (additional layers for DIV/MOD pipeline)

### C Runtime (`onnx_runner_c4.c`)

The C runtime implements neural arithmetic:

| Operation | Method | Status |
|-----------|--------|--------|
| Multiply | SwiGLU: `silu(a)*b + silu(-a)*(-b)` | OK |
| Add | Nibble table + carry | OK |
| Divide | Newton-Raphson reciprocal | OK |
| Softmax | exp table + division | OK |

### Running Open-Source LLMs

To run models like Llama/GPT-2:

1. **Export to ONNX** (PyTorch → ONNX)
2. **Compile to VM instructions**:
   - MatMul → loop with MUL + ADD
   - Softmax → exp table + DIV
   - LayerNorm → mean/var via ADD/MUL
3. **Execute** with neural arithmetic

All ONNX ops reduce to our 46 opcodes via subroutines.

---

## Architecture Evolution

| Version | File | Dim | Layers | Non-Zero Weights | Key Innovation |
|---------|------|-----|--------|------------------|----------------|
| V1 | pure_gen_vm.py | 1600 | 4 | ~2B | Dense baked experts |
| V2 | pure_gen_vm_v2.py | 2048 | 40 | ~9.5B | Sparse lookup + unrolled DIV |
| V3 | pure_gen_vm_v3.py | 1024 | 7 | ~100K | Value-encoded + subroutine triggers |
| V4 | pure_gen_vm_v4.py | 896 | 8 | ~20K | Difference-based equality testing |
| V5 | pure_gen_vm_v5.py | 512 | 8 | **~1,336** | **SiLU arithmetic formulas** |

**V5 achieves 74× weight reduction vs V3 with 215/215 tests passing.**

---

## Equality Testing Approaches

### V1-V3: One-Hot Lookup (SwiGLU Pattern)

```
Input:  One-hot encoded (a[nib] at position p → dim[p] = 1.0)
Method: Address decoding via sparse weights

For checking if a[nib] == 5 AND b[nib] == 3 AND opcode == ADD:
  gate.weight[row, a_dim_5] = SCALE           # Fires when a=5
  up.weight[row, b_dim_3] = SCALE             # Fires when b=3
  up.weight[row, opcode_ADD] = SCALE          # Fires when opcode=ADD
  up.bias[row] = -SCALE * 1.95                # Threshold: need both b AND opcode

Activation:
  Both b+opcode match:  up = 2*SCALE - 1.95*SCALE = 0.05*SCALE → silu > 0 ✓
  Only one matches:     up = SCALE - 1.95*SCALE = -0.95*SCALE → silu ≈ 0 ✗
```

### V4: Difference-Based (Your Suggestion)

```
Input:  Value-encoded (a_val = nibble_value * SCALE, scalar)
Method: Compute diff, use sigmoid/silu windows

For checking if a == b (equality):
  diff = a - b                                           # Linear projection
  up = SCALE * (diff + threshold)                        # Positive when diff > -T
  gate = SCALE * (threshold - diff)                      # Positive when diff < +T
  eq_hidden = silu(up) * gate                            # Large only when |diff| < T

For checking if a < b:
  lt_hidden = silu(SCALE * (-diff - threshold)) * SCALE  # Fires when diff < -T

For 32-bit comparison (cascaded MSB-first):
  eq_cascade = 1.0
  for nib in [7, 6, 5, 4, 3, 2, 1, 0]:                   # MSB to LSB
    lt_final += eq_cascade * lt_nib[nib]                 # Only if higher nibs equal
    eq_cascade *= eq_nib[nib]                            # This nib must also equal
```

**Comparison:**
| Aspect | One-Hot Lookup | Difference-Based |
|--------|----------------|------------------|
| Input encoding | 16 dims/nibble | 1 dim/nibble |
| Non-zero weights | ~6,144/op | ~300/op |
| 32-bit compare | Per-nibble lookup | Cascaded MSB-first |
| Best for | Arbitrary lookups | Arithmetic comparisons |

---

## Opcode Tables by Version

### V1 (dim=1600, Dense Experts)

V1 uses **dense** baked weights. MUL/DIV/SHL/SHR use Python loops.

| Opcode | c4 | Layers | Non-Zero | Notes |
|--------|-----|--------|----------|-------|
| ADD | Yes | 1 | 30,720,000 | Dense expert |
| SUB | Yes | 1 | 30,720,000 | Dense expert |
| MUL | Yes | 1×N | 30,720,000 | + Python nibble loop |
| DIV | Yes | 1×N | 30,720,000 | + Python bit loop |
| MOD | Yes | 1×N | 30,720,000 | Uses DIV |
| AND | Yes | 1 | 30,720,000 | Dense expert |
| OR | Yes | 1 | 30,720,000 | Dense expert |
| XOR | Yes | 1 | 30,720,000 | Dense expert |
| NOT | Yes | 1 | 30,720,000 | Dense expert |
| SHL | Yes | 1×N | 30,720,000 | + Python bit loop |
| SHR | Yes | 1×N | 30,720,000 | + Python bit loop |
| EQ/NE/LT/GT/LE/GE | Yes | 1 | 30,720,000 | Dense expert |
| SI/LI/SC/LC | Yes | 1 | 10,240,000 | Attention |
| PSH/POP | Yes | 1 | 10,240,000 | Attention |

**V1 Totals:** 4 layers, ~2B non-zero params

---

### V2 (dim=2048, Sparse Lookup Tables)

V2 uses **sparse** baked weights. All ops fully neural with unrolled loops.

| Opcode | c4 | Layers | Non-Zero | Notes |
|--------|-----|--------|----------|-------|
| ADD | Yes | 1 | 6,144 | Sparse lookup |
| SUB | Yes | 1 | 6,144 | Sparse lookup |
| MUL | Yes | 4 | 50,000 | 64 parallel products |
| DIV | Yes | 32 | 19,400 | Weight-tied iterations |
| MOD | Yes | 32 | 19,400 | Uses DIV result |
| AND/OR/XOR | Yes | 1 | 6,144 | Sparse lookup |
| NOT | Yes | 1 | 3,072 | Sparse lookup |
| SHL/SHR | Yes | 1 | 134,480 | 32 shifts parallel |
| EQ/NE/LT/GT/LE/GE | Yes | 1 | 6,144 | Sparse lookup |
| SI/LI/SC/LC | Yes | 1 | 16,777,216 | Attention |
| PSH/POP | Yes | 1 | 16,777,216 | Attention |
| MSET/MCMP/MCPY | No | 1 | 2,057 | Subroutine trigger |

**V2 Totals:** 40 effective layers, ~300K non-zero FFN + ~100M attention

---

### V3 (dim=1024, Hybrid Value-Encoded)

V3 uses **hybrid encoding** + **subroutine triggers** for DIV/MOD.

**Key innovations:**
- SCALE=20, THRESHOLD=1.95 for sharp equality testing
- Skip LayerNorm for predictable one-hot values
- Value-encoded outputs (8 floats vs 128-dim one-hot)
- All basic ops verified working ✓

| Opcode | c4 | Layers | Non-Zero | Notes |
|--------|-----|--------|----------|-------|
| ADD | Yes | 1 | 10,240 | Value-encoded (verified ✓) |
| SUB | Yes | 1 | 10,240 | Value-encoded (verified ✓) |
| MUL | Yes | 2 | 65,536 | 64 parallel products |
| DIV | Yes | 1 | 80 | **Subroutine trigger** |
| MOD | Yes | 1 | 80 | **Subroutine trigger** |
| AND/OR/XOR | Yes | 1 | 10,240 | Value-encoded (verified ✓) |
| NOT | Yes | 1 | 5,120 | Value-encoded |
| SHL/SHR | Yes | 1 | 67,000 | Value-encoded |
| EQ/NE/LT/GT/LE/GE | Yes | 1 | 6,144 | Sparse lookup |
| SI/LI/SC/LC | Yes | 1 | 4,194,304 | Attention |
| PSH/POP | Yes | 1 | 4,194,304 | Attention |
| MSET/MCMP/MCPY | No | 1 | 80 | Subroutine trigger |
| Value→OneHot | - | 1 | 1,664 | Final conversion |

**V3 Totals:** 7 layers, ~230K non-zero FFN + ~25M attention

**V3 Weight breakdown (BasicOps):**
```
Per entry (nibble × a_val × b_val × opcode):
  gate: 1 weight (checks a[nib])
  up: 2 weights (checks b[nib] + opcode)
  up.bias: 1 bias (threshold = -SCALE × 1.95)
  down: 1 weight (value-encoded result)
  Total: 5 weights/entry × 2048 entries = 10,240 per operation
```

---

### V4 (dim=896, Difference-Based Comparisons)

V4 uses **difference-based equality testing** for comparisons.

**Key innovations:**
- diff = a - b computed as scalar per nibble
- SwiGLU window: `silu(SCALE*(diff+T)) * (SCALE*(T-diff))` fires when |diff| < T
- Cascaded MSB-first: process nibbles 7→0, only contribute when higher nibs equal
- Clamp LT/GT to non-negative to prevent equal-nibble interference

| Opcode | c4 | Layers | Non-Zero | Notes |
|--------|-----|--------|----------|-------|
| EQ | Yes | 1 | ~300 | Diff + SwiGLU window (verified ✓) |
| NE | Yes | 1 | ~300 | 1 - EQ |
| LT | Yes | 1 | ~400 | Cascaded MSB-first (verified ✓) |
| GT | Yes | 1 | ~400 | Cascaded MSB-first (verified ✓) |
| LE | Yes | 1 | ~400 | LT OR EQ |
| GE | Yes | 1 | ~400 | GT OR EQ |
| ADD | Yes | 2 | 10,240 | Nibble-wise + carry prop |
| SUB | Yes | 2 | 10,240 | Nibble-wise + borrow prop |

**V4 Totals:** 8 layers, ~150K non-zero FFN + ~20M attention

**V4 Comparison implementation:**
```python
# Step 1: Compute per-nibble differences
diff = a_val - b_val  # Linear projection: 16 inputs → 8 outputs

# Step 2: Per-nibble equality (SwiGLU window for |diff| < threshold)
eq_up = self.eq_up(diff)      # SCALE * (diff + T)
eq_gate = self.eq_gate(diff)  # SCALE * (T - diff)
eq_nib = silu(eq_up) * eq_gate  # Large when |diff| < T

# Step 3: Per-nibble LT/GT (fires when clearly less/greater)
lt_nib = clamp(silu(-SCALE*diff - SCALE*T) * SCALE, min=0)
gt_nib = clamp(silu(SCALE*diff - SCALE*T) * SCALE, min=0)

# Step 4: Cascade from MSB (proper 32-bit comparison)
eq_cascade = 1.0
for nib in [7, 6, 5, 4, 3, 2, 1, 0]:
    lt_final += eq_cascade * lt_nib[nib] / SCALE
    gt_final += eq_cascade * gt_nib[nib] / SCALE
    eq_cascade *= eq_nib_norm[nib]  # Clamp and normalize
eq_final = eq_cascade  # Product of all nibble equalities
```

---

## Layer Types and Sparsity

| Layer Type | Total Params | Non-Zero | Sparsity |
|------------|--------------|----------|----------|
| MulProductsLayer (32 products) | 50M | 25,632 | 99.9% |
| DivIterationLayer | 75M | 19,402 | 99.97% |
| ShiftParallelLayer | 403M | 134,480 | 99.97% |
| SubroutineTriggerLayer | 400K | 80 | 99.98% |
| ValueEncodedBasicOps (V3) | 25M | 10,240 | 99.96% |
| DifferenceComparisonLayer (V4) | 5M | ~300 | 99.99% |
| **EfficientAddSubLayer (V5)** | 16K | **64** | **99.6%** |
| **EfficientMulProductsLayer (V5)** | 65K | **384** | **99.4%** |
| **EfficientBitwiseLayer (V5)** | - | **224** | forward() |
| **EfficientShiftLayer (V5)** | 32K | **32** | **99.9%** |
| **EfficientComparisonLayer (V5)** | 32K | **376** | **98.8%** |
| **ControlFlowLayer (V5)** | 12K | **88** | **99.3%** |
| **StackFrameLayer (V5)** | 8K | **64** | **99.2%** |

---

## Memory Operations (Attention-Based)

Memory operations use softmax1 attention for content-addressable lookup:

```
softmax1(x) = exp(x) / (1 + sum(exp(x)))
```

This allows "attend to nothing" when address not found (cache miss).

| Dim | Q/K/V Projections | Output Projection | Total Attention |
|-----|-------------------|-------------------|-----------------|
| 512 (V5) | 0.8M | 0.3M | **1.1M** |
| 896 (V4) | 2.4M | 0.8M | **3.2M** |
| 1024 (V3) | 3.1M | 1.0M | **4.2M** |
| 1600 (V1) | 7.7M | 2.6M | **10.2M** |
| 2048 (V2) | 12.6M | 4.2M | **16.8M** |

---

## Encoding Summary

| Encoding | Dims/Value | Range | Used For |
|----------|------------|-------|----------|
| One-hot nibble | 16 | 0-15 | V1-V3 input address decoding |
| Value-encoded (scaled) | 1 | 0-1 | V3 intermediate results |
| Difference | 1 | -1 to +1 | V4 comparison operations |
| **Raw value** | 1 | 0-15 | **V5 direct arithmetic** |

**V4 flow:** One-hot input → Value-encoded diff → Cascaded comparison → Scalar output

**V5 flow:** Value input → SiLU arithmetic → Value output → OneHot conversion

---

## V5: SiLU Arithmetic Formulas

V5 achieves **~200-300× parameter reduction** by computing arithmetic directly with SiLU rather than lookup tables.

### V5 Addition Formula

```
SiLU(SCALE*(a+b)) / SCALE ≈ a + b
```

For large SCALE (e.g., 20):
- SiLU(x) ≈ x when x >> 0
- SiLU(SCALE*(a+b)) ≈ SCALE*(a+b)
- Result: SCALE*(a+b) / SCALE = a+b ✓

**Weight layout (4 weights per nibble):**
```
up.weight[row, a_dim] = SCALE
up.weight[row, b_dim] = SCALE
gate.weight[row, opcode_ADD] = SCALE
down.weight[out, row] = 1/SCALE²
```

### V5 Multiplication Formula

```
(SiLU(SCALE*a) + SiLU(-SCALE*a)) * b / SCALE ≈ a * b
```

Key insight:
- SiLU(SCALE*a) ≈ SCALE*a for a > 0
- SiLU(-SCALE*a) ≈ 0 for a > 0
- Sum ≈ SCALE*a
- Result: SCALE*a * b / SCALE = a*b ✓

**Weight layout (6 weights per product):**
```
Row 1: up.weight = +SCALE*a, gate.weight = b, down.weight = 1/SCALE
Row 2: up.weight = -SCALE*a, gate.weight = b, down.weight = 1/SCALE
```

### V5 Subtraction

```
SiLU(SCALE*(a-b)) / SCALE ≈ a - b  (for a >= b)
```

For borrow cases (a < b), add offset of 16 and detect borrow separately.

---

### V5 (dim=512, SiLU Arithmetic) - COMPREHENSIVE

V5 uses **direct SiLU arithmetic formulas** instead of lookup tables.

**Key innovations:**
- Value-encoded inputs AND outputs (no one-hot lookups)
- Addition: SiLU(SCALE*(a+b))/SCALE ≈ a+b (4 weights/nibble)
- Multiplication: (SiLU(+SCALE*a) + SiLU(-SCALE*a))*b/SCALE ≈ a*b (6 weights/product)
- Bitwise: Bit decomposition + boolean formulas (AND=a*b, OR=a+b-ab, XOR=a+b-2ab)
- Shifts: Power of 2 multiplication/division
- Control flow: SiLU zero detection for BZ/BNZ
- Stack frame: SiLU arithmetic for SP/BP operations

| Opcode | c4 | Layers | Non-Zero | Formula/Notes |
|--------|-----|--------|----------|---------------|
| ADD | Yes | 1 | **64** | SiLU(SCALE*(a+b))/SCALE² + carry prop ✓ |
| SUB | Yes | 1 | **64** | SiLU(SCALE*(a-b))/SCALE² + borrow prop ✓ |
| MUL | Yes | 1 | **384** | (SiLU(±SCALE*a))*b/SCALE, 64 products ✓ |
| DIV | Yes | 1 | **16** | Reciprocal table + forward compute ✓ |
| MOD | Yes | 1 | **16** | Uses DIV remainder ✓ |
| AND | Yes | 1 | **~75** | Bit decomposition: bit_i(a)*bit_i(b) ✓ |
| OR | Yes | 1 | **~75** | Bit decomposition: a+b-ab per bit ✓ |
| XOR | Yes | 1 | **~75** | Bit decomposition: a+b-2ab per bit ✓ |
| SHL | Yes | 1 | **32** | A * 2^B (power lookup table) ✓ |
| SHR | Yes | 1 | **32** | A / 2^B (integer division) ✓ |
| EQ | Yes | 1 | **~63** | SiLU(diff+ε)-2*SiLU(diff)+SiLU(diff-ε) ✓ |
| NE | Yes | 1 | **~63** | 1 - EQ ✓ |
| LT | Yes | 1 | **~63** | Cascaded MSB-first SiLU(-diff) ✓ |
| GT | Yes | 1 | **~63** | Cascaded MSB-first SiLU(+diff) ✓ |
| LE | Yes | 1 | **~63** | LT OR EQ ✓ |
| GE | Yes | 1 | **~63** | GT OR EQ ✓ |
| JMP | Yes | 1 | **~20** | PC = immediate ✓ |
| JSR | Yes | 1 | **~20** | Push PC+8, PC = immediate ✓ |
| BZ | Yes | 1 | **88** | SiLU zero detection on AX ✓ |
| BNZ | Yes | 1 | **88** | Inverse of BZ ✓ |
| ENT | Yes | 1 | **~20** | BP=SP, SP-=imm (SiLU arithmetic) ✓ |
| ADJ | Yes | 1 | **~20** | SP+=imm (SiLU arithmetic) ✓ |
| LEV | Yes | 1 | **~20** | SP=BP (register copy) ✓ |

**V5 Weight Summary:**
```
Category              | Weights | Notes
──────────────────────|─────────|──────────────────────
Arithmetic (ADD/SUB)  |      64 | 8 nibbles × 8 weights
Multiplication        |     384 | 64 products × 6 weights
Division/Modulo       |      16 | Reciprocal table
Comparisons (6 ops)   |     376 | Cascaded MSB-first
Zero Detection        |      88 | BZ/BNZ layer
Bitwise (AND/OR/XOR)  |     224 | Bit decomposition
Shifts (SHL/SHR)      |      32 | Power of 2 table
Control Flow          |      88 | JMP/BZ/BNZ
Stack Frame           |      64 | ENT/ADJ/LEV
──────────────────────|─────────|──────────────────────
TOTAL                 |   1,336 | ~74× reduction vs V3
```

**Comprehensive Test Results (215/215 PASS):**
```
ADD (21 tests):  0+0, 5+3=8, 15+15=0x1E, 0xFF+1=0x100, 0xFFFFFFFF+1=0 ✓
SUB (17 tests):  0-0, 5-3=2, 0-1=0xFFFFFFFF, 0x100-1=0xFF ✓
MUL (23 tests):  0*0, 5*3=15, 15*15=225, all squares 2²-14² ✓
DIV (16 tests):  0/1, 100/7=14, 255/3=85, 0xFFFF/0xFF=0x101 ✓
MOD (16 tests):  5%3=2, 100%7=2, 255%3=0, 0xFFFF%0x100=0xFF ✓
EQ  (11 tests):  0==0, 5!=3, 0x12345678==0x12345678, max==max ✓
NE  (4 tests):   0!=0=F, 5!=3=T ✓
LT  (10 tests):  0<1, 3<5, 0x1234<0x1235, MSB comparison ✓
GT  (9 tests):   1>0, 5>3, 0x1235>0x1234 ✓
LE  (6 tests):   0<=0, 0<=1, 5<=5 ✓
GE  (6 tests):   0>=0, 1>=0, 5>=5 ✓
AND (10 tests):  0&0, FF&55=55, mask operations ✓
OR  (8 tests):   0|0, AA|55=FF, combine operations ✓
XOR (8 tests):   0^0, FF^FF=0, self^self=0 ✓
SHL (10 tests):  1<<4=16, 0xFFFFFFFF<<1=0xFFFFFFFE ✓
SHR (10 tests):  16>>4=1, 0xFFFFFFFF>>1=0x7FFFFFFF ✓
BZ  (11 tests):  0=zero, 1≠zero, all nibble positions ✓
JMP (3 tests):   Unconditional jump ✓
BZ  (4 tests):   Branch if AX==0 ✓
BNZ (3 tests):   Branch if AX!=0 ✓
ADJ (4 tests):   SP adjustment with carry ✓
ENT (3 tests):   Function entry (BP=SP, SP-=n) ✓
LEV (2 tests):   Function leave (SP=BP) ✓
```

**Comparison to Previous Versions:**
| Operation | V3 (Lookup) | V5 (SiLU) | Reduction |
|-----------|-------------|-----------|-----------|
| ADD/SUB | 20,480 | 64 | **320×** |
| MUL | 65,536 | 384 | **170×** |
| DIV/MOD | 160 | 16 | **10×** |
| CMP (all 6) | 6,144 | 376 | **16×** |
| Bitwise | 30,720 | 224 | **137×** |
| Shifts | 67,000 | 32 | **2,094×** |
| Control | ~500 | 152 | **3×** |
| **TOTAL** | ~100,000 | **1,336** | **~74×** |

---

## 32-bit I/O with 4-bit Cascade

Each I/O operation (GETCHAR/PUTCHAR) uses **8 cascaded layers** to extract a 32-bit position, one nibble per layer:

```
Layer 0: nibble[7] = offset / 2^28  (268M threshold)
Layer 1: nibble[6] = remaining / 2^24
...
Layer 7: nibble[0] = remaining
```

Per layer: 16 branch scores + 4 extraction bits + 1 remainder update = **21 weights**.
8 layers = **168 weights** for full 32-bit extraction.

### Multi-Bit Cascade Options

| Bits/Layer | Branches | Layers | Weights | Use Case |
|------------|----------|--------|---------|----------|
| 1 | 2 | 32 | 224 | Maximum error tolerance |
| 2 | 4 | 16 | 320 | Balanced |
| **4** | **16** | **8** | **168** | **Optimal (nibble-aligned)** |
| 8 | 256 | 4 | 1024 | Very wide, 256-way softmax |

### Binary Key Matching (32-bit)

```
Query: offset → 32 binary bits (from 8 nibbles)
Score = sum of matching bits:
  Exact match: score = 32 (maximum)
  Hamming distance d: score = 32 - 2d
Softmax with temperature 0.1: exact match >99.9999% confidence
```

---

## Version History

| Version | I/O Bits | Weights | Max Layers | Key Innovation |
|---------|----------|---------|------------|----------------|
| V5 | 8 | ~1,336 | ~8 | SiLU arithmetic |
| V6 | 16 | ~8,917 | ~17 | 1-bit cascaded I/O |
| V7 | 16 | ~8,797 | ~17 | + Dynamic heap |
| V8 | 16 | ~5,100 | ~9 | 4-bit cascade, fixed comparison |
| **V9** | **32** | **~5,247** | **9** | **32-bit I/O, nibble cascade** |
