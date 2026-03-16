# Neural VM Opcode Table (16-bit I/O)

This document describes all opcodes with weight counts and layer requirements for 16-bit I/O support.

## Architecture Overview

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Word Size** | 32 bits | 8 nibbles |
| **I/O Offset** | 16 bits | 4 nibbles, cascaded extraction |
| **Embedding Dim** | 512 | VMState dimension |
| **Scale** | 100 | SiLU activation scale |
| **Position Bits** | 16 | For I/O buffer indexing |

---

## Cascaded Binary I/O (16-bit)

Each I/O operation uses **16 cascaded layers** to extract position bits:

```
Layer  0: bit[15] >= 32768? → subtract if yes
Layer  1: bit[14] >= 16384? → subtract if yes
...
Layer 15: bit[0]  >= 1?     → subtract if yes
```

### Per-Layer Weights

| Component | Weights | Notes |
|-----------|---------|-------|
| Threshold check | 3 | W_up, b_up, threshold |
| Bit extraction | 2 | sigmoid gate, output |
| Remaining update | 2 | subtract threshold |
| **Total per layer** | **7** | |
| **16 layers** | **112** | For full 16-bit extraction |

---

## Complete Opcode Table

### Stack/Address (0-8)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 0 | LEA | 1 | 20 | AX = BP + immediate |
| 1 | IMM | 0 | 0 | Embedding injection (no FFN) |
| 2 | JMP | 1 | 12 | PC = immediate |
| 3 | JSR | 2 | 30 | Push PC+8, PC = immediate |
| 4 | BZ | 1 | 10 | Branch if AX == 0 |
| 5 | BNZ | 1 | 10 | Branch if AX != 0 |
| 6 | ENT | 2 | 40 | BP=SP, SP -= immediate |
| 7 | ADJ | 1 | 16 | SP += immediate |
| 8 | LEV | 2 | 40 | SP=BP, pop BP, pop PC |

**Subtotal: 11 layers, 178 weights**

### Memory (9-13)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 9 | LI | 1 | 500 | Load int via KV attention |
| 10 | LC | 1 | 500 | Load char (byte) |
| 11 | SI | 1 | 500 | Store int via KV write |
| 12 | SC | 1 | 500 | Store char (byte) |
| 13 | PSH | 2 | 80 | Push AX to stack |

**Subtotal: 6 layers, 2,080 weights**

### Bitwise (14-16)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 14 | OR | 6 | 150 | Bit decomposition: a+b-ab per bit |
| 15 | XOR | 6 | 150 | Bit decomposition: a+b-2ab per bit |
| 16 | AND | 6 | 150 | Bit decomposition: a*b per bit |

**Subtotal: 18 layers, 450 weights**

### Comparison (17-22)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 17 | EQ | 4 | 200 | Diff + SiLU window per nibble |
| 18 | NE | 4 | 200 | 1 - EQ |
| 19 | LT | 10 | 400 | Cascaded MSB-first borrow chain |
| 20 | GT | 10 | 400 | Swap operands + LT |
| 21 | LE | 11 | 420 | LT OR EQ |
| 22 | GE | 11 | 420 | GT OR EQ |

**Subtotal: 50 layers, 2,040 weights**

### Shift (23-24)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 23 | SHL | 12 | 800 | Bit extract + conditional shift |
| 24 | SHR | 12 | 800 | Bit extract + conditional shift |

**Subtotal: 24 layers, 1,600 weights**

### Arithmetic (25-29)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 25 | ADD | 8 | 300 | RawSum + CarryChain (8 nibbles) |
| 26 | SUB | 8 | 300 | RawDiff + BorrowChain |
| 27 | MUL | 16 | 600 | Partial products + carry prop |
| 28 | DIV | 6 | 250 | Newton-Raphson (6 iterations) |
| 29 | MOD | 9 | 350 | DIV + MUL + SUB |

**Subtotal: 47 layers, 1,800 weights**

### System Calls (30-38)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 30 | OPEN | 1 | 15 | Set tool call type (file ops) |
| 31 | READ | 1 | 15 | Set tool call type |
| 32 | CLOS | 1 | 10 | Set tool call type |
| 33 | PRTF | 1 | 20 | Printf (deferred to handler) |
| 34 | MALC | 1 | 51 | Bump HEAP_PTR, return old |
| 35 | FREE | 1 | 27 | Write zeros (softmax1 eviction) |
| 36 | MSET | loop | 16 | Memset subroutine (uses SI) |
| 37 | MCMP | loop | 16 | Memcmp subroutine (uses LI) |
| 38 | EXIT | 1 | 9 | Set IO_PROGRAM_END |

**Subtotal: 8 layers, 179 weights** (excluding loop ops)

### Control (39-42)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 39 | NOP | 0 | 0 | Identity (no-op) |
| 40 | POP | 1 | 16 | SP += 4 |
| 41 | BLT | 2 | 17 | SetSignFlag + BranchLT |
| 42 | BGE | 2 | 17 | SetSignFlag + BranchGE |

**Subtotal: 5 layers, 50 weights**

### I/O with Cascaded Binary (64-66)

| # | Name | Layers | Weights | Neural Implementation |
|---|------|--------|---------|----------------------|
| 64 | GETCHAR | 5 | 180 | 4 nibble layers (4-bit cascade) + KV lookup |
| 65 | PUTCHAR | 5 | 180 | 4 nibble layers (4-bit cascade) + KV write |
| 66 | PRINTF2 | - | - | Not implemented |

**Subtotal: 10 layers, 360 weights** (using 4-bit cascade)

*Alternative 1-bit cascade: 34 layers, 480 weights*

---

## I/O Implementation Details

### GETCHAR with 4-bit Cascade (5 layers, 180 weights)

```
Layer 0:  Mark input read (2 weights)
Layer 1:  Extract nibble[3] = offset / 4096 (36 weights, 16 branches)
Layer 2:  Extract nibble[2] = remaining / 256 (36 weights)
Layer 3:  Extract nibble[1] = remaining / 16 (36 weights)
Layer 4:  Extract nibble[0] = remaining (36 weights)
Layer 5:  Query input KV with 16-bit binary key (34 weights)

Total: 2 + 4*36 + 34 = 180 weights
```

### PUTCHAR with 4-bit Cascade (5 layers, 180 weights)

```
Layer 0:  Mark output write (2 weights)
Layer 1-4: Extract 4 nibbles from offset (4*36 = 144 weights)
Layer 5:  Write to output KV with 16-bit binary key (34 weights)

Total: 180 weights
```

### Alternative: 1-bit Cascade (17 layers, 240 weights)

```
Layer 0:  Mark input read (2 weights)
Layer 1:  Extract bit[15] from offset (7 weights)
Layer 2:  Extract bit[14] from remaining (7 weights)
...
Layer 16: Extract bit[0] from remaining (7 weights)
Layer 17: Query input KV with 16-bit binary key (128 weights)

Total: 2 + 15*7 + 7 + 128 = 240 weights
```

### Input Flow (User → VM)

```
Sequence: [BOS] ... <USER_INPUT> H e l l o </USER_INPUT> ... [VM]
                    ↓            ↓ ↓ ↓ ↓ ↓  ↓
                    start_pos    writes to input KV cache
                                 key = cascaded 16-bit offset
                                 value = char

GETCHAR execution:
  1. Cascaded extraction: offset → 16 binary bits (16 layers)
  2. Query input KV with binary key
  3. Softmax attention selects exact position
  4. Return character value
```

### Output Flow (VM → User)

```
VM: putchar('H') → cascaded extraction → write to output KV
    putchar('i') → cascaded extraction → write to output KV
    ...

Sequence: ... <USER_OUTPUT> [generated tokens attend to output KV]
               ↓
               output_start_pos

Each output token:
  1. Compute offset = my_pos - output_start_pos
  2. Cascaded extraction → 16 binary bits
  3. Query output KV with binary key
  4. Attend and output character: </think>X<think>
```

---

## Summary by Category

| Category | Opcodes | Layers | Weights | % of Total |
|----------|---------|--------|---------|------------|
| Stack/Address | 9 | 11 | 178 | 2.0% |
| Memory | 5 | 6 | 2,080 | 23.6% |
| Bitwise | 3 | 18 | 450 | 5.1% |
| Comparison | 6 | 50 | 2,040 | 23.2% |
| Shift | 2 | 24 | 1,600 | 18.2% |
| Arithmetic | 5 | 47 | 1,800 | 20.4% |
| System Calls | 9 | 8 | 179 | 2.0% |
| Control | 4 | 5 | 50 | 0.6% |
| I/O (4-bit cascade) | 2 | 10 | 360 | 4.1% |
| **TOTAL** | **45** | **179** | **8,797** | **100%** |

*With 1-bit cascade: 203 layers, 8,917 weights*

---

## Multi-Bit Cascaded Extraction

Instead of extracting 1 bit per layer (binary decision), extract 2-4 bits per layer by enumerating all possibilities.

### Comparison Table

| Bits/Layer | Branches | Layers (16-bit) | Weights/Layer | Total Weights | Pros |
|------------|----------|-----------------|---------------|---------------|------|
| 1 | 2 | 16 | 7 | 112 | Simple, robust |
| 2 | 4 | 8 | 14 | 112 | Balanced depth/width |
| 3 | 8 | 6 | 28 | 168 | Fast extraction |
| **4** | **16** | **4** | **36** | **144** | **Matches nibble arithmetic** |

### Why 4 Bits Per Layer is Optimal

1. **Nibble alignment**: Neural arithmetic already operates on nibbles (8 per 32-bit word)
2. **Only 4 layers**: Minimal depth for I/O extraction
3. **16 branches = 1 nibble**: Each layer extracts one hex digit
4. **Direct division**: `remaining / base` gives nibble value directly

### 4-Bit Extraction (Nibble Mode)

```
Input: offset = 43721 (hex: AAC9)

Layer 0: 43721 / 4096 → nibble=A, remaining=2761
Layer 1:  2761 / 256  → nibble=A, remaining=201
Layer 2:   201 / 16   → nibble=C, remaining=9
Layer 3:     9 / 1    → nibble=9, remaining=0

Result: 0xAAC9 ✓ (4 layers total)
```

### I/O Opcode Layers with Multi-Bit Cascade

| Mode | Extract Layers | + KV Lookup | Total GETCHAR/PUTCHAR |
|------|----------------|-------------|----------------------|
| 1-bit cascade | 16 | 1 | 17 layers |
| 2-bit cascade | 8 | 1 | 9 layers |
| 4-bit cascade | 4 | 1 | **5 layers** |

---

## Cascaded vs Direct Binary Comparison

| Approach | Layers for 16-bit | Weights | Pros | Cons |
|----------|-------------------|---------|------|------|
| **Direct Binary** | 1 | ~256 | Single-layer extraction | All-or-nothing |
| **1-bit Cascaded** | 16 | ~112 | Natural depth fit, error correction | More layers |
| **4-bit Cascaded** | 4 | ~144 | Fast, nibble-aligned | More branches per layer |

### 1-Bit Cascaded Layer-by-Layer

```
Input: offset = 43721 (binary: 1010101011001001)

Layer  0: 43721 >= 32768? YES → bit[15]=1, rem=10953
Layer  1: 10953 >= 16384? NO  → bit[14]=0, rem=10953
Layer  2: 10953 >=  8192? YES → bit[13]=1, rem=2761
Layer  3:  2761 >=  4096? NO  → bit[12]=0, rem=2761
Layer  4:  2761 >=  2048? YES → bit[11]=1, rem=713
Layer  5:   713 >=  1024? NO  → bit[10]=0, rem=713
Layer  6:   713 >=   512? YES → bit[9]=1, rem=201
Layer  7:   201 >=   256? NO  → bit[8]=0, rem=201
Layer  8:   201 >=   128? YES → bit[7]=1, rem=73
Layer  9:    73 >=    64? YES → bit[6]=1, rem=9
Layer 10:     9 >=    32? NO  → bit[5]=0, rem=9
Layer 11:     9 >=    16? NO  → bit[4]=0, rem=9
Layer 12:     9 >=     8? YES → bit[3]=1, rem=1
Layer 13:     1 >=     4? NO  → bit[2]=0, rem=1
Layer 14:     1 >=     2? NO  → bit[1]=0, rem=1
Layer 15:     1 >=     1? YES → bit[0]=1, rem=0

Result: 1010101011001001 ✓
```

### 2-Bit Cascaded (4-way Branch per Layer)

```
Input: offset = 43721 (binary: 1010101011001001)

Layer 0: 43721 >= 2*16384? → bits[15:14]=10, rem=10953
Layer 1: 10953 >= 2*4096?  → bits[13:12]=10, rem=2761
Layer 2:  2761 >= 2*1024?  → bits[11:10]=10, rem=713
Layer 3:   713 >= 2*256?   → bits[9:8]=10, rem=201
Layer 4:   201 >= 3*64?    → bits[7:6]=11, rem=9
Layer 5:     9 >= 0*16?    → bits[5:4]=00, rem=9
Layer 6:     9 >= 2*4?     → bits[3:2]=10, rem=1
Layer 7:     1 >= 1*1?     → bits[1:0]=01, rem=0

Result: 1010101011001001 ✓ (8 layers total)
```

---

## Memory Model

### KV Cache Structure

| Head | Purpose | Key | Value |
|------|---------|-----|-------|
| INPUT_DATA | User input chars | 16-bit binary offset | char (8 bits) |
| OUTPUT_BUFFER | VM output chars | 16-bit binary offset | char (8 bits) |
| INPUT_MARKER | Start/end positions | fixed | position |
| PROGRAM_MEMORY | VM heap/stack | 32-bit address | 32-bit value |

### Binary Key Matching

```
Query: offset 42 → binary key [0,1,0,1,0,1,0,0,0,0,1,0,1,0,1,0]

Score = sum of matching bits:
  - Exact match: score = 16 (maximum)
  - 1 bit different: score = 14
  - Hamming distance d: score = 16 - 2d

Softmax with temperature 0.1:
  - Exact match dominates (exp(160) >> exp(140))
  - Selects correct entry with >99.99% confidence
```

---

## Weight Breakdown by Operation Type

### FFN Weights (SwiGLU pattern)

```python
# Per hidden unit:
W_up[h, input_dim]     # 1 weight
W_gate[h, input_dim]   # 1 weight
W_down[output_dim, h]  # 1 weight
b_up[h]                # 1 bias
b_gate[h]              # 1 bias
# Total: ~5 non-zero weights per hidden unit
```

### Attention Weights (KV cache)

```python
# Q/K/V projections for dim=512:
Q_proj: 512 x 512 = 262,144 (but sparse: ~500 non-zero)
K_proj: 512 x 512 = 262,144 (but sparse: ~500 non-zero)
V_proj: 512 x 512 = 262,144 (but sparse: ~500 non-zero)
# Total: ~1,500 non-zero for memory ops
```

---

## Version History

| Version | I/O Bits | Total Weights | Layers | Key Innovation |
|---------|----------|---------------|--------|----------------|
| V5 | 8 | ~1,336 | ~8 | SiLU arithmetic |
| V6 | 16 | ~8,917 | ~203 | 1-bit cascaded binary I/O |
| V7 | 16 | ~8,917 | ~203 | + Dynamic heap allocation |
| V8 | 16 | ~8,797 | ~179 | 4-bit cascade (nibble mode) |

---

## Implementation Notes

### Why 16-bit I/O?

- 65,536 character buffer (sufficient for most programs)
- With 4-bit cascade: only 4 extraction layers (fits in any transformer)
- Nibble-aligned matches neural arithmetic design
- Compatible with standard transformer architectures

### Multi-Bit Cascade Trade-offs

| Approach | Layers | Branches/Layer | Best For |
|----------|--------|----------------|----------|
| 1-bit | 16 | 2 | Deep transformers, error tolerance |
| 2-bit | 8 | 4 | Balanced depth/width |
| 4-bit | 4 | 16 | Shallow models, nibble arithmetic |

### Softmax1 for Memory

```
softmax1(x) = exp(x) / (1 + sum(exp(x)))

Key property: zero values are ignored
- Free memory = write zeros
- Zero entries can be evicted from KV cache
- ZFOD (Zero Fill On Demand) for free
```

### Think Tag Output

```
VM putchar('H') → "</think>H<think>"
VM putchar('i') → "</think>i<think>"

Final output (after filtering think tags): "Hi"
```
