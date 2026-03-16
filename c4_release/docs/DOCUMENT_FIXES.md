# Document Fixes and Additions

This file tracks remaining TODOs and technical notes for the c4vm documentation.

## Code Convention Note

This document contains both:
1. **Neural implementations** - Weight matrices (`W_up`, `W_down`, `W_q`, `W_k`, `W_v`) that define pure neural operations
2. **Semantic descriptions** - Python code showing WHAT an operation does (not HOW it's implemented)
3. **PyTorch module code** - The actual PyTorch classes that implement neural ops

**Key distinction:**
- `_bake_weights()` code shows how neural weights are configured - this IS the neural implementation
- Functions with Python for-loops over data are usually SEMANTIC (reference), not neural
- Bytecode (`.asm`) shows the neural execution path - each instruction is a neural forward pass

When in doubt, the actual neural implementation is in `neural_vm/*.py` files.

---

## 0.1 Argv Memory Layout

**C Code Accessing argv:**

```c
int main(int argc, char **argv) {
    // argc = 3
    // argv = 0x7F000 (pointer to array)
    // argv[0] = *(0x7F000) = 0x80000 → "program"
    // argv[1] = *(0x7F008) = 0x80008 → "arg1"
    // argv[2] = *(0x7F010) = 0x8000D → "arg2"
}
```

**Memory Writes Needed:**

1. **String storage** (at STRING_BASE):
   - Write each arg as bytes with null terminators
   - Total: sum of (strlen(arg) + 1) for each arg

2. **Pointer array** (at ARGV_BASE):
   - Write 8-byte little-endian pointers
   - Each points to corresponding string
   - Total: argc × 8 bytes

3. **Stack setup**:
   - Push argc (4 or 8 bytes depending on word size)
   - Push argv_base (pointer to pointer array)

```python
from neural_vm.io_handler import ArgvMemorySetup

setup = ArgvMemorySetup(["program", "arg1", "arg2"])
for addr, value, width in setup.get_memory_writes():
    memory.write(addr, value, width)  # width: 1=byte, 8=int

# Stack: push argc, push argv_base
for op, val in setup.get_stack_setup():
    stack.push(val)
```

**Test coverage**: 26 tests in `test_argv.py` covering both ArgvHandler and ArgvMemorySetup

---
---

## 0.3 ALiBi Binary Position Matching (Detailed)

**ALiBi (Attention with Linear Biases)** adds position-dependent biases to attention scores rather than encoding position in Q/K vectors. For the neural VM, ALiBi enables exact binary address matching for memory operations.

### Standard ALiBi

Standard ALiBi adds a bias proportional to distance:
$$\text{score}_{i,j} = Q_i \cdot K_j - m \cdot |i - j|$$

where $m$ is the slope (typically a geometric sequence: $m_h = \frac{1}{2^h}$ for head $h$).

### Binary ALiBi for Memory Addressing

For 32-bit address matching, we use **bit-specific slopes**:

$$\text{slope}_k = \frac{1}{2^k}$$

Each attention head extracts one bit of the relative position. For position $p$:
- Head 0 ($k=0$): slope = 1, extracts LSB (bit 0)
- Head 1 ($k=1$): slope = 0.5, extracts bit 1
- Head 2 ($k=2$): slope = 0.25, extracts bit 2
- Head $k$: slope = $2^{-k}$, extracts bit $k$

**Bit Extraction Mechanism:**

For query at position $q$ attending to key at position $k$:
$$\text{bias}_{\text{head}_i} = -\frac{|q - k|}{2^i}$$

At different relative distances:
- Distance 0: all biases = 0 (perfect match)
- Distance 1: head_0 bias = -1, others ≈ 0
- Distance 2: head_1 bias = -1, head_0 bias = -2
- Distance 4: head_2 bias = -1, lower heads ≈ -2 or -4

After softmax, the attention pattern reveals binary position:
- If head_$k$ attends strongly → bit $k$ is 0
- If head_$k$ attends weakly → bit $k$ is 1

### ALiBi for Input Buffer Indexing (GETCHAR)

For reading character $n$ from input buffer:

```
Cascaded ALiBi Attention (8 layers):
Layer 0: slope = 1/128   → extracts bit 7 (MSB)
Layer 1: slope = 1/64    → extracts bit 6
Layer 2: slope = 1/32    → extracts bit 5
...
Layer 7: slope = 1       → extracts bit 0 (LSB)

After 8 layers: binary offset [b7,b6,...,b0] assembled in embedding
Final attention: fetch INPUT token at that offset
```

## 0.4 Position Offset Extraction via Comparison Cascade

Given a scalar offset (e.g., a relative distance from ALiBi, a memory address in a
temp slot, or an I/O buffer pointer), we need to recover its binary or nibble
representation without division or logarithms. The approach is a sequential
comparison cascade that extracts the offset digit-by-digit using neural step functions.

### Bitwise Extraction (32 layers)

Process the offset one bit at a time, from MSB to LSB. Each layer applies a single
step function as a threshold comparison:

```
residual = offset
for b = 31 down to 0:
    bit_b = step(residual - 2^b)           # 1 if residual >= 2^b, else 0
    residual = residual - bit_b * 2^b      # subtract contribution
    output[b] = bit_b
```

Each layer uses one neural step function (2 hidden units in a SwiGLU FFN). The step
fires when the residual exceeds the threshold $2^b$, extracting that bit. Subtracting
the contribution leaves the residual for the next layer.

This is simple and robust — each layer only needs a binary decision — but requires
32 sequential layers for a 32-bit offset.

### Nibblewise Extraction (8 layers)

Extract 4 bits per layer by classifying the residual into one of 16 buckets at
scale $16^j$:

```
residual = offset
for j = 7 down to 0:
    d_j = Σ_{k=1}^{15} step(residual - k · 16^j)    # counts 0–15
    residual = residual - d_j · 16^j                   # subtract contribution
    output[j] = d_j
```

Each layer uses 15 neural step functions (30 hidden units). The sum counts how many
multiples of $16^j$ fit into the residual, giving the hexadecimal digit
$d_j \in \{0,\dots,15\}$. This is mathematically identical to the quotient-digit
computation in long division (`ComputeQuotientNibbleFFN`), with $16^j$ playing the
role of the divisor.

Both the digit computation and residual subtraction can be merged into a single layer
(same technique as `SubtractAndWriteQFFN` in division), since they compute the same
step functions with different output routing:

```
Units 0–29:  step(residual - k·16^j) gated by 1     → writes output[j]
Units 30–59: step(residual - k·16^j) gated by 16^j  → subtracts from residual
```

### Precision Concerns

**Error propagation.** If a high-order digit is extracted incorrectly, the residual
is wrong by at least $16^j$, and every subsequent digit inherits the error. Bitwise
extraction is more forgiving here — a single-bit error affects only $2^b$ — while
nibblewise extraction amplifies errors by up to $15 \cdot 16^j$.

**Dynamic range.** Higher-order comparisons involve thresholds up to $15 \cdot 16^7
\approx 4 \times 10^9$. The step function `silu(S · (residual - threshold))` requires
the product $S \cdot \text{threshold}$ to be representable. In float32 (mantissa ≈ 7
decimal digits), large thresholds force a smaller scale $S$, which widens the step
transition and can blur adjacent buckets. This is the same issue that motivates
`DIV_Q_SCALE` in the division implementation.

**16-way separation.** Each nibble layer must cleanly separate 16 cases. With step
functions of width $\sim 1/S$, adjacent thresholds at distance $16^j$ are well-separated
when $S \cdot 16^j \gg 1$. For the lowest nibble ($j=0$), thresholds are spaced by 1,
requiring $S \gg 1$. For the highest nibble ($j=7$), thresholds are spaced by $16^7$,
so even a small $S$ suffices. The hardest layer is always the least significant digit.

### Comparison with Parallel Approaches

The cascade is sequential (each layer depends on the previous residual). Two
alternatives extract bits in parallel:

**ALiBi multi-head (Section 0.3).** With slope $m_k = 1/2^k$ for head $k$, each
attention head is naturally sensitive to bit $k$ of the relative distance. All 32 bits
can be extracted in a single attention layer with 32 heads. However, this requires
attention (not pure FFN) and the bit values must be decoded from attention patterns
rather than read directly.

**RoPE binary frequencies.** With $\theta_k = 2^k$, the rotation
$\cos(2^k \cdot p)$ encodes bit $k$ of position $p$ directly in the Q/K vectors.
Dot-product matching gives score = $d_{\text{key}} - 2 \cdot \text{Hamming}(p, q)$,
peaking at exact position match. This is the cleanest route for position matching
since the binary structure is explicit in the encoding, but it solves a different
problem (position matching) rather than offset extraction.

### Implementation

The cascade is implemented in `cascaded_binary_io.py` (bitwise, 32 layers) and
`multibit_cascaded_io.py` (nibblewise, 8 layers). The nibblewise variant is
structurally identical to the remainder extraction in the MOD operation
(`ExtractRemainderNibbleFFN` + `SubtractHigherNibblesFFN` in `long_division_ops.py`).

---

## 7. Position Encoding Test Results

Position encoding methods for nibble positions (0-7):

| Method  | Max Error | Status |
|---------|----------:|:------:|
| ALiBi   |   0.00034 |  PASS  |
| RoPE    |  <0.00001 |  PASS  |
| Direct  |   0.00000 |  PASS  |

ALiBi and RoPE both derive position from attention patterns rather than
storing position directly in embedding slots, enabling pure positional
encoding without dedicated storage.

---

## 8. Division Implementation: Nibble-Wise Long Division

Division uses base-16 long division implemented entirely in `FlattenedPureFFN` layers
(file: `long_division_ops.py`). No attention is needed — all cross-position communication
happens through scalar slots at position 0.

### Algorithm

Standard long division in base 16. For each nibble from MSB (7) to LSB (0):

```
1. remainder = remainder * 16 + dividend_nibble[i]
2. q = floor(remainder / divisor)      # 0–15
3. remainder = remainder - q * divisor
4. quotient_nibble[i] = q
```

The key insight: `q = Σ_{k=1}^{15} step(remainder - k·divisor)`. Each neural step
function adds 1 when `remainder ≥ k·divisor`, so summing 15 steps counts how many
times the divisor fits (0 to 15).

### Slot Layout

All slots live at position 0 (scalar values accessed via FlattenedPureFFN):

| Slot | Embedding Index | Purpose |
|------|----------------|---------|
| SLOT_DIVIDEND | E.TEMP | Gathered dividend (read nibbles directly instead) |
| SLOT_DIVISOR | E.TEMP+1 | Gathered 32-bit divisor scalar |
| SLOT_REMAINDER | E.TEMP+2 | Running remainder |
| SLOT_QUOTIENT | E.TEMP+3 | (unused — quotient written directly to RESULT) |
| SLOT_CURR_Q | E.TEMP+4 | Current quotient nibble (0–15) |

### Layer Structure (26 total)

```
Setup (2 layers):
  ClearDivSlotsFFN          — zero all temp slots
  GatherScalarFFN           — gather 8 divisor nibbles into scalar: Σ nib[i]·16^i

Iteration (3 layers × 8 nibbles = 24 layers):
  ShiftRemainderAndClearQFFN  — remainder = remainder·16 + dividend_nib[i]; curr_q = 0
  ComputeQuotientNibbleFFN    — curr_q = Σ step(remainder - k·divisor) for k=1..15
  SubtractAndWriteQFFN        — remainder -= curr_q·divisor; RESULT[i] = curr_q
```

### Merged Layers

Two pairs of originally separate layers were merged because they write to independent slots:

- **ShiftRemainderAndClearQFFN** (6 hidden units): merges ShiftRemainderAddNibble
  (writes SLOT_REMAINDER) + ClearCurrQ (writes SLOT_CURR_Q)
- **SubtractAndWriteQFFN** (32 hidden units): merges SubtractQTimesDivisor
  (writes SLOT_REMAINDER) + WriteQuotientNibble (writes RESULT[i])

### DIV_Q_SCALE

`ComputeQuotientNibbleFFN` uses `E.DIV_Q_SCALE` (a lower scale than `E.DIV_SCALE`)
to avoid float32 precision issues. The step functions compute `silu(S·(remainder - k·divisor + 1))`,
and with large remainders (up to ~4 billion) and scale factors, the product can exceed
float32 range. The lower scale keeps intermediate values representable.

### MOD Extension

For MOD, after the 8 division iterations the remainder holds the correct value as a scalar
in SLOT_REMAINDER. Cascade extraction decomposes it to nibbles (8 layers):

```
For pos = 7 (MSB) down to 0 (LSB):
  ExtractAndSubtractRemainderFFN(pos)  — clear RESULT[pos], extract nibble via 15
                                         step functions, subtract nibble·16^pos from
                                         SLOT_REMAINDER (all in one 32-unit FFN)
```

MOD total: 26 (division) + 8 (extraction) = 34 layers.

### Comparison with Alternatives

| Algorithm       | Layers | Hidden Units | Precision | Notes |
|-----------------|-------:|-------------:|----------:|:------|
| Long Division   |     26 |         ~280 |    exact  | Current implementation |
| Newton-Raphson  |     12 |         ~216 |   ~32 bit | Q16 fixed-point, inexact for large values |
| SRT (radix-4)   |      8 |        ~300  |    4 bits | Quotient digit selection table |

Newton-Raphson (`newton_raphson_div.py`) was implemented but is **not used** — Q16
fixed-point cannot produce exact 32-bit quotients, and the final multiply would itself
require the full MUL pipeline.


### Carry Propagation (FFN via Flattening)

Carry propagation is done via FFN by flattening positions:

```
Input: [batch, 8 positions, dim]
Flatten: [batch, 8 * dim]
FFN: Weights connect position i's CARRY_OUT to position i+1's CARRY_IN
Reshape: [batch, 8, dim]

FFN weight structure (flattened):
  W_up reads: slot (i * dim + CARRY_OUT) for each i
  W_down writes: slot ((i+1) * dim + CARRY_IN) for each i

Result: CARRY_IN[i] = CARRY_OUT[i-1] for all i > 0
        CARRY_IN[0] = 0 (no carry into LSB)
```

This is pure FFN - no attention mechanism needed. The flattening allows
cross-position communication within a single FFN layer.


### Integration Status

**Current state:**
- `byte_alu.py` and `byte_ops.py` exist with complete byte-level implementations
- `pure_alu.py` still uses nibble-level operations (8 nibbles)
- Byte-level integration is a TODO for future optimization

**Why not integrated yet:**
- Nibble-level is simpler and well-tested (163 tests passing)
- Byte-level requires attention to gather nibbles into bytes
- Performance gain (3.6×) may not justify complexity for ONNX export

**All test programs use integers:**
- Mandelbrot uses int types (Q16.16 fixed-point represented as int)
- No floating-point operations in the neural VM
- All arithmetic is 32-bit integer (signed/unsigned via reinterpretation)

---

## 14. Mandelbrot Fixed-Point Arithmetic

The Mandelbrot set renderer uses fixed-point complex arithmetic.

### Fixed-Point Format

Q16.16 format: 16 integer bits, 16 fractional bits.

```
Value = raw_int / 65536

Examples:
  1.0     = 65536
  -2.0    = -131072
  0.5     = 32768
  0.0625  = 4096
```

### Complex Iteration

For each pixel (px, py), map to complex c and iterate:

```
z = 0
c = complex(map(px, -2, 1), map(py, -1.5, 1.5))

for i in range(MAX_ITER):
    z_real² = mul_fp(z.real, z.real)  # Fixed-point multiply
    z_imag² = mul_fp(z.imag, z.imag)

    if z_real² + z_imag² > 4 * 65536:  # |z|² > 4
        return i  # Escape time

    z_new_real = z_real² - z_imag² + c.real
    z_new_imag = 2 * mul_fp(z.real, z.imag) + c.imag
    z = complex(z_new_real, z_new_imag)

return MAX_ITER  # Didn't escape
```

### Fixed-Point Multiply

```c
int mul_fp(int a, int b) {
    long long product = (long long)a * b;
    return (int)(product >> 16);  // Shift back to Q16.16
}
```

In neural VM, this becomes:
1. SwiGLU multiply: `temp = silu(S*a) * b / S`
2. Right-shift by 16: Implemented as divide by 65536


## 17. I/O Implementation Details

I/O uses embedding "mailbox" slots, not system calls.

### Embedding Slots for I/O

| Slot | Name | Purpose |
|------|------|---------|
| E.IO_CHAR | 368-375 | Character as nibbles (8 bits) |
| E.IO_OUTPUT_READY | 376 | 1.0 when PUTCHAR has output |
| E.IO_INPUT_READY | 377 | 1.0 when input is available |
| E.IO_NEED_INPUT | 378 | 1.0 when GETCHAR needs input |
| E.IO_PROGRAM_END | 379 | 1.0 when EXIT called |
| E.IO_TOOL_CALL_TYPE | 380 | Tool call type enum |
| E.IO_EXIT_CODE | 381 | Exit code for EXIT |

### GETCHAR Flow

```
1. FFN detects GETCHAR opcode
2. If IO_INPUT_READY == 0:
   - Set IO_NEED_INPUT = 1.0
   - External handler sees this, provides input
   - Handler sets IO_CHAR and IO_INPUT_READY = 1.0
3. If IO_INPUT_READY == 1:
   - Copy IO_CHAR to RESULT
   - Clear IO_INPUT_READY and IO_NEED_INPUT
```

### PUTCHAR Flow

```
1. FFN detects PUTCHAR opcode
2. Copy NIB_A (character) to IO_CHAR
3. Set IO_OUTPUT_READY = 1.0
4. External handler reads IO_CHAR, outputs character
5. Handler clears IO_OUTPUT_READY
```

### Streaming Mode

Uses text markers:
```
Hello!          ← output
<NEED_INPUT/>   ← VM requests input
Alice           ← user provides input
Nice to meet you, Alice!
<PROGRAM_END/>  ← EXIT called
```

### Tool-Use Mode

Uses structured tool calls:
```
TOOL_CALL:putchar:42:{"char":72}
TOOL_CALL:getchar:43:{}
TOOL_RESPONSE:43:{"success":true,"result":65}
TOOL_CALL:exit:44:{"code":0}
```

This enables agentic execution with explicit pause/resume.

---

## 19. Conclusion

The Neural VM demonstrates that **transformers can execute arbitrary computation**
through carefully constructed weight matrices. The C runtimes achieve fully neural
execution for all arithmetic; with baked models, instruction fetch and data reads
also flow through the model. The Python `full_neural_vm.py` achieves the same
with attention-based memory and neural instruction fetch.

### Key Insights

1. **SwiGLU enables multiplication**: `silu(S*a) * b` approximates `a * b` for positive values
2. **Step functions from SiLU pairs**: `silu(S*(x+0.5)) - silu(S*(x-0.5))` creates sharp thresholds
3. **Attention is memory**: KV cache with binary position encoding implements addressable RAM
4. **softmax1 enables ZFOD**: Zero values gracefully return zero, no initialization needed
5. **MoE without routing**: Opcode one-hot weights blend all experts, inactive ones contribute zero

### Architecture Summary

| Component | Implementation |
|-----------|----------------|
| ALU | Stacked PureFFN layers with baked weights |
| Memory | KV cache attention with binary address keys |
| Control | PC in embedding, branch via conditional copy |
| I/O | Embedding mailbox slots + external handler |
| Routing | Soft MoE with opcode-weighted blending |

### By the Numbers

- **46 opcodes** defined (0-42 + 64-66), 42 emitted by compiler
- **~3,000 non-zero weights** for core ALU operations
- **95%+ sparsity** in weight matrices
- **3,079 GFLOPs** for 256×256 Mandelbrot (fully neural)
- **100-500x speedup** from speculative execution



#### Store Char (SC) - Neural KV Cache Write

```
┌────────────────────────────────────────────────────────────────┐
│                    SC Neural Forward Pass                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Address (popped from stack), Value in AX               │
│                                                                 │
│  STEP 1: Pop Address from Stack (FFN + KV Attention)           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Read SP from register slot, decrement by 4               │   │
│  │ Attention lookup at stack pointer → address              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  STEP 2: Extract Low Byte from AX (FFN)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ byte_value = AX & 0xFF                                   │   │
│  │                                                          │   │
│  │ FFN extracts nibbles 0-1:                                │   │
│  │   W_up[hidden, NIB_A_0] = 1.0                            │   │
│  │   W_up[hidden, NIB_A_1] = 16.0                           │   │
│  │   (ignores nibbles 2-7)                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  STEP 3: Binary Encode Address (same as LC)                    │
│                                                                 │
│  STEP 4: KV Cache Write (Append to KV cache)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ New K entry: binary(address)                             │   │
│  │ New V entry: byte_value                                  │   │
│  │                                                          │   │
│  │ If address already exists in cache:                      │   │
│  │   - New entry shadows old (newer KV entries have higher  │   │
│  │     attention scores due to causal masking)              │   │
│  │   - Or: eviction policy removes old entry                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  OUTPUT: Updated KV cache with new (address, value) entry      │
└────────────────────────────────────────────────────────────────┘
```




## 22. c4onnx v3 Binary Format

The `.c4onnx` format stores neural VM weights and computation graphs.

### File Layout

```
Header:
  [4] magic     = 0x584E4E4F ("ONNX")
  [4] version   = 3
  [4] n_tensors
  [4] n_subgraphs

Tensors (repeated n_tensors times):
  [4] name_len
  [*] name_bytes (UTF-8, max 63)
  [4] ndims
  [4×ndims] dims
  [4] storage_type   (0=dense, 1=sparse_coo)
  if dense:
    [4] size
    [4×size] values (16.16 fixed-point int32)
  if sparse_coo:
    [4] nnz
    [4×nnz] indices
    [4×nnz] values

Subgraphs (repeated n_subgraphs times):
  [4] name_len
  [*] name_bytes
  [4] num_inputs
  [4] num_outputs
  [4] num_temps
  [4×num_temps] temp_sizes
  [4] num_nodes
  Nodes (repeated num_nodes times):
    [4] op_type
    [4] num_inputs
    [4×num_inputs] input_refs
    [4] num_outputs
    [4×num_outputs] output_refs
```

### Reference Encoding

Subgraph node references use negative indices for local buffers:
- `>= 0`: global tensor index
- `-1..-num_inputs`: subgraph inputs
- `-(num_inputs+1)..-(num_inputs+num_temps)`: temp buffers
- `-(num_inputs+num_temps+1)..`: subgraph outputs

### Op Types

| Value | Op | Description |
|------:|:---|:------------|
| 1 | ADD_NN | Element-wise add |
| 5 | MATMUL | Matrix multiply (dense or sparse) |
| 7 | SIGMOID_NN | Element-wise sigmoid |
| 10 | SOFTMAX | Softmax with temperature |
| 14 | CONCAT | Concatenate vectors |
| 15 | SLICE | Slice [start, end) |
| 16 | SCALE | Multiply by scalar constant |


### Tools

- `tools/export_vm_weights.py`: Export base model (ALU weights + subgraphs)
- `tools/bake_program.py`: Bake program into model (adds baked_fetch/baked_data subgraphs)

---













## 23. Baked Mode (Neural Instruction Fetch)

Baked mode encodes the program's instruction table and data segment as neural
weight matrices inside the .c4onnx model, replacing array lookups with
MATMUL + sigmoid routing.

### The Math

Any address→value lookup becomes a 5-step neural operation:

```
Input: address decomposed into 8 one-hot nibbles [128]

1. MATMUL  [128] @ select[128, N]  → scores[N]    (count matching nibbles)
2. ADD     scores + bias[N]        → shifted[N]    (threshold at -7.5)
3. SCALE   shifted × 100           → scaled[N]     (sharpen)
4. SIGMOID scaled                  → router[N]     (≈1.0 for exact match)
5. MATMUL  router @ values[N, V]   → output[V]     (select value)
```

Where N = number of entries, V = value vector size.

### Why It Works

The selection matrix has 1.0 at `(nib_pos*16 + nib_val, entry_idx)` for each
nibble of each address. MATMUL sums these: an exact match scores 8 (all 8 nibbles
match), any mismatch scores < 8. The bias of -7.5 makes only exact matches
positive. After sigmoid with temperature 100, the router is effectively one-hot.
The final MATMUL selects the corresponding value.

### Tensors Per Baked Table

| Tensor | Shape | Sparsity | Description |
|--------|-------|----------|-------------|
| `select` | [128, N] | 8N nnz / 128N total (93.75%) | Nibble match matrix |
| `bias` | [N] | Dense (all = -7.5 × SCALE) | Threshold |
| `val` | [N, V] | Varies | Value vectors |

### What Gets Baked

| Subgraph | Entries | Value Size | Source |
|----------|---------|------------|--------|
| `baked_fetch` | N instructions | 1025 (opcode + NVal immediate) | `program_code[]` |
| `baked_data` | M data bytes | 1 (byte value) | `program_data[]` |

Instruction addresses: `instr_idx × 8`. Data addresses: `0x10000 + offset`.

### Parameter Cost

Per instruction: 8 (select nnz) + 1 (bias) + 1 (opcode) + 4 (immediate NVal) = **14 nonzero parameters**.

Per data byte: 8 (select nnz) + 1 (bias) + 1 (value) = **10 nonzero parameters**.

### Usage

```bash
# Bake a program into the model
python3 tools/bake_program.py models/transformer_vm.c4onnx source.c -o models/baked.c4onnx

# Bundle with baked model (auto-detects baked_fetch subgraph)
python3 bundler/neural_bundler.py models/baked.c4onnx source.c > bundled.c
```

The C runtime auto-detects baked mode: if a `baked_fetch` subgraph exists in the
model, it uses neural instruction fetch instead of `program_code[]` array lookup.
The `program_code[]` array is still emitted for backward compatibility.

---


























## 24. C Runtimes

Three C runtime implementations, each with different trade-offs:

| | `neural_runtime.c` | `onnx_vm_runtime.c` | `optimized_runtime.c` |
|---|---|---|---|
| **Purpose** | Reference / self-hosting | General ONNX graph executor | Production / speculative |
| **Lines** | ~1,255 | ~2,135 | ~1,301 |
| **C dialect** | C4-compatible (`int`/`char`, `while` only) | C4-compatible | Standard C (`double`, `for`, `switch`) |
| **Arithmetic** | 16.16 fixed-point | 16.16 fixed-point | `double` float |
| **Softmax** | Full exp-normalize | Full exp-normalize | Argmax shortcut |
| **Multiply** | Schoolbook nibble | Schoolbook nibble | SwiGLU shortcut |
| **Subgraphs** | No | Yes | No |
| **Sparse matmul** | No | Yes (COO at runtime) | No (sparse-input shortcut) |
| **Baked mode** | No | Yes | No |
| **Compile modes** | 1 | 1 | 3 (neural, speculative, validate) |

### neural_runtime.c

The reference implementation. Written in C4-compatible syntax so it can be
compiled by the C4 compiler itself (self-hosting). All computation via 16.16
fixed-point integer arithmetic. Each neural primitive (b2n, n2b, nib_add, etc.)
is a hardcoded function that calls matmul/softmax with specific weight tensors.
Also used as the inline default in `neural_bundler.py`.

### onnx_vm_runtime.c

The most general runtime. Includes a full ONNX-style tensor execution engine
with 7 op types, plus a subgraph parser/executor. Neural primitives are defined
as named subgraphs in the model file and executed by `run_subgraph()`. This
makes it data-driven — baked tables work without C code changes. Supports sparse
COO matmul at runtime for large weight matrices.

### optimized_runtime.c

The fast runtime with three compile-time modes:
- **Default**: Full neural execution using `double` float with shortcuts (argmax
  instead of softmax, SwiGLU for multiply)
- **`-DSPECULATIVE`**: Native C arithmetic, no model loading, near-native speed
- **`-DNEURAL_VALIDATE=N`**: Speculative execution with periodic neural validation
  every N operations

Also supports `-DPRUNED_MEMORY` for sparse KV-cache hash table with overwrite
elimination and LRU eviction.

---

## 25. Main Document vs Implementation Discrepancies

Systematic comparison of the main writeup against the actual code in `neural_vm/`.

### 25.1 Opcode Table — Layer Counts (L column) are Wrong

The L column in the opcode table does not match `pure_alu.py`. Actual layer counts
from `build_parallel_alu()` and its pipeline builders:

| Op | Doc L | Actual L | Pipeline | Notes |
|----|------:|--------:|----------|-------|
| LEA | 1 | 1 | single_step | ✓ |
| IMM | 0 | 1 | single_step | Doc says 0, actual is 1 MoE expert |
| JMP | 1 | 1 | single_step | ✓ |
| JSR | 2 | 1 | single_step (CallFFN) | Doc says 2 |
| BZ | 1 | 3 | bz_bnz (detect→reduce→branch) | |
| BNZ | 1 | 3 | bz_bnz | |
| ENT | 2 | 1 | single_step | Doc says 2 |
| ADJ | 1 | 1 | single_step | ✓ |
| LEV | 2 | 1 | single_step (RetFFN) | Doc says 2 |
| LI | 1 | 1 | single_step (LoadFFN) | ✓ |
| LC | 1 | 1 | single_step | ✓ |
| SI | 1 | 1 | single_step (StoreFFN) | ✓ |
| SC | 1 | 1 | single_step | ✓ |
| PSH | 2 | 1 | single_step (PushFFN) | Doc says 2 |
| OR | 1 | 7 | bitwise (clear→bit3→2→1→0→combine→clear) | |
| XOR | 1 | 7 | bitwise | |
| AND | 1 | 7 | bitwise | |
| EQ | 1 | 5 | eq_ne (diff→nibble_cmp→clear→reduce_new→reduce_old) | |
| NE | 1 | 5 | eq_ne | |
| LT | 2 | 3 | cmp (diff+gen→borrow-lookahead→cleanup) | |
| GT | 2 | 3 | cmp | |
| LE | 2 | 3 | cmp (+invert in step 3) | |
| GE | 2 | 3 | cmp (+invert) | |
| SHL | 2 | 2 | shift (precompute→select) | ✓ |
| SHR | 2 | 2 | shift | ✓ |
| ADD | 2 | 3 | add (raw+gen→carry-lookahead→final) | |
| SUB | 2 | 3 | sub (raw+gen→borrow-lookahead→final) | |
| MUL | 4 | 7 | mul (schoolbook→carry×3→gen_prop→lookahead→correction) | |
| DIV | 6 | 26 | div (2 setup + 8×3 long division) | Major error |
| MOD | 7 | 34 | mod (26 long division + 8 remainder extraction) | FIXED (was 17) |
| GETCHAR | 9 | 1 | single_step (mailbox slot) | Doc says 9 |
| PUTCHAR | 9 | 1 | single_step (mailbox slot) | Doc says 9 |
| EXIT | 1 | 1 | single_step | ✓ |
| BLT | 1 | 1 | single_step | ✓ |
| BGE | 1 | 1 | single_step | ✓ |

**Note:** All single-step ops share one MoE layer. The "L" column should represent
the pipeline depth within the parallel ALU, not the total sequential layers.

### 25.2 Opcode Table — Weight Counts (W column) are Wrong

The W column numbers don't correspond to the actual hidden unit counts in the FFN
classes. Key examples:

- LI/LC/SI/SC: Doc says 500 weights each. Actual LoadFFN/StoreFFN are simple PureFFN
  cancel-pairs with 2 hidden units that prepare address/value in embedding slots.
  The actual memory read/write happens via standard autoregressive attention to the
  KV cache (binary keys with ±SCALE, ALiBi for recency). The 500-weight figure likely
  counted the attention projection weights, not just the FFN.
- GETCHAR/PUTCHAR: Doc says 220 weights. Actual is a single FFN that sets/reads
  embedding mailbox slots (E.IO_CHAR, E.IO_NEED_INPUT, etc.), not KV attention.
- DIV: Doc says 250 weights. Actual is ~563 hidden units across 26 layers.
- MUL: Doc says 320 weights. Actual SchoolbookFlatFFN alone has ~246 hidden units.

### 25.3 Summary Tables are Wrong

The summary table claims:
- "Total 45 opcodes" → actual is 46 (0-42 = 43, plus 64-66 = 3)
- "Max L = 9" → actual max is 26 (DIV)
- "Total weights 5,487" → outdated given actual layer/weight counts
- "~800 unique non-zero weights" → not verified against current implementation

### 25.4 Redundancy Table References Newton-Raphson DIV

The redundancy table says "6 iterations" for DIV with "250 weights" labeled as
Newton-Raphson. Newton-Raphson is NOT used — the actual implementation is
nibble-wise long division (8 iterations × 3 layers = 26 total). See Section 8.

### 25.5 Step Function Formula is Wrong

Document says: `SiLU(SCALE*(x+ε)) − SiLU(SCALE*(x−ε))`
(symmetric/centered difference)

Actual implementation uses forward difference:
`[silu(S*(x+ε)) − silu(S*x)] / (S*ε)`

Both produce step-like behavior but differ in transition shape and normalization.
The code in `long_division_ops.py`, `fast_arithmetic.py`, etc. all use the forward
difference with explicit ε parameter.

### 25.6 Addition Shows Bytes, Actual Uses Nibbles

Document shows "Position: 0 1 2 3" with "BYTE_A, BYTE_B" (4 positions = byte-level).
Actual implementation uses 8 positions with NIB_A, NIB_B (nibble-level). The carry
propagation operates on nibbles via `AddCarryLookaheadFFN` (FlattenedPureFFN), not bytes.

File: `fast_arithmetic.py` — `AddRawAndGenFFN`, `AddCarryLookaheadFFN`, `AddFinalResultFFN`.

### 25.7 Multiplication Describes Byte-Level, Actual is Nibble-Level

Document says "byte-level (8-bit) reduces partial products from 36 with nibbles to 10"
and shows a 4-byte layout. Actual implementation:

- Uses nibble-level schoolbook multiplication (`SchoolbookFlatFFN` in `fast_mul.py`)
- All 36 partial products computed in one FlattenedPureFFN layer
- 7 total layers: 1 schoolbook + 3 carry passes + gen/prop + lookahead + correction
- NOT byte-level, NOT 10 products

The document's byte-level description appears to be an unrealized optimization idea.

### 25.8 Division Description is Completely Wrong

Document says "6 layers" with a simple algorithm:
"Extract top 4 bits of remainder → compare with divisor → subtract → shift left"

Actual: 26 layers of base-16 long division processing MSB to LSB:
- 2 setup layers (ClearDivSlots + GatherScalar)
- 8 iterations × 3 layers: ShiftRemainderAndClearQ → ComputeQuotientNibble → SubtractAndWriteQ
- ComputeQuotientNibbleFFN uses 15 step functions: `q = Σ step(remainder - k·divisor)`

See Section 8 for the full description.

### 25.9 MOD Implementation — FIXED (v3: direct extraction)

**Was:** Old iterative pipeline (`ModInitFFN` + 15 × `ModIterFFN` + `ModResultFFN` = 17 layers).

**Now:** Long division + cascade remainder extraction:
- 26 layers: long division computes quotient nibbles to RESULT and scalar remainder to
  SLOT_REMAINDER (same loop as DIV)
- 8 layers: `ExtractAndSubtractRemainderFFN` for each nibble position 7 (MSB) down to 0.
  Each layer clears the old quotient from RESULT[pos], extracts the nibble value using
  15 step functions on SLOT_REMAINDER, and subtracts nibble×16^pos from SLOT_REMAINDER
  — all in a single 32-hidden-unit FlattenedPureFFN. The cascade ensures each position's
  floor(remainder/16^pos) is ≤ 15, so 15 step functions always suffice.

= **34 layers total**. MOD is still the ALU critical path. VM step = 45 layers.

**Known benign slot collision:** `SLOT_DIVISOR = E.TEMP + 1 = 7 = E.OP_START`. The
GatherScalarFFN writes the divisor value to slot 7 (overwriting opcode 0's weight).
This is harmless because no opcode-0 expert writes to division-relevant slots, and
opcode weights are only used for MoE gating (never as data after the ALU).

### 25.10 Bitwise Ops are NOT Lookup Tables

Document says "just lookup tables per nibble embedded in the FFNs" with "256 entries."

Actual: uses bounded step functions for bit extraction. Four layers extract bits 3→0
using increasing numbers of thresholds:
- ExtractBit3FFN: 4 hidden units (2 thresholds)
- ExtractBit2FFN: 12 hidden units (6 thresholds)
- ExtractBit1FFN: 28 hidden units (14 thresholds)
- ExtractBit0FFN: 64 hidden units (32 thresholds)

Then a combine layer (BitwiseAndCombineFFN etc.) reconstructs the nibble. Total: 7 layers.
File: `bitwise_ops.py`.

### 25.11 I/O Implementation Differs

Document describes GETCHAR/PUTCHAR as 9-layer operations using KV attention for
reading user input. Actual:

- GETCHAR/PUTCHAR are single-step FFN experts (1 layer each in the MoE)
- They set/read embedding mailbox slots (E.IO_CHAR, E.IO_NEED_INPUT, E.IO_OUTPUT_READY)
- The I/O handler (`io_handler.py`) manages the external interface
- No multi-layer KV attention cascade for I/O

See Section 17 (existing) for the correct I/O description.

### 25.12 VM Step Total Layer Count — NOW CORRECT

With MOD using long division (26 layers) + cascade remainder extraction (8 layers)
= 34 layers total, MOD is the ALU critical path. Total: 2 (register load) + 34
(PureALU, MOD critical path) + 1 (writeback 3a) + 7 (PC increment) + 1 (writeback 3b)
= **45 layers**. Matches the docstring.

### 25.13 Floor Function — Single-SiLU vs Sum-of-Steps

Document shows: `floor(x) ≈ silu(SCALE*(x-1+eps))/SCALE + 1 - eps`

This single-SiLU formula only captures one step transition at x=1. For a full
staircase floor function, the actual approach sums step functions:
`floor(x) = Σ_{k=1}^{K} step(x - k)`

The single-SiLU formula is valid as a building block but the document should clarify
it gives one threshold, not a general floor.

### 25.14 PureAttention Code Missing mask Init

The PureAttention code shown references `self.mask[:S, :S]` in forward() but the
`__init__` doesn't show the mask buffer registration. The actual `base_layers.py`
registers a causal mask buffer.

### 25.15 Minor Issues

- IMM listed as "L=0, W=0" but it's a real FFN expert (ImmFFN) with 1 layer
- The "CALL" alias for JSR and "RET" alias for LEV should be noted
- BEQ/BNE (aliases for BZ/BNZ) are used internally but not listed in the opcode table
- PRINTF2 (opcode 66) not listed in the table
- The MoE routing description says "opcode one-hot weights blend all experts" but
  the actual code uses `expert_out - x` (weighted residual), not weighted output


## 26. Bundler

The bundler takes a C source file and the neural VM's weight matrices and produces a single standalone C file. The generated file contains everything needed to run the program: the compiled bytecode, the model weights as embedded byte arrays, and a complete neural VM runtime. When compiled with `gcc -O2 -o program bundled.c -lm`, the result is a native executable where every arithmetic operation — addition, multiplication, division — flows through the neural model's SwiGLU layers via matrix multiply and activation, rather than using the CPU's ALU directly. The typical workflow is two commands:

```bash
python3 bundler/neural_bundler.py model.c4onnx source.c > bundled.c
gcc -O2 -o program bundled.c -lm
```

For self-hosting, the fixed-point bundler (`tools/neural_bundle_fixedpoint.py`) uses integer-only arithmetic — all weights are scaled by 2^12 and SiLU is approximated via a 17-entry lookup table with linear interpolation, so the generated C contains no floating point and can be compiled and run under the C4 VM. The bundled executable accepts a `-n` runtime flag to route ADD and MUL through the sparse SwiGLU pipeline; without it, native arithmetic is used. Separately, `bundler/neural_bundler.c` is a C4-compatible compiler and packager: it takes a model file, source file, and runtime template, compiles the source to bytecode, and emits a single C file with the model weights, bytecode, and runtime concatenated — no neural arithmetic of its own. Additional tools handle different input formats: `tools/bundle_onnx.py model.onnx output --compile` auto-converts from ONNX and optionally compiles in one step, `tools/bundle_executable.py` offers a modular `--runtime`/`--weights`/`--program` interface with a `--minimal` mode that strips out neural components entirely, and `bundler/bundle_c4.c` produces freestanding output with raw syscall wrappers instead of libc.
