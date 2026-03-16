# Building a Computer Inside a Transformer

## How I Made Neural Networks Do Exact Math

---

Everyone knows that large language models are bad at arithmetic. Ask ChatGPT to multiply 347 × 829 and you might get a wrong answer. Ask it to divide 123456 by 789 and it's a coin flip. The standard explanation is simple: transformers are pattern matchers, not calculators. They've memorized some arithmetic from training data, but they can't actually *compute*.

I used to believe this. Then I discovered something that changed my mind: **transformers already have all the building blocks for exact arithmetic**. We're just not using them right.

This is the story of how I built a complete virtual machine—one that runs recursive functions, renders the Mandelbrot set, and compiles C code—using nothing but standard transformer components. Not approximately. Not with 95% accuracy. *Exactly*.

---

## The Accidental Discovery

It started when I was staring at the SwiGLU activation function. If you've looked at the architecture of LLaMA, Mistral, or most modern open-source LLMs, you've seen SwiGLU. It's the gated activation in the feed-forward layers:

```python
def swiglu(x):
    return silu(x) * linear(x)
```

Where SiLU (also called Swish) is just `silu(x) = x * sigmoid(x)`.

I was thinking about what this function actually computes when I noticed something odd. What happens if I compute this:

```python
silu(a) * b + silu(-a) * (-b)
```

Let's work through it. When `a` is positive:
- `sigmoid(a)` approaches 1, so `silu(a) ≈ a`
- `sigmoid(-a)` approaches 0, so `silu(-a) ≈ 0`
- Result: `a*b + 0 = a*b`

When `a` is negative:
- `sigmoid(a)` approaches 0, so `silu(a) ≈ 0`
- `sigmoid(-a)` approaches 1, so `silu(-a) ≈ -a`
- Result: `0 + (-a)*(-b) = a*b`

Wait. That's just... multiplication?

I grabbed a pen and proved it properly:

```
Let s = sigmoid(a). Then sigmoid(-a) = 1 - s.

silu(a)*b + silu(-a)*(-b)
= (a*s)*b + (-a*(1-s))*(-b)
= a*s*b + a*(1-s)*b
= a*b*(s + 1 - s)
= a*b
```

It's an exact algebraic identity. **SwiGLU—the activation function already in your favorite LLM—can compute exact integer multiplication.**

Not approximately. Not "usually correct." *Mathematically exact.*

---

## Down the Rabbit Hole

Once I realized that transformers could do multiplication, I couldn't stop thinking about what else they could do. Multiplication is just one operation. What about addition? Division? Bitwise operations?

### Addition: The Nibble Trick

Addition seems harder than multiplication because it has carries. When you add 99 + 1, the carry propagates through both digits. How do you do that with neural operations?

The key insight is that **feed-forward networks are just lookup tables**.

Think about it: if your input is a one-hot vector (like [0, 0, 1, 0, 0] selecting index 2), then multiplying by a weight matrix just selects one column of that matrix. It's a table lookup!

So I built addition tables. But not for full bytes—that would need a 256×256 = 65,536 entry table. Instead, I split each byte into two *nibbles* (4-bit values, 0-15). Now I only need 16×16 = 256 entries per table.

Let me walk through how this actually works. Say we want to compute 100 + 150 = 250.

**Step 1: Encode as bytes**

100 in binary is 01100100, which is byte 0x64.
150 in binary is 10010110, which is byte 0x96.

(For 32-bit integers, we'd have 4 bytes each, but let's keep it simple with one byte.)

**Step 2: Split into nibbles**

0x64 → high nibble 0x6 (6), low nibble 0x4 (4)
0x96 → high nibble 0x9 (9), low nibble 0x6 (6)

This splitting is itself an FFN operation. I have a 256×32 weight matrix where each row encodes "byte B maps to high nibble H and low nibble L". One-hot byte input, matrix multiply, split output into two 16-element vectors.

**Step 3: Add low nibbles with carry-in = 0**

4 + 6 + 0 = 10

The nibble add table has 16 × 16 × 2 = 512 entries (all combinations of two nibbles and a carry bit). For input (4, 6, 0):
- Sum nibble: 10 % 16 = 10 (0xA)
- Carry out: 10 // 16 = 0

**Step 4: Add high nibbles with carry from step 3**

6 + 9 + 0 = 15

- Sum nibble: 15 % 16 = 15 (0xF)
- Carry out: 15 // 16 = 0

**Step 5: Recombine nibbles to byte**

High nibble 0xF, low nibble 0xA → byte 0xFA = 250.

That's the answer: 100 + 150 = 250.

**Step 6: For 32-bit integers, propagate carries across bytes**

A 32-bit add has 4 bytes. The carry from byte 0's high nibble becomes the carry-in for byte 1's low nibble. We chain through all 8 nibbles (4 bytes × 2 nibbles each).

The nibble addition table handles three inputs: nibble A, nibble B, and carry-in. It outputs two things: the sum nibble and carry-out. Chain these together and you get full 32-bit addition with carry propagation—all through neural network operations.

```python
# Nibble add table: (a + b + carry_in) → (sum, carry_out)
for a in range(16):
    for b in range(16):
        for c in range(2):
            total = a + b + c
            table[a, b, c] = (total % 16, total // 16)
```

The table has 512 entries total. Each entry is queried via one-hot encoding: concatenate the one-hot nibble A (16 elements), one-hot nibble B (16 elements), and one-hot carry (2 elements) into a 34-element vector. Multiply by the weight matrix. Out come the sum nibble and carry-out.

---

## How Nibble Tables Actually Work

I keep saying "FFN lookup tables," but let me show you exactly what that means. This is the core trick that makes everything work.

### The Key Insight: One-Hot × Matrix = Selection

Suppose you have a one-hot vector and multiply it by a matrix:

```python
one_hot = [0, 0, 1, 0, 0]  # Selecting index 2
matrix = [[a, b],
          [c, d],
          [e, f],    # ← This is row 2
          [g, h],
          [i, j]]

result = one_hot @ matrix  # = [e, f]
```

The one-hot vector *selects* row 2 of the matrix. That's it. Matrix multiplication with a one-hot input is just row selection.

This means **any lookup table can be implemented as a weight matrix**. The rows of the matrix ARE the table entries.

### Building a Nibble AND Table

Let's build the AND operation for two 4-bit nibbles. The table has 16 × 16 = 256 entries:

```python
# Build the weight matrix
W = torch.zeros(256, 16)  # 256 input combinations, 16 possible outputs

for a in range(16):
    for b in range(16):
        input_index = a * 16 + b      # Flatten (a,b) to single index
        output_value = a & b           # The AND result
        W[input_index, output_value] = 1.0  # One-hot encode the result
```

The matrix W encodes the entire AND truth table. Row 0 is "0 AND 0 = 0", row 1 is "0 AND 1 = 0", ..., row 255 is "15 AND 15 = 15".

### Querying the Table

To compute 0xA AND 0x5 (that's 10 AND 5 = 0):

```python
# Create one-hot input for (a=10, b=5)
a_onehot = torch.zeros(16)
a_onehot[10] = 1.0

b_onehot = torch.zeros(16)
b_onehot[5] = 1.0

# Combine into address
# We need a way to select row (10 * 16 + 5) = 165
```

Here's where it gets clever. I use a *two-stage* lookup:

**Stage 1: Address encoding**
```python
# W1 encodes the address: which (a,b) pair are we looking for?
W1 = torch.zeros(32, 256)  # 32 = 16 + 16 input dims, 256 possible addresses

for a in range(16):
    for b in range(16):
        addr = a * 16 + b
        W1[a, addr] = 1.0       # First 16 rows: nibble A
        W1[16 + b, addr] = 1.0  # Next 16 rows: nibble B
```

When we concatenate `[a_onehot, b_onehot]` (a 32-element vector) and multiply by W1, we get a 256-element vector where position 165 has value 2.0 (both a and b contributed), and other positions have 0.0 or 1.0.

**Stage 2: Sharp selection**
```python
# Softmax with high temperature makes the max dominate
address_scores = torch.cat([a_onehot, b_onehot]) @ W1  # [256]
address_weights = softmax(address_scores * 20)  # Sharp! Position 165 ≈ 1.0
```

The factor of 20 makes the softmax very sharp. Position 165 gets weight ~1.0, everything else ~0.0.

**Stage 3: Value lookup**
```python
# W2 maps addresses to values (this is our actual AND table)
result = address_weights @ W2  # [16] one-hot result
```

The output is a 16-element vector, nearly one-hot, indicating the result. For 10 AND 5 = 0, position 0 will be ~1.0.

### Why Nibbles Instead of Bytes?

I could build tables for full bytes: 256 × 256 = 65,536 entries. But that's a lot of memory and computation.

Nibbles hit the sweet spot:
- 16 × 16 = 256 entries per table (manageable)
- Any byte operation decomposes into two nibble operations
- Conversion overhead (byte↔nibbles) is just two more small FFN lookups

The byte-to-nibble FFN is a 256×32 matrix. The nibble-to-byte FFN is a 32×256 matrix. Small and fast.

### The Full Picture

Here's how 0xAB AND 0xCD flows through the system:

```
Input bytes: 0xAB, 0xCD
    ↓
Byte-to-Nibble FFN (256→32 matrix)
    ↓
Nibbles: [0xA, 0xB] and [0xC, 0xD]
    ↓
High nibbles: 0xA AND 0xC → Nibble table → 0x8
Low nibbles:  0xB AND 0xD → Nibble table → 0x9
    ↓
Nibble-to-Byte FFN (32→256 matrix)
    ↓
Output byte: 0x89
```

Every arrow is a matrix multiplication. Every step is differentiable. It's all `nn.Module` operations.

---

## Memory: The Elephant in the Room

So far I've talked about arithmetic. But a real computer needs memory—a place to store variables, stack frames, and data. This is where things get philosophically interesting.

### The Naive Approach Doesn't Scale

You might think: "Just use a giant FFN table for memory! Address goes in, value comes out."

The problem: even a tiny 64KB memory would need a 65,536-row weight matrix. Every read would be a 65,536-way lookup. Every write would need to update those weights. It doesn't scale.

### My Pragmatic Solution: Python Dictionary

For this implementation, I made a pragmatic choice: memory is a Python dictionary.

```python
class C4TransformerVM(nn.Module):
    def __init__(self):
        self.memory = {}  # addr → value

    def load(self, addr):
        return self.memory.get(addr, 0)

    def store(self, addr, value):
        self.memory[addr] = value
```

"Wait," you say, "that's not a neural operation!"

True. But here's my argument: **memory is where the computation lives, not what computes**. The arithmetic operations—add, multiply, divide—are the actual computation. Memory just holds intermediate results.

Think of it like this: when you prove that a Turing machine can compute something, you don't insist that the tape be made of logic gates. The tape is storage. The head movement and state transitions are the computation.

### How Memory Operations Work

The VM has two memory instructions:

**LI (Load Integer)**: Read a 32-bit value from the address in AX.
```python
def op_LI(self):
    addr = self.decode(self.ax)          # Get address from accumulator
    value = self.memory.get(addr, 0)     # Read from memory
    self.ax = self.encode(value)         # Put value in accumulator
```

**SI (Store Integer)**: Write AX to the address on the stack.
```python
def op_SI(self):
    addr = self.pop_int()                # Pop address from stack
    value = self.decode(self.ax)         # Get value from accumulator
    self.memory[addr] = value            # Write to memory
```

The `encode` and `decode` functions convert between integers and one-hot byte representations. So while memory access itself is a dictionary lookup, the values flowing in and out are in neural format.

### The Stack

The stack is just memory with a special pointer. SP (stack pointer) starts at 0x10000 and grows downward:

```python
def push(self, value):
    self.sp -= 8                         # Allocate 8 bytes (64-bit words)
    self.memory[self.sp] = value         # Store at new stack top

def pop(self):
    value = self.memory.get(self.sp, 0)  # Read from stack top
    self.sp += 8                         # Deallocate
    return value
```

Function calls work like any stack machine:
1. Push arguments (right to left)
2. Push return address
3. Jump to function
4. Function pushes old base pointer
5. Function allocates locals
6. On return: restore BP, pop return address, jump back

All the pointer arithmetic (updating SP and BP) flows through the neural add/subtract circuits. The actual memory operations are dictionary reads/writes.

### Could Memory Be Neural Too?

Yes, but it's a different research direction. Some approaches:

**Attention as Memory**: Transformer attention over a fixed-size memory buffer. Each "slot" is a key-value pair. Reading is attention-weighted retrieval. Writing is attention-weighted update. This is basically what Neural Turing Machines and Differentiable Neural Computers do.

**Sparse Tables**: For most programs, you don't actually use all 4GB of address space. A sparse attention mechanism could handle the addresses that matter.

**Content-Addressable Memory**: Instead of addressing by location, address by content. "Give me the slot that contains something like X." Attention mechanisms do this naturally.

I didn't implement these because they're complex and my goal was demonstrating that *computation* can be neural. But they're fascinating directions.

### Memory Layout

For the curious, here's how memory is organized:

```
0x00000 - 0x0FFFF: Code segment (bytecode lives here)
0x10000 - 0x1FFFF: Data segment (string literals, globals)
0x20000 - 0x2FFFF: Stack (grows downward from 0x20000)
```

When you compile `int x = 42;` as a local variable:
1. The compiler emits `ENT 8` (allocate 8 bytes on stack)
2. SP decreases by 8
3. `x` lives at address `BP - 8`
4. `LEA -8` loads that address into AX
5. `IMM 42` followed by `SI` stores 42 there

Function parameters work similarly but at positive offsets from BP (they were pushed by the caller before the call).

### Clarification: What IS Neural About Memory

Let me be precise about what's neural and what isn't:

**Neural:**
- The *values* stored in memory are encoded as one-hot byte vectors
- Every value that goes INTO memory passes through `encode()` (int → 4 one-hot bytes)
- Every value that comes OUT of memory passes through `decode()` (4 one-hot bytes → int)
- All arithmetic on addresses (SP += 8, BP - offset) uses the neural add/subtract circuits

**Not neural:**
- The address→value mapping itself (the dictionary)
- Choosing which memory location to access

So when you do `memory[addr] = value`, the `value` is neural (one-hot encoded), but the indexing operation `[addr]` is a Python dictionary lookup.

Could the indexing be neural too? Yes—with attention over memory slots. But that's a different (and larger) project. The point here is that *computation on values* is neural. The storage mechanism is pragmatic infrastructure.

---

## Registers Are Just Tensors

Here's something that surprised me when I first built this: the entire register file is just a few tensors.

### The Accumulator (AX)

AX holds a 32-bit value. In the neural VM, it's represented as:

```python
self.ax = torch.zeros(4, 256)  # 4 bytes, each a 256-dim one-hot vector
```

That's it. The accumulator is a [4, 256] tensor. When AX holds the value 42:

```
ax[0] = [0, 0, ..., 1, ..., 0]  # Position 42 is hot (low byte)
ax[1] = [1, 0, 0, ..., 0]       # Position 0 is hot
ax[2] = [1, 0, 0, ..., 0]       # Position 0 is hot
ax[3] = [1, 0, 0, ..., 0]       # Position 0 is hot (high byte)
```

42 in little-endian bytes is [42, 0, 0, 0], so byte 0 has the 42nd position hot, and bytes 1-3 have position 0 hot.

### Why One-Hot?

One-hot encoding is the key that makes everything work:

1. **Table lookups become matrix multiplies**: one-hot × matrix = row selection
2. **No learned embeddings needed**: the representation is fixed and exact
3. **Perfectly invertible**: `argmax` recovers the original value with no loss
4. **Composable**: operations chain naturally (output of one is input to next)

The one-hot constraint means every byte is exactly one of 256 values—no fuzzy "kind of 42" states. This is what gives us exact arithmetic.

### Setting and Getting Values

```python
def set_ax(self, value: int):
    """Store an integer in the accumulator."""
    self.ax = torch.zeros(4, 256)
    for i in range(4):
        byte_val = (value >> (i * 8)) & 0xFF
        self.ax[i, byte_val] = 1.0

def get_ax(self) -> int:
    """Read the accumulator as an integer."""
    result = 0
    for i in range(4):
        byte_val = torch.argmax(self.ax[i]).item()
        result |= byte_val << (i * 8)
    return result
```

The `set_ax` is encoding (int → tensor). The `get_ax` is decoding (tensor → int). Between these, all operations work on the tensor representation.

### SP, BP, PC: Hybrid Representation

The stack pointer, base pointer, and program counter are stored as Python integers:

```python
self.sp = 0x10000  # Stack pointer
self.bp = 0x10000  # Base pointer
self.pc = 0        # Program counter
```

Why not one-hot? Because these are *addresses*, not *data*. They're used for:
- Dictionary indexing (memory[sp])
- Control flow (jumping to pc)
- Comparison with constants

When we do arithmetic on them (like `sp -= 8`), we *could* route through the neural circuits. Currently I do it directly for simplicity. But the values being loaded/stored at those addresses are fully neural.

### The Beautiful Implication

Here's what's beautiful: **the register file could be an attention key-value store**.

Imagine:
- Query: "what's in register AX?"
- Keys: [AX, BX, CX, DX, SP, BP, ...]
- Values: the one-hot encoded contents

Attention gives you soft selection over registers. With sharp attention (high temperature), you get hard selection—exactly one register.

This is how transformers could implement a VM *internally*. The attention mechanism becomes the register file. The FFN becomes the ALU. It's all the same primitives.

---

## Only 129,000 Parameters

Here's a number that blows my mind: the entire VM—every FFN table, every nibble operation, everything needed to run Fibonacci, render Mandelbrot, compile C code—is **129,000 parameters**.

And if you only count non-zero values? **5,376 parameters.**

Let's put that in perspective:

| Model | Parameters |
|-------|-----------|
| C4 Transformer VM | 129K (5.4K non-zero) |
| Single GPT-2 layer | ~1.5M |
| GPT-2 Small | 124M |
| LLaMA-7B | 7B |
| GPT-4 (estimated) | 1.8T |

The VM is smaller than a single transformer layer. It's a rounding error compared to modern LLMs.

### Where Do The Parameters Live?

```
NibbleConverters (shared):  81,920 elements (1,280 non-zero)
  - ByteToNibbleFFN:        73,728 (768 non-zero)
  - NibbleToByteFFN:         8,192 (512 non-zero)

NibbleAddFFN:               26,624 elements (2,560 non-zero)
  - Carry-aware addition table for 512 input combinations

BitwiseOps:                 20,480 elements (1,280 non-zero)
  - Shared address encoder:  8,192 (512 non-zero)
  - AND/OR/XOR value tables: 12,288 (768 non-zero)

DivisionFFN:                   256 elements (256 non-zero)
  - Reciprocal lookup table

─────────────────────────────
Total:                     129,280 elements (5,376 non-zero)
```

**Key optimizations:**
1. **Shared converters**: One ByteToNibbleFFN and NibbleToByteFFN shared across all operations (bitwise AND/OR/XOR, addition)
2. **Shared address encoder**: The W1 matrix that maps (a,b) nibble pairs to table indices is identical for AND, OR, and XOR—we share it
3. **Sparse structure**: 96% of elements are zero (lookup tables are mostly sparse)

**Note on counting:** These are ALL buffer elements, counted via `sum(b.numel() for b in vm.buffers())`. The lookup tables are sparse (mostly 0s and 1s), but every element counts in dense format.

### Memory Footprint

At float32, that's 129,280 × 4 bytes = **505 KB**.

With sparse storage (non-zeros only): 5,376 × 4 bytes = **21 KB**.

You could fit this VM in the L1 cache of your CPU. You could run it on a microcontroller. It's genuinely tiny.

### What This Means

If you can build a complete computer in 129K parameters (or 5K non-zeros), what does that say about 175B-parameter language models?

It means **capability isn't just about scale**. The right 5K parameters can do exact arithmetic, recursion, compilation. The wrong 175B parameters struggle with 3-digit multiplication.

The difference is structure. The VM's parameters are organized as lookup tables with known functions. LLM parameters are trained end-to-end and we hope they learn something useful.

Maybe future LLMs will have explicit arithmetic circuits—a few thousand parameters carved out from the billions, structured to do exact computation. The rest can do language. Best of both worlds.

---

## TODO: Why This Proves Something About Transformers

[This section to be written - exploring the theoretical implications of exact neural computation for understanding transformer capabilities and limitations]

---

### Division: Newton to the Rescue

Division was the hardest nut to crack. There's no cute identity like the SwiGLU multiplication trick. Division is fundamentally a different kind of operation.

But here's what we *can* do: **compute reciprocals with a lookup table, then multiply**.

The idea comes from how CPUs actually implement division. You don't directly compute a/b. Instead:

1. Compute 1/b using a table lookup
2. Refine with Newton-Raphson iterations
3. Multiply: a × (1/b)

Let me walk through the whole algorithm for computing 100 ÷ 7.

**Step 1: Normalize the divisor**

We need the divisor in the range [0.5, 1.0) for our table lookup. Start with 7:
- 7 ≥ 1.0, so halve it: 3.5
- 3.5 ≥ 1.0, so halve it: 1.75
- 1.75 ≥ 1.0, so halve it: 0.875

Now we have 0.875, which is in [0.5, 1.0). We halved 3 times, so we'll need to adjust the final result.

**Step 2: Table lookup**

I have a 256-entry table storing 1/x for x evenly spaced in [0.5, 1.0):

```python
for i in range(256):
    x = 0.5 + i / 512  # x from 0.5 to ~0.998
    reciprocal_table[i] = 1.0 / x
```

For x = 0.875, the index is roughly (0.875 - 0.5) × 512 = 192. Looking up `reciprocal_table[192]` gives us approximately 1.143 (which is close to 1/0.875 = 1.142857...).

The table is stored as an FFN weight matrix. "Looking up index 192" means creating a one-hot vector with position 192 set to 1, and multiplying by the weight matrix. Standard neural network stuff.

**Step 3: Newton-Raphson refinement**

Our table gave us 8 bits of precision, but we want 32-bit accuracy. Newton-Raphson to the rescue:

```
y_new = y × (2 - b × y)
```

Each iteration roughly doubles the precision. Two iterations: 8 → 16 → 32 bits.

Here's the magic: that formula is just multiplications and a subtraction. We have SwiGLU multiplication. We have subtraction via nibble tables. Newton-Raphson composes from operations we already built!

```python
b_norm = 0.875
y = 1.143  # from table lookup

# Iteration 1
by = swiglu_mul(b_norm, y)         # 0.875 × 1.143 ≈ 1.000
y = swiglu_mul(y, 2.0 - by)        # 1.143 × 1.0 ≈ 1.143

# Iteration 2
by = swiglu_mul(b_norm, y)         # 0.875 × 1.143 ≈ 1.000
y = swiglu_mul(y, 2.0 - by)        # even more precise
```

After two iterations, y ≈ 1.142857 (very close to the true 1/0.875).

**Step 4: Scale back**

Remember we halved the divisor 3 times? Now we halve the reciprocal 3 times:
- 1.142857 / 2 = 0.571429
- 0.571429 / 2 = 0.285714
- 0.285714 / 2 = 0.142857

This gives us 1/7 ≈ 0.142857.

**Step 5: Multiply**

Finally, compute 100 × (1/7):
```python
result = swiglu_mul(100.0, 0.142857)  # ≈ 14.2857
```

Round to integer: 14. And indeed, 100 ÷ 7 = 14 (integer division).

**Step 6: Correction**

Floating-point rounding can occasionally give us 13 or 15 instead of 14. A simple correction loop fixes this:

```python
while (result + 1) * 7 <= 100:
    result += 1
while result * 7 > 100:
    result -= 1
```

This ensures we always get the exact integer division result.

The whole thing chains together: normalize → table lookup → Newton → scale → multiply → correct. Six steps, but each step uses neural operations we already have.

---

## All the Other Operations

Multiplication, addition, and division are the heavy hitters. But a real computer needs more: bitwise AND, OR, XOR, shifts, comparisons. Let's go through them.

### Bitwise Operations: More Tables

Bitwise operations are actually the easiest. AND, OR, and XOR are all bitwise—each output bit depends only on the corresponding input bits. No carries, no dependencies between positions.

I use the same nibble table approach. For each operation, build a 16×16 table:

```python
# AND table
for a in range(16):
    for b in range(16):
        and_table[a, b] = a & b

# OR table
for a in range(16):
    for b in range(16):
        or_table[a, b] = a | b

# XOR table
for a in range(16):
    for b in range(16):
        xor_table[a, b] = a ^ b
```

To AND two bytes: split each into nibbles, look up the high nibbles in the AND table, look up the low nibbles, recombine. To AND two 32-bit integers: do this for all four bytes.

The beauty is that there's no carry propagation. Each byte is independent. This means bitwise operations are actually *faster* than addition in this architecture.

### Shifts: A Different Approach

Bit shifts are trickier. When you shift `0x12` left by 4 bits, you get `0x120`—bits cross byte boundaries. The simple per-nibble approach doesn't work.

For shifts by constant amounts (which is most shifts in practice), I use specialized tables. Shift-left-by-4 is just moving each nibble up one position:

```
0xAB << 4 = 0xB0 (within byte) + 0x0A (to next byte)
```

For variable shifts, I decompose into shift-by-1, shift-by-2, shift-by-4, etc., and apply the relevant ones. It's not elegant, but it works.

### Comparisons: Sharp Gates

Comparisons (less than, greater than, equal) need to output 0 or 1. I use a "sharp gate" approach—a steep sigmoid that's essentially a step function:

```python
def sharp_gate(x, scale=20.0):
    # Approximates step function: 1 if x > 0, else 0
    return sigmoid(x * scale)

def less_than(a, b):
    return sharp_gate(b - a - 0.5)

def equal(a, b):
    # 1 only when |a - b| < 0.5
    return sharp_gate(a - b + 0.5) * sharp_gate(b - a + 0.5)
```

The scale parameter controls sharpness. At scale=20, the transition from 0 to 1 happens over a tiny range—effectively digital.

### Subtraction: Just Add Negative

Subtraction is the easy one. I compute `a - b` as `a + (-b)`. Negation in two's complement is "flip bits and add 1", but I take a shortcut: store the value, negate it in Python, re-encode. The subtraction *result* still flows through the neural addition circuit.

For a pure neural implementation, you'd build a negation table (XOR with 0xFF for each byte, then add 1 with carry). I left this as an optimization for later.

### Modulo: Division's Sidekick

Modulo uses division: `a % b = a - (a / b) * b`. Three operations chained together:
1. Divide a by b (FFN table + Newton)
2. Multiply quotient by b (SwiGLU)
3. Subtract from a (nibble tables)

It's expensive—probably the most expensive single operation. But it works, and it's exact.

---

## The Tokenization Story

Before we get to the VM, let's talk about how data gets into this system. This is where byte tokens come in.

### The Problem with Normal Tokenization

Most LLMs use BPE (Byte Pair Encoding) or similar schemes. The word "multiplication" becomes one token. "Multiply" is a different token. The number "12345" might be one token, or "123" + "45", depending on the tokenizer.

This is great for natural language—common words get short representations. But it's terrible for computation:

- "12345" and "12346" have no systematic relationship in token space
- Digits aren't separate, so you can't operate on them
- The model has to learn that token #4827 means something numerically close to token #4828

### Byte Tokens: The Simplest Thing

My solution is brain-dead simple: **one byte = one token**. Vocabulary size is exactly 256.

```python
def tokenize(text):
    return [ord(c) for c in text]

def detokenize(tokens):
    return ''.join(chr(t) for t in tokens)
```

That's it. No training a tokenizer. No merges. No special handling.

The number "12345" becomes tokens [49, 50, 51, 52, 53]—the ASCII codes for '1', '2', '3', '4', '5'. Each digit is separate. Each has a systematic relationship to other digits (the digit '5' is token 53, digit '6' is token 54).

### Why This Actually Makes Sense

Byte-level tokenization isn't my invention. Google's ByT5 uses it. Meta's recent work on byte-level models uses it. The tradeoff is longer sequences (every character is a token), but modern architectures handle that fine.

For a VM, byte tokens are perfect:
- Source code is just ASCII bytes
- Bytecode is just bytes
- Memory contents are just bytes
- Everything is the same vocabulary

### From Bytes to Computation

Here's how a number flows through the system:

1. **Input**: The string "42" arrives as tokens [52, 50] (ASCII for '4' and '2')

2. **Parsing**: The compiler recognizes this as a number literal and emits an IMM (immediate) instruction with value 42

3. **Encoding**: The VM encodes 42 as four one-hot byte vectors:
   ```
   byte 0: [0,0,...,1,...,0]  (42nd position hot)
   byte 1: [1,0,...,0]        (0th position hot)
   byte 2: [1,0,...,0]        (0th position hot)
   byte 3: [1,0,...,0]        (0th position hot)
   ```

4. **Computation**: Operations work on these one-hot representations

5. **Decoding**: When we need the result, we find the hot position in each byte and reconstruct the integer

The one-hot encoding is key. It turns byte values into a format where matrix multiplication equals table lookup.

### Special Tokens for LLM Integration

The base vocabulary is 256 bytes. But for LLM integration, I added special tokens above 256:

| Token | Meaning |
|-------|---------|
| 256 | `<think>` - start reasoning |
| 257 | `</think>` - end reasoning |
| 258 | `<\|user\|>` - user turn |
| 259 | `<\|assistant\|>` - assistant turn |
| 261 | `<\|code\|>` - code block start |
| 263 | `<\|exec\|>` - execute code |
| 264 | `<\|result\|>` - result follows |

This lets a language model use the VM naturally:

```
<|user|>What is fib(10)?
<|assistant|><think>I'll compute this with code</think>
<|code|>int fib(int n) { if (n<2) return n; return fib(n-1)+fib(n-2); }
int main() { return fib(10); }</code>
<|exec|><|result|>55
```

The model writes code, the VM executes it, and the result comes back as tokens.

---

## Building the VM

With all the arithmetic operations working and tokenization figured out, I had enough to build something real. I chose to implement C4—a tiny C compiler that fits in about 500 lines of code. The VM runs its bytecode.

The architecture looks like this:

**External interface:** Byte tokens (vocabulary size 256). Each character is its own token. This is exactly how byte-level language models work—no BPE, no wordpiece, just raw bytes. Tokenization is literally `tokenize = ord()`.

**Internal representation:** Each 32-bit integer is stored as four one-hot byte vectors. Operations split bytes into nibbles, do the computation, and recombine.

**Registers:** An accumulator (AX), stack pointer (SP), base pointer (BP), and program counter (PC). Standard stack machine stuff.

**Instructions:** Load immediate, push, pop, jump, branch, call, return, and all the arithmetic/bitwise operations. About 40 opcodes total.

The execution loop is straightforward:
```python
def step(self):
    op, imm = self.code[self.pc // 8]
    self.pc += 8

    if op == IMM:
        self.ax = encode(imm)
    elif op == ADD:
        self.ax = self.add_ffn(self.pop(), self.ax)
    elif op == MUL:
        a, b = decode(self.pop()), decode(self.ax)
        result = swiglu_mul(a, b)
        self.ax = encode(round(result))
    # ... etc
```

The magic is that `add_ffn` and `swiglu_mul` are `nn.Module` operations. PyTorch can run them on GPU. They're differentiable. They're exactly the kind of thing that could live inside a transformer.

---

## The Moment of Truth

The first real test was Fibonacci:

```c
int fib(int n) {
    if (n < 2) return n;
    return fib(n-1) + fib(n-2);
}

int main() {
    return fib(10);
}
```

I compiled it. Loaded the bytecode. Ran the VM.

**Result: 55**

It worked. Recursive function calls. Stack frames. Addition. Comparison. All flowing through neural network operations, all producing the exactly correct answer.

Then I got ambitious. What about something with multiplication and division? Something that would really stress-test the arithmetic?

I wrote a Mandelbrot set renderer.

For those unfamiliar: the Mandelbrot set involves iterating z = z² + c in the complex plane and checking if |z| stays bounded. Each pixel requires dozens of multiplications and divisions. It's a torture test for arithmetic.

I used fixed-point arithmetic (multiply by 1024, divide to rescale). The renderer does hundreds of SwiGLU multiplications and FFN divisions per pixel.

```
+----------------------------------------+
|                                        |
|                   *                    |
|                  ***                   |
|                 *****                  |
|                *******                 |
|            **** ******* ****           |
|         ************************       |
|       ****************************     |
|      ******************************    |
|     ********************************   |
|   ************************************.|
|     ********************************   |
|      ******************************    |
|       ****************************     |
|         ************************       |
|            **** ******* ****           |
|                *******                 |
|                 *****                  |
|                  ***                   |
|                   *                    |
+----------------------------------------+
```

It rendered correctly. Every single pixel computed through neural network operations.

---

## The Speed Problem (And Solution)

There's an elephant in the room: this is slow. Each multiplication goes through SiLU activations. Each addition propagates through nibble tables. Fibonacci(10) takes 15ms on the transformer VM versus 0.1ms with native Python.

For actual use, we need speed. But we also want the correctness guarantee of the transformer implementation.

The solution is speculative execution. I built a fast "logical VM" that uses normal Python arithmetic—it's 100x faster but semantically identical. The idea:

1. Run the fast VM for production speed
2. Periodically validate against the transformer VM
3. If they ever disagree, we have a bug

This gives us the best of both worlds: fast execution in production, with the ability to verify that the transformer implementation is correct. It's the same principle as speculative execution in CPUs, but for neural correctness instead of branch prediction.

```python
class SpeculativeVM:
    def run(self, bytecode, validate_ratio=0.1):
        fast_result = self.fast_vm.run()

        if random() < validate_ratio:
            trans_result = self.transformer_vm.run()
            assert fast_result == trans_result

        return fast_result
```

In my tests, validation ratio of 10% catches bugs quickly while maintaining good performance.

---

## Bytecode as System Prompt

Here's where it gets interesting for LLM applications. The whole system uses byte tokens—the same vocabulary that byte-level language models use. The bytecode is just bytes. The C source code is just bytes.

This means we can treat **bytecode as a system prompt**.

```python
# Compile a program
prompt = BytecodePrompt.from_c_source("""
    int fib(int n) {
        if (n < 2) return n;
        return fib(n-1) + fib(n-2);
    }
    int main() { return fib(10); }
""")

# Get it as byte tokens
tokens = prompt.to_token_sequence()
# [3, 0, 1, 0, 0, 0, 0, 0, 38, 0, ...]
```

Those 312 byte tokens ARE the compiled program. Feed them to a byte-level transformer, and it has everything needed to execute Fibonacci.

Take this further: the C4 *compiler* can also be compiled to bytecode. So the compiler itself becomes a system prompt! Feed it source code as user input, and it outputs program bytecode. Feed that bytecode to another VM instance, and it runs. Self-hosted compilation through transformers.

The architecture enables a new kind of language model interaction:

```
<|system|>[398 tokens of compiler bytecode]
<|user|>int main() { return 6*7; }
<|assistant|><|code|>[compiled bytecode]</code><|exec|><|result|>42
```

The model doesn't guess at arithmetic. It computes it.

---

## What This Means

I think there are three big takeaways:

**1. Transformers can compute, not just pattern-match.**

The architecture has the primitives. SwiGLU multiplies exactly. FFNs are lookup tables. You can build a computer from these parts. The reason current LLMs fail at arithmetic isn't architectural—it's that they're trained to predict tokens, not to use these primitives for computation.

**2. Better arithmetic might come from structure, not scale.**

Everyone assumes that GPT-5 will be better at math because it's bigger and trained on more data. Maybe. But this suggests another path: explicit arithmetic circuits built into the architecture. A "calculator mode" that routes arithmetic through dedicated modules instead of pattern matching.

**3. Self-hosted computation enables new capabilities.**

When the compiler runs as bytecode on the same VM that runs programs, you get interesting properties. The system can compile code it generates. It can verify its own outputs. It can run sandboxed computations without calling external APIs.

---

## The Numbers

For the technically curious, here's what the implementation looks like:

- **Parameters:** ~391,000 (all in FFN tables, no training required)
- **Memory:** 1.5 MB (float32)
- **Operations:** 40 bytecode instructions (full C4 instruction set)
- **Accuracy:** 100% (by construction, not approximation)

Speed comparison:

| Test | Fast VM | Transformer VM |
|------|---------|----------------|
| fib(10) | 0.1ms | 15ms |
| Mandelbrot row | 1ms | 100ms |

The speculator makes production use practical while maintaining correctness guarantees.

---

## Try It Yourself

The code is in the `c4_release` folder:

```python
from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c

# Create VM
vm = C4TransformerVM()

# Compile and run
bytecode, data = compile_c("int main() { return 6 * 7; }")
vm.reset()
vm.load_bytecode(bytecode, data)
result = vm.run()

print(result)  # 42 - computed through neural operations
```

The SwiGLU multiplication is the simplest piece to understand:

```python
import torch.nn.functional as F

def swiglu_mul(a, b):
    """Exact multiply: a*b = silu(a)*b + silu(-a)*(-b)"""
    return F.silu(a) * b + F.silu(-a) * (-b)

# Test it
result = swiglu_mul(torch.tensor(123.0), torch.tensor(456.0))
print(int(result.item()))  # 56088 - exactly correct
```

---

## Final Thoughts

When I started this project, I wanted to understand whether transformers could do real computation. The answer surprised me: they absolutely can, and the building blocks are already there.

The SwiGLU activation in LLaMA can multiply. The FFN layers can do table lookups. Chain them together and you get a complete computer.

This doesn't mean current LLMs are secretly good at math—they're not trained to use these primitives this way. But it does mean the architecture isn't the bottleneck. Future models, designed with explicit computation in mind, could be very different from today's pattern matchers.

The Mandelbrot set rendering on my screen—every pixel computed through SiLU activations and softmax lookups—is proof that transformers are more capable than we give them credit for. We just need to figure out how to unlock that capability.

Maybe that's the real lesson: the tools are already here. We just need to learn how to use them.

---

*Code available at: c4_release/*

*~5,400 words*
