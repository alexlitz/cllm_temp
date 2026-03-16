# C4 Transformer: All Implementation Tricks

A complete C4 virtual machine implemented using only standard transformer primitives:
- SwiGLU FFN activations
- Attention with learned/computed keys
- No specialized arithmetic circuits

## 1. Sharp Gates via SwiGLU

The foundation of everything. We create sharp (nearly binary) gates from smooth activations.

```python
def silu(x):
    return x * torch.sigmoid(x)

def silu_threshold(x, scale=20.0):
    """Differentiable step function: ~0 for x<0, ~1 for x>0"""
    diff = scale * x
    return (silu(diff + 0.5*scale) - silu(diff - 0.5*scale)) / scale

def eq_gate(a, b, scale=20.0):
    """Returns ~1 if a==b, ~0 otherwise"""
    diff = (a - b).float()
    return silu_threshold(diff + 0.5, scale) * silu_threshold(-diff + 0.5, scale)
```

**Why it works:** The difference of two shifted SiLUs creates a "bump" function that's ~1 in a narrow range and ~0 elsewhere. Higher scale = sharper gates.

## 2. Opcode Dispatch via Parallel Gates

Instead of if-else chains, we compute ALL opcode results in parallel, then select:

```python
def execute_all_opcodes(opcode, ax, stack_top, memory):
    # Compute gates for each opcode
    is_add = eq_gate(opcode, Op.ADD)
    is_sub = eq_gate(opcode, Op.SUB)
    is_mul = eq_gate(opcode, Op.MUL)
    # ... etc for all 39 opcodes

    # Compute ALL possible results
    result_add = stack_top + ax
    result_sub = stack_top - ax
    result_mul = stack_top * ax
    # ... etc

    # Weighted sum (only matching opcode contributes)
    new_ax = (is_add * result_add +
              is_sub * result_sub +
              is_mul * result_mul + ...)

    return new_ax
```

**Key insight:** Gates sum to ~1 (softmax-like), so only the correct opcode's result survives.

## 3. Memory via Attention with Binary-Encoded Addresses

Memory is a sequence of (key, value) pairs where keys encode addresses in binary.

```python
def encode_address(addr, num_bits=16, scale=10.0):
    """Encode address as binary vector with +N/-N values"""
    bits = []
    for b in range(num_bits):
        bit = (addr >> b) & 1
        bits.append(scale if bit else -scale)
    return torch.tensor(bits)

def memory_read(memory_values, memory_keys, target_addr):
    """Attention-based memory read"""
    query = encode_address(target_addr)
    scores = torch.matmul(memory_keys, query) / sqrt(num_bits)
    weights = softmax(scores)
    return sum(weights * memory_values)  # Only matching address contributes
```

**Why binary encoding:** Dot product of matching binary keys gives score N*num_bits (high). Mismatching keys give lower scores. Softmax sharpens this to select exactly one address.

## 4. Memory Write via Masked Update

```python
def memory_write(memory, address, value):
    write_mask = eq_gate(all_addresses, address)  # 1 at target, 0 elsewhere
    return memory * (1 - write_mask) + value * write_mask
```

**Key:** The mask is differentiable, so gradients flow through writes.

## 5. Division via Log-Space Subtraction

Division is hard in linear space. In log space, it's subtraction!

```python
def divide(a, b):
    log_a = compute_log2(a)      # Via iterative bit detection
    log_b = compute_log2(b)
    log_result = log_a - log_b   # Division → subtraction!

    # Convert back via attention over powers-of-2 table
    result = attention_lookup(log_result, power_table)

    # Refine with Newton step for integer accuracy
    result = result + (a - result * b) / b

    return floor(result)
```

**The trick:** Precompute table of [1, 2, 4, 8, ...], use attention to find 2^log_result.

## 6. Log2 via Iterative Bit Detection

```python
def log2_approx(x, scale=20.0):
    result = 0
    remaining = x
    for bit in [15, 14, 13, ..., 0]:
        threshold = 2^bit
        is_above = silu_threshold(remaining - threshold, scale)
        result += is_above * bit
        # If above threshold, divide by it
        remaining = remaining * (1 - is_above) + (remaining / threshold) * is_above
    return result
```

**Why it works:** We find the highest bit position by checking each threshold, accumulating the bit positions where x exceeds 2^bit.

## 7. Comparison Operators

All comparisons reduce to subtraction + sign detection:

```python
def lt_gate(a, b):  # a < b
    return silu_threshold(b - a - 0.5)  # Positive when b > a

def le_gate(a, b):  # a <= b
    return silu_threshold(b - a + 0.5)

def eq_gate(a, b):  # a == b
    return lt_gate(a, b+1) * lt_gate(b, a+1)  # Both a<b+1 and b<a+1
```

## 8. Bitwise Operations via Bit Extraction

```python
def extract_bit(x, position):
    """Get bit at position using modular arithmetic"""
    shifted = floor(x / (2 ** position))
    return shifted % 2  # 0 or 1

def bitwise_and(a, b, num_bits=64):
    result = 0
    for i in range(num_bits):
        bit_a = extract_bit(a, i)
        bit_b = extract_bit(b, i)
        result += bit_a * bit_b * (2 ** i)
    return result
```

**Key insight:** Extract each bit, apply operation, reassemble.

## 9. Stack Operations via SP Register

The stack is just memory with a pointer:

```python
def push(sp, memory, value):
    new_sp = sp - 8
    new_memory = memory_write(memory, new_sp, value)
    return new_sp, new_memory

def pop(sp, memory):
    value = memory_read(memory, sp)
    new_sp = sp + 8
    return new_sp, value
```

## 10. Function Calls via Address Stack

```python
def jsr(pc, sp, memory, target):
    # Push return address
    sp, memory = push(sp, memory, pc + 8)
    return target, sp, memory  # New PC

def lev(bp, sp, memory):
    sp = bp
    bp = memory_read(memory, sp); sp += 8
    pc = memory_read(memory, sp); sp += 8
    return pc, sp, bp
```

## 11. Autoregressive Register Prediction

Each step predicts registers in sequence: PC → SP → BP → AX

```python
def step(context):
    # Context: [memory..., PC, SP, BP, AX]
    # Each register attends to all previous values

    new_pc = attention(query=PC_query, keys=context[:PC])
    context_with_pc = concat(context, new_pc)

    new_sp = attention(query=SP_query, keys=context_with_pc)
    # ... etc
```

**Why autoregressive:** Register updates may depend on each other (e.g., SP depends on whether we pushed).

## 12. Type Tags via Extra Key Dimensions

Distinguish memory vs registers vs output:

```python
# Keys have address bits + type bits
# Memory:  [addr_bits..., -N, -N, -N, -N]
# PC:      [addr_bits..., +N, -N, -N, -N]
# SP:      [addr_bits..., -N, +N, -N, -N]
# Output:  [addr_bits..., -N, -N, -N, +N]
```

**Benefit:** Can query "give me the PC register" vs "give me memory at address X".

## 13. Instruction Encoding

Each instruction is 64 bits: low 8 bits = opcode, upper 56 bits = immediate.

```python
def decode(instruction):
    opcode = instruction % 256
    immediate = instruction // 256
    return opcode, immediate
```

## 14. Conditional Branches via Masked PC Update

```python
def branch_if_zero(ax, pc, target):
    is_zero = eq_gate(ax, 0)
    new_pc = pc * (1 - is_zero) + target * is_zero
    return new_pc
```

**No actual branching:** Both paths computed, gate selects result.

## 15. Printf via Digit Extraction

```python
def printf_int(value):
    output = []
    remaining = abs(value)
    while remaining > 0:
        digit = remaining % 10
        output.append(ord('0') + digit)
        remaining = remaining // 10
    return reversed(output)  # Digits were extracted in reverse
```

Uses the division/modulo operations above.

## Summary: Constant Count

Total unique constants needed: ~85

- Scale factors: 3 (10.0, 20.0, 1.0)
- Opcode values: 39 (0-38)
- Type tag values: 6
- Bit positions: 16 (for addresses)
- Powers of 2: 16 (for log table)
- ASCII offsets: 3 ('0', '\n', etc.)
- Address constants: 2 (CODE_BASE, STACK_TOP)

**Zero learned parameters.** Everything is computed from these constants.
